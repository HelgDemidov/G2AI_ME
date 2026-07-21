"""Автоматический QA-проход над результатом конвертации (C1, spec convert-hardening).

Операционализирует заморозку калибровки: ручной аудит каждого документа
заменяется авто-смоком + точечным взглядом только на флаги. Чистые функции,
CI-safe (без сети/модели/pdfplumber) — вход уже готовый ``doc.md``-текст.

``fmt`` в сигнатуре ``lint_conversion`` сейчас не влияет на набор проверок
(text-loss уже управляется через ``raw_text_chars is None`` для html) — задел
на расширение convert-cloud-tier (witness-проверки облачного OCR, применимые
только к ``fmt == "pdf"``); список проверок остаётся плоским и легко
дополняемым (см. гармонизацию convert-cloud-tier §0).
"""
from __future__ import annotations

import re
from collections import Counter

from index.chunking import strip_frontmatter

LINT_MIN_TEXT_RATIO = 0.5   # doc.md должен сохранить >= половины извлекаемых символов raw

_TABLE_LINE_RE = re.compile(r"^\|.*\|$")
_HEADING_STRIP_RE = re.compile(r"^#{1,6}\s*")
_LIST_QUOTE_STRIP_RE = re.compile(r"^[-*>]\s+")

# --- witness-линт (spec convert-cloud-tier §3): свидетель = pdfplumber.extract_text
# нормализованного raw (tesseract-слой), сверяется с ОБЛАЧНЫМ doc.md ---

WITNESS_MIN_TOKEN_RECALL = 0.80   # доля словарных токенов свидетеля, найденных в облачном тексте

_WORD_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # буквенные токены (Unicode — диакритика)
_NUMBER_TOKEN_RE = re.compile(r"\d+")


def _text_length_excluding_markup(body: str) -> int:
    """Длина текста БЕЗ markdown-разметки (заголовки/маркеры списков-цитат/
    таблицы) — сопоставимо с ``raw_text_chars`` (сырой текст raw, без какой-либо
    разметки), иначе символы разметки завышали бы числитель и маскировали
    реальную потерю текста."""
    chars = 0
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        s = _HEADING_STRIP_RE.sub("", s)
        s = _LIST_QUOTE_STRIP_RE.sub("", s)
        s = s.replace("|", "").replace("---", "")
        chars += len(s)
    return chars


def _count_ragged_table_rows(body: str) -> int:
    """Строки-данные таблиц, где число ячеек не совпадает с числом ячеек шапки —
    след дробления/OCR-шума в конвертации таблиц (по блокам, разделённым
    пустой строкой — тот же формат блока, что и ``merge_split_tables``)."""
    ragged = 0
    for block in body.split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(lines) < 3 or not all(_TABLE_LINE_RE.match(ln) for ln in lines):
            continue
        header_cells = lines[0].count("|") - 1
        for row in lines[2:]:  # пропуск шапки и разделителя ``|---|``
            if row.count("|") - 1 != header_cells:
                ragged += 1
    return ragged


def lint_conversion(md_text: str, *, raw_text_chars: int | None, fmt: str) -> list[str]:
    """Список строк-дефектов (пустой = чисто). Никогда не роняет конвертацию —
    только сигнализирует (сомнение ⇒ проза, дефект ⇒ лог/state, см. §6 good-enough).

    Проверки:
    - нет ни одного ``#``-заголовка -> ``"no-headings"``
    - ``raw_text_chars`` задан и отношение длины текста (без разметки) к нему
      ниже ``LINT_MIN_TEXT_RATIO`` -> ``"text-loss: <ratio>"``
    - строки-обломки таблиц (число ячеек данных != числу ячеек шапки) ->
      ``"table-ragged: <n> строк"``
    """
    defects: list[str] = []
    body = strip_frontmatter(md_text)

    if not any(line.startswith("#") for line in body.splitlines()):
        defects.append("no-headings")

    if raw_text_chars is not None and raw_text_chars > 0:
        ratio = _text_length_excluding_markup(body) / raw_text_chars
        if ratio < LINT_MIN_TEXT_RATIO:
            defects.append(f"text-loss: {ratio:.2f}")

    ragged = _count_ragged_table_rows(body)
    if ragged:
        defects.append(f"table-ragged: {ragged} строк")

    return defects


def token_recall(reference: str, candidate: str) -> float:
    """Доля УНИКАЛЬНЫХ буквенных токенов ``reference``, найденных где-либо в
    ``candidate`` (регистронезависимо; юникодные буквы — диакритика — как
    обычные буквы, см. ``_WORD_TOKEN_RE``). ``reference`` без единого буквенного
    токена -> 1.0 (терять нечего, recall тривиально полный) — так свап
    источника местами не меняет семантику: recall всегда «сколько из
    reference нашлось в candidate»."""
    reference_words = {w.lower() for w in _WORD_TOKEN_RE.findall(reference)}
    if not reference_words:
        return 1.0
    candidate_words = {w.lower() for w in _WORD_TOKEN_RE.findall(candidate)}
    return len(reference_words & candidate_words) / len(reference_words)


def _numeric_counter(text: str) -> Counter[str]:
    return Counter(_NUMBER_TOKEN_RE.findall(text))


def numeric_delta(reference: str, candidate: str) -> tuple[int, int]:
    """``(missing, added)`` — мультимножество числовых токенов (``\\d+``):
    вхождения ``reference`` без пары в ``candidate``, и наоборот. Порядок/
    позиция чисел в тексте не учитывается, только счёт по значению."""
    reference_nums, candidate_nums = _numeric_counter(reference), _numeric_counter(candidate)
    missing = sum((reference_nums - candidate_nums).values())
    added = sum((candidate_nums - reference_nums).values())
    return missing, added


_NUMERIC_DIVERGENCE_TOKEN_CAP = 10  # токенов на сторону в строке дефекта — .state.yaml не резиновый


def _format_missing_side(nums: Counter[str], other: Counter[str]) -> str:
    """Сами числа стороны ``nums``, отсутствующие в ``other`` — множество (не
    мультимножество: повтор значения неинформативен в списке «что разошлось»),
    отсортировано численно для детерминизма отчёта, капается на ``_NUMERIC_DIVERGENCE_TOKEN_CAP``."""
    missing = sorted((nums - other).keys(), key=int)
    if not missing:
        return "none"
    shown = missing[:_NUMERIC_DIVERGENCE_TOKEN_CAP]
    rest = len(missing) - len(shown)
    return ",".join(shown) + (f"…+{rest}" if rest > 0 else "")


def witness_checks(witness_text: str, cloud_text: str) -> list[str]:
    """Сверка облачного OCR-вывода с независимым свидетелем (tesseract-текст-слой
    нормализованного raw, spec convert-cloud-tier §3). **Сигнал, не отказ**: живая
    приёмка чекпоинта 1 показала ненулевую дельту у ЗАВЕДОМО корректного облака
    (свидетель сам шумит) — расхождение маркирует «посмотреть глазами» на Стадии 2,
    финальный арбитр — человек, не этот линт.

    - Словарный token-recall (``token_recall``, доля УНИКАЛЬНЫХ буквенных токенов
      свидетеля, найденных где-либо в облачном тексте) ниже ``WITNESS_MIN_TOKEN_RECALL``
      -> ``"cloud-ocr-text-loss: <ratio>"`` — ловит выпавшие/пропущенные куски.
    - Мультимножества числовых токенов расходятся -> ВСЕГДА (любая ненулевая
      дельта) ``"cloud-ocr-numeric-divergence: witness_only=[...] cloud_only=[...]"``
      — перечисляет САМИ расходящиеся числа (не только счётчик), капается на
      ``_NUMERIC_DIVERGENCE_TOKEN_CAP`` на сторону (spec ocr-eval-harness §8.2:
      живой разбор боевого флага без списка токенов занял ~20 мин на одном
      документе — счётчик `-12/+18` не говорит, КТО прав). Самый опасный для
      юридического корпуса класс: тихая подмена цифры в номере статьи/дате/сумме.
    """
    if not witness_text.strip():
        return []  # свидетель пуст (сбой extract_text) — сигнал неинформативен, не 0.0-recall

    defects: list[str] = []
    recall = token_recall(witness_text, cloud_text)
    if recall < WITNESS_MIN_TOKEN_RECALL:
        defects.append(f"cloud-ocr-text-loss: {recall:.2f}")

    witness_nums, cloud_nums = _numeric_counter(witness_text), _numeric_counter(cloud_text)
    if witness_nums != cloud_nums:
        defects.append(
            "cloud-ocr-numeric-divergence: "
            f"witness_only=[{_format_missing_side(witness_nums, cloud_nums)}] "
            f"cloud_only=[{_format_missing_side(cloud_nums, witness_nums)}]"
        )

    return defects
