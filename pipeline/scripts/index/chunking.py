"""Разбиение текста документа на канонические чанки ~512 токенов для поиска.

Чанки КАНОНИЧНЫ: одни и те же чанки индексируются и в FTS5, и в векторном слое,
поэтому попадание по ключевому слову и семантическое попадание ссылаются на один chunk.

Логика чанковки НЕ зависит от конкретного токенизатора — функция подсчёта токенов
инжектируется (``count_tokens``). В рантайме передаётся токенизатор bge-m3
(см. bge_tokenizer.py); в тестах — простой счётчик слов. Так логику можно
проверять в CI без модели.

Секционирование (spec analyze-retrieval §1): текст режется на секции по строкам-
заголовкам markdown ДО packing-логики — чанк никогда не пересекает границу секции,
у каждого чанка есть breadcrumb (цепочка заголовков-предков).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from core import markers, schema

TokenCounter = Callable[[str], int]

strip_frontmatter = schema.strip_frontmatter
"""Реэкспорт (spec convert-knowledge-seam-hardening §6): грамматика frontmatter живёт
в ``core.schema`` РЯДОМ с порождающей её ``render_frontmatter``; имя остаётся здесь для
существующих потребителей (``corpus_index``/``run_pipeline``), как ``store.state_dir``
после переезда в schema (knowledge-hardening §2)."""

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])\s*|\n")
# Вторая альтернатива — CJK-пунктуация (knowledge-hardening §8): в пробельных
# языках граница предложения требует \s+ ПОСЛЕ знака, но CJK-текст пробелов
# между предложениями не ставит вовсе — с \s+ эта форма никогда не совпадала бы,
# и весь абзац оставался бы одним «предложением» (живой замер: 1140-токенный
# CJK-абзац -> один чанк 2.2× бюджета). \s* (не \s+) даёт пустое совпадение сразу
# после знака — разрез происходит независимо от наличия пробела.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class Chunk:
    """Канонический чанк: принадлежит документу doc_id, порядковый index.

    ``breadcrumb`` — цепочка заголовков-предков (`"H1 › H2 › H3"`), "" для текста
    до первого заголовка или документа без заголовков вовсе.

    ``reconstruction`` — чанк несёт машинную реконструкцию (VLM-описание фигуры), а не
    verbatim-текст издателя (spec convert-knowledge-seam-hardening §2). Смешанный чанк
    считается реконструированным: занизить доверие честнее, чем завысить.
    """

    doc_id: str
    index: int
    text: str
    n_tokens: int
    breadcrumb: str = ""
    reconstruction: bool = False


def _paragraphs(text: str) -> list[str]:
    """Абзацы текста; fenced-блок (```...```) — ОДИН атомарный абзац.

    Без фенс-осознанности пустые строки ВНУТРИ code-фенса рвали блок на несколько
    «абзацев», и packing мог развести половинки по разным чанкам (живой дефект
    приёмки convert-cloud-tier чекпоинт 2: mermaid-блок VLM-фигуры sg p.6 разрезан
    посередине — открытый вопрос спека подтвердился). Бонус: строка `# comment`
    внутри фенса больше не может быть принята _sections за markdown-заголовок.
    Фенс крупнее max_tokens по-прежнему деградирует в нарезку (_split_long_paragraph)
    — целостность гарантируется только в пределах бюджета чанка."""
    out: list[str] = []
    fence: list[str] | None = None
    plain: list[str] = []

    def flush_plain() -> None:
        nonlocal plain
        if plain:
            block = "\n".join(plain)
            out.extend(p.strip() for p in _PARA_RE.split(block) if p.strip())
            plain = []

    for line in text.split("\n"):
        if fence is None and line.lstrip().startswith("```"):
            flush_plain()
            fence = [line]
        elif fence is not None:
            fence.append(line)
            if line.strip().startswith("```"):
                out.append("\n".join(fence).strip())
                fence = None
        else:
            plain.append(line)
    if fence is not None:
        out.append("\n".join(fence).strip())  # незакрытый фенс — честно до конца текста
    flush_plain()
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _tag_reconstruction(paras: list[str]) -> list[tuple[str, bool]]:
    """Пометить абзацы, принадлежащие блоку VLM-инъекции (spec §2).

    Границы читаются ИЗ САМОГО ТЕКСТА (``core.markers``), а не восстанавливаются по
    кэшу ``.figures.yaml``: сайдкар принадлежит слою convert, и лезть в него из индекса
    значило бы завести ту самую межслойную связь, которую спек снимает, — плюс матчинг
    по подстроке хрупок (текст чанка — нормализованная проекция, Б21).

    Легаси-блок БЕЗ терминатора (инъекции до этого спека, живут до первой реконверсии)
    закрывается на ближайшем заголовке: пометить до конца документа было бы
    неоправданно широко, а не пометить вовсе — вернуть исходный дефект.
    """
    out: list[tuple[str, bool]] = []
    inside = False
    for para in paras:
        first = para.split("\n", 1)[0]
        if markers.is_injection_open(first):
            inside = True
            out.append((para, True))
        elif markers.is_injection_end(first):
            inside = False
            out.append((para, True))
        elif _HEADING_RE.match(first.strip()):
            inside = False  # страховка для легаси-блоков без терминатора
            out.append((para, False))
        else:
            out.append((para, inside))
    return out


def _sections(text: str) -> list[tuple[str, list[tuple[str, bool]]]]:
    """Разбить абзацы текста на секции по строкам-заголовкам.

    Заголовок = абзац, чья ПЕРВАЯ строка матчит ``^#{1,6}\\s+.*$`` (pdf_to_markdown
    эмитит заголовок отдельным абзацем). Стек заголовков: уровень N сбрасывает все
    уровни >= N, затем пишет себя; breadcrumb секции = "› "-цепочка стека. Строка
    заголовка остаётся первым абзацем своей секции (искомый контент, не только
    метаданные). Документ без заголовков -> одна секция с breadcrumb "".

    Абзацы идут парами ``(текст, реконструкция)`` — флаг едет рядом до самого чанка,
    не влияя на packing (границы чанков байт-в-байт те же, что до §2).
    """
    sections: list[tuple[str, list[tuple[str, bool]]]] = []
    stack: list[tuple[int, str]] = []
    breadcrumb = ""
    current: list[tuple[str, bool]] = []
    for para, reconstructed in _tag_reconstruction(_paragraphs(text)):
        m = _HEADING_RE.match(para.splitlines()[0])
        if m:
            if current:
                sections.append((breadcrumb, current))
            level, title = len(m.group(1)), m.group(2).strip()
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, title))
            breadcrumb = " › ".join(t for _, t in stack)
            current = [(para, reconstructed)]
        else:
            current.append((para, reconstructed))
    if current:
        sections.append((breadcrumb, current))
    return sections


def _split_by_chars(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Разрез БЕСПРОБЕЛЬНОЙ строки по символам — фолбэк ``_hard_split`` для текста
    без словных границ (knowledge-hardening §8): сплошной CJK-абзац токенизатор
    ``str.split()`` видит как ОДНО «слово» размером с абзац, и без этого фолбэка
    оно принималось бы целиком поверх лимита (эмбеддер молча усекал бы чанк —
    подтверждено живым запуском, 1140-токенный абзац -> один чанк 2.2× бюджета).

    Тот же приём, что у словного пути выше: дёшевая посимвольная сумма + бинарный
    поиск точки разреза при расхождении оценки. Разрез по индексу Python ``str``
    НИКОГДА не рвёт кодовую точку посередине (в отличие от UTF-16, Python хранит
    строки как последовательность code point'ов, не суррогатных пар) — граница
    чанка может лишь разделить пару «базовый символ + комбинирующийся диакритик»,
    что не создаёт невалидный текст, только менее удачную визуальную границу.
    """
    if not text:
        return []
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        j, budget = i, 0
        while j < n:
            c_tokens = count_tokens(text[j])
            if budget + c_tokens > max_tokens and j > i:
                break
            budget += c_tokens
            j += 1
        candidate = text[i:j]
        if j == i + 1 or count_tokens(candidate) <= max_tokens:
            out.append(candidate)
            i = j
            continue
        lo, hi, best = i + 1, j, i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(text[i:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        out.append(text[i:best])
        i = best
    return out


def _hard_split(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Жёсткая нарезка по словам для аномально длинных предложений (типовой выход
    pdf_to_markdown на дампах диаграмм/таблиц без пунктуации).

    Бюджет накапливается ДЕШЁВОЙ суммой per-word ``count_tokens(word)`` — O(1)
    работы на слово, а не повторной токенизацией всей растущей строки на каждом
    добавленном слове (O(n²) символов: предложение на 5000 слов давало ~12М
    токенизированных "слово-эквивалентов" вместо 5К). Когда оценка превышает
    ``max_tokens`` — ОДНА проверка ``count_tokens`` на полном кандидате; если она
    тоже превышает (subword-склейка на границах слов может дать больше суммы по
    словам, чем реальный совместный счёт) — бинарный поиск точки разреза (O(log n)
    энкодов). Свойство «чанк <= max_tokens» — точное (финальная верификация),
    меняется только СТОИМОСТЬ его достижения.

    Единственное «слово» (нет пробелов вовсе — беспробельный язык, напр. сплошной
    CJK-текст) больше лимита — фолбэк ``_split_by_chars`` (knowledge-hardening §8),
    а не молчаливое принятие поверх бюджета: без него граница чанка была видна
    только эмбеддеру, который её просто тихо усекал.
    """
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    i, n = 0, len(words)
    while i < n:
        j, budget = i, 0
        while j < n:
            w_tokens = count_tokens(words[j])
            if budget + w_tokens > max_tokens and j > i:
                break
            budget += w_tokens
            j += 1
        candidate = " ".join(words[i:j])
        if j == i + 1:
            if count_tokens(candidate) <= max_tokens:
                out.append(candidate)
            else:
                out.extend(_split_by_chars(candidate, count_tokens, max_tokens))
            i = j
            continue
        if count_tokens(candidate) <= max_tokens:
            # честная проверка подтвердила оценку — принимаем как есть
            out.append(candidate)
            i = j
            continue
        # оценка соврала (subword-склейка) -> бинарный поиск точки разреза внутри [i, j)
        lo, hi, best = i + 1, j, i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(" ".join(words[i:mid])) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        out.append(" ".join(words[i:best]))
        i = best
    return out


def _table_header(para: str) -> tuple[str, str, list[str]] | None:
    """Если ``para`` — GFM-таблица (строка `|...|` + строка-разделитель `|---|---|`
    следом), вернуть ``(header, separator, data_rows)``; иначе ``None`` (вызывающая
    сторона деградирует в generic sentence-split). Все конвертеры проекта (PDF
    ``pdf_graphics.py``, DOCX через ``markdownify``) эмитят именно эту форму —
    проверено на реальных `doc.md` корпуса, не только на синтетике."""
    lines = para.split("\n")
    if len(lines) < 3:
        return None
    header, sep = lines[0], lines[1]
    if not (_TABLE_ROW_RE.match(header) and _TABLE_SEP_RE.match(sep)):
        return None
    return header, sep, lines[2:]


def _split_table_paragraph(
    header: str,
    sep: str,
    rows: list[str],
    count_tokens: TokenCounter,
    max_tokens: int,
    first_budget: int | None = None,
) -> list[str]:
    """GFM-таблица больше лимита -> резать ПО СТРОКАМ, повторяя заголовок+
    разделитель в каждом куске (row+header — практика RAG для табличных данных:
    каждый чанк остаётся самодостаточной валидной таблицей). Раньше сюда попадал
    ``_split_long_paragraph`` (режет по ``\\n`` как «предложения», затем склеивает
    результат ЧЕРЕЗ ПРОБЕЛ) — валидный markdown таблицы ломался (строки съезжались
    в одну без переносов). Одна строка данных крупнее лимита сама по себе (редкий
    вырожденный случай — аномально широкая строка) — оставляется цельной поверх
    бюджета, тот же принцип, что у ``_hard_split`` для оверсайз-предложений:
    целостность строки важнее строгого лимита."""
    prefix = f"{header}\n{sep}"
    prefix_tokens = count_tokens(prefix)
    out: list[str] = []
    current: list[str] = []
    current_tokens = prefix_tokens
    budget = first_budget if first_budget is not None else max_tokens
    for row in rows:
        n = count_tokens(row)
        if current and current_tokens + n > budget:
            out.append("\n".join([prefix, *current]))
            current, current_tokens = [], prefix_tokens
            budget = max_tokens  # уменьшённый бюджет — только у ПЕРВОГО куска
        current.append(row)
        current_tokens += n
    if current:
        out.append("\n".join([prefix, *current]))
    return out or [prefix]  # таблица без строк данных (вырожденный случай) — как есть


def _split_long_paragraph(
    para: str, count_tokens: TokenCounter, max_tokens: int, first_budget: int | None = None
) -> list[str]:
    """Абзац больше лимита -> нарезать по предложениям (с fallback на слова);
    GFM-таблица — по строкам с повторением заголовка (см. ``_split_table_paragraph``
    выше).

    ``first_budget`` (spec convert-knowledge-seam-hardening §10) — уменьшённый бюджет
    ПЕРВОГО куска: он поедет в один чанк со строкой-заголовком секции, чтобы та не
    осталась чанком-сиротой. Остальные куски получают полный ``max_tokens``."""
    table = _table_header(para)
    if table is not None:
        return _split_table_paragraph(*table, count_tokens, max_tokens, first_budget)
    out: list[str] = []
    current: list[str] = []
    current_tokens = 0
    budget = first_budget if first_budget is not None else max_tokens
    for sent in _sentences(para):
        n = count_tokens(sent)
        if n > max_tokens:
            if current:
                out.append(" ".join(current))
                current, current_tokens = [], 0
            out.extend(_hard_split(sent, count_tokens, max_tokens))
            budget = max_tokens
            continue
        if current and current_tokens + n > budget:
            out.append(" ".join(current))
            current, current_tokens = [], 0
            budget = max_tokens
        current.append(sent)
        current_tokens += n
    if current:
        out.append(" ".join(current))
    return out


def _merge_heading_with_first_piece(
    current: list[tuple[str, bool]],
    current_tokens: int,
    para: str,
    reconstructed: bool,
    count_tokens: TokenCounter,
    max_tokens: int,
) -> tuple[tuple[str, bool], list[str]] | None:
    """Приклеить строку-заголовок секции к первому куску следующего (оверсайз) абзаца —
    иначе она осталась бы чанком-сиротой (spec convert-knowledge-seam-hardening §10,
    Б15: секция «заголовок + длинная таблица» давала отдельный чанк из одной строки,
    полноправного кандидата выдачи и отдельный вектор без содержания).

    ``None`` — приклеивать нечего или невыгодно: накопленный чанк не является ровно
    строкой-заголовком, запаса бюджета нет, либо склейка всё равно не влезла в
    ``max_tokens`` (post-check — гарантия инварианта независимо от внутренностей
    сплиттера). Во всех этих случаях вызывающая сторона идёт прежним путём.

    ⚠ НЕ трогает секцию, состоящую ТОЛЬКО из заголовка (следом сразу другой заголовок):
    там сироту убрать нечем — контента у секции нет вовсе, а склейка через границу
    секции нарушила бы инвариант «чанк не пересекает границу секции». Живой замер
    корпуса: этот подслучай и есть основная масса коротких чанков (см. спек §10).
    """
    if len(current) != 1:
        return None
    head_text, head_flag = current[0]
    if not _HEADING_RE.match(head_text.splitlines()[0]):
        return None
    first_budget = max_tokens - current_tokens
    if first_budget <= 0:
        return None
    pieces = _split_long_paragraph(para, count_tokens, max_tokens, first_budget)
    if not pieces:
        return None
    merged_text = f"{head_text}\n\n{pieces[0]}"
    if count_tokens(merged_text) > max_tokens:
        return None
    return (merged_text, head_flag or reconstructed), pieces[1:]


def _pack_paragraphs(
    paras: list[tuple[str, bool]], count_tokens: TokenCounter, max_tokens: int
) -> list[tuple[str, bool]]:
    """Упаковать абзацы ОДНОЙ секции в чанки <= max_tokens, стараясь не резать
    абзацы/предложения (packing-логика, вынесена из ``chunk_text`` для секционирования).

    Флаг реконструкции (spec §2) едет РЯДОМ с текстом и в решения упаковки не входит:
    сам текст чанка собирается ровно теми же операциями над теми же строками, что до
    введения флага, поэтому границы чанков (а значит ``content_hash`` и векторы)
    побитово те же. Чанк реконструирован, если реконструирован ЛЮБОЙ его абзац;
    разрезанный оверсайз-абзац передаёт свой флаг всем кускам.
    """
    raw: list[tuple[str, bool]] = []
    current: list[tuple[str, bool]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            raw.append(("\n\n".join(text for text, _ in current), any(flag for _, flag in current)))
            current, current_tokens = [], 0

    for para, reconstructed in paras:
        n = count_tokens(para)
        if n > max_tokens:
            merged = _merge_heading_with_first_piece(
                current, current_tokens, para, reconstructed, count_tokens, max_tokens
            )
            if merged is not None:
                head_chunk, pieces = merged
                raw.append(head_chunk)
                current, current_tokens = [], 0
                raw.extend((piece, reconstructed) for piece in pieces)
                continue
            flush()
            raw.extend(
                (piece, reconstructed)
                for piece in _split_long_paragraph(para, count_tokens, max_tokens)
            )
            continue
        if current and current_tokens + n > max_tokens:
            flush()
        current.append((para, reconstructed))
        current_tokens += n
    flush()
    return raw


def chunk_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int = 512,
    doc_id: str = "",
) -> list[Chunk]:
    """Разбить текст на чанки <= max_tokens, стараясь не резать абзацы/предложения.

    Секционируется ПЕРЕД packing (см. ``_sections``): чанк никогда не пересекает
    границу секции, у каждого чанка breadcrumb секции, к которой он принадлежит.
    """
    raw: list[tuple[str, str, bool]] = [
        (breadcrumb, chunk, reconstructed)
        for breadcrumb, paras in _sections(text)
        for chunk, reconstructed in _pack_paragraphs(paras, count_tokens, max_tokens)
    ]
    # count_tokens(chunk) здесь пересчитывает готовый текст ЗАНОВО, хотя суммы уже
    # накапливались по пути (_pack_paragraphs/_split_long_paragraph) — избыточно, но
    # НЕ квадратично (один линейный проход по уже собранным чанкам, не по n²
    # растущих префиксов, как было в _hard_split). Оставлено как есть: чтобы нести
    # накопленное значение через flush()/_split_long_paragraph/_hard_split, все три
    # должны были бы возвращать (text, n_tokens) вместо str — не стоит сложности
    # ради устранения уже-линейной работы (см. spec code-consolidation §5).
    return [
        Chunk(doc_id, i, chunk, count_tokens(chunk), breadcrumb, reconstructed)
        for i, (breadcrumb, chunk, reconstructed) in enumerate(raw)
    ]


_MARKUP_LINE_RE = re.compile(r"^\s*(\||```|> \[)")
MARKUP_HEAVY_RATIO = 0.6
"""Доля строк-разметки, выше которой чанк считается markup-heavy (spec §2, Б18).
Живой замер аудита: 172 из 768 чанков корпуса (22%) — таблицы/фенсы/маркеры. Это не
дефект (табличный контент легитимен), но и не проза: счётчик делает класс видимым, а
будущий table-retrieval — измеримым."""


def is_markup_heavy(text: str) -> bool:
    """Чанк состоит преимущественно из разметки (строки таблиц, код-фенсы, маркеры),
    а не из прозы — наблюдаемость формы чанков, не фильтр.

    Содержимое код-фенса считается разметкой ЦЕЛИКОМ: тело ```mermaid — исходник
    диаграммы, прозой он не является ни в каком прочтении (иначе фенс из четырёх строк
    давал бы ровно 50% и не проходил порог из-за собственных ограничителей)."""
    markup = total = 0
    in_fence = False
    for line in text.split("\n"):
        if not line.strip():
            continue
        total += 1
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            markup += 1
            continue
        if in_fence or _MARKUP_LINE_RE.match(line):
            markup += 1
    if not total:
        return False
    return markup / total > MARKUP_HEAVY_RATIO


def embed_input(breadcrumb: str, text: str) -> str:
    """Текст, который видит эмбеддер: breadcrumb-контекст + тело чанка."""
    return f"{breadcrumb}\n{text}" if breadcrumb else text
