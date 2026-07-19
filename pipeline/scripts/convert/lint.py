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

from index.chunking import strip_frontmatter

LINT_MIN_TEXT_RATIO = 0.5   # doc.md должен сохранить >= половины извлекаемых символов raw

_TABLE_LINE_RE = re.compile(r"^\|.*\|$")
_HEADING_STRIP_RE = re.compile(r"^#{1,6}\s*")
_LIST_QUOTE_STRIP_RE = re.compile(r"^[-*>]\s+")


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
