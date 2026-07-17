"""Layout-free восстановление заголовков из плоского OCR-текста (precision-first).

OCR-нормализация (`_ocr_normalize`) кладёт невидимый текст-слой шрифтом GlyphLessFont,
масштабированным под bounding-box каждого слова — считываемый `pdf_to_markdown` размер
шрифта на таком слое шумная геометрическая подгонка, а не типографский кегль исходника,
и детектор заголовков `pdf_to_markdown` реагирует только на размер. Поэтому его
кластеризация на скане ненадёжна, и OCR-документ выходит из pdf_convert плоским.

Этот модуль — отдельный markdown->markdown пост-проход, вызываемый ТОЛЬКО из OCR-ветки
`_convert_pdf` (см. converters.py). Цифровой PDF-путь и `pdf_to_markdown.py` не трогаются:
регекс-по-нумерации для цифровых PDF мы сознательно отвергли (там кластеризация по
размеру чище), а на скане размер ненадёжен и layout-независимые сигналы надёжнее.

Precision-first (решение пользователя, спек convert-ocr v2): калибровочного скана в
корпусе ещё нет, ложные заголовки (особенно из нумерации под-клауз закона) дали бы
структуру ХУЖЕ честного плоского текста. Три тира по убыванию точности — при
неоднозначности строка остаётся телом, а не промоутится вслепую.
"""
from __future__ import annotations

import re

HEADING_MAX_LEN = 80  # длиннее -> скорее предложение тела, чем заголовок
CAPS_MAX_LEN = 60      # Тир 2 короче Тир 1 — CAPS-эвристика менее надёжна, требует строже

_TIER1_LEVEL1 = frozenset({"ANNEX", "CHAPTER", "TITLE", "PART"})
_TIER1_LEVEL2 = frozenset({"SECTION", "ARTICLE", "APPENDIX"})
_TIER1_LEADING_WORD = re.compile(r"^([A-Za-z]+)\b")

_TIER3_NUMBERING = re.compile(r"^(\d+)(\.(\d+))?\.?\s+(.+)$")


def _tier1_heading(stripped: str) -> str | None:
    """ANNEX/CHAPTER/TITLE/PART -> #; SECTION/Article/Appendix -> ##.

    Guard против тела-предложения («Article 6 shall apply to…»): короче HEADING_MAX_LEN
    И не оканчивается точкой/точкой-с-запятой.
    """
    if len(stripped) >= HEADING_MAX_LEN or stripped.endswith((".", ";")):
        return None
    m = _TIER1_LEADING_WORD.match(stripped)
    if not m:
        return None
    word = m.group(1).upper()
    if word in _TIER1_LEVEL1:
        return f"# {stripped}"
    if word in _TIER1_LEVEL2:
        return f"## {stripped}"
    return None


def _tier2_heading(stripped: str, next_line: str) -> str | None:
    """Короткая CAPS-строка + непустое тело следом -> ##.

    Guard против акронимов/обломков таблиц: минимум 2 слова (одинокий «GDPR» —
    не заголовок) И хотя бы одно «настоящее» слово из >=2 букв (не только цифры/символы).
    """
    if not stripped or len(stripped) > CAPS_MAX_LEN or not next_line:
        return None
    if stripped[-1] in ".,;:":
        return None
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 2:
        return None
    if sum(1 for c in letters if c.isupper()) / len(letters) < 0.7:
        return None
    words = stripped.split()
    if len(words) < 2:
        return None  # одно слово (акроним/обломок таблицы) — не заголовок
    if not any(sum(1 for c in w if c.isalpha()) >= 2 for w in words):
        return None  # ни одного «настоящего» слова — только цифры/символы/одиночные буквы
    return f"## {stripped}"


def _tier3_heading(stripped: str, next_line: str) -> str | None:
    """Голая нумерация глубиной <=2 («1.» -> ##, «1.1» -> ###) + тело следом.

    Самый опасный тир: в юридическом корпусе нумерация есть у каждой под-клаузы
    статьи. Guard'ы: тайтл с заглавной, без финальной точки, короче HEADING_MAX_LEN —
    иначе типичная длинная под-клауза («1. The provider shall ensure that…») дала
    бы структуру хуже честного плоского текста.
    """
    if len(stripped) >= HEADING_MAX_LEN or stripped.endswith(".") or not next_line:
        return None
    m = _TIER3_NUMBERING.match(stripped)
    if not m:
        return None
    title = m.group(4)
    if not title or not title[0].isupper():
        return None
    depth = 3 if m.group(2) else 2
    return f"{'#' * depth} {stripped}"


def promote_flat_headings(md: str) -> str:
    """Восстановить высокоуверенный скелет заголовков в плоском OCR-markdown.

    Строки, уже начинающиеся с `#`, не трогаются. При неоднозначности — строка
    остаётся как есть (precision-first). Идемпотентно: промоутнутая строка (уже с
    `#`-префиксом) на повторном прогоне снова пропускается тем же инвариантом.
    """
    lines = md.split("\n")
    n = len(lines)
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        next_line = lines[i + 1].strip() if i + 1 < n else ""
        heading = (
            _tier1_heading(stripped)
            or _tier2_heading(stripped, next_line)
            or _tier3_heading(stripped, next_line)
        )
        out.append(heading if heading is not None else line)
    return "\n".join(out)
