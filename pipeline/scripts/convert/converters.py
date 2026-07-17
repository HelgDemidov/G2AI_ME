"""Реестр конвертеров raw.* -> doc.md (Strategy). Ключ — расширение raw-файла.

Оркестратор (_do_convert) не знает форматов: resolve_converter(raw) -> Converter.
Новый формат = новая запись в _CONVERTERS (+ свой convert-модуль), ноль правок
оркестратора (чартер convert/architecture.md §3.1).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from convert import eli
from convert.pdf_to_markdown import convert as pdf_convert

logger = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    """Базовый типизированный отказ конвертации (per-doc изоляция ловит как обычно)."""


class UnsupportedFormat(ConversionError):
    """Расширение raw.* не имеет зарегистрированного конвертера."""


class NeedsOCR(ConversionError):
    """PDF без текстового слоя — нужен OCR-путь (спек convert-ocr)."""


@dataclass(frozen=True)
class Converter:
    name: str      # стабильный id пути конвертации ("pdf")
    version: str   # бамп => авто-реконверсия всех документов формата (needed_stages)
    convert: Callable[[Path, Path, str | None], None]  # (raw, out, language)


SCAN_MIN_CHARS_PER_PAGE = 50      # страница «с текстом», если извлечено >= стольких символов
SCAN_MIN_TEXTPAGE_FRACTION = 0.5  # доля страниц с текстом ниже порога => скан


def _detect_scan(raw: Path) -> None:
    """Поднять NeedsOCR, если у большинства страниц нет текст-слоя.

    Отдельный дешёвый проход pdfplumber ДО pdf_convert: полный конвертер на скане
    дал бы пустой вывод с враньём «пустой файл» — диагноз должен называть причину
    (агенда §4: «явный флаг „нужен OCR“»). Двойной парс PDF — секунды, конвертация
    редка (раз на документ).
    """
    with pdfplumber.open(raw) as pdf:
        n = len(pdf.pages)
        if n == 0:
            return  # пустой PDF диагностирует pdf_convert («PDF без страниц»)
        with_text = sum(
            1 for p in pdf.pages
            if len((p.extract_text() or "").strip()) >= SCAN_MIN_CHARS_PER_PAGE
        )
    if with_text / n < SCAN_MIN_TEXTPAGE_FRACTION:
        raise NeedsOCR(
            f"{raw.name}: текст-слой лишь на {with_text}/{n} страниц — вероятен скан; "
            f"OCR-путь не реализован (см. docs/pipeline/convert/tech_specs/convert-ocr)"
        )


# rec.language (schema.py: ISO 639-1, либо 639-3 где нет 639-1, напр. cnr) -> tesseract langcode.
TESSERACT_LANGS = {
    "en": "eng", "et": "est", "sr": "srp_latn", "cnr": "srp_latn",
    "hr": "hrv", "bs": "bos", "sl": "slv", "sq": "sqi", "mk": "mkd",
    "de": "deu", "fr": "fra", "it": "ita", "es": "spa",
    "ru": "rus", "ar": "ara", "zh": "chi_sim", "ja": "jpn",
    # zh по умолчанию упрощённый (материк); традиционный (HK/TW) — chi_tra, добавить при нужде.
}
# CJK/арабский — БЕЗ +eng: удваивает проход и иногда интерферирует (иные скрипты).
_NO_ENG_SUFFIX = frozenset({"chi_sim", "chi_tra", "jpn", "ara"})


def _tesseract_langs(language: str | None) -> str:
    """rec.language -> tesseract -l аргумент. Латиница получает +eng (гос-документы часто
    со вставками EN); CJK/арабский — нет (см. _NO_ENG_SUFFIX). Неизвестный код -> честный
    eng-fallback с предупреждением (не молчаливая порча качества)."""
    mapped = TESSERACT_LANGS.get(language or "en")
    if mapped is None:
        logger.warning("неизвестный языковой код %r для OCR — используется eng", language)
        mapped = "eng"
    if mapped == "eng" or mapped in _NO_ENG_SUFFIX:
        return mapped
    return f"{mapped}+eng"


def _convert_pdf(raw: Path, out: Path, language: str | None) -> None:
    _detect_scan(raw)
    pdf_convert(str(raw), str(out))  # существующий конвертер, без изменений


def _convert_html(raw: Path, out: Path, language: str | None) -> None:
    import trafilatura  # ленивый импорт: pdf-путь не платит за html-зависимость

    html = eli.promote_eli_headings(raw.read_bytes())  # ELI (EUR-Lex/CELLAR) -> <hN>, иначе no-op
    text = trafilatura.extract(
        html,                            # bytes: charset определяет trafilatura
        output_format="markdown",
        include_tables=True,
        include_links=False,            # URL-хвосты — шум для чанков/эмбеддера
        include_images=False,
        favor_recall=True,              # гос-страницы: лучше лишний блок, чем потерянная статья
        with_metadata=False,            # frontmatter — производная meta.yaml, не trafilatura
    )
    if not text or not text.strip():
        raise ConversionError(f"{raw.name}: trafilatura не извлекла контента")
    out.write_text(text, encoding="utf-8")


_CONVERTERS: dict[str, Converter] = {
    "pdf": Converter("pdf", "2", _convert_pdf),  # v2: графика-пасс (spec convert-graphics)
    "html": Converter("html", "1", _convert_html),
}


def resolve_converter(raw: Path) -> Converter:
    ext = raw.suffix.lstrip(".").lower()
    conv = _CONVERTERS.get(ext)
    if conv is None:
        known = ", ".join(sorted(_CONVERTERS))
        raise UnsupportedFormat(f"{raw.name}: формат '{ext}' не поддержан (есть: {known})")
    return conv
