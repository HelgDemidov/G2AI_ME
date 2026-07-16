"""Реестр конвертеров raw.* -> doc.md (Strategy). Ключ — расширение raw-файла.

Оркестратор (_do_convert) не знает форматов: resolve_converter(raw) -> Converter.
Новый формат = новая запись в _CONVERTERS (+ свой convert-модуль), ноль правок
оркестратора (чартер convert/architecture.md §3.1).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from convert.pdf_to_markdown import convert as pdf_convert


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
            f"OCR-путь не реализован (см. pipeline/setup/convert-ocr)"
        )


def _convert_pdf(raw: Path, out: Path, language: str | None) -> None:
    _detect_scan(raw)
    pdf_convert(str(raw), str(out))  # существующий конвертер, без изменений


_CONVERTERS: dict[str, Converter] = {
    "pdf": Converter("pdf", "1", _convert_pdf),
}


def resolve_converter(raw: Path) -> Converter:
    ext = raw.suffix.lstrip(".").lower()
    conv = _CONVERTERS.get(ext)
    if conv is None:
        known = ", ".join(sorted(_CONVERTERS))
        raise UnsupportedFormat(f"{raw.name}: формат '{ext}' не поддержан (есть: {known})")
    return conv
