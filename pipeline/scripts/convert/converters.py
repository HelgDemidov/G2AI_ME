"""Реестр конвертеров raw.* -> doc.md (Strategy). Ключ — расширение raw-файла.

Оркестратор (_do_convert) не знает форматов: resolve_converter(raw) -> Converter.
Новый формат = новая запись в _CONVERTERS (+ свой convert-модуль), ноль правок
оркестратора (чартер convert/architecture.md §3.1).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from convert import eli, ocr_headings
from convert.pdf_to_markdown import convert as pdf_convert
from core import fsio

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


def _was_ocr_normalized(raw: Path) -> bool:
    """PDF уже прошёл OCR-нормализацию РАНЬШЕ (по метаданным ocrmypdf).

    `_ocr_normalize` мутирует `raw` in-place (один файл, не сайдкар) — после первого
    успеха текст-слой уже есть, и `_detect_scan` больше НЕ поднимет `NeedsOCR` на
    повторных конвертациях (`--force`, бамп версии конвертера). Без этой проверки
    `ocr_headings` перестал бы применяться после первого прогона — метаданные ocrmypdf
    (`Creator: ocrmypdf ...`) переживают мутацию текст-слоя и остаются надёжным маркером.
    """
    with pdfplumber.open(raw) as pdf:
        creator = (pdf.metadata.get("Creator") or "").lower()
    return "ocrmypdf" in creator


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


OCR_TIMEOUT = 7200      # 2 ч — потолок для ~200-страничного скана на i5-6200U
OCR_PAGE_WARN = 200     # страниц > порога -> лог оценки времени до запуска
_OCR_STDERR_TAIL = 500  # символов stderr в ConversionError — достаточно для диагноза


def _check_langs_available(langs: str) -> None:
    """tesseract --list-langs -> ConversionError с apt-командой, если traineddata нет.

    Проверяется ДО (потенциально долгого) ocrmypdf — быстрый честный отказ вместо
    невнятной ошибки из недр ocrmypdf/tesseract.
    """
    result = subprocess.run(
        ["tesseract", "--list-langs"], check=False, capture_output=True, text=True
    )
    installed = set(result.stdout.splitlines()[1:])  # первая строка — заголовок списка
    missing = [code for code in langs.split("+") if code not in installed]
    if missing:
        apt_pkgs = " ".join(f"tesseract-ocr-{code.replace('_', '-')}" for code in missing)
        raise ConversionError(f"нет traineddata для {', '.join(missing)} — sudo apt install {apt_pkgs}")


def _ocr_normalize(raw: Path, language: str | None) -> None:
    """OCR-нормализовать скан IN-PLACE: `raw` заменяется версией с невидимым текст-слоем.

    Один PDF-файл на документ, без сайдкара `.ocr.pdf` (раньше кэш жил отдельным файлом —
    двойное хранение того же документа; убрано по решению пользователя). Кэширование
    получается «бесплатно» иначе: после успеха `raw` САМ содержит текст-слой, поэтому
    следующий `_detect_scan(raw)` больше не поднимет `NeedsOCR`, и `_ocr_normalize` не
    вызовется повторно без явного `--force`/бампа версии конвертера. Вызывающий
    (`_do_convert` в run_pipeline.py) ОБЯЗАН пересчитать sha256/размер/mtime в
    `.state.yaml` после конвертации — raw физически изменился.
    """
    if shutil.which("ocrmypdf") is None:
        raise NeedsOCR(
            f"{raw.name}: ocrmypdf не установлен — sudo apt install ocrmypdf "
            f"tesseract-ocr-srp-latn tesseract-ocr-est"
        )

    langs = _tesseract_langs(language)
    _check_langs_available(langs)

    with pdfplumber.open(raw) as pdf:
        n = len(pdf.pages)
    if n > OCR_PAGE_WARN:
        logger.warning(
            "%s: %d страниц — OCR займёт ориентировочно %d–%d мин",
            raw.name, n, n * 20 // 60, n * 40 // 60,
        )

    staging = fsio.staging_path(raw)
    result = subprocess.run(
        [
            "ocrmypdf", "--skip-text", "-l", langs, "--output-type", "pdf", "--quiet",
            str(raw), str(staging),
        ],
        check=False, capture_output=True, text=True, timeout=OCR_TIMEOUT,
    )
    if result.returncode != 0:
        raise ConversionError(
            f"{raw.name}: ocrmypdf завершился с кодом {result.returncode}: "
            f"{result.stderr[-_OCR_STDERR_TAIL:]}"
        )
    staging.replace(raw)


def _convert_pdf(raw: Path, out: Path, language: str | None) -> None:
    try:
        _detect_scan(raw)
        scanned = _was_ocr_normalized(raw)  # текст есть — но, может, уже был нормализован раньше
    except NeedsOCR:
        _ocr_normalize(raw, language)   # мутирует raw IN-PLACE (текст-слой встроен)
        scanned = True
    pdf_convert(str(raw), str(out))  # существующий конвертер, без изменений
    if scanned:  # только OCR-ветка: цифровой путь не трогаем (размер-кластеризация там чище)
        out.write_text(
            ocr_headings.promote_flat_headings(out.read_text(encoding="utf-8")),
            encoding="utf-8",
        )


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
    "pdf": Converter("pdf", "4", _convert_pdf),  # v4: bold-фолбэк заголовков + таблицы (convert-hardening)
    "html": Converter("html", "1", _convert_html),
}


def resolve_converter(raw: Path) -> Converter:
    ext = raw.suffix.lstrip(".").lower()
    conv = _CONVERTERS.get(ext)
    if conv is None:
        known = ", ".join(sorted(_CONVERTERS))
        raise UnsupportedFormat(f"{raw.name}: формат '{ext}' не поддержан (есть: {known})")
    return conv
