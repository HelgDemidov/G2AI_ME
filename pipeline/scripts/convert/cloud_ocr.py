"""Облачный OCR сканов (spec convert-cloud-tier §2): рендер страниц -> батчи ->
OpenRouter VLM-транскрипция -> markdown. Witness-слой (tesseract, `_ocr_normalize`)
остаётся ОБЯЗАТЕЛЬНЫМ и не трогается этим модулем — он вызывающая сторона
(``converters._convert_pdf``), не отсюда.

Рендер — через ``pdfplumber.Page.to_image()``/``Page.crop().to_image()`` (обёртка
над pypdfium2, уже транзитивной зависимостью pdfplumber) — не сырой pypdfium2 API:
тот же слой, что весь остальной конвертер уже использует, минус второй прямой
импорт ради идентичного результата.

Эта функция ПРОПАГИРУЕТ исключение при неисправимом отказе батча (после ретраев
``core.openrouter.chat_request``) — решение о локальном фолбэке принимает
вызывающая сторона (``converters._cached_or_call_cloud``); частичный чекпоинт
``.cloudocr.parts.yaml`` остаётся на диске для добора следующим прогоном.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any

import pdfplumber
import yaml

from core import fsio, openrouter

logger = logging.getLogger(__name__)

# --- константы (стартовые значения калибровки пилота — см. спек §2.1: любая
# правка ТОЛЬКО через 🔶-чекпоинт, фиксируется в «Статусе выполнения» спека) ---
DEFAULT_VLM_MODEL = "google/gemini-3-flash-preview"  # победитель пилота ОБЕИХ задач (OCR + фигуры)
ACTIVE_MODEL = DEFAULT_VLM_MODEL  # settable override (--vlm-model, run_pipeline.main())

OCR_RENDER_DPI = 200          # разрешение рендера страницы
OCR_JPEG_QUALITY = 85         # grayscale L; ~250-330 КБ/стр — LTE-бюджет
# Батч ограничен ДВУМЯ потолками — что наступит раньше:
OCR_BATCH_PAGES = 20          # страниц на запрос (потолок по счёту)
OCR_BATCH_MAX_MB = 8          # суммарный base64-объём картинок запроса (потолок по байтам):
# у OpenRouter существует 413 PayloadTooLarge (лимит тела не документирован публично);
# 8 МБ ≈ 1.4× от живьём проверенного пилотом запроса (13 стр. / 5.7 МБ) — консервативно.
OCR_MAX_TOKENS = 24000        # ≈20 стр. × 600 ткн/стр × 2 запаса
OCR_REQUEST: dict[str, Any] = {"reasoning": {"effort": "minimal"}, "temperature": 0}

# Зеркало ключей converters.TESSERACT_LANGS — не импортируется оттуда (разные
# словари назначения: tesseract-код vs человекочитаемое имя для промпта).
CLOUD_LANG_NAMES = {
    "en": "English", "et": "Estonian", "sr": "Serbian (Latin script)",
    "cnr": "Montenegrin (Latin script)", "hr": "Croatian", "bs": "Bosnian (Latin script)",
    "sl": "Slovenian", "sq": "Albanian", "mk": "Macedonian",
    "de": "German", "fr": "French", "it": "Italian", "es": "Spanish",
    "ru": "Russian", "ar": "Arabic", "zh": "Chinese (Simplified)", "ja": "Japanese",
}

# Дословно из пилота (доказано живой построчной сверкой со сканом) — verbatim-
# транскрипция, НЕ исправление/угадывание (принцип: осторожность важнее для
# юридического корпуса, чем косметическая неточность).
OCR_PROMPT = """Transcribe this scanned {lang_name} document to Markdown, VERBATIM.
Rules:
- Transcribe exactly what is printed. Do NOT correct, normalize, translate, or guess.
  If a word is truly illegible, write [illegible].
- Preserve all diacritics exactly as printed.
- Preserve all numbers exactly as printed.
- Use # heading levels reflecting the document's visual structure (document title,
  section headings, article headings and their short title labels if visually present).
- Output ONLY the transcription, no commentary."""

_OUTLINE_PREAMBLE = (
    "\n\nContinuation of the same document. Heading outline so far "
    "(keep the SAME depth conventions):\n"
)


def _lang_name(language: str | None) -> str:
    if language is None:
        return "English"
    name = CLOUD_LANG_NAMES.get(language)
    if name is None:
        logger.warning("неизвестный языковой код %r для облачного OCR — 'the source language'", language)
        return "the source language"
    return name


def _render_page(page: Any) -> tuple[str, int]:
    """Одна страница -> (data-URI JPEG, вес base64 в байтах). PIL-изображение
    отбрасывается сразу после кодирования — дисциплина 8 ГБ (не копится по
    всему документу в декомпрессированном виде)."""
    img = page.to_image(resolution=OCR_RENDER_DPI).original.convert("L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=OCR_JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", len(b64)


def _plan_batches(rendered: list[tuple[str, int]]) -> list[tuple[int, int, list[str]]]:
    """Нарезка 1-based списка отрендеренных страниц на батчи по двум потолкам
    §2.1 (страницы И байты — что раньше). Чистая функция над уже готовыми
    (data_uri, byte_len) — тестируется синтетическими весами без PDF."""
    batches: list[tuple[int, int, list[str]]] = []
    current: list[str] = []
    current_bytes = 0
    start = 1
    max_bytes = OCR_BATCH_MAX_MB * 1024 * 1024
    for page_num, (data_uri, byte_len) in enumerate(rendered, start=1):
        would_exceed_pages = len(current) >= OCR_BATCH_PAGES
        would_exceed_bytes = bool(current) and (current_bytes + byte_len) > max_bytes
        if current and (would_exceed_pages or would_exceed_bytes):
            batches.append((start, start + len(current) - 1, current))
            current, current_bytes = [], 0
            start = page_num
        current.append(data_uri)
        current_bytes += byte_len
    if current:
        batches.append((start, start + len(current) - 1, current))
    return batches


def _ordered_texts(parts: dict[str, str]) -> list[str]:
    """Тексты батчей в порядке страниц — числовая сортировка по началу диапазона
    ("100-118" ПОСЛЕ "19-36", лексикографическая сортировка их бы перепутала)."""
    return [parts[k] for k in sorted(parts, key=lambda k: int(k.split("-")[0]))]


def _parts_path(raw: Path) -> Path:
    return raw.parent / ".cloudocr.parts.yaml"


def _header(model: str, raw_sha256: str) -> dict[str, Any]:
    return {"model": model, "raw_sha256": raw_sha256, "dpi": OCR_RENDER_DPI, "quality": OCR_JPEG_QUALITY}


def _load_parts(raw: Path, *, model: str, raw_sha256: str) -> dict[str, str]:
    """Прочитать чекпоинт батчей; несовпадение заголовка (модель/sha/dpi/качество
    изменились) -> частичный кэш отбрасывается ЦЕЛИКОМ (§2.3: куски несовместимы)."""
    path = _parts_path(raw)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict) or data.get("header") != _header(model, raw_sha256):
        return {}
    parts = data.get("parts", {})
    return parts if isinstance(parts, dict) else {}


def _save_parts(raw: Path, *, model: str, raw_sha256: str, parts: dict[str, str]) -> None:
    payload = {"header": _header(model, raw_sha256), "parts": parts}
    fsio.atomic_write_text(_parts_path(raw), yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def _finalize_parts(raw: Path) -> None:
    _parts_path(raw).unlink(missing_ok=True)


def _build_payload(model: str, prompt: str, data_uris: list[str]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": uri}} for uri in data_uris)
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": OCR_MAX_TOKENS,
        **OCR_REQUEST,
    }


def convert_scan(raw: Path, language: str | None, *, model: str) -> str:
    """Облачный OCR скана: рендер -> батчи (§2.1) -> запросы с outline-контекстом
    для батча N>1 (§2.3) -> чекпоинт после КАЖДОГО батча -> сборка. Пропагирует
    исключение при отказе батча — вызывающая сторона решает о фолбэке."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("нет OPENROUTER_API_KEY (см. .env / .env.example)")

    raw_sha256 = fsio.sha256_file(raw)
    parts = _load_parts(raw, model=model, raw_sha256=raw_sha256)
    prompt_base = OCR_PROMPT.format(lang_name=_lang_name(language))

    with pdfplumber.open(raw) as pdf:
        rendered = [_render_page(p) for p in pdf.pages]
    batches = _plan_batches(rendered)

    for start, end, data_uris in batches:
        key = f"{start}-{end}"
        if key in parts:
            continue  # уже добыт предыдущим (прерванным) прогоном — сеть не трогаем
        prompt = prompt_base
        if start > 1:
            outline = [ln for text in _ordered_texts(parts) for ln in text.splitlines() if ln.startswith("#")]
            if outline:
                prompt += _OUTLINE_PREAMBLE + "\n".join(outline)
        response = openrouter.chat_request(_build_payload(model, prompt, data_uris), api_key=api_key)
        parts[key] = response["choices"][0]["message"]["content"]
        _save_parts(raw, model=model, raw_sha256=raw_sha256, parts=parts)  # чекпоинт СРАЗУ после батча

    full_text = "\n\n".join(_ordered_texts(parts))
    _finalize_parts(raw)
    return full_text
