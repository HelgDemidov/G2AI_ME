"""VLM-пасс фигур (spec convert-cloud-tier §5): отдельная идемпотентная стадия
поверх УЖЕ сконвертированного doc.md — конвертеры (pdf_to_markdown) продолжают
честно маркировать нереконструированную графику (``> [Figure, ... — structure
not reconstructed]``/``> [Image, ... — raster content not analyzed]``), этот
модуль сканирует doc.md на такие ГОЛЫЕ маркеры, пере-детектирует регион ТЕМ ЖЕ
детерминированным кодом (``pdf_to_markdown.compute_page_graphics``), рендерит
пиксели по требованию (только на cache-miss), вызывает VLM, кэширует ответ в
``.figures.yaml`` и детерминированно инъецирует его в doc.md.

Идемпотентность БЕЗ отдельного механизма "уже обработано": инъецированный блок
несёт ДРУГУЮ грамматику маркера («VLM interpretation», не «structure not
reconstructed»/«raster content not analyzed») — маркерные регексы этого модуля
её попросту не находят, повторный прогон на уже инъецированном тексте не видит
совпадений и возвращает False (файл не тронут, байт-в-байт). Реконверсия
регенерирует ГОЛЫЕ маркеры заново — следующий прогон этого пасса ре-инъецирует
их ИЗ КЭША офлайн (``region_id`` самоописателен: содержит текст/bbox региона,
изменившийся raw даёт другой id — устаревшая запись кэша просто не находится).
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import io
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import pdfplumber
import yaml

from convert import pdf_graphics, pdf_to_markdown
from core import fsio, openrouter

logger = logging.getLogger(__name__)

# Скан-грамматика: ТОЧНОЕ зеркало pdf_graphics._render_opaque/render_raster_marker.
_FIGURE_MARKER_RE = re.compile(
    r"^> \[Figure, p\. (?P<page>\d+), region (?P<id>[0-9a-f]{12}) — structure not reconstructed\]\n"
    r"> Labels \(reading order not guaranteed\): .*$",
    re.MULTILINE,
)
_IMAGE_MARKER_RE = re.compile(
    r"^> \[Image, p\. (?P<page>\d+), image (?P<id>[0-9a-f]{12}) — raster content not analyzed\]$",
    re.MULTILINE,
)
# docx (spec convert-docx §2-bis): зеркало converters._docx_image_markers — БЕЗ
# номера страницы (docx reflowable, надёжного понятия страницы нет).
_DOCX_IMAGE_MARKER_RE = re.compile(
    r"^> \[Image, docx media (?P<id>[0-9a-f]{12}) — raster content not analyzed\]$",
    re.MULTILINE,
)

# Мимо-типы растровых форматов, которые word/media/* реально несёт (spec §2-bis:
# "картинка уже отдельный файл" — рендер не нужен, только определить content-type
# для data-URI). Легаси-векторные OLE-превью (wmf/emf) и svg — НЕ растр, VLM как
# vision-input их не примет; такой маркер честно пропускается (см. _docx_media_uri).
_DOCX_IMAGE_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "bmp": "image/bmp", "tif": "image/tiff", "tiff": "image/tiff",
}

# Рендер региона (spec §5: "кроп pypdfium2, scale 2.0") — через pdfplumber.Page.crop().
# to_image() (обёртка над pypdfium2, уже транзитивной зависимостью pdfplumber), тот же
# design rationale, что cloud_ocr._render_page: минус второй прямой импорт ради
# идентичного детерминированного контракта. scale 2.0 = 2x нативных 72 DPI PDF-страницы.
FIGURE_RENDER_DPI = 144
FIGURE_JPEG_QUALITY = 90  # выше OCR (85): фигуры цветные/мелкодетальные, объём на документ мал
FIG_MAX_TOKENS = 8000
FIG_REQUEST: dict[str, Any] = {"reasoning": {"effort": "low"}}

FIG_PROMPT = """Describe this figure/diagram, cropped from a document page.
Output in English, in two parts:

1. Prose description (ALWAYS include this): full sentences describing what the
   figure shows. Transcribe every text label exactly as printed (verbatim, do
   not translate or paraphrase). Describe spatial and logical relationships
   between elements (what connects to what, what contains what, ordering).

2. Mermaid diagram — ONLY if and only if the figure is a flowchart, sequence,
   or hierarchy (omit entirely for matrices, grids, photos, or anything without
   a clear directional/hierarchical structure). Include ONLY edges that are
   visually present in the figure (arrows/connectors you can actually see) —
   never infer or guess a connection that is not drawn. Wrap every node label
   in double quotes, e.g. A["Label"] (unquoted labels containing punctuation
   break the mermaid parser).

Output ONLY the prose description, optionally followed by a ```mermaid code
fence — no other commentary."""


def has_bare_markers(text: str) -> bool:
    """Есть ли в doc.md хотя бы один необработанный (не инъецированный) маркер —
    дешёвая проверка для реконсиляции стадии (``run_pipeline.needed_stages``,
    spec §6: свежая конвертация ВСЕГДА регенерирует голые маркеры, а уже
    существующий, ещё не обработанный документ должен самовосстановиться без
    форсированной полной реконверсии — desired-state, не in-run флаг)."""
    return bool(
        _FIGURE_MARKER_RE.search(text)
        or _IMAGE_MARKER_RE.search(text)
        or _DOCX_IMAGE_MARKER_RE.search(text)
    )


def _cache_path(raw: Path) -> Path:
    return raw.parent / ".figures.yaml"


def _load_cache(raw: Path) -> dict[str, dict[str, Any]]:
    path = _cache_path(raw)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(raw: Path, cache: dict[str, dict[str, Any]]) -> None:
    fsio.atomic_write_text(_cache_path(raw), yaml.safe_dump(cache, allow_unicode=True, sort_keys=False))


def _render_crop(page: Any, bbox: pdf_graphics.BBox) -> str:
    """bbox региона -> data-URI JPEG. RGB (не grayscale, в отличие от cloud_ocr):
    фигуры несут смысловой цвет (SWOT-квадранты, статусные цвета флоучартов).

    bbox КЛАМПИТСЯ к границам страницы: реальные PDF несут изображения, чей bbox
    выходит за MediaBox на доли пункта или больше (живой случай — обложка sg,
    (-1.25, -0.65, 611.74, 806.45) на странице 612x792), а ``pdfplumber.crop``
    на таком bbox поднимает ValueError."""
    px0, ptop, px1, pbottom = page.bbox
    clamped = (max(bbox[0], px0), max(bbox[1], ptop), min(bbox[2], px1), min(bbox[3], pbottom))
    img = page.crop(clamped).to_image(resolution=FIGURE_RENDER_DPI).original.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=FIGURE_JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_payload(model: str, data_uri: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": FIG_PROMPT},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": FIG_MAX_TOKENS,
        **FIG_REQUEST,
    }


def _call_vlm_uri(data_uri: str, *, model: str, api_key: str, raw_name: str) -> str | None:
    """Общий payload+chat_request+обработка отказа — используется и pdf-кроп-путём
    (``_call_vlm`` сперва рендерит ``data_uri`` из bbox), и docx-путём (``data_uri``
    уже готов, извлечён из zip без рендера, см. ``_docx_media_uri``). Отказ ОДНОГО
    региона (после ретраев ``core.openrouter.chat_request``) НЕ должен ронять весь
    пасс/документ — маркер остаётся честным «не реконструировано» (чартер §2.6)."""
    try:
        response = openrouter.chat_request(_build_payload(model, data_uri), api_key=api_key)
        return response["choices"][0]["message"]["content"]  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001 — см. docstring
        logger.warning("%s: VLM-вызов для региона не удался (%s) — маркер оставлен как есть", raw_name, exc)
        return None


def _call_vlm(page: Any, bbox: pdf_graphics.BBox, *, model: str, api_key: str, raw_name: str) -> str | None:
    try:
        data_uri = _render_crop(page, bbox)
    except Exception as exc:  # noqa: BLE001 — рендер тоже может отказать (битый bbox/страница)
        logger.warning("%s: рендер региона не удался (%s) — маркер оставлен как есть", raw_name, exc)
        return None
    return _call_vlm_uri(data_uri, model=model, api_key=api_key, raw_name=raw_name)


def _docx_media_uri(raw: Path, marker_id: str) -> str | None:
    """Найти в ``word/media/*`` файл с данным id (spec §2-bis: id = 12 hex sha256
    байт файла — та же схема, что ``converters._docx_image_markers``) и вернуть
    его как data-URI. Кроп/рендер НЕ нужен — файл уже отдельное растровое
    изображение (в отличие от pdf-пути, где кропается регион страницы). Нерастровый
    формат (не в ``_DOCX_IMAGE_MIME`` — svg/wmf/emf, легаси-векторные OLE-превью)
    -> None + warning: VLM как vision-input принимает только растр."""
    with zipfile.ZipFile(raw) as z:
        for name in z.namelist():
            if not name.startswith("word/media/"):
                continue
            data = z.read(name)
            if hashlib.sha256(data).hexdigest()[:12] != marker_id:
                continue
            ext = name.rsplit(".", 1)[-1].lower()
            mime = _DOCX_IMAGE_MIME.get(ext)
            if mime is None:
                logger.warning(
                    "%s: media %s — формат .%s не растр (VLM не примет), маркер пропущен",
                    raw.name, marker_id, ext,
                )
                return None
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    logger.warning(
        "%s: media %s не найдено в word/media/* при пере-детекции (raw изменился?) — маркер пропущен",
        raw.name, marker_id,
    )
    return None


def _find_region(doc: pdf_to_markdown.DocGraphics, page_num: int, region_id: str) -> pdf_graphics.Region | None:
    if not 1 <= page_num <= len(doc.pages):
        return None
    return next((r for r in doc.pages[page_num - 1].regions if r.id == region_id and r.kind == "opaque"), None)


def _find_raster_image(
    doc: pdf_to_markdown.DocGraphics, page_num: int, marker_id: str
) -> pdf_graphics.Element | None:
    if not 1 <= page_num <= len(doc.pages):
        return None
    targets = doc.pages[page_num - 1].raster_targets
    return next((img for img in targets if pdf_graphics.image_id(img, page_num) == marker_id), None)


def _render_injected_figure(page: int, region_id: str, model: str, markdown: str) -> str:
    return (
        f"> [Figure, p. {page}, region {region_id} — VLM interpretation ({model}); "
        f"reconstruction, verify against original]\n\n{markdown}"
    )


def _render_injected_image(page: int, marker_id: str, model: str, markdown: str) -> str:
    return (
        f"> [Image, p. {page}, image {marker_id} — VLM interpretation ({model}); "
        f"reconstruction, verify against original]\n\n{markdown}"
    )


def _render_injected_docx_image(marker_id: str, model: str, markdown: str) -> str:
    return (
        f"> [Image, docx media {marker_id} — VLM interpretation ({model}); "
        f"reconstruction, verify against original]\n\n{markdown}"
    )


def apply_figures_pass(md_path: Path, raw: Path, *, model: str) -> bool:
    """Сканирует ``md_path`` на голые маркеры pdf_graphics, инъецирует VLM-
    интерпретацию (кэш-хит — офлайн; кэш-мисс — рендер+вызов+кэш). Возвращает
    True, если файл переписан (вызывающая сторона решает о реиндексе), False —
    маркеров нет ИЛИ все уже инъецированы (истинный no-op, файл не тронут)."""
    text = md_path.read_text(encoding="utf-8")
    figure_matches = list(_FIGURE_MARKER_RE.finditer(text))
    image_matches = list(_IMAGE_MARKER_RE.finditer(text))
    docx_image_matches = list(_DOCX_IMAGE_MARKER_RE.finditer(text))
    if not figure_matches and not image_matches and not docx_image_matches:
        return False

    # Ключ требуется ЛЕНИВО — только когда реально нужен облачный вызов (cache-miss,
    # см. _require_key ниже): реинъекция с тёплым кэшем полностью офлайн и работает
    # без ключа вовсе — на этом стоит golden-самосверка (@corpus): свежая конвертация
    # + офлайн-реинъекция обязаны воспроизводить doc.md без единого касания сети.
    api_key = os.environ.get("OPENROUTER_API_KEY") or None

    def _require_key() -> str:
        if api_key is None:
            raise RuntimeError("нет OPENROUTER_API_KEY (см. .env / .env.example)")
        return api_key

    cache = _load_cache(raw)
    cache_dirty = False
    doc: pdf_to_markdown.DocGraphics | None = None
    pdf_doc: Any = None  # ленивый pdfplumber.open — только на cache-miss (реальный рендер)
    replacements: list[tuple[int, int, str]] = []

    try:
        for m in figure_matches:
            page_num, rid = int(m.group("page")), m.group("id")
            entry = cache.get(rid)
            if entry is None:
                key = _require_key()
                doc = doc or pdf_to_markdown.compute_page_graphics(str(raw))
                region = _find_region(doc, page_num, rid)
                if region is None:
                    logger.warning(
                        "%s: регион %s (p.%d) не найден при пере-детекции — маркер пропущен",
                        raw.name, rid, page_num,
                    )
                    continue
                pdf_doc = pdf_doc or pdfplumber.open(raw)
                markdown = _call_vlm(
                    pdf_doc.pages[page_num - 1], region.bbox, model=model, api_key=key, raw_name=raw.name
                )
                if markdown is None:
                    continue
                entry = {"model": model, "markdown": markdown, "requested": _dt.date.today().isoformat()}
                cache[rid] = entry
                cache_dirty = True
            replacements.append(
                (m.start(), m.end(), _render_injected_figure(page_num, rid, entry["model"], entry["markdown"]))
            )

        for m in image_matches:
            page_num, iid = int(m.group("page")), m.group("id")
            entry = cache.get(iid)
            if entry is None:
                key = _require_key()
                doc = doc or pdf_to_markdown.compute_page_graphics(str(raw))
                image = _find_raster_image(doc, page_num, iid)
                if image is None:
                    logger.warning(
                        "%s: изображение %s (p.%d) не найдено при пере-детекции — маркер пропущен",
                        raw.name, iid, page_num,
                    )
                    continue
                pdf_doc = pdf_doc or pdfplumber.open(raw)
                bbox = (image.x0, image.top, image.x1, image.bottom)
                markdown = _call_vlm(pdf_doc.pages[page_num - 1], bbox, model=model, api_key=key, raw_name=raw.name)
                if markdown is None:
                    continue
                entry = {"model": model, "markdown": markdown, "requested": _dt.date.today().isoformat()}
                cache[iid] = entry
                cache_dirty = True
            replacements.append(
                (m.start(), m.end(), _render_injected_image(page_num, iid, entry["model"], entry["markdown"]))
            )

        for m in docx_image_matches:
            did = m.group("id")
            entry = cache.get(did)
            if entry is None:
                key = _require_key()
                data_uri = _docx_media_uri(raw, did)  # None -> уже залогировано (не найден/не растр)
                if data_uri is None:
                    continue
                markdown = _call_vlm_uri(data_uri, model=model, api_key=key, raw_name=raw.name)
                if markdown is None:
                    continue
                entry = {"model": model, "markdown": markdown, "requested": _dt.date.today().isoformat()}
                cache[did] = entry
                cache_dirty = True
            replacements.append(
                (m.start(), m.end(), _render_injected_docx_image(did, entry["model"], entry["markdown"]))
            )
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    if cache_dirty:
        _save_cache(raw, cache)
    if not replacements:
        return False

    new_text = text
    for start, end, replacement in sorted(replacements, key=lambda t: t[0], reverse=True):
        new_text = new_text[:start] + replacement + new_text[end:]
    fsio.atomic_write_text(md_path, new_text)
    return True
