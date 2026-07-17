"""Live-smoke: полный OCR-путь на PIL-генерированном image-only PDF (реальный ocrmypdf/
tesseract subprocess, не мок). Требует установленный `ocrmypdf` (в CI отсутствует ->
skipif; полевой прогон на реальном скане корпуса — отдельный, более сильный гейт
закрытия спека, см. docs/pipeline/convert/tech_specs/convert-ocr §План коммитов/PR)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from convert import converters

pytestmark = [
    pytest.mark.ocr,
    pytest.mark.skipif(shutil.which("ocrmypdf") is None, reason="ocrmypdf не установлен (см. spec §1)"),
]


def _make_scan_pdf(path: Path) -> None:
    """Image-only PDF (без единой строки текст-слоя) — синтетический аналог скана."""
    img = Image.new("RGB", (1000, 400), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=48)
    draw.text((40, 40), "ANNEX I", fill="black", font=font)
    draw.text((40, 140), "OCR SMOKE TEST DOCUMENT", fill="black", font=font)
    img.save(path)


def test_ocr_path_extracts_text_and_restores_annex_heading(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    _make_scan_pdf(raw)
    out = tmp_path / "out.md"

    conv = converters.resolve_converter(raw)
    conv.convert(raw, out, "en")

    text = out.read_text(encoding="utf-8")
    assert "smoke" in text.lower()
    assert "# ANNEX I" in text  # ocr_headings восстановил Тир-1 заголовок
    assert (raw.parent / ".ocr.pdf").exists()  # кэш-сайдкар создан
