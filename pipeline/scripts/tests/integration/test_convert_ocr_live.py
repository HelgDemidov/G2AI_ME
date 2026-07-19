"""Live-smoke: полный OCR-путь на PIL-генерированном image-only PDF (реальный ocrmypdf/
tesseract subprocess, не мок). Требует установленный `ocrmypdf` (в CI отсутствует ->
skipif; полевой прогон на реальном скане корпуса — отдельный, более сильный гейт
закрытия спека, см. docs/pipeline/convert/tech_specs/convert-ocr §План коммитов/PR)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

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


def test_ocr_path_extracts_text_and_restores_annex_heading(tmp_path: Path, monkeypatch: Any) -> None:
    """Смок-тест целостности проводки (реальный subprocess), не точности layout.

    Синтетическая PIL-страница — один сплошной растровый объект (так работает
    Image.save(pdf)), что взаимодействует с ДРУГОЙ, не-OCR эвристикой (детекция
    инфографики convert-graphics, RASTER_MIN_PAGE_FRACTION) непредсказуемо для
    точного текста/переносов строк — реальный полевой прогон на настоящем скане
    корпуса (13-стр. закон, Черногория, 2026-07-17) уже подтвердил точность
    извлечения текста и восстановления заголовков на реалистичном документе;
    здесь проверяем только, что реальный ocrmypdf/tesseract subprocess
    отработал и что Тир 1 хоть где-то сработал — не точный текст/geometry.

    Облако (spec convert-cloud-tier) явно отключено: этот тест — про локальный
    tesseract-путь конкретно, а живой ``.env``-ключ на машине разработчика иначе
    увёл бы синтетический скан в реальный (пусть и грошовый) облачный вызов.
    """
    monkeypatch.setattr("convert.converters._cloud_allowed", lambda record: False)
    raw = tmp_path / "raw.pdf"
    _make_scan_pdf(raw)
    out = tmp_path / "out.md"

    conv = converters.resolve_converter(raw)
    conv.convert(raw, out, "en")

    text = out.read_text(encoding="utf-8")
    assert "smoke" in text.lower()
    assert "# ANNEX" in text  # ocr_headings: Тир 1 сработал (ключевое слово ANNEX)
    assert raw.exists()  # один файл на документ — не сайдкар .ocr.pdf
    assert converters._was_ocr_normalized(raw)  # raw мутирован in-place, метка ocrmypdf на месте
