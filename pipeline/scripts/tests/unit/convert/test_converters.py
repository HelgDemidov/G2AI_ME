"""Тесты реестра конвертеров: resolve_converter, детекция скана (_detect_scan)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from convert.converters import (
    NeedsOCR,
    UnsupportedFormat,
    _convert_pdf,
    _detect_scan,
    resolve_converter,
)


def test_resolve_converter_pdf(tmp_path: Path) -> None:
    conv = resolve_converter(tmp_path / "raw.pdf")
    assert conv.name == "pdf"


def test_resolve_converter_unsupported_lists_known_formats(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormat, match="pdf"):
        resolve_converter(tmp_path / "raw.xyz")


def test_resolve_converter_uppercase_extension(tmp_path: Path) -> None:
    conv = resolve_converter(tmp_path / "raw.PDF")
    assert conv.name == "pdf"


# --- _detect_scan: fake pdfplumber (паттерн test_pdf_to_markdown._FakeEmptyPdf) ---


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakePdf:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _patch_open(monkeypatch: Any, pages: list[Any]) -> None:
    monkeypatch.setattr("convert.converters.pdfplumber.open", lambda path: _FakePdf(pages))


def test_detect_scan_raises_when_all_pages_empty(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch, [_FakePage(""), _FakePage(""), _FakePage("")])
    with pytest.raises(NeedsOCR, match="0/3"):
        _detect_scan(tmp_path / "raw.pdf")


def test_detect_scan_passes_when_half_pages_have_text(monkeypatch: Any, tmp_path: Path) -> None:
    """Ровно порог (with_text/n == 0.5) не должен считаться сканом (строгое '<')."""
    long_text = "x" * 60  # >= SCAN_MIN_CHARS_PER_PAGE
    _patch_open(monkeypatch, [_FakePage(long_text), _FakePage("")])
    _detect_scan(tmp_path / "raw.pdf")  # не бросает


def test_detect_scan_noop_on_zero_pages(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch, [])
    _detect_scan(tmp_path / "raw.pdf")  # диагностирует pdf_convert, не _detect_scan


def test_detect_scan_none_extract_text_treated_as_empty(monkeypatch: Any, tmp_path: Path) -> None:
    class _NonePage:
        def extract_text(self) -> str | None:
            return None

    _patch_open(monkeypatch, [_NonePage(), _NonePage()])
    with pytest.raises(NeedsOCR):
        _detect_scan(tmp_path / "raw.pdf")


def test_convert_pdf_calls_pdf_convert_only_after_scan_check(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[str] = []
    _patch_open(monkeypatch, [_FakePage("x" * 60)])
    monkeypatch.setattr(
        "convert.converters.pdf_convert",
        lambda src, dst: calls.append("pdf_convert"),
    )
    _convert_pdf(tmp_path / "raw.pdf", tmp_path / "out.md", "en")
    assert calls == ["pdf_convert"]


def test_convert_pdf_propagates_needs_ocr_without_calling_pdf_convert(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _patch_open(monkeypatch, [_FakePage(""), _FakePage("")])

    def fail_if_called(src: str, dst: str) -> None:
        raise AssertionError("pdf_convert не должен вызываться на скане")

    monkeypatch.setattr("convert.converters.pdf_convert", fail_if_called)
    with pytest.raises(NeedsOCR):
        _convert_pdf(tmp_path / "raw.pdf", tmp_path / "out.md", "en")
