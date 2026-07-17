"""Тесты реестра конвертеров: resolve_converter, детекция скана (_detect_scan)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from convert.converters import (
    ConversionError,
    NeedsOCR,
    UnsupportedFormat,
    _convert_html,
    _convert_pdf,
    _detect_scan,
    _tesseract_langs,
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


def test_resolve_converter_html(tmp_path: Path) -> None:
    conv = resolve_converter(tmp_path / "raw.html")
    assert conv.name == "html"


# --- _convert_html: реальная trafilatura на инлайн-фикстуре (ни сети, ни модели) ---

_HTML_FIXTURE = """<!doctype html>
<html><head><title>Test Act</title></head>
<body>
<nav><a href="/">Home</a><a href="/about">About</a></nav>
<header>Site chrome — should not survive extraction</header>
<article>
<h1>Article 1</h1>
<p>This is the operative text of the article, long enough to be recognised as
real content by trafilatura's boilerplate-vs-content heuristics rather than
being pruned as a short unimportant fragment.</p>
<table>
<tr><th>Term</th><th>Definition</th></tr>
<tr><td>AI system</td><td>A machine-based system.</td></tr>
</table>
</article>
<footer>Copyright footer — should not survive extraction</footer>
</body></html>"""


def test_convert_html_extracts_article_and_table(tmp_path: Path) -> None:
    raw = tmp_path / "raw.html"
    raw.write_text(_HTML_FIXTURE, encoding="utf-8")
    out = tmp_path / "out.md"
    _convert_html(raw, out, "en")
    text = out.read_text(encoding="utf-8")
    assert "Article 1" in text
    assert "operative text of the article" in text
    assert "AI system" in text and "machine-based system" in text  # таблица сохранена
    assert "Site chrome" not in text  # nav — не контент
    assert "Copyright footer" not in text


def test_convert_html_empty_content_raises(tmp_path: Path) -> None:
    """С favor_recall=True даже голая nav-строка выживет как fallback-контент (боязнь
    потерь > боязнь шума, см. spec §design rationale) — по-настоящему пустой ConversionError
    ловит случай, где extract() вернул None целиком (напр. пустой <body>)."""
    raw = tmp_path / "raw.html"
    raw.write_text("<html><body></body></html>", encoding="utf-8")
    out = tmp_path / "out.md"
    with pytest.raises(ConversionError, match="не извлекла"):
        _convert_html(raw, out, "en")


# --- _tesseract_langs: rec.language -> tesseract -l аргумент ---


def test_tesseract_langs_english() -> None:
    assert _tesseract_langs("en") == "eng"


def test_tesseract_langs_latin_script_gets_eng_suffix() -> None:
    assert _tesseract_langs("cnr") == "srp_latn+eng"
    assert _tesseract_langs("et") == "est+eng"
    assert _tesseract_langs("es") == "spa+eng"


def test_tesseract_langs_cjk_and_arabic_no_eng_suffix() -> None:
    assert _tesseract_langs("zh") == "chi_sim"
    assert _tesseract_langs("ja") == "jpn"
    assert _tesseract_langs("ar") == "ara"


def test_tesseract_langs_unknown_code_falls_back_to_eng(caplog: Any) -> None:
    result = _tesseract_langs("xx")
    assert result == "eng"
    assert "xx" in caplog.text


def test_tesseract_langs_none_defaults_to_eng_silently(caplog: Any) -> None:
    result = _tesseract_langs(None)
    assert result == "eng"
    assert caplog.text == ""


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
