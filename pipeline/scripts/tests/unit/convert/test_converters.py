"""Тесты реестра конвертеров: resolve_converter, детекция скана (_detect_scan)."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from convert.converters import (
    ConversionError,
    NeedsOCR,
    UnsupportedFormat,
    _check_langs_available,
    _convert_html,
    _convert_pdf,
    _detect_scan,
    _ocr_normalize,
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


def _touch_after(target: Path, reference: Path) -> None:
    """Выставить mtime target заведомо позже reference (для проверки кэш-свежести)."""
    ref_mtime = reference.stat().st_mtime
    os.utime(target, (ref_mtime + 10, ref_mtime + 10))


def _touch_before(target: Path, reference: Path) -> None:
    """Выставить mtime target заведомо раньше reference (для проверки протухшего кэша)."""
    ref_mtime = reference.stat().st_mtime
    os.utime(target, (ref_mtime - 10, ref_mtime - 10))


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


def test_convert_pdf_routes_scan_through_ocr_normalize(monkeypatch: Any, tmp_path: Path) -> None:
    """С convert-ocr NeedsOCR больше не пропагируется наружу — скан идёт через
    _ocr_normalize, а pdf_convert затем вызывается на её результате, и вывод проходит
    post-проход ocr_headings (только OCR-ветка)."""
    _patch_open(monkeypatch, [_FakePage(""), _FakePage("")])
    ocr_result = tmp_path / ".ocr.pdf"
    ocr_result.write_bytes(b"fake ocr-normalized pdf")

    normalize_calls: list[tuple[Path, str | None]] = []

    def fake_normalize(raw: Path, language: str | None) -> Path:
        normalize_calls.append((raw, language))
        return ocr_result

    monkeypatch.setattr("convert.converters._ocr_normalize", fake_normalize)
    convert_calls: list[tuple[str, str]] = []

    def fake_pdf_convert(src: str, dst: str) -> None:
        convert_calls.append((src, dst))
        Path(dst).write_text("ANNEX I\nSome body text.\n", encoding="utf-8")

    monkeypatch.setattr("convert.converters.pdf_convert", fake_pdf_convert)
    out = tmp_path / "out.md"
    _convert_pdf(tmp_path / "raw.pdf", out, "en")
    assert normalize_calls == [(tmp_path / "raw.pdf", "en")]
    assert convert_calls == [(str(ocr_result), str(out))]
    assert out.read_text(encoding="utf-8") == "# ANNEX I\nSome body text.\n"  # ocr_headings применён


def test_convert_pdf_digital_path_skips_ocr_headings_postprocessing(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Цифровой путь (текст-слой есть) НЕ проходит ocr_headings — во избежание
    регресса калиброванной размер-кластеризации pdf_to_markdown (Сингапур/Эстония)."""
    _patch_open(monkeypatch, [_FakePage("x" * 60)])  # текст есть — не скан

    def fake_pdf_convert(src: str, dst: str) -> None:
        Path(dst).write_text("ANNEX I\nSome body text.\n", encoding="utf-8")

    monkeypatch.setattr("convert.converters.pdf_convert", fake_pdf_convert)
    out = tmp_path / "out.md"
    _convert_pdf(tmp_path / "raw.pdf", out, "en")
    assert out.read_text(encoding="utf-8") == "ANNEX I\nSome body text.\n"  # без изменений


# --- _check_langs_available / _ocr_normalize ---


def test_check_langs_available_all_present(monkeypatch: Any) -> None:
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout="List of available languages (3):\neng\nsrp_latn\nest\n", stderr="",
    )
    monkeypatch.setattr("convert.converters.subprocess.run", lambda *a, **kw: fake)
    _check_langs_available("srp_latn+eng")  # не бросает


def test_check_langs_available_missing_raises_with_apt_command(monkeypatch: Any) -> None:
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="List of available languages (1):\neng\n", stderr=""
    )
    monkeypatch.setattr("convert.converters.subprocess.run", lambda *a, **kw: fake)
    with pytest.raises(ConversionError, match="tesseract-ocr-chi-sim"):
        _check_langs_available("chi_sim")


def test_ocr_normalize_missing_binary_raises_needs_ocr_with_apt_command(
    monkeypatch: Any, tmp_path: Path
) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("convert.converters.shutil.which", lambda name: None)
    with pytest.raises(NeedsOCR, match="apt install ocrmypdf"):
        _ocr_normalize(raw, "en")


def test_ocr_normalize_uses_fresh_cache_without_subprocess(monkeypatch: Any, tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    cached = tmp_path / ".ocr.pdf"
    cached.write_bytes(b"%PDF-1.4 cached")
    _touch_after(cached, raw)  # кэш заведомо свежее raw

    def fail_if_called(*a: Any, **kw: Any) -> Any:
        raise AssertionError("subprocess не должен вызываться при свежем кэше")

    monkeypatch.setattr("convert.converters.subprocess.run", fail_if_called)
    assert _ocr_normalize(raw, "en") == cached


def test_ocr_normalize_stale_cache_triggers_ocr(monkeypatch: Any, tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    cached = tmp_path / ".ocr.pdf"
    cached.write_bytes(b"stale")
    _touch_before(cached, raw)  # кэш старее raw -> протух

    monkeypatch.setattr("convert.converters.shutil.which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr("convert.converters._check_langs_available", lambda langs: None)
    _patch_open(monkeypatch, [_FakePage("")])

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"%PDF-1.4 ocr result")  # ocrmypdf пишет staging
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("convert.converters.subprocess.run", fake_run)
    result = _ocr_normalize(raw, "en")
    assert calls  # subprocess реально вызван
    assert result == cached
    assert cached.read_bytes() == b"%PDF-1.4 ocr result"


def test_ocr_normalize_nonzero_exit_raises_conversion_error_with_stderr_tail(
    monkeypatch: Any, tmp_path: Path
) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("convert.converters.shutil.which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr("convert.converters._check_langs_available", lambda langs: None)
    _patch_open(monkeypatch, [_FakePage("")])

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom: bad scan")

    monkeypatch.setattr("convert.converters.subprocess.run", fake_run)
    with pytest.raises(ConversionError, match="boom: bad scan"):
        _ocr_normalize(raw, "en")


def test_ocr_normalize_warns_on_large_page_count(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("convert.converters.shutil.which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr("convert.converters._check_langs_available", lambda langs: None)
    _patch_open(monkeypatch, [_FakePage("")] * 250)  # > OCR_PAGE_WARN

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        Path(cmd[-1]).write_bytes(b"ocr result")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("convert.converters.subprocess.run", fake_run)
    with caplog.at_level(logging.WARNING):
        _ocr_normalize(raw, "en")
    assert "250" in caplog.text
