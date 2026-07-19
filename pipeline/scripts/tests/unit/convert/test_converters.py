"""Тесты реестра конвертеров: resolve_converter, детекция скана (_detect_scan)."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from convert.converters import (
    ConversionError,
    NeedsOCR,
    UnsupportedFormat,
    _cached_or_call_cloud,
    _check_langs_available,
    _cloud_allowed,
    _convert_html,
    _convert_pdf,
    _detect_scan,
    _ocr_normalize,
    _tesseract_langs,
    _was_ocr_normalized,
    resolve_converter,
)
from core.schema import SourceRecord
from tests.support import valid_record


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


def test_html_heading_loss_guard_warns(monkeypatch: Any, tmp_path: Path, caplog: Any) -> None:
    """B2: исходный HTML несёт <h2>, но trafilatura (замокана) отдала выход без
    единого markdown-заголовка — генерическая ловушка «главный контент без
    <article> теряет <hN>» (эмпирика convert-html); НЕ отказ, только warning."""
    raw = tmp_path / "raw.html"
    raw.write_text(
        "<html><body><h2>Some Heading</h2><p>Body text long enough to survive.</p></body></html>",
        encoding="utf-8",
    )
    out = tmp_path / "out.md"
    monkeypatch.setattr("trafilatura.extract", lambda *a, **kw: "Some Heading\n\nBody text long enough to survive.")
    with caplog.at_level(logging.WARNING):
        _convert_html(raw, out, "en")
    assert "вероятна потеря структуры" in caplog.text


def test_html_heading_preserved_no_warning(monkeypatch: Any, tmp_path: Path, caplog: Any) -> None:
    raw = tmp_path / "raw.html"
    raw.write_text("<html><body><h2>Some Heading</h2><p>Body text.</p></body></html>", encoding="utf-8")
    out = tmp_path / "out.md"
    monkeypatch.setattr("trafilatura.extract", lambda *a, **kw: "## Some Heading\n\nBody text.")
    with caplog.at_level(logging.WARNING):
        _convert_html(raw, out, "en")
    assert caplog.text == ""


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
    def __init__(self, pages: list[_FakePage], metadata: dict[str, str] | None = None) -> None:
        self.pages = pages
        self.metadata = metadata or {}

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _patch_open(monkeypatch: Any, pages: list[Any], metadata: dict[str, str] | None = None) -> None:
    monkeypatch.setattr(
        "convert.converters.pdfplumber.open", lambda path: _FakePdf(pages, metadata)
    )


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


# --- _was_ocr_normalized: метка ocrmypdf в метаданных переживает мутацию raw ---


def test_was_ocr_normalized_true_when_creator_mentions_ocrmypdf(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch, [], metadata={"Creator": "ocrmypdf 15.2.0+dfsg1 / Tesseract OCR-PDF 5.3.4"})
    assert _was_ocr_normalized(tmp_path / "raw.pdf") is True


def test_was_ocr_normalized_false_for_born_digital_pdf(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch, [], metadata={"Creator": "Microsoft® Word 2019"})
    assert _was_ocr_normalized(tmp_path / "raw.pdf") is False


def test_was_ocr_normalized_false_when_no_metadata_at_all(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch, [])  # metadata=None -> {}
    assert _was_ocr_normalized(tmp_path / "raw.pdf") is False


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
    _ocr_normalize (мутирует raw IN-PLACE, один файл — не сайдкар), pdf_convert
    затем вызывается на ТОМ ЖЕ raw, и вывод проходит post-проход ocr_headings
    (только OCR-ветка)."""
    _patch_open(monkeypatch, [_FakePage(""), _FakePage("")])
    monkeypatch.setattr("convert.converters._cloud_allowed", lambda record: False)  # локальный путь явно
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"fake scanned pdf")

    normalize_calls: list[tuple[Path, str | None]] = []

    def fake_normalize(raw_arg: Path, language: str | None) -> None:
        normalize_calls.append((raw_arg, language))

    monkeypatch.setattr("convert.converters._ocr_normalize", fake_normalize)
    convert_calls: list[tuple[str, str]] = []

    def fake_pdf_convert(src: str, dst: str) -> None:
        convert_calls.append((src, dst))
        Path(dst).write_text("ANNEX I\nSome body text.\n", encoding="utf-8")

    monkeypatch.setattr("convert.converters.pdf_convert", fake_pdf_convert)
    out = tmp_path / "out.md"
    _convert_pdf(raw, out, "en")
    assert normalize_calls == [(raw, "en")]
    assert convert_calls == [(str(raw), str(out))]  # тот же raw, не отдельный файл
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


def test_convert_pdf_reapplies_ocr_headings_on_already_normalized_raw(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Регресс: raw уже нормализован ПРЕДЫДУЩИМ прогоном (текст-слой есть, _detect_scan
    больше не поднимет NeedsOCR) — но ocr_headings обязан примениться СНОВА на этой
    повторной конвертации (--force/бамп версии конвертера), иначе восстановление
    заголовков теряется после первого же прогона. Метка — метаданные ocrmypdf,
    переживающие мутацию текст-слоя (см. _was_ocr_normalized)."""
    _patch_open(
        monkeypatch, [_FakePage("x" * 60)],
        metadata={"Creator": "ocrmypdf 15.2.0+dfsg1 / Tesseract OCR-PDF 5.3.4"},
    )
    monkeypatch.setattr("convert.converters._cloud_allowed", lambda record: False)  # локальный путь явно

    def fake_pdf_convert(src: str, dst: str) -> None:
        Path(dst).write_text("ANNEX I\nSome body text.\n", encoding="utf-8")

    monkeypatch.setattr("convert.converters.pdf_convert", fake_pdf_convert)
    out = tmp_path / "out.md"
    _convert_pdf(tmp_path / "raw.pdf", out, "en")
    assert out.read_text(encoding="utf-8") == "# ANNEX I\nSome body text.\n"  # ocr_headings применён


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


def test_ocr_normalize_mutates_raw_in_place_no_sidecar_file(monkeypatch: Any, tmp_path: Path) -> None:
    """Один PDF-файл на документ — не сайдкар .ocr.pdf (решение пользователя, ревизия
    convert-ocr от 2026-07-17): raw.pdf заменяется версией с текст-слоем НА МЕСТЕ."""
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"%PDF-1.4 fake scan")

    monkeypatch.setattr("convert.converters.shutil.which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr("convert.converters._check_langs_available", lambda langs: None)
    _patch_open(monkeypatch, [_FakePage("")])

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"%PDF-1.4 ocr result")  # ocrmypdf пишет staging
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("convert.converters.subprocess.run", fake_run)
    _ocr_normalize(raw, "en")
    assert calls  # subprocess реально вызван
    assert raw.read_bytes() == b"%PDF-1.4 ocr result"  # raw САМ теперь содержит OCR-результат
    assert not (tmp_path / ".ocr.pdf").exists()  # никакого сайдкара


def test_ocr_normalize_always_calls_subprocess_no_cache_check(monkeypatch: Any, tmp_path: Path) -> None:
    """Без сайдкара нечего и проверять на свежесть — _ocr_normalize безусловно
    вызывает subprocess при каждом вызове (кэш получается «бесплатно» иначе: после
    успеха raw САМ содержит текст, и _detect_scan больше не поднимет NeedsOCR — то
    есть _ocr_normalize вообще не будет вызван на следующих прогонах, без кэш-файла)."""
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"%PDF-1.4 fake scan")
    monkeypatch.setattr("convert.converters.shutil.which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr("convert.converters._check_langs_available", lambda langs: None)
    _patch_open(monkeypatch, [_FakePage("")])

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"ocr result")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("convert.converters.subprocess.run", fake_run)
    _ocr_normalize(raw, "en")
    assert len(calls) == 1


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


# --- _cloud_allowed / _cached_or_call_cloud / _convert_pdf облачная маршрутизация
# (spec convert-cloud-tier §6) ---


def _record(**over: Any) -> SourceRecord:
    data = valid_record()
    data.update(over)
    return SourceRecord.model_validate(data)


def _reset_cloud_module_state(monkeypatch: Any) -> None:
    monkeypatch.setattr("convert.converters._CLOUD_DISABLED", False)
    monkeypatch.setattr("convert.converters._CLOUD_KEY_WARNED", False)


def test_cloud_allowed_false_when_disabled_flag_set(monkeypatch: Any) -> None:
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.setattr("convert.converters._CLOUD_DISABLED", True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert _cloud_allowed(None) is False


def test_cloud_allowed_false_for_confidential_record(monkeypatch: Any) -> None:
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = _record(sensitivity="confidential")
    assert _cloud_allowed(rec) is False


def test_cloud_allowed_false_without_key(monkeypatch: Any) -> None:
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("convert.converters.load_dotenv", lambda: None)  # не подхватить реальный .env
    assert _cloud_allowed(None) is False


def test_cloud_allowed_true_for_normal_record_with_key(monkeypatch: Any) -> None:
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = _record(sensitivity="normal")
    assert _cloud_allowed(rec) is True


def test_cached_or_call_cloud_hit_skips_network(monkeypatch: Any, tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"normalized scan bytes")
    from core import fsio, schema

    cache = raw.parent / ".cloudocr.md"
    cache.write_text("# Cached\n\nBody.", encoding="utf-8")
    state = schema.OperationalState(cloud_ocr_model="m", cloud_ocr_raw_sha256=fsio.sha256_file(raw))
    schema.save_state(raw.parent / ".state.yaml", state)

    monkeypatch.setattr(
        "convert.converters.cloud_ocr.convert_scan",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("сеть не должна была вызываться")),
    )
    text = _cached_or_call_cloud(raw, "en", model="m")
    assert text == "# Cached\n\nBody."


def test_cached_or_call_cloud_model_mismatch_uses_cache_without_recall(
    monkeypatch: Any, tmp_path: Path, caplog: Any
) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"normalized scan bytes")
    from core import fsio, schema

    cache = raw.parent / ".cloudocr.md"
    cache.write_text("# From old model", encoding="utf-8")
    state = schema.OperationalState(cloud_ocr_model="old-model", cloud_ocr_raw_sha256=fsio.sha256_file(raw))
    schema.save_state(raw.parent / ".state.yaml", state)

    monkeypatch.setattr(
        "convert.converters.cloud_ocr.convert_scan",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("авто-перевызов при смене модели запрещён")),
    )
    with caplog.at_level(logging.WARNING):
        text = _cached_or_call_cloud(raw, "en", model="new-model")
    assert text == "# From old model"
    assert "old-model" in caplog.text and "new-model" in caplog.text


def test_cached_or_call_cloud_miss_calls_cloud_and_persists(monkeypatch: Any, tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"normalized scan bytes")
    from core import fsio, schema

    monkeypatch.setattr("convert.converters.cloud_ocr.convert_scan", lambda raw_, lang, *, model: "# Fresh\n\nText.")
    text = _cached_or_call_cloud(raw, "en", model="m")
    assert text == "# Fresh\n\nText."
    assert (raw.parent / ".cloudocr.md").read_text(encoding="utf-8") == "# Fresh\n\nText."
    state = schema.load_state(raw.parent / ".state.yaml")
    assert state.cloud_ocr_model == "m"
    assert state.cloud_ocr_raw_sha256 == fsio.sha256_file(raw)


def test_cached_or_call_cloud_failure_returns_none(monkeypatch: Any, tmp_path: Path, caplog: Any) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"normalized scan bytes")

    def failing(*a: Any, **kw: Any) -> str:
        raise RuntimeError("OpenRouter: исчерпаны попытки")

    monkeypatch.setattr("convert.converters.cloud_ocr.convert_scan", failing)
    with caplog.at_level(logging.WARNING):
        text = _cached_or_call_cloud(raw, "en", model="m")
    assert text is None
    assert not (raw.parent / ".cloudocr.md").exists()


def test_convert_pdf_confidential_record_skips_cloud(monkeypatch: Any, tmp_path: Path) -> None:
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _patch_open(monkeypatch, [_FakePage("x" * 60)], metadata={"Creator": "ocrmypdf 15.2.0"})  # уже нормализован

    monkeypatch.setattr(
        "convert.converters._cached_or_call_cloud",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("облако не должно было вызываться")),
    )

    def fake_pdf_convert(src: str, dst: str) -> None:
        Path(dst).write_text("ANNEX I\nBody.\n", encoding="utf-8")

    monkeypatch.setattr("convert.converters.pdf_convert", fake_pdf_convert)
    out = tmp_path / "out.md"
    rec = _record(sensitivity="confidential")
    _convert_pdf(tmp_path / "raw.pdf", out, "en", record=rec)
    assert out.read_text(encoding="utf-8") == "# ANNEX I\nBody.\n"  # локальный путь + ocr_headings


def test_convert_pdf_digital_never_calls_cloud(monkeypatch: Any, tmp_path: Path) -> None:
    """Цифровой PDF (не скан) — облако не вызывается вовсе, независимо от гейтов."""
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _patch_open(monkeypatch, [_FakePage("x" * 60)])  # текст есть — не скан

    monkeypatch.setattr(
        "convert.converters._cached_or_call_cloud",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("облако не должно было вызываться")),
    )

    def fake_pdf_convert(src: str, dst: str) -> None:
        Path(dst).write_text("body", encoding="utf-8")

    monkeypatch.setattr("convert.converters.pdf_convert", fake_pdf_convert)
    out = tmp_path / "out.md"
    _convert_pdf(tmp_path / "raw.pdf", out, "en", record=_record())
    assert out.read_text(encoding="utf-8") == "body"


def test_convert_pdf_cloud_success_skips_ocr_headings(monkeypatch: Any, tmp_path: Path) -> None:
    """Облачный вывод НЕ проходит ocr_headings (иерархия уже есть и лучше, §Design rationale)."""
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _patch_open(monkeypatch, [_FakePage("x" * 60)], metadata={"Creator": "ocrmypdf 15.2.0"})

    monkeypatch.setattr(
        "convert.converters._cached_or_call_cloud",
        lambda raw, lang, *, model: "# Cloud Title\n\nBody, unflagged.",
    )
    monkeypatch.setattr(
        "convert.converters.pdf_convert",
        lambda src, dst: (_ for _ in ()).throw(AssertionError("локальный путь не должен был вызываться")),
    )
    out = tmp_path / "out.md"
    _convert_pdf(tmp_path / "raw.pdf", out, "en", record=_record())
    assert out.read_text(encoding="utf-8") == "# Cloud Title\n\nBody, unflagged."


def test_convert_pdf_cloud_failure_falls_back_to_local_with_ocr_headings(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _reset_cloud_module_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _patch_open(monkeypatch, [_FakePage("x" * 60)], metadata={"Creator": "ocrmypdf 15.2.0"})

    monkeypatch.setattr("convert.converters._cached_or_call_cloud", lambda raw, lang, *, model: None)

    def fake_pdf_convert(src: str, dst: str) -> None:
        Path(dst).write_text("ANNEX I\nBody.\n", encoding="utf-8")

    monkeypatch.setattr("convert.converters.pdf_convert", fake_pdf_convert)
    out = tmp_path / "out.md"
    _convert_pdf(tmp_path / "raw.pdf", out, "en", record=_record())
    assert out.read_text(encoding="utf-8") == "# ANNEX I\nBody.\n"  # локальный фолбэк + ocr_headings, без краха
