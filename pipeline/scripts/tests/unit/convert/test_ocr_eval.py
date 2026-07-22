"""Тесты харнесса качества OCR (spec ocr-eval-harness). Тир 1/2 полностью
CI-safe (без сети/модели) — синтетические тексты, не реальный корпус.
run_document/run_pages тестируются с мокнутым cloud_ocr.convert_scan
(сеть запрещена в unit-тестах, spec §7); run_tesseract сети не требует
вовсе — реальный многостраничный PDF через reportlab."""
from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pdfplumber
import pytest
from reportlab.pdfgen import canvas

import yaml

from convert import cloud_ocr, ocr_eval
from core import fsio
from convert.ocr_eval import (
    CandidateResult,
    Divergence,
    diverge,
    extract_headings,
    format_report,
    levenshtein,
    main,
    normalize_for_cer,
    run_document,
    run_pages,
    run_tesseract,
    score_page,
)


def _multi_page_pdf(texts: list[str], *, creator: str | None = None) -> bytes:
    """Реальный многостраничный PDF через reportlab — run_pages/run_tesseract
    нуждаются в настоящих объектах pdfplumber/pypdfium2, не в моке (та же
    дисциплина, что tests/support.py::build_pdf; своя копия здесь — build_pdf
    однострочна и не поддерживает несколько страниц)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(300, 300))
    if creator is not None:
        c.setCreator(creator)
    for text in texts:
        c.drawString(20, 250, text)
        c.showPage()
    c.save()
    return buf.getvalue()

# --- normalize_for_cer ---


def test_normalize_strips_heading_hashes() -> None:
    assert normalize_for_cer("# Naslov\n\nTijelo teksta.") == "Naslov Tijelo teksta."


def test_normalize_collapses_whitespace() -> None:
    assert normalize_for_cer("a   b\n\n\tc") == "a b c"


def test_normalize_nfc() -> None:
    decomposed = "e" + "́"  # e + combining acute accent
    assert normalize_for_cer(decomposed) == "é"  # é (NFC precomposed)


def test_normalize_preserves_case_and_diacritics() -> None:
    assert normalize_for_cer("Član ČLAN član") == "Član ČLAN član"


# --- levenshtein ---


def test_levenshtein_identical_is_zero() -> None:
    assert levenshtein("abc", "abc") == 0


def test_levenshtein_empty_strings() -> None:
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3


def test_levenshtein_single_substitution() -> None:
    assert levenshtein("kitten", "kitten".replace("k", "s", 1)) == 1


def test_levenshtein_insertion_deletion() -> None:
    assert levenshtein("ab", "abc") == 1  # вставка
    assert levenshtein("abc", "ab") == 1  # удаление


def test_levenshtein_symmetric() -> None:
    a, b = "Predmet Član 1", "Predmet Clan I"
    assert levenshtein(a, b) == levenshtein(b, a)


def test_levenshtein_classic_kitten_sitting() -> None:
    assert levenshtein("kitten", "sitting") == 3


# --- extract_headings ---


def test_extract_headings_levels_and_text() -> None:
    md = "# Naslov\n\n## Glava I\n\nBody.\n\n### Član 1\n"
    assert extract_headings(md) == [(1, "Naslov"), (2, "Glava I"), (3, "Član 1")]


def test_extract_headings_ignores_fenced_code() -> None:
    md = "# Real\n\n```\n# not a heading\n```\n\n## Also Real"
    assert extract_headings(md) == [(1, "Real"), (2, "Also Real")]


def test_extract_headings_none() -> None:
    assert extract_headings("Just prose, no headings.") == []


def test_extract_headings_unclosed_fence_swallows_rest() -> None:
    """Незакрытый фенс — честно: всё до конца документа считается кодом,
    заголовки внутри не всплывают (симметрично chunking._paragraphs)."""
    md = "# Before\n\n```\n# swallowed\n## also swallowed"
    assert extract_headings(md) == [(1, "Before")]


# --- score_page ---


def test_score_page_perfect_match() -> None:
    text = "# Član 1\n\nOvim zakonom uređuje se registracija broj 42."
    score = score_page(text, text)
    assert score.cer == 0.0
    assert score.diacritics_recall == 1.0
    assert score.numeric_missing == 0 and score.numeric_added == 0
    assert score.headings_matched == score.headings_gold == 1


def test_score_page_diacritics_loss_only_affects_diacritics_metric() -> None:
    gold = "Član vođenje registra"
    candidate = "Clan vodenje registra"  # диакритика потеряна, остальное идентично
    score = score_page(gold, candidate)
    assert score.diacritics_recall == 0.0
    assert score.numeric_missing == 0 and score.numeric_added == 0


def test_score_page_swapped_digit_is_a_silent_substitution() -> None:
    """Тихая подмена цифры (26 -> 25): numeric_missing/added ловят её РОВНО,
    CER остаётся почти нулевым (один символ на длинной строке) — это и есть
    самый опасный класс, ради которого числовая метрика существует отдельно."""
    gold = "Odložena primjena Član 26 ovog zakona"
    candidate = "Odložena primjena Član 25 ovog zakona"
    score = score_page(gold, candidate)
    assert score.numeric_missing == 1 and score.numeric_added == 1
    assert score.cer < 0.1


def test_score_page_missing_heading_reflected_in_ratio() -> None:
    gold = "# Glava I\n\n## Član 1\n\nBody."
    candidate = "Glava I\n\nBody."  # заголовки потеряны целиком
    score = score_page(gold, candidate)
    assert score.headings_gold == 2
    assert score.headings_matched == 0


def test_score_page_duplicate_headings_use_multiset_matching() -> None:
    gold = "# Član 1\n\n# Član 1\n\nBody."  # дубль в эталоне (редкий, но легитимный вырожденный случай)
    candidate = "# Član 1\n\nBody."
    score = score_page(gold, candidate)
    assert score.headings_gold == 2
    assert score.headings_matched == 1  # только одна пара найдена, не обе


# --- diverge ---


def _candidate(name: str, document_text: str, failed: str | None = None) -> CandidateResult:
    return CandidateResult(name=name, document_text=document_text, page_text={}, scores=[], failed=failed)


def test_diverge_identical_documents_no_divergence() -> None:
    text = "# Član 1\n\nOvim zakonom broj 42."
    (d,) = diverge([_candidate("a", text), _candidate("b", text)])
    assert not any((d.numeric_only_left, d.numeric_only_right, d.headings_only_left, d.headings_only_right))


def test_diverge_swapped_digit_appears_on_both_sides() -> None:
    """Тихая подмена 26 -> 25: «26» есть только у left, «25» — только у right
    (симметрично тому же классу в witness_checks/numeric_delta)."""
    (d,) = diverge([_candidate("gemini", "Član 26"), _candidate("tesseract", "Član 25")])
    assert d.numeric_only_left == ("26",)
    assert d.numeric_only_right == ("25",)


def test_diverge_heading_only_on_one_side() -> None:
    (d,) = diverge([_candidate("a", "# Glava I\n\nBody."), _candidate("b", "Glava I\n\nBody.")])
    assert d.headings_only_left == ("1:Glava I",)
    assert d.headings_only_right == ()


def test_diverge_excludes_failed_candidates() -> None:
    """Упавший кандидат не участвует ни в одной паре — сравнивать нечего."""
    results = [_candidate("a", "text"), _candidate("b", "text", failed="429 rate limit")]
    assert diverge(results) == []


def test_diverge_three_candidates_builds_three_pairs() -> None:
    results = [_candidate("a", "1"), _candidate("b", "2"), _candidate("c", "3")]
    pairs = {(d.left, d.right) for d in diverge(results)}
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_diverge_numeric_tokens_sorted_numerically_not_lexically() -> None:
    """«2» перед «10» — численная сортировка, не строковая (иначе «10» < «2»)."""
    (d,) = diverge([_candidate("a", "2 10 30"), _candidate("b", "")])
    assert d.numeric_only_left == ("2", "10", "30")


# --- format_report ---


def _result(name: str, pages: dict[int, tuple[str, str]], failed: str | None = None) -> CandidateResult:
    scores = [replace(score_page(gold, candidate), page=page) for page, (gold, candidate) in pages.items()]
    return CandidateResult(name=name, document_text="", page_text={}, scores=scores, failed=failed)


def test_format_report_contains_all_candidate_names_including_failed() -> None:
    ok = _result("gemini", {1: ("# T\n\nx", "# T\n\nx")})
    failed = CandidateResult(name="broken-model", document_text="", page_text={}, scores=[], failed="429 rate limit")
    out = format_report([ok, failed], [])
    assert "gemini" in out and "broken-model" in out and "429 rate limit" in out


def test_format_report_prints_divergent_tokens_not_only_counts() -> None:
    d = Divergence(
        left="gemini", right="tesseract",
        numeric_only_left=("26",), numeric_only_right=("25",),
        headings_only_left=(), headings_only_right=(),
    )
    out = format_report([], [d])
    assert "26" in out and "25" in out
    assert "gemini_only=[26]" in out and "tesseract_only=[25]" in out


def test_format_report_collapses_zero_divergence_pair_to_one_line() -> None:
    d = Divergence(
        left="gemini", right="gemini-3.6",
        numeric_only_left=(), numeric_only_right=(),
        headings_only_left=(), headings_only_right=(),
    )
    out = format_report([], [d])
    assert "gemini vs gemini-3.6: совпали" in out
    lines = [ln for ln in out.splitlines() if "gemini" in ln and "3.6" in ln]
    assert len(lines) == 1  # свёрнуто в одну строку, не расписано по метрикам


def test_format_report_tier1_header_lists_actual_pages() -> None:
    r = _result("gemini", {1: ("# T\n\nx", "# T\n\nx"), 7: ("# U\n\ny", "# U\n\ny")})
    out = format_report([r], [])
    assert "стр. 1, 7" in out.splitlines()[0]


# --- run_document / run_pages: изоляция (§5) — сеть замокана, convert_scan не касается сети ---


def test_run_document_calls_convert_scan_with_path_inside_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(_multi_page_pdf(["p1"]))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    seen: list[Path] = []

    def fake_convert_scan(path: Path, language: str | None, *, model: str) -> str:
        seen.append(path)
        return "cloud text"

    monkeypatch.setattr(cloud_ocr, "convert_scan", fake_convert_scan)
    result = run_document(raw, "cnr", "test-model", workdir)

    assert result == "cloud text"
    assert len(seen) == 1
    assert seen[0].parent == workdir
    assert seen[0] != raw


def test_run_document_never_touches_original_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(_multi_page_pdf(["p1"]))
    before = raw.stat()
    workdir = tmp_path / "wd"
    workdir.mkdir()

    monkeypatch.setattr(cloud_ocr, "convert_scan", lambda path, language, *, model: "x")
    run_document(raw, "cnr", "test-model", workdir)

    after = raw.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


def test_run_pages_calls_convert_scan_once_per_page_with_single_page_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(_multi_page_pdf(["p1", "p2", "p3"]))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    calls: list[Path] = []

    def fake_convert_scan(path: Path, language: str | None, *, model: str) -> str:
        calls.append(path)
        with pdfplumber.open(path) as pdf:  # каждый вызов — валидный ОДНОСТРАНИЧНЫЙ PDF
            assert len(pdf.pages) == 1
        return f"text-for-{path.name}"

    monkeypatch.setattr(cloud_ocr, "convert_scan", fake_convert_scan)
    result = run_pages(raw, [1, 3], "cnr", "test-model", workdir)

    assert len(calls) == 2  # ровно по одному вызову на страницу, не один на весь список
    assert set(result.keys()) == {1, 3}
    assert all(p.parent == workdir for p in calls)


def test_run_pages_never_touches_original_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(_multi_page_pdf(["p1", "p2"]))
    before = raw.stat()
    workdir = tmp_path / "wd"
    workdir.mkdir()

    monkeypatch.setattr(cloud_ocr, "convert_scan", lambda path, language, *, model: "x")
    run_pages(raw, [1, 2], "cnr", "test-model", workdir)

    after = raw.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


# --- run_tesseract: сети не требует, реальный текст-слой через reportlab ---


def test_run_tesseract_extracts_all_pages_when_normalized(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(
        _multi_page_pdf(
            ["Clan 26 potpis", "Clan 7 broj"],
            creator="ocrmypdf 15.2.0+dfsg1 / Tesseract OCR-PDF 5.3.4",
        )
    )
    workdir = tmp_path / "wd"
    workdir.mkdir()

    result = run_tesseract(raw, workdir)

    assert result is not None
    assert set(result) == {1, 2}
    assert "Clan 26" in result[1]
    assert "Clan 7" in result[2]


def test_run_tesseract_returns_none_for_born_digital(tmp_path: Path) -> None:
    """Нет метки ocrmypdf в Creator -> нет текст-слоя от OCR -> кандидат
    пропускается с None, а не роняет прогон (spec §3)."""
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(_multi_page_pdf(["born digital text"], creator="Microsoft Word"))
    workdir = tmp_path / "wd"
    workdir.mkdir()

    assert run_tesseract(raw, workdir) is None


def test_run_tesseract_never_touches_original_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(_multi_page_pdf(["x"], creator="ocrmypdf 15.2.0"))
    before = raw.stat()
    workdir = tmp_path / "wd"
    workdir.mkdir()

    run_tesseract(raw, workdir)

    after = raw.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


# --- _resolve_raw / _load_gold_page: helpers CLI (§6) ---


def _fake_sources(tmp_path: Path, document: str, raw_bytes: bytes) -> Path:
    """Минимальная sources/-раскладка (track/entity/doc-id/raw.pdf) — только
    то, что нужно _resolve_raw, без валидного meta.yaml (харнесс не читает реестр)."""
    root = tmp_path / "sources"
    doc_dir = root / "track" / "entity" / document
    doc_dir.mkdir(parents=True)
    (doc_dir / "raw.pdf").write_bytes(raw_bytes)
    return root


def test_resolve_raw_finds_file_by_document_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources_root = _fake_sources(tmp_path, "me-crps-registration-law-2025", b"%PDF-1.4 stub")
    monkeypatch.setattr(ocr_eval, "DEFAULT_SOURCES", sources_root)

    found = ocr_eval._resolve_raw("me-crps-registration-law-2025")
    assert found == sources_root / "track" / "entity" / "me-crps-registration-law-2025" / "raw.pdf"


def test_resolve_raw_raises_when_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_eval, "DEFAULT_SOURCES", tmp_path / "sources")
    with pytest.raises(FileNotFoundError):
        ocr_eval._resolve_raw("does-not-exist")


def test_load_gold_page_reads_matching_suffix_file(tmp_path: Path) -> None:
    (tmp_path / "me-crps-p01.md").write_text("# Naslov\n\nBody.", encoding="utf-8")
    assert ocr_eval._load_gold_page(tmp_path, 1) == "# Naslov\n\nBody."


def test_load_gold_page_raises_on_ambiguous_match(tmp_path: Path) -> None:
    (tmp_path / "a-p01.md").write_text("x", encoding="utf-8")
    (tmp_path / "b-p01.md").write_text("y", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ocr_eval._load_gold_page(tmp_path, 1)


def test_load_gold_page_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ocr_eval._load_gold_page(tmp_path, 99)


# --- main: CLI (§6) — все вызовы с явными tmp-путями, сеть только мокнутая ---


def _write_gold(gold_dir: Path, document: str, raw_sha256: str, pages: dict[int, str]) -> None:
    gold_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "document": document, "raw_sha256": raw_sha256, "language": "cnr",
        "pages": sorted(pages), "verified_by": "test", "verified_at": "2026-07-22",
    }
    (gold_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    for page, text in pages.items():
        (gold_dir / f"doc-p{page:02d}.md").write_text(text, encoding="utf-8")


def test_main_missing_gold_dir_returns_2(tmp_path: Path) -> None:
    assert main(["--gold", str(tmp_path / "nope")]) == 2


def test_main_gold_dir_without_manifest_returns_2(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    gold.mkdir()
    assert main(["--gold", str(gold)]) == 2


def test_main_dry_run_prints_request_count_and_makes_no_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sources_root = _fake_sources(tmp_path, "doc-x", b"%PDF-1.4 stub")
    monkeypatch.setattr(ocr_eval, "DEFAULT_SOURCES", sources_root)
    gold = tmp_path / "gold"
    _write_gold(gold, "doc-x", "irrelevant-in-dry-run", {1: "p1", 7: "p7"})

    called = []
    monkeypatch.setattr(ocr_eval, "run_document", lambda *a, **k: called.append("doc"))
    monkeypatch.setattr(ocr_eval, "run_pages", lambda *a, **k: called.append("pages"))
    monkeypatch.setattr(ocr_eval, "run_tesseract", lambda *a, **k: called.append("tesseract"))

    code = main(["--gold", str(gold), "--models", "model-a,model-b", "--dry-run"])

    assert code == 0
    assert called == []  # ни одного сетевого/tesseract вызова
    out = capsys.readouterr().out
    assert "6 сетевых запросов" in out  # 2 модели x (1 документ + 2 стр.)
    assert "tesseract (бесплатно" in out


def test_main_dry_run_no_tesseract_flag_omits_tesseract_from_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sources_root = _fake_sources(tmp_path, "doc-x", b"%PDF-1.4 stub")
    monkeypatch.setattr(ocr_eval, "DEFAULT_SOURCES", sources_root)
    gold = tmp_path / "gold"
    _write_gold(gold, "doc-x", "irrelevant", {1: "p1"})

    code = main(["--gold", str(gold), "--models", "model-a", "--no-tesseract", "--dry-run"])
    assert code == 0
    # узкая проверка фразы плана, не голого "tesseract": tmp_path пифтеста сам
    # содержит "no_tesseract" (из имени теста) — блэкет-серч по подстроке ложно
    # сработал бы на пути в выводе, не на логике кода.
    assert "tesseract (бесплатно" not in capsys.readouterr().out


def test_main_full_run_wires_report_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Полный прогон (без --dry-run), сеть замокана: проверяет склейку
    _build_candidates -> _score_candidates -> diverge -> format_report -> print,
    не только отдельные функции по отдельности."""
    raw_bytes = _multi_page_pdf(["Član 1 broj 42", "Član 7 broj 100"], creator="ocrmypdf 15.2.0")
    sources_root = _fake_sources(tmp_path, "doc-x", raw_bytes)
    monkeypatch.setattr(ocr_eval, "DEFAULT_SOURCES", sources_root)
    raw_path = sources_root / "track" / "entity" / "doc-x" / "raw.pdf"

    gold = tmp_path / "gold"
    _write_gold(
        gold, "doc-x", fsio.sha256_file(raw_path),
        {1: "Član 1 broj 42", 2: "Član 7 broj 100"},
    )

    monkeypatch.setattr(cloud_ocr, "convert_scan", lambda path, language, *, model: "# Cloud\n\nČlan 1 broj 42")

    code = main(["--gold", str(gold), "--models", "cloud-model", "--no-tesseract"])

    assert code == 0
    out = capsys.readouterr().out
    assert "cloud-model" in out
    assert "Против эталона" in out and "Расхождения кандидатов" in out
    assert "⚠ raw_sha256" not in out  # sha256 совпал с манифестом


def test_main_warns_on_raw_sha256_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sources_root = _fake_sources(tmp_path, "doc-x", _multi_page_pdf(["x"], creator="ocrmypdf 15.2.0"))
    monkeypatch.setattr(ocr_eval, "DEFAULT_SOURCES", sources_root)
    gold = tmp_path / "gold"
    _write_gold(gold, "doc-x", "0" * 64, {1: "x"})  # заведомо неверный sha256

    monkeypatch.setattr(cloud_ocr, "convert_scan", lambda path, language, *, model: "x")

    code = main(["--gold", str(gold), "--models", "cloud-model", "--no-tesseract"])
    assert code == 0
    assert "⚠ raw_sha256 разошёлся" in capsys.readouterr().out
