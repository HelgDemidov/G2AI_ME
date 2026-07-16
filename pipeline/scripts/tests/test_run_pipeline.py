"""Тесты оркестратора: реконсиляция стадий, синк frontmatter, dry-run, изоляция отказов.

Всё CI-safe — без сети (download), pdfplumber (convert) и модели (index).
Раскладка — папка-документ (corpus-layout-v2): пути выводятся из <root>/<track>/<entity>/<id>/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import acquisition
import schema
from acquisition import AcquisitionOutcome, ClassifiedResponse
from run_pipeline import Stage, _compose_md, _do_download, _do_frontmatter, needed_stages, process_docs
from schema import SourceRecord, render_frontmatter
from test_schema import valid_record


def make(**over: Any) -> SourceRecord:
    data = valid_record()
    data.update(over)
    return SourceRecord.model_validate(data)


def _place(
    rec: SourceRecord,
    root: Path,
    *,
    raw: bytes | None = None,
    md: str | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    """Разложить raw.pdf/doc.md/.state.yaml в выведенную папку-документ."""
    import yaml as _yaml

    d = schema.doc_dir(rec, root)
    d.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (d / "raw.pdf").write_bytes(raw)
    if md is not None:
        (d / "doc.md").write_text(md, encoding="utf-8")
    if state is not None:
        (d / ".state.yaml").write_text(_yaml.safe_dump(state, allow_unicode=True), encoding="utf-8")


def test_needs_all_when_nothing_exists(tmp_path: Path) -> None:
    assert needed_stages(make(), tmp_path) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_needs_convert_when_md_missing(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")  # raw есть, sha неизвестен -> download не нужен
    assert needed_stages(rec, tmp_path) == [Stage.convert, Stage.frontmatter]


def test_sha256_mismatch_triggers_download(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", state={"sha256": "0" * 64})  # sha не совпадёт с содержимым
    assert Stage.download in needed_stages(rec, tmp_path)


def test_up_to_date_no_stages(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))  # синхронный frontmatter, sha неизвестен
    assert needed_stages(rec, tmp_path) == []


def test_force_redoes_all(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))
    assert needed_stages(rec, tmp_path, force=True) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_frontmatter_drift_detected(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md="---\nid: stale-old\n---\n\nBody.\n")
    assert needed_stages(rec, tmp_path) == [Stage.frontmatter]


def test_do_frontmatter_syncs_and_idempotent(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, md="Body only, no frontmatter.\n")
    assert _do_frontmatter(rec, tmp_path) is True
    content = schema.md_file(rec, tmp_path).read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "Body only" in content
    assert _do_frontmatter(rec, tmp_path) is False  # второй раз — уже синхронно


def test_dry_run_no_side_effects(tmp_path: Path) -> None:
    rec = make()
    results, changed = process_docs(
        [rec], tmp_path,
        force=False, dry_run=True, no_download=False, pause=0,
    )
    assert changed is False
    assert results[0].done == [Stage.download, Stage.convert, Stage.frontmatter]
    assert not schema.doc_dir(rec, tmp_path).exists()  # ничего не создано


def test_failure_isolation(tmp_path: Path) -> None:
    a = make(id="a-doc-2026", entity_id="aa")  # нужен download
    b = make(id="b-doc-2026", entity_id="bb")
    _place(b, tmp_path, raw=b"pdf", md=_compose_md(b, ""))  # b актуален

    results, changed = process_docs(
        [a, b], tmp_path,
        force=False, dry_run=False, no_download=True, pause=0,  # a падает на download
    )
    ra = next(r for r in results if r.doc_id == "a-doc-2026")
    rb = next(r for r in results if r.doc_id == "b-doc-2026")
    assert ra.error is not None and "download" in ra.error  # a упал, но не оборвал батч
    assert rb.up_to_date is True
    assert changed is False


def test_do_download_writes_state(tmp_path: Path, monkeypatch: Any) -> None:
    """Сквозная проводка: _do_download -> run_ladder -> запись .state.yaml (не meta.yaml)."""
    rec = make()
    ok = ClassifiedResponse(AcquisitionOutcome.ok, 200, "valid PDF")

    def fake_fetch(url: str, dest: Path, *, user_agent: str, timeout: int = 30) -> ClassifiedResponse:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 fake content")
        return ok

    monkeypatch.setattr("acquisition.fetch_and_classify", fake_fetch)

    _do_download(rec, tmp_path, pause=0)

    assert (schema.doc_dir(rec, tmp_path) / "raw.pdf").read_bytes() == b"%PDF-1.4 fake content"
    st = schema.load_state(schema.state_file(rec, tmp_path))
    assert st.acquisition_method is not None and st.acquisition_method.value == "direct"
    assert st.fidelity is not None and st.fidelity.value == "live"
    assert st.sha256 is not None


def test_do_download_cleans_staging_on_batch_block(tmp_path: Path, monkeypatch: Any) -> None:
    """При AcquisitionBlocked в батче (interactive=False) staging убирается —
    challenge-тело (реалистично: curl -o пишет его в part ДО классификации) не
    остаётся под именем, которое schema.raw_file мог бы усыновить как оригинал."""
    rec = make()

    def fake_run_ladder(rec_: SourceRecord, dest: Path, *, user_agent: str) -> Any:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<html>Attention Required</html>")  # как реальный curl -o на challenge
        raise acquisition.AcquisitionBlocked(rec_.source_url, "direct blocked (WAF challenge)")

    monkeypatch.setattr("acquisition.run_ladder", fake_run_ladder)

    with pytest.raises(acquisition.AcquisitionBlocked):
        _do_download(rec, tmp_path, pause=0, interactive=False)

    doc_dir = schema.doc_dir(rec, tmp_path)
    assert list(doc_dir.glob(".*.part")) == []  # staging убран в finally
    assert schema.raw_file(rec, tmp_path) is None  # challenge не усыновлён как оригинал


def test_do_download_replaces_prior_raw_of_different_ext(tmp_path: Path, monkeypatch: Any) -> None:
    """Смена канала/формата: новый raw.pdf публикуется, прежний raw.html удаляется (single-raw)."""
    rec = make()
    doc_dir = schema.doc_dir(rec, tmp_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "raw.html").write_bytes(b"<html>old html original</html>")

    ok = ClassifiedResponse(AcquisitionOutcome.ok, 200, "valid PDF")

    def fake_fetch(url: str, dest: Path, *, user_agent: str, timeout: int = 30) -> ClassifiedResponse:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 fake content")
        return ok

    monkeypatch.setattr("acquisition.fetch_and_classify", fake_fetch)

    _do_download(rec, tmp_path, pause=0)

    assert schema.raw_file(rec, tmp_path) == doc_dir / "raw.pdf"  # ровно один raw.* — новый
    assert not (doc_dir / "raw.html").exists()


def test_process_docs_cleans_stale_staging_before_planning(tmp_path: Path) -> None:
    """Реконсиляционная чистка (§1d): останки упавшего прогона убираются сами."""
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))  # актуален
    doc_dir = schema.doc_dir(rec, tmp_path)
    (doc_dir / ".raw.pdf.part").write_bytes(b"stale leftover")

    results, changed = process_docs(
        [rec], tmp_path, force=False, dry_run=False, no_download=False, pause=0
    )

    assert not (doc_dir / ".raw.pdf.part").exists()
    assert results[0].up_to_date is True
    assert changed is False


def test_planning_failure_isolated_from_batch(tmp_path: Path) -> None:
    """Папка с двумя raw.* ломает планирование ОДНОГО документа (ValueError у
    raw_file), но не рвёт батч — вопреки прежнему поведению, где needed_stages()
    вызывался вне per-doc try и такая папка убивала весь прогон."""
    broken = make(id="broken-doc-2026", entity_id="bb")
    broken_dir = schema.doc_dir(broken, tmp_path)
    broken_dir.mkdir(parents=True, exist_ok=True)
    (broken_dir / "raw.pdf").write_bytes(b"pdf")
    (broken_dir / "raw.html").write_bytes(b"html")  # два raw.* -> ValueError у raw_file

    ok = make(id="ok-doc-2026", entity_id="oo")
    _place(ok, tmp_path, raw=b"pdf", md=_compose_md(ok, ""))  # актуален

    results, changed = process_docs(
        [broken, ok], tmp_path, force=False, dry_run=False, no_download=False, pause=0
    )

    r_broken = next(r for r in results if r.doc_id == "broken-doc-2026")
    r_ok = next(r for r in results if r.doc_id == "ok-doc-2026")
    assert r_broken.error is not None and r_broken.error.startswith("planning:")
    assert r_ok.up_to_date is True  # батч не оборван
    assert changed is False


def test_render_frontmatter_used_in_compose(tmp_path: Path) -> None:
    rec = make()
    composed = _compose_md(rec, "old body")
    assert composed.startswith(render_frontmatter(rec))
    assert composed.rstrip().endswith("old body")
