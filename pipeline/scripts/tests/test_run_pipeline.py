"""Тесты оркестратора: реконсиляция стадий, синк frontmatter, dry-run, изоляция отказов.

Всё CI-safe — без сети (download), pdfplumber (convert) и модели (index).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from schema import SourceRecord, render_frontmatter
from run_pipeline import Stage, _compose_md, _do_frontmatter, needed_stages, process_docs
from test_schema import valid_record


def make(**over: Any) -> SourceRecord:
    data = valid_record()
    data.update(over)
    return SourceRecord.model_validate(data)


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_needs_all_when_nothing_exists(tmp_path: Path) -> None:
    rec = make(raw_path="raw/d.pdf", md_path="md/d.md")
    assert needed_stages(rec, tmp_path) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_needs_convert_when_md_missing(tmp_path: Path) -> None:
    rec = make(raw_path="raw/d.pdf", md_path="md/d.md")  # sha256 не задан
    _touch(tmp_path / "raw/d.pdf")
    assert needed_stages(rec, tmp_path) == [Stage.convert, Stage.frontmatter]


def test_sha256_mismatch_triggers_download(tmp_path: Path) -> None:
    rec = make(raw_path="raw/d.pdf", md_path="md/d.md", sha256="0" * 64)
    _touch(tmp_path / "raw/d.pdf")  # содержимое не совпадёт с sha256
    assert Stage.download in needed_stages(rec, tmp_path)


def test_up_to_date_no_stages(tmp_path: Path) -> None:
    rec = make(raw_path="raw/d.pdf", md_path="md/d.md")
    _touch(tmp_path / "raw/d.pdf")
    md = tmp_path / "md/d.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_compose_md(rec, ""), encoding="utf-8")  # синхронный frontmatter
    assert needed_stages(rec, tmp_path) == []


def test_force_redoes_all(tmp_path: Path) -> None:
    rec = make(raw_path="raw/d.pdf", md_path="md/d.md")
    _touch(tmp_path / "raw/d.pdf")
    md = tmp_path / "md/d.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_compose_md(rec, ""), encoding="utf-8")
    assert needed_stages(rec, tmp_path, force=True) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_frontmatter_drift_detected(tmp_path: Path) -> None:
    rec = make(raw_path="raw/d.pdf", md_path="md/d.md")
    _touch(tmp_path / "raw/d.pdf")
    md = tmp_path / "md/d.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("---\nid: stale-old\n---\n\nBody.\n", encoding="utf-8")  # frontmatter разошёлся
    assert needed_stages(rec, tmp_path) == [Stage.frontmatter]


def test_do_frontmatter_syncs_and_idempotent(tmp_path: Path) -> None:
    rec = make(md_path="md/d.md")
    md = tmp_path / "md/d.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("Body only, no frontmatter.\n", encoding="utf-8")
    assert _do_frontmatter(rec, tmp_path) is True
    content = md.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "Body only" in content
    assert _do_frontmatter(rec, tmp_path) is False  # второй раз — уже синхронно


def test_dry_run_no_side_effects(tmp_path: Path) -> None:
    rec = make(raw_path="raw/d.pdf", md_path="md/d.md")
    results, changed = process_docs(
        [rec], tmp_path, force=False, dry_run=True, no_download=False, pause=0
    )
    assert changed is False
    assert results[0].done == [Stage.download, Stage.convert, Stage.frontmatter]
    assert not (tmp_path / "raw/d.pdf").exists()  # ничего не создано


def test_failure_isolation(tmp_path: Path) -> None:
    a = make(id="a-doc-2026", raw_path="raw/a.pdf", md_path="md/a.md")  # нужен download
    b = make(id="b-doc-2026", raw_path="raw/b.pdf", md_path="md/b.md")
    _touch(tmp_path / "raw/b.pdf")
    (tmp_path / "md/b.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "md/b.md").write_text(_compose_md(b, ""), encoding="utf-8")  # b актуален

    results, changed = process_docs(
        [a, b], tmp_path, force=False, dry_run=False, no_download=True, pause=0
    )
    ra = next(r for r in results if r.doc_id == "a-doc-2026")
    rb = next(r for r in results if r.doc_id == "b-doc-2026")
    assert ra.error is not None and "download" in ra.error  # a упал, но не оборвал батч
    assert rb.up_to_date is True
    assert changed is False


def test_render_frontmatter_used_in_compose(tmp_path: Path) -> None:
    rec = make(md_path="md/d.md")
    composed = _compose_md(rec, "old body")
    assert composed.startswith(render_frontmatter(rec))
    assert composed.rstrip().endswith("old body")
