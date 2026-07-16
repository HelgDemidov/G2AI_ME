"""Тесты staging-политики: dot-префикс вне глобов raw.*, самовосстановление, атомарная запись."""
from __future__ import annotations

from pathlib import Path

from fsio import atomic_write_text, cleanup_staging, staging_path


def test_staging_path_is_dot_prefixed(tmp_path: Path) -> None:
    target = tmp_path / "raw.pdf"
    part = staging_path(target)
    assert part.name == ".raw.pdf.part"
    assert part.parent == target.parent


def test_staging_path_not_matched_by_raw_glob(tmp_path: Path) -> None:
    target = tmp_path / "raw.pdf"
    staging_path(target).write_bytes(b"challenge body")
    assert list(tmp_path.glob("raw.*")) == []


def test_cleanup_staging_removes_only_part_files(tmp_path: Path) -> None:
    (tmp_path / ".raw.pdf.part").write_bytes(b"stale")
    (tmp_path / ".doc.md.part").write_bytes(b"stale")
    (tmp_path / "raw.pdf").write_bytes(b"real")
    cleanup_staging(tmp_path)
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"raw.pdf"}


def test_cleanup_staging_missing_directory_is_noop(tmp_path: Path) -> None:
    cleanup_staging(tmp_path / "does-not-exist")  # не должно бросать


def test_atomic_write_text_creates_target(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "doc.md"
    atomic_write_text(target, "содержимое")
    assert target.read_text(encoding="utf-8") == "содержимое"
    assert not staging_path(target).exists()  # staging убран после rename


def test_atomic_write_text_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("старое", encoding="utf-8")
    atomic_write_text(target, "новое")
    assert target.read_text(encoding="utf-8") == "новое"
