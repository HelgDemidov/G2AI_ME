"""Тесты staging-политики: dot-префикс вне глобов raw.*, самовосстановление, атомарная запись."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.fsio import AlreadyLocked, atomic_write_text, cleanup_staging, exclusive_flock, staging_path


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


# --- exclusive_flock: взаимоисключение писателей общего мутируемого состояния
# (spec acquire-convert-seam-hardening §3, В3) ---


def test_exclusive_flock_grants_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "run_pipeline.lock"
    with exclusive_flock(path):
        pass  # не бросает; лок снят по выходу из контекста
    with exclusive_flock(path):
        pass  # повторный захват после освобождения проходит


def test_exclusive_flock_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "run_pipeline.lock"
    with exclusive_flock(path):
        assert path.exists()


def test_exclusive_flock_second_open_conflicts_within_same_process(tmp_path: Path) -> None:
    """Живая проба 2026-07-27: два независимых open()+flock() конфликтуют и ВНУТРИ
    одного процесса (per-open-file-description locking) — лок тестируем герметично,
    без запуска второго процесса/subprocess."""
    path = tmp_path / "run_pipeline.lock"
    with exclusive_flock(path):
        with pytest.raises(AlreadyLocked, match="уже занято"):
            with exclusive_flock(path):
                pass  # не должно дойти сюда


def test_exclusive_flock_available_again_after_release(tmp_path: Path) -> None:
    path = tmp_path / "run_pipeline.lock"
    with exclusive_flock(path):
        pass
    with exclusive_flock(path):
        pass  # прежний лок снят ядром по выходу из первого контекста — не протухает
