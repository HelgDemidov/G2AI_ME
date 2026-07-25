"""Тест самого guard-механизма ``tests/unit/conftest.py`` (spec discovery-snowball §7):
искусственная мутация guarded-артефакта БЕЗ tmp-root детектируется — не только
предполагается по конструкции. Зовёт ``_artifact_snapshot``/``_assert_artifacts_unchanged``
напрямую (обычный Python-импорт conftest как модуля пакета `tests.unit`, НЕ через
внутренности generator-фикстуры pytest — устойчивее к версиям pytest)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.unit import conftest as guard_conftest


def test_snowball_leads_path_is_in_guarded_artifacts() -> None:
    """discovery-snowball §7: ``.snowball_leads.yaml`` — тот же класс ловушки
    (main()-путь без явного ``--root``), что ``candidates.yaml``/``.discovery_cursors.yaml``."""
    from core import schema

    assert (schema.DEFAULT_SOURCES / ".snowball_leads.yaml") in guard_conftest._GUARDED_REAL_ARTIFACTS


def test_candidates_store_is_guarded_both_layouts() -> None:
    """Шардированный store сторожится КАТАЛОГОМ, легаси-монолит — файлом (оба до конца
    миграционного периода): пока монолит существует, именно он источник истины store."""
    from discovery import store

    assert store.LEGACY_CANDIDATES_PATH in guard_conftest._GUARDED_REAL_ARTIFACTS
    assert store.CANDIDATES_DIR in guard_conftest._GUARDED_REAL_DIRS


def test_guard_detects_content_mutation_of_existing_artifact(tmp_path: Path) -> None:
    """Ядро теста: снимок ДО -> реальная мутация файла (без tmp-root у ГИПОТЕТИЧЕСКОГО
    вызывающего теста) -> снимок ПОСЛЕ -> сравнение обязано бросить AssertionError.
    ``_GUARDED_REAL_ARTIFACTS`` подменяется на фейковый путь внутри ``tmp_path`` — сам
    этот тест не трогает НИКАКОЙ реальный боевой файл."""
    fake = tmp_path / "fake_guarded.yaml"
    fake.write_text("original", encoding="utf-8")

    real_paths = guard_conftest._GUARDED_REAL_ARTIFACTS
    guard_conftest._GUARDED_REAL_ARTIFACTS = (fake,)
    try:
        before = guard_conftest._artifact_snapshot()
        fake.write_text("mutated content, different size", encoding="utf-8")
        after = guard_conftest._artifact_snapshot()
        with pytest.raises(AssertionError, match="БОЕВОЙ артефакт"):
            guard_conftest._assert_artifacts_unchanged(before, after)
    finally:
        guard_conftest._GUARDED_REAL_ARTIFACTS = real_paths


def test_guard_passes_when_nothing_mutated(tmp_path: Path) -> None:
    fake = tmp_path / "untouched.yaml"
    fake.write_text("stable", encoding="utf-8")

    real_paths = guard_conftest._GUARDED_REAL_ARTIFACTS
    guard_conftest._GUARDED_REAL_ARTIFACTS = (fake,)
    try:
        before = guard_conftest._artifact_snapshot()
        after = guard_conftest._artifact_snapshot()
        guard_conftest._assert_artifacts_unchanged(before, after)  # не должно бросить
    finally:
        guard_conftest._GUARDED_REAL_ARTIFACTS = real_paths


def test_guard_passes_when_artifact_absent_both_times() -> None:
    """CI-сценарий: артефакт вовсе не существует ни до, ни после — не ошибка."""
    missing = Path("/nonexistent/definitely/not/here.yaml")
    real_paths = guard_conftest._GUARDED_REAL_ARTIFACTS
    guard_conftest._GUARDED_REAL_ARTIFACTS = (missing,)
    try:
        before = guard_conftest._artifact_snapshot()
        after = guard_conftest._artifact_snapshot()
        guard_conftest._assert_artifacts_unchanged(before, after)
    finally:
        guard_conftest._GUARDED_REAL_ARTIFACTS = real_paths


# --- guarded КАТАЛОГИ: шардированный store (spec discovery-candidates-sharding §6) ---


@pytest.fixture()
def guarded_dir(tmp_path: Path, monkeypatch: Any) -> Path:
    """Фейковый guarded-каталог внутри ``tmp_path``: тесты ниже не трогают реальный
    ``sources/candidates/`` вовсе (ни чтением stat, ни записью)."""
    directory = tmp_path / "shards"
    directory.mkdir()
    monkeypatch.setattr(guard_conftest, "_GUARDED_REAL_ARTIFACTS", ())
    monkeypatch.setattr(guard_conftest, "_GUARDED_REAL_DIRS", (directory,))
    return directory


def test_guard_detects_shard_content_mutation(guarded_dir: Path) -> None:
    """Правка шарда НА МЕСТЕ — ровно то, что один файл-guard поймать не мог бы."""
    shard = guarded_dir / "manual.yaml"
    shard.write_text("- title: original\n", encoding="utf-8")

    before = guard_conftest._artifact_snapshot()
    shard.write_text("- title: mutated, different size\n", encoding="utf-8")
    after = guard_conftest._artifact_snapshot()

    with pytest.raises(AssertionError, match="БОЕВОЙ артефакт"):
        guard_conftest._assert_artifacts_unchanged(before, after)


def test_guard_detects_new_shard_appearing(guarded_dir: Path) -> None:
    """Появление шарда (тест «прогнал» коннектор по боевому корню) — тоже мутация store."""
    before = guard_conftest._artifact_snapshot()
    (guarded_dir / "oecd.yaml").write_text("- title: leaked\n", encoding="utf-8")
    after = guard_conftest._artifact_snapshot()

    with pytest.raises(AssertionError, match="БОЕВОЙ артефакт"):
        guard_conftest._assert_artifacts_unchanged(before, after)


def test_guard_detects_shard_disappearing(guarded_dir: Path) -> None:
    """Исчезновение шарда (опустевший шард удалён боевым save) — симметрично появлению."""
    shard = guarded_dir / "agora.yaml"
    shard.write_text("- title: present\n", encoding="utf-8")

    before = guard_conftest._artifact_snapshot()
    shard.unlink()
    after = guard_conftest._artifact_snapshot()

    with pytest.raises(AssertionError, match="БОЕВОЙ артефакт"):
        guard_conftest._assert_artifacts_unchanged(before, after)


def test_guard_passes_when_guarded_dir_untouched(guarded_dir: Path) -> None:
    (guarded_dir / "manual.yaml").write_text("- title: stable\n", encoding="utf-8")

    before = guard_conftest._artifact_snapshot()
    after = guard_conftest._artifact_snapshot()

    guard_conftest._assert_artifacts_unchanged(before, after)  # не должно бросить


def test_guard_passes_when_guarded_dir_absent(tmp_path: Path, monkeypatch: Any) -> None:
    """CI/свежий клон: каталога шардов нет вовсе — не ошибка, а нормальный сценарий."""
    monkeypatch.setattr(guard_conftest, "_GUARDED_REAL_ARTIFACTS", ())
    monkeypatch.setattr(guard_conftest, "_GUARDED_REAL_DIRS", (tmp_path / "never-created",))

    before = guard_conftest._artifact_snapshot()
    after = guard_conftest._artifact_snapshot()

    assert before == after == []
    guard_conftest._assert_artifacts_unchanged(before, after)
