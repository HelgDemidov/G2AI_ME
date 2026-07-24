"""Тест самого guard-механизма ``tests/unit/conftest.py`` (spec discovery-snowball §7):
искусственная мутация guarded-артефакта БЕЗ tmp-root детектируется — не только
предполагается по конструкции. Зовёт ``_artifact_snapshot``/``_assert_artifacts_unchanged``
напрямую (обычный Python-импорт conftest как модуля пакета `tests.unit`, НЕ через
внутренности generator-фикстуры pytest — устойчивее к версиям pytest)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit import conftest as guard_conftest


def test_snowball_leads_path_is_in_guarded_artifacts() -> None:
    """discovery-snowball §7: ``.snowball_leads.yaml`` — тот же класс ловушки
    (main()-путь без явного ``--root``), что ``candidates.yaml``/``.discovery_cursors.yaml``."""
    from core import schema

    assert (schema.DEFAULT_SOURCES / ".snowball_leads.yaml") in guard_conftest._GUARDED_REAL_ARTIFACTS


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
