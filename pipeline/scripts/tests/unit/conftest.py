"""Герметичность unit-тестов от локального окружения разработчика.

На РАБОЧЕЙ машине в ``.env`` лежит настоящий рабочий ``OPENROUTER_API_KEY``
(см. CLAUDE.md) — без явной изоляции поведение ``convert.converters.
cloud_allowed`` (а через него — ``needed_stages``/маршрутизация ``_convert_pdf``)
недетерминированно менялось бы в зависимости от того, кто и где запускает тесты:
в CI ключа нет (гейт закрыт), локально — есть (гейт открыт), один и тот же тест
давал бы разные результаты. Autouse-фикстура фиксирует CI-базовый сценарий
(«ключа нет») для ВСЕХ unit-тестов по умолчанию; тесты, которым нужен открытый
гейт, переопределяют ``OPENROUTER_API_KEY`` в своём теле как обычно (monkeypatch
внутри теста применяется ПОСЛЕ фикстуры и имеет приоритет).

``OPENROUTER_API_KEY=""`` (не ``delenv``) — намеренно: ``core.env.load_dotenv``
использует ``os.environ.setdefault``, который сработал бы на ОТСУТСТВУЮЩЕМ ключе
и тихо подтянул бы настоящий из файла; пустая строка физически ПРИСУТСТВУЕТ в
``os.environ`` (``setdefault`` — no-op) и одновременно фальшива для гейта
(``not os.environ.get(...)``).
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from convert import converters
from core import schema
from discovery import registry_store
from index import corpus_index


@pytest.fixture(autouse=True)
def _hermetic_cloud_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr(converters, "_CLOUD_DISABLED", False)
    monkeypatch.setattr(converters, "_CLOUD_KEY_WARNED", False)


# Боевые машиннописаные артефакты, которые ни один unit-тест не имеет права трогать.
# Прецедент (2026-07-21): два main()-теста (эпоха PR #18) забыли --db — argparse-дефолт
# указал на РЕАЛЬНУЮ pipeline/index/corpus.db, и пустой tmp-корпус «исчезновением» всех
# документов вычистил из боевого индекса chunks/doc_state. CI не ловил (без bge-токенизатора
# стадия индекса «пропущена»), а самовосстановление пайплайна маскировало ущерб тихой полной
# пере-чанковкой на следующем прогоне. Guard превращает тихую порчу в громкий красный тест.
_GUARDED_REAL_ARTIFACTS = (
    corpus_index.DEFAULT_DB,
    schema.DEFAULT_SOURCES / "candidates.yaml",
    schema.DEFAULT_SOURCES / ".discovery_cursors.yaml",
    registry_store.DEFAULT_DB_PATH,
)


def _artifact_snapshot() -> list[tuple[str, tuple[int, int] | None]]:
    """(size, mtime_ns) каждого guarded-файла; None — файла нет (CI-сценарий)."""
    snapshot: list[tuple[str, tuple[int, int] | None]] = []
    for path in _GUARDED_REAL_ARTIFACTS:
        try:
            st = path.stat()
            snapshot.append((str(path), (st.st_size, st.st_mtime_ns)))
        except FileNotFoundError:
            snapshot.append((str(path), None))
    return snapshot


@pytest.fixture(autouse=True)
def _guard_real_artifacts() -> Iterator[None]:
    before = _artifact_snapshot()
    yield
    after = _artifact_snapshot()
    assert after == before, (
        "unit-тест мутировал БОЕВОЙ артефакт — тесту не хватает явного tmp-пути "
        f"(--db / --root / root=tmp_path):\n  до:    {before}\n  после: {after}"
    )
