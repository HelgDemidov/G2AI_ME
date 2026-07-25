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
from pathlib import Path
from typing import Any

import pytest

from acquire import acquisition
from convert import converters
from discovery import registry_store, store
from index import corpus_index


@pytest.fixture(autouse=True)
def _hermetic_cloud_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr(converters, "_CLOUD_DISABLED", False)
    monkeypatch.setattr(converters, "_CLOUD_KEY_WARNED", False)


@pytest.fixture(autouse=True)
def _hermetic_savepagenow(monkeypatch: Any) -> None:
    """Проактивный снимок (spec post-acquisition-lifecycle §4) — единственный сетевой
    вызов, который ``_do_download`` делает САМ, вне замоканной лестницы: любой уже
    существующий тест успешной добычи иначе начал бы молча дёргать web.archive.org.

    Заглушка возвращает True (успех), поэтому наблюдаемое поведение — как у рабочего
    SPN; тесты, проверяющие сам вызов/отказ, ставят свой спай поверх (monkeypatch
    внутри теста применяется ПОСЛЕ фикстуры)."""
    monkeypatch.setattr(acquisition, "request_snapshot", lambda url, **kw: True)


# Боевые машиннописаные артефакты, которые ни один unit-тест не имеет права трогать.
# Прецедент (2026-07-21): два main()-теста (эпоха PR #18) забыли --db — argparse-дефолт
# указал на РЕАЛЬНУЮ pipeline/index/corpus.db, и пустой tmp-корпус «исчезновением» всех
# документов вычистил из боевого индекса chunks/doc_state. CI не ловил (без bge-токенизатора
# стадия индекса «пропущена»), а самовосстановление пайплайна маскировало ущерб тихой полной
# пере-чанковкой на следующем прогоне. Guard превращает тихую порчу в громкий красный тест.
_GUARDED_REAL_ARTIFACTS: tuple[Path, ...] = (
    corpus_index.DEFAULT_DB,
    # Легаси-монолит слоя кандидатов: остаётся под guard'ом до конца миграционного
    # периода (spec discovery-candidates-sharding §6) — пока он существует, именно он
    # источник истины store (правило прецедентности `store.load`), и тихая мутация
    # именно его была бы порчей боевых данных.
    store.LEGACY_CANDIDATES_PATH,
    registry_store.DEFAULT_DB_PATH,
)

# Боевые КАТАЛОГИ, чей состав тоже неприкосновенен (spec discovery-candidates-sharding §6):
# для шардированного store недостаточно сторожить один файл — правку шарда на месте И
# появление/исчезновение шарда ловит только снимок всего каталога. `.state/` (курсоры,
# лиды snowball, в будущем dangling-цитаты graph-v2) сторожится тем же механизмом с
# 2026-07-25: раньше это были два отдельных файла в списке выше, и КАЖДЫЙ новый
# операционный артефакт требовал правки списка — каталог закрывает класс целиком.
_GUARDED_REAL_DIRS: tuple[Path, ...] = (store.CANDIDATES_DIR, store.STATE_DIR)


def _artifact_snapshot() -> list[tuple[str, tuple[int, int] | None]]:
    """(size, mtime_ns) каждого guarded-файла; None — файла нет (CI-сценарий).

    Для guarded-КАТАЛОГОВ снимок берётся по каждому их ``*.yaml`` (шарду) — состав
    каталога входит в сравнение, поэтому появившийся или исчезнувший шард виден так же,
    как правка существующего.
    """
    snapshot: list[tuple[str, tuple[int, int] | None]] = []
    paths = list(_GUARDED_REAL_ARTIFACTS)
    for directory in _GUARDED_REAL_DIRS:
        paths.extend(sorted(directory.glob("*.yaml")))
    for path in paths:
        try:
            st = path.stat()
            snapshot.append((str(path), (st.st_size, st.st_mtime_ns)))
        except FileNotFoundError:
            snapshot.append((str(path), None))
    return snapshot


def _assert_artifacts_unchanged(
    before: list[tuple[str, tuple[int, int] | None]], after: list[tuple[str, tuple[int, int] | None]]
) -> None:
    """Вынесено из фикстуры в отдельную функцию (spec discovery-snowball §7): тест на
    сам guard-механизм зовёт ``_artifact_snapshot``/эту функцию напрямую, не полагаясь
    на внутренности generator-фикстуры pytest."""
    assert after == before, (
        "unit-тест мутировал БОЕВОЙ артефакт — тесту не хватает явного tmp-пути "
        f"(--db / --root / root=tmp_path):\n  до:    {before}\n  после: {after}"
    )


@pytest.fixture(autouse=True)
def _guard_real_artifacts() -> Iterator[None]:
    before = _artifact_snapshot()
    yield
    after = _artifact_snapshot()
    _assert_artifacts_unchanged(before, after)
