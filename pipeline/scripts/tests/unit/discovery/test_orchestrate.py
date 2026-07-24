"""Тесты discovery/orchestrate.py: прогон с изоляцией отказов + идемпотентность (spec §5)."""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest

from core import schema
from discovery import registry, store
from discovery.base import ConnectorCursor, DiscoverResult
from discovery.orchestrate import discover


def _candidate(connector_id: str, ref: str, **overrides: object) -> schema.CandidateRecord:
    fields: dict[str, object] = {
        "connector_id": connector_id,
        "retrieved_at": dt.date(2026, 7, 21),
        "raw_hash": ref,
        "normalized_url": f"https://example.gov/{ref}",
    }
    fields.update(overrides)
    return schema.CandidateRecord.model_validate(fields)


class _StaticConnector:
    """Всегда возвращает один и тот же набор кандидатов — идемпотентность-пробa."""

    def __init__(self, cid: str, candidates: list[schema.CandidateRecord]) -> None:
        self.id = cid
        self.kind = schema.ConnectorKind.manual
        self.enabled = True
        self._candidates = candidates
        self.calls: list[ConnectorCursor | None] = []

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        self.calls.append(cursor)
        return DiscoverResult(candidates=list(self._candidates), cursor={"n": len(self.calls)})


class _FailingConnector:
    id = "broken"
    kind = schema.ConnectorKind.manual
    enabled = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        raise RuntimeError("upstream недоступен")


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    saved = dict(registry.CONNECTORS)
    registry.CONNECTORS.clear()
    yield
    registry.CONNECTORS.clear()
    registry.CONNECTORS.update(saved)


def test_discover_persists_fresh_candidates(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a", [_candidate("a", "doc1")]))

    summary = discover(root=tmp_path)

    assert summary.total_fresh == 1
    assert summary.failed == []
    loaded = store.load(tmp_path / "candidates.yaml")
    assert len(loaded) == 1
    assert loaded[0].raw_hash == "doc1"


def test_discover_persists_cursor_returned_by_connector(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a", [_candidate("a", "doc1")]))
    discover(root=tmp_path)
    assert store.load_cursors(tmp_path / ".discovery_cursors.yaml") == {"a": {"n": 1}}


def test_failing_connector_does_not_abort_run(tmp_path: Path) -> None:
    registry.register(_FailingConnector())
    registry.register(_StaticConnector("ok", [_candidate("ok", "doc1")]))

    summary = discover(root=tmp_path)

    assert [c.connector_id for c in summary.failed] == ["broken"]
    assert summary.failed[0].error == "upstream недоступен"
    ok_summary = next(c for c in summary.connectors if c.connector_id == "ok")
    assert ok_summary.fresh == 1
    # отказавший коннектор не должен помешать персисту результата рабочего
    assert len(store.load(tmp_path / "candidates.yaml")) == 1


def test_repeat_run_with_unchanged_upstream_is_idempotent(tmp_path: Path) -> None:
    conn = _StaticConnector("a", [_candidate("a", "doc1")])
    registry.register(conn)

    first = discover(root=tmp_path)
    second = discover(root=tmp_path)

    assert first.total_fresh == 1
    assert second.total_fresh == 0  # dedup против уже персистнутого — no-op
    assert len(store.load(tmp_path / "candidates.yaml")) == 1  # без дублей на диске


def test_dry_run_does_not_write_store_or_cursors(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a", [_candidate("a", "doc1")]))

    summary = discover(root=tmp_path, dry_run=True)

    assert summary.total_fresh == 1  # сводка честная...
    assert not (tmp_path / "candidates.yaml").exists()  # ...но диск не тронут
    assert not (tmp_path / ".discovery_cursors.yaml").exists()


def test_two_connectors_same_run_fold_into_one_candidate(tmp_path: Path) -> None:
    """Один и тот же документ, найденный двумя коннекторами ЗА ОДИН прогон, схлопывается
    (чартер §4.4) — не только против уже персистнутого, но и внутри самого прогона."""
    registry.register(_StaticConnector("a", [_candidate("a", "doc1")]))
    registry.register(_StaticConnector("b", [_candidate("b", "doc1")]))

    summary = discover(root=tmp_path)

    assert summary.total_fresh == 1
    b_summary = next(c for c in summary.connectors if c.connector_id == "b")
    assert b_summary.merged == 1
    assert len(store.load(tmp_path / "candidates.yaml")) == 1


def test_only_narrows_which_connectors_run(tmp_path: Path) -> None:
    conn_a = _StaticConnector("a", [_candidate("a", "doc1")])
    conn_b = _StaticConnector("b", [_candidate("b", "doc2")])
    registry.register(conn_a)
    registry.register(conn_b)

    discover(root=tmp_path, only=["a"])

    assert conn_a.calls == [None]
    assert conn_b.calls == []


# --- connectors_override (discovery-snowball §3): CLI-подкоманда `snowball` строит
# коннектор с переопределённым конфигом на один прогон, минуя реестр целиком ---


def test_connectors_override_used_when_registry_is_empty(tmp_path: Path) -> None:
    """Реестр пуст (ничего не зарегистрировано) — override всё равно работает, dedup/
    персист/cursor-запись идут тем же путём, что и для обычных registry-коннекторов."""
    assert registry.CONNECTORS == {}
    override_conn = _StaticConnector("snowball", [_candidate("snowball", "found1")])

    summary = discover(root=tmp_path, connectors_override=[override_conn])

    assert summary.total_fresh == 1
    assert override_conn.calls == [None]
    assert len(store.load(tmp_path / "candidates.yaml")) == 1
    cursors = store.load_cursors(tmp_path / ".discovery_cursors.yaml")
    assert cursors["snowball"] == {"n": 1}


def test_connectors_override_ignores_only_param(tmp_path: Path) -> None:
    """``only`` с неизвестным реестру id обычно кидает ValueError (registry.enabled_
    connectors) — override обходит эту проверку целиком, реестр вообще не читается."""
    override_conn = _StaticConnector("snowball", [])

    summary = discover(root=tmp_path, only=["nonexistent-id"], connectors_override=[override_conn])

    assert summary.connectors[0].connector_id == "snowball"


def test_connectors_override_absent_falls_back_to_registry(tmp_path: Path) -> None:
    """Регресс-тест: параметр отсутствует (``None``, дефолт) -> поведение НЕ меняется,
    обычный путь через ``registry.enabled_connectors`` работает как раньше."""
    registry.register(_StaticConnector("a", [_candidate("a", "doc1")]))

    summary = discover(root=tmp_path)

    assert summary.connectors[0].connector_id == "a"
    assert summary.total_fresh == 1
