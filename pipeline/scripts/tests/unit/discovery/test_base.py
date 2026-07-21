"""Тесты discovery/base.py: контракт Connector/DiscoverResult (spec discovery-core §1)."""
from __future__ import annotations

import datetime as dt

from core import schema
from discovery.base import Connector, ConnectorCursor, DiscoverResult


def _candidate(**overrides: object) -> schema.CandidateRecord:
    fields: dict[str, object] = {
        "connector_id": "manual",
        "connector_kind": schema.ConnectorKind.manual,
        "retrieved_at": dt.date(2026, 7, 21),
        "source_ref": "https://example.gov/doc",
        "raw_hash": "abc123",
    }
    fields.update(overrides)
    return schema.CandidateRecord.model_validate(fields)


def test_discover_result_holds_candidates_cursor_and_default_diagnostics() -> None:
    cand = _candidate()
    result = DiscoverResult(candidates=[cand], cursor={"seen": ["abc123"]})

    assert result.candidates == [cand]
    assert result.cursor == {"seen": ["abc123"]}
    assert result.diagnostics == {}  # default_factory, не мутирует между инстансами


def test_discover_result_is_frozen() -> None:
    result = DiscoverResult(candidates=[], cursor={})
    try:
        result.cursor = {"x": 1}  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("DiscoverResult должен быть frozen")


def test_discover_result_diagnostics_not_shared_between_instances() -> None:
    a = DiscoverResult(candidates=[], cursor={})
    b = DiscoverResult(candidates=[], cursor={})
    assert a.diagnostics is not b.diagnostics


class _FakeConnector:
    """Минимальная реализация Connector protocol — доказательство структурной типизации."""

    id = "fake"
    kind = schema.ConnectorKind.manual
    enabled = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        return DiscoverResult(candidates=[_candidate()], cursor={"seen": True}, diagnostics={"found": 1})


def test_fake_connector_satisfies_protocol_structurally() -> None:
    connector: Connector = _FakeConnector()  # mypy: структурная проверка; рантайм: просто вызов
    result = connector.discover(None)
    assert connector.id == "fake"
    assert connector.kind == schema.ConnectorKind.manual
    assert connector.enabled is True
    assert result.diagnostics == {"found": 1}
