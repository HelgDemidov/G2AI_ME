"""Тесты discovery/base.py: контракт Connector/DiscoverResult (spec discovery-core §1)."""
from __future__ import annotations

import datetime as dt

from core import schema
from discovery.base import Connector, DiscoverResult


def _candidate(**overrides: object) -> schema.CandidateRecord:
    fields: dict[str, object] = {
        "connector_id": "manual",
        "retrieved_at": dt.date(2026, 7, 21),
        "raw_hash": "abc123",
    }
    fields.update(overrides)
    return schema.CandidateRecord.model_validate(fields)


def test_discover_result_holds_candidates_and_default_diagnostics() -> None:
    cand = _candidate()
    result = DiscoverResult(candidates=[cand])

    assert result.candidates == [cand]
    assert result.diagnostics == {}  # default_factory, не мутирует между инстансами


def test_discover_result_is_frozen() -> None:
    result = DiscoverResult(candidates=[])
    try:
        result.candidates = []  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("DiscoverResult должен быть frozen")


def test_discover_result_diagnostics_not_shared_between_instances() -> None:
    a = DiscoverResult(candidates=[])
    b = DiscoverResult(candidates=[])
    assert a.diagnostics is not b.diagnostics


class _FakeConnector:
    """Минимальная реализация Connector protocol — доказательство структурной типизации.

    ``discover()`` без аргументов: коннектор — чистая функция от ИСТОЧНИКА, ядро не
    передаёт ему состояния и не принимает от него (spec drop-cursors-and-decision-overlay §1).
    """

    id = "fake"
    kind = schema.ConnectorKind.manual
    enabled = True

    def discover(self) -> DiscoverResult:
        return DiscoverResult(candidates=[_candidate()], diagnostics={"found": 1})


def test_fake_connector_satisfies_protocol_structurally() -> None:
    connector: Connector = _FakeConnector()  # mypy: структурная проверка; рантайм: просто вызов
    result = connector.discover()
    assert connector.id == "fake"
    assert connector.kind == schema.ConnectorKind.manual
    assert connector.enabled is True
    assert result.diagnostics == {"found": 1}
