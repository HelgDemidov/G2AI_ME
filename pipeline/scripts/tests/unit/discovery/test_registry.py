"""Тесты discovery/registry.py: register/enabled_connectors (spec discovery-core §2).

Приёмочный тест требования чартера §4: fake-коннектор подхватывается БЕЗ правок реестра.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from core import schema
from discovery import registry
from discovery.base import ConnectorCursor, DiscoverResult


class _FakeConnector:
    def __init__(self, cid: str, *, enabled: bool = True) -> None:
        self.id = cid
        self.kind = schema.ConnectorKind.manual
        self.enabled = enabled

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        return DiscoverResult(candidates=[], cursor={})


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Реестр — модульный синглтон; изолировать тесты друг от друга и от реальных коннекторов."""
    saved = dict(registry.CONNECTORS)
    registry.CONNECTORS.clear()
    yield
    registry.CONNECTORS.clear()
    registry.CONNECTORS.update(saved)


def test_register_adds_connector_by_id() -> None:
    conn = _FakeConnector("fake")
    registry.register(conn)
    assert registry.CONNECTORS["fake"] is conn


def test_register_duplicate_id_raises() -> None:
    registry.register(_FakeConnector("fake"))
    with pytest.raises(ValueError, match="fake"):
        registry.register(_FakeConnector("fake"))


def test_enabled_connectors_filters_disabled() -> None:
    registry.register(_FakeConnector("on", enabled=True))
    registry.register(_FakeConnector("off", enabled=False))
    ids = {c.id for c in registry.enabled_connectors()}
    assert ids == {"on"}


def test_enabled_connectors_only_narrows_selection() -> None:
    registry.register(_FakeConnector("a"))
    registry.register(_FakeConnector("b"))
    ids = {c.id for c in registry.enabled_connectors(only=["b"])}
    assert ids == {"b"}


def test_enabled_connectors_only_disabled_excluded_even_if_requested() -> None:
    registry.register(_FakeConnector("off", enabled=False))
    assert registry.enabled_connectors(only=["off"]) == []


def test_enabled_connectors_only_unknown_id_raises() -> None:
    registry.register(_FakeConnector("a"))
    with pytest.raises(ValueError, match="ghost"):
        registry.enabled_connectors(only=["ghost"])


def test_fake_connector_registers_without_core_edits() -> None:
    """Доказательство требования чартера §4: реестр ничего не знает о _FakeConnector заранее."""
    registry.register(_FakeConnector("proof"))
    result = registry.enabled_connectors(only=["proof"])[0].discover(None)
    assert result.candidates == []
