"""discovery/base.py — connector-agnostic контракт (Connector protocol, DiscoverResult).

Чартер `docs/pipeline/discovery/charters/architecture.md` §4.2; спек discovery-core §1.
Коннектор реализует ровно одно: как из своего источника породить `CandidateRecord`-ы.
Он НЕ пишет в store и НЕ знает о dedup — персист и кросс-коннекторное слияние решает
оркестратор ядра (`discovery/orchestrate.py`); это делает коннектор тестируемым как
чистую функцию от курсора.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core import schema

ConnectorCursor = dict[str, Any]
"""Непрозрачный для ядра watermark коннектора (JSON/YAML-сериализуемый — персистится как есть)."""


@dataclass(frozen=True)
class DiscoverResult:
    """Результат одного прогона коннектора: свежие кандидаты + новый курсор + диагностика."""

    candidates: list[schema.CandidateRecord]
    cursor: ConnectorCursor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Connector(Protocol):
    """Протокол discovery-коннектора. Реализации — модули `discovery/connectors/*`."""

    id: str
    kind: schema.ConnectorKind
    enabled: bool

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        """Породить кандидатов с учётом курсора предыдущего прогона (``None`` — первый прогон)."""
        ...
