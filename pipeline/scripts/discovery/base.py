"""discovery/base.py — connector-agnostic контракт (Connector protocol, DiscoverResult).

Чартер `docs/pipeline/discovery/charters/architecture.md` §4.2; спек discovery-core §1.
Коннектор реализует ровно одно: как из своего источника породить `CandidateRecord`-ы.
Он НЕ пишет в store и НЕ знает о dedup — персист и кросс-коннекторное слияние решает
оркестратор ядра (`discovery/orchestrate.py`).

**Коннектор — чистая функция от ИСТОЧНИКА** (spec drop-cursors-and-decision-overlay §1):
ядро не передаёт ему состояния и не принимает от него. Коннектор всегда отдаёт полный
текущий вид источника, а отсев уже известного — единственная работа `dedup`, который
делает это строго лучше: кросс-коннекторно, по URL и по паре issuer+title, а не по
собственному id-пространству одного источника.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core import schema


@dataclass(frozen=True)
class DiscoverResult:
    """Результат одного прогона коннектора: полный текущий вид источника + диагностика."""

    candidates: list[schema.CandidateRecord]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Connector(Protocol):
    """Протокол discovery-коннектора. Реализации — модули `discovery/connectors/*`."""

    id: str
    kind: schema.ConnectorKind
    enabled: bool

    def discover(self) -> DiscoverResult:
        """Породить кандидатов по текущему состоянию источника."""
        ...
