"""discovery/orchestrate.py — прогон enabled-коннекторов с изоляцией отказов (spec §5).

Зеркалит `run_pipeline.process_docs`: отказ одного коннектора не рвёт прогон, а
логируется в сводку. Реконсиляционный инвариант: повторный `discover()` по
неизменённому upstream-состоянию (курсоры + dedup против уже персистнутых
кандидатов) — no-op, ноль новых `fresh`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core import schema
from discovery import registry, store
from discovery.base import Connector
from discovery.dedup import dedup


@dataclass
class ConnectorRunSummary:
    """Итог прогона одного коннектора: сколько нашёл / сколько реально свежих / сколько
    поглотил dedup / ошибка (если коннектор упал — прогон остальных не прерван)."""

    connector_id: str
    found: int = 0
    fresh: int = 0
    merged: int = 0
    error: str | None = None


@dataclass
class DiscoverySummary:
    connectors: list[ConnectorRunSummary] = field(default_factory=list)
    dry_run: bool = False

    @property
    def total_fresh(self) -> int:
        return sum(c.fresh for c in self.connectors)

    @property
    def failed(self) -> list[ConnectorRunSummary]:
        return [c for c in self.connectors if c.error is not None]


def discover(
    only: list[str] | None = None,
    *,
    root: Path = schema.DEFAULT_SOURCES,
    dry_run: bool = False,
    connectors_override: list[Connector] | None = None,
) -> DiscoverySummary:
    """Прогнать enabled-коннекторы (или подмножество ``only``), dedup'нуть и персистить.

    Кросс-коннекторный dedup: каждый коннектор сверяется против (уже персистнутых +
    уже собранных в ЭТОМ прогоне более ранними коннекторами) кандидатов — документ,
    найденный двумя коннекторами за один прогон, схлопывается так же, как если бы
    они запускались раздельно (чартер §4.4).

    ``connectors_override`` — явный список коннекторов ВМЕСТО чтения реестра (спек
    discovery-snowball §3): CLI-подкоманда `snowball` строит один переопределённый
    экземпляр `SnowballConnector` из yaml+флагов на один прогон, не мутируя ни реестр,
    ни файл конфига. ``only`` в этом режиме игнорируется — сам список уже финальный.
    Без параметра (``None``, дефолт) поведение НЕ меняется — обычный путь через реестр.
    """
    candidates_path = root / "candidates.yaml"
    cursors_path = root / ".discovery_cursors.yaml"

    existing = store.load(candidates_path)
    cursors = store.load_cursors(cursors_path)

    summaries: list[ConnectorRunSummary] = []
    fresh_this_run: list[schema.CandidateRecord] = []

    connectors = connectors_override if connectors_override is not None else registry.enabled_connectors(only)
    for connector in connectors:
        try:
            result = connector.discover(cursors.get(connector.id))
        except Exception as exc:  # noqa: BLE001 — изоляция отказов, зеркало run_pipeline
            summaries.append(ConnectorRunSummary(connector_id=connector.id, error=str(exc)))
            continue

        fresh, merged = dedup(result.candidates, existing + fresh_this_run)
        fresh_this_run.extend(fresh)
        cursors[connector.id] = result.cursor
        summaries.append(
            ConnectorRunSummary(
                connector_id=connector.id,
                found=len(result.candidates),
                fresh=len(fresh),
                merged=merged,
            )
        )

    if not dry_run:
        store.save(existing + fresh_this_run, candidates_path)
        store.save_cursors(cursors, cursors_path)

    return DiscoverySummary(connectors=summaries, dry_run=dry_run)
