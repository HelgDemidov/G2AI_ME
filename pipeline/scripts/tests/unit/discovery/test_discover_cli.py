"""Тесты discover.py CLI: подкоманда `discover` — argparse + вызов orchestrate.discover
(spec discovery-core §5)."""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest

from core import schema
from discover import main
from discovery import registry, store
from discovery.base import ConnectorCursor, DiscoverResult


class _StaticConnector:
    def __init__(self, cid: str) -> None:
        self.id = cid
        self.kind = schema.ConnectorKind.manual
        self.enabled = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        cand = schema.CandidateRecord.model_validate(
            {
                "connector_id": self.id,
                "connector_kind": schema.ConnectorKind.manual,
                "retrieved_at": dt.date(2026, 7, 21),
                "source_ref": f"doc-{self.id}",
                "raw_hash": f"doc-{self.id}",
            }
        )
        return DiscoverResult(candidates=[cand], cursor={})


class _BoomConnector:
    id = "boom"
    kind = schema.ConnectorKind.manual
    enabled = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        raise RuntimeError("down")


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    saved = dict(registry.CONNECTORS)
    registry.CONNECTORS.clear()
    yield
    registry.CONNECTORS.clear()
    registry.CONNECTORS.update(saved)


def test_discover_subcommand_runs_and_persists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry.register(_StaticConnector("a"))

    code = main(["discover", "--root", str(tmp_path)])

    assert code == 0
    assert len(store.load(tmp_path / "candidates.yaml")) == 1
    assert "1 новых кандидат" in capsys.readouterr().out


def test_discover_subcommand_dry_run_does_not_write(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))

    code = main(["discover", "--root", str(tmp_path), "--dry-run"])

    assert code == 0
    assert not (tmp_path / "candidates.yaml").exists()


def test_discover_subcommand_only_narrows_connectors(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))
    registry.register(_StaticConnector("b"))

    main(["discover", "--root", str(tmp_path), "--only", "a"])

    loaded = store.load(tmp_path / "candidates.yaml")
    assert [c.connector_id for c in loaded] == ["a"]


def test_discover_subcommand_nonzero_exit_on_connector_failure(tmp_path: Path) -> None:
    registry.register(_BoomConnector())

    assert main(["discover", "--root", str(tmp_path)]) == 1


def test_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
