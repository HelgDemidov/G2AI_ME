"""Тесты discovery/store.py: персист candidates.yaml + .discovery_cursors.yaml
(spec discovery-core §4)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from core import schema
from discovery import store


def _candidate(**overrides: object) -> schema.CandidateRecord:
    fields: dict[str, object] = {
        "connector_id": "manual",
        "connector_kind": schema.ConnectorKind.manual,
        "retrieved_at": dt.date(2026, 7, 21),
        "source_ref": "https://example.gov/doc",
        "raw_hash": "h0",
        "title": "Example Document",
        "native_tags": ["ai-governance"],
    }
    fields.update(overrides)
    return schema.CandidateRecord.model_validate(fields)


# --- candidates -------------------------------------------------------------------


def test_load_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert store.load(tmp_path / "candidates.yaml") == []


def test_save_load_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    cand = _candidate()
    cand.merged_connector_ids = ["agora"]  # type: ignore[attr-defined]  # extra="allow"

    store.save([cand], path)
    loaded = store.load(path)

    assert len(loaded) == 1
    assert loaded[0].source_ref == cand.source_ref
    assert loaded[0].title == cand.title
    assert loaded[0].native_tags == ["ai-governance"]
    assert loaded[0].merged_connector_ids == ["agora"]  # type: ignore[attr-defined]


def test_save_overwrites_previous_content(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    store.save([_candidate(source_ref="a", raw_hash="ha")], path)
    store.save([_candidate(source_ref="b", raw_hash="hb")], path)

    loaded = store.load(path)
    assert [c.source_ref for c in loaded] == ["b"]


def test_save_leaves_no_staging_file(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    store.save([_candidate()], path)
    leftovers = list(tmp_path.glob(".*.part"))
    assert leftovers == []


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "candidates.yaml"
    store.save([_candidate()], path)
    assert path.exists()


def test_default_candidates_path_under_default_sources() -> None:
    assert store.CANDIDATES_PATH == schema.DEFAULT_SOURCES / "candidates.yaml"


# --- cursors ------------------------------------------------------------------------


def test_load_cursors_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert store.load_cursors(tmp_path / ".discovery_cursors.yaml") == {}


def test_save_load_cursors_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ".discovery_cursors.yaml"
    cursors = {"agora": {"dataset_version": "2026-05-16"}, "manual": {}}

    store.save_cursors(cursors, path)
    loaded = store.load_cursors(path)

    assert loaded == cursors


def test_default_cursors_path_is_dot_file_under_default_sources() -> None:
    assert store.CURSORS_PATH == schema.DEFAULT_SOURCES / ".discovery_cursors.yaml"
