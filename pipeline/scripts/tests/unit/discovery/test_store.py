"""Тесты discovery/store.py: персист слоя кандидатов + .discovery_cursors.yaml
(spec discovery-core §4; раскладка — spec discovery-candidates-sharding §1–§3)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from core import schema
from discovery import store


def _candidate(**overrides: object) -> schema.CandidateRecord:
    fields: dict[str, object] = {
        "connector_id": "manual",
        "retrieved_at": dt.date(2026, 7, 21),
        "raw_hash": "h0",
        "title": "Example Document",
        "native_tags": ["ai-governance"],
    }
    fields.update(overrides)
    return schema.CandidateRecord.model_validate(fields)


def _store_text(root: Path) -> str:
    """Сырой текст store — сумма всех его файлов (тесты содержимого не зависят от
    того, один это файл или каталог шардов; раскладку проверяют отдельные тесты)."""
    return "".join(
        p.read_text(encoding="utf-8") for p in sorted(store.candidates_path(root).parent.glob("*.yaml"))
    )


# --- candidates -------------------------------------------------------------------


def test_load_missing_store_returns_empty_list(tmp_path: Path) -> None:
    assert store.load(tmp_path) == []


def test_save_load_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    cand = _candidate()
    cand.merged_connector_ids = ["agora"]  # type: ignore[attr-defined]  # extra="allow"

    store.save([cand], tmp_path)
    loaded = store.load(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].raw_hash == cand.raw_hash
    assert loaded[0].title == cand.title
    assert loaded[0].native_tags == ["ai-governance"]
    assert loaded[0].merged_connector_ids == ["agora"]  # type: ignore[attr-defined]


def test_save_overwrites_previous_content(tmp_path: Path) -> None:
    store.save([_candidate(raw_hash="ha")], tmp_path)
    store.save([_candidate(raw_hash="hb")], tmp_path)

    loaded = store.load(tmp_path)
    assert [c.raw_hash for c in loaded] == ["hb"]


def test_save_leaves_no_staging_file(tmp_path: Path) -> None:
    store.save([_candidate()], tmp_path)
    leftovers = list(tmp_path.rglob(".*.part"))
    assert leftovers == []


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "dir"  # корня ещё нет — save обязан его создать
    store.save([_candidate()], root)
    assert len(store.load(root)) == 1


def test_default_candidates_path_under_default_sources() -> None:
    assert store.candidates_path() == store.CANDIDATES_PATH
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


# --- слим CandidateRecord + человекочитаемый дамп (2026-07-21) ---


def test_save_separates_candidates_with_blank_line(tmp_path: Path) -> None:
    store.save([_candidate(raw_hash="ha"), _candidate(raw_hash="hb")], tmp_path)
    assert "\n\n- " in _store_text(tmp_path)  # пустая строка между записями
    assert len(store.load(tmp_path)) == 2  # round-trip не страдает


def test_save_puts_title_first(tmp_path: Path) -> None:
    store.save([_candidate()], tmp_path)
    assert _store_text(tmp_path).startswith("- title:")


def test_save_omits_empty_list_fields(tmp_path: Path) -> None:
    """native_tags/matched_vocab_tags дефолтятся None -> в YAML не пишутся вовсе
    (раньше каждый ручной кандидат тащил шумную строку 'native_tags: []')."""
    store.save([_candidate(native_tags=None)], tmp_path)
    text = _store_text(tmp_path)
    assert "native_tags" not in text
    assert "matched_vocab_tags" not in text
