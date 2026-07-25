"""Тесты discovery/store.py: персист слоя кандидатов + курсоров (.state/cursors.yaml)
(spec discovery-core §4; шардированная раскладка — spec discovery-candidates-sharding §1–§3)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from core import fsio, schema
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
    """Сырой текст store — сумма всех шардов (тесты СОДЕРЖИМОГО не зависят от того,
    в каком шарде лежит запись; раскладку проверяют отдельные тесты ниже)."""
    return "".join(
        p.read_text(encoding="utf-8") for p in sorted(store.candidates_dir(root).glob("*.yaml"))
    )


def _shard_names(root: Path) -> list[str]:
    return sorted(p.name for p in store.candidates_dir(root).glob("*.yaml"))


# --- candidates: базовый round-trip -------------------------------------------------


def test_load_missing_store_returns_empty_list(tmp_path: Path) -> None:
    """Ни монолита, ни каталога шардов — пустой список, не исключение."""
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

    assert [c.raw_hash for c in store.load(tmp_path)] == ["hb"]


def test_save_leaves_no_staging_file(tmp_path: Path) -> None:
    store.save([_candidate()], tmp_path)
    assert list(tmp_path.rglob(".*.part")) == []


def test_save_creates_missing_directories(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "dir"  # корня ещё нет — save обязан его создать
    store.save([_candidate()], root)
    assert len(store.load(root)) == 1


def test_default_paths_under_default_sources() -> None:
    assert store.candidates_dir() == store.CANDIDATES_DIR == schema.DEFAULT_SOURCES / "candidates"
    assert store.legacy_candidates_path() == schema.DEFAULT_SOURCES / "candidates.yaml"


# --- шардинг: партиционирование, санитизация, детерминизм ----------------------------


def test_save_partitions_by_connector_id(tmp_path: Path) -> None:
    store.save(
        [
            _candidate(connector_id="manual", raw_hash="hm"),
            _candidate(connector_id="oecd", raw_hash="ho"),
            _candidate(connector_id="oecd", raw_hash="ho2"),
        ],
        tmp_path,
    )

    assert _shard_names(tmp_path) == ["manual.yaml", "oecd.yaml"]
    assert len(schema.load_candidates(store.shard_path("oecd", tmp_path))) == 2
    assert len(store.load(tmp_path)) == 3


@pytest.mark.parametrize(
    ("connector_id", "expected"),
    [
        ("manual", "manual"),
        ("search:western-balkans", "search__western-balkans"),  # ":" грамматики connector_id
        ("search:a/b", "search__a__b"),  # "/" не создаёт подкаталог/выход из store
        (".hidden", "__hidden"),  # ведущая точка: Path.glob("*.yaml") матчит скрытые файлы
        ("../escape", "______escape"),  # каждый небезопасный символ -> "__"; traversal невозможен
    ],
)
def test_shard_name_sanitizes_unsafe_characters(connector_id: str, expected: str) -> None:
    assert store.shard_name(connector_id) == expected


def test_save_writes_sanitized_shard_and_reads_it_back(tmp_path: Path) -> None:
    store.save([_candidate(connector_id="search:western-balkans")], tmp_path)

    assert _shard_names(tmp_path) == ["search__western-balkans.yaml"]
    assert store.load(tmp_path)[0].connector_id == "search:western-balkans"


def test_save_raises_on_shard_name_collision(tmp_path: Path) -> None:
    """Теоретическая коллизия санитизации падает громко, а не сливает шарды молча."""
    with pytest.raises(ValueError, match="коллизия имён шардов"):
        store.save(
            [
                _candidate(connector_id="search:x", raw_hash="ha"),
                _candidate(connector_id="search__x", raw_hash="hb"),
            ],
            tmp_path,
        )


def test_load_order_is_deterministic(tmp_path: Path) -> None:
    """Шарды — по имени файла, внутри шарда — порядок записи."""
    store.save(
        [
            _candidate(connector_id="oecd", raw_hash="ho1"),
            _candidate(connector_id="agora", raw_hash="hа1"),
            _candidate(connector_id="oecd", raw_hash="ho2"),
        ],
        tmp_path,
    )

    assert [c.raw_hash for c in store.load(tmp_path)] == ["hа1", "ho1", "ho2"]


def test_unchanged_shard_is_not_rewritten(tmp_path: Path) -> None:
    """Byte-compare: неизменившийся шард не получает ложный mtime-чурн."""
    records = [_candidate(connector_id="manual", raw_hash="hm"), _candidate(connector_id="oecd", raw_hash="ho")]
    store.save(records, tmp_path)
    untouched = store.shard_path("oecd", tmp_path)
    before = untouched.stat().st_mtime_ns

    records[0] = _candidate(connector_id="manual", raw_hash="hm-changed")
    store.save(records, tmp_path)

    assert untouched.stat().st_mtime_ns == before  # чужой шард не тронут
    assert {c.raw_hash for c in store.load(tmp_path)} == {"hm-changed", "ho"}


def test_emptied_shard_is_removed(tmp_path: Path) -> None:
    store.save([_candidate(connector_id="manual"), _candidate(connector_id="oecd", raw_hash="ho")], tmp_path)
    store.save([_candidate(connector_id="manual")], tmp_path)

    assert _shard_names(tmp_path) == ["manual.yaml"]


def test_save_empty_list_clears_store(tmp_path: Path) -> None:
    store.save([_candidate()], tmp_path)
    store.save([], tmp_path)

    assert store.load(tmp_path) == []
    assert _shard_names(tmp_path) == []


# --- авто-миграция монолита (§3) -----------------------------------------------------


def _write_monolith(root: Path, records: list[schema.CandidateRecord]) -> Path:
    path = store.legacy_candidates_path(root)
    fsio.atomic_write_text(path, store._dump_records(records))
    return path


def test_load_reads_legacy_monolith(tmp_path: Path) -> None:
    _write_monolith(tmp_path, [_candidate(raw_hash="hlegacy")])

    assert [c.raw_hash for c in store.load(tmp_path)] == ["hlegacy"]


def test_first_save_splits_monolith_into_shards_and_removes_it(tmp_path: Path) -> None:
    monolith = _write_monolith(
        tmp_path,
        [_candidate(connector_id="manual", raw_hash="hm"), _candidate(connector_id="oecd", raw_hash="ho")],
    )

    store.save(store.load(tmp_path), tmp_path)

    assert not monolith.exists()  # удаляется ПОСЛЕДНИМ шагом успешного save
    assert _shard_names(tmp_path) == ["manual.yaml", "oecd.yaml"]
    assert {c.raw_hash for c in store.load(tmp_path)} == {"hm", "ho"}


def test_migration_is_idempotent(tmp_path: Path) -> None:
    _write_monolith(tmp_path, [_candidate(connector_id="manual", raw_hash="hm")])
    store.save(store.load(tmp_path), tmp_path)
    shard = store.shard_path("manual", tmp_path)
    before = shard.stat().st_mtime_ns

    store.save(store.load(tmp_path), tmp_path)  # повторный прогон

    assert shard.stat().st_mtime_ns == before  # no-op: байты те же, mtime не тронут
    assert [c.raw_hash for c in store.load(tmp_path)] == ["hm"]


def test_live_monolith_takes_precedence_over_shards(tmp_path: Path) -> None:
    """Прецедентность (§3): пока монолит жив — шарды игнорируются целиком.

    Ровно это спасает от окна краша «шарды записаны, монолит ещё жив»: наивная
    конкатенация вернула бы каждую мигрированную запись ДВАЖДЫ.
    """
    records = [_candidate(connector_id="manual", raw_hash="hm")]
    fsio.atomic_write_text(store.shard_path("manual", tmp_path), store._dump_records(records))
    _write_monolith(tmp_path, records)

    loaded = store.load(tmp_path)

    assert [c.raw_hash for c in loaded] == ["hm"]  # не ["hm", "hm"]


def test_crash_between_shards_converges_on_next_save(tmp_path: Path, monkeypatch: Any) -> None:
    """Краш посреди записи шардов (монолит ещё жив) → повторный save сходится без потерь."""
    records = [
        _candidate(connector_id="manual", raw_hash="hm"),
        _candidate(connector_id="oecd", raw_hash="ho"),
    ]
    monolith = _write_monolith(tmp_path, records)

    real_write = fsio.atomic_write_text
    calls = {"n": 0}

    def flaky(target: Path, text: str) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("диск отвалился между шардами")
        real_write(target, text)

    monkeypatch.setattr(fsio, "atomic_write_text", flaky)  # store зовёт через тот же модуль
    with pytest.raises(OSError):
        store.save(store.load(tmp_path), tmp_path)

    assert monolith.exists()  # не удалён — миграция не завершена
    assert {c.raw_hash for c in store.load(tmp_path)} == {"hm", "ho"}  # монолит, без дублей

    monkeypatch.setattr(fsio, "atomic_write_text", real_write)
    store.save(store.load(tmp_path), tmp_path)

    assert not monolith.exists()
    assert _shard_names(tmp_path) == ["manual.yaml", "oecd.yaml"]
    assert {c.raw_hash for c in store.load(tmp_path)} == {"hm", "ho"}


# --- cursors ------------------------------------------------------------------------


def test_load_cursors_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert store.load_cursors(tmp_path) == {}


def test_save_load_cursors_round_trip(tmp_path: Path) -> None:
    cursors = {"agora": {"dataset_version": "2026-05-16"}, "manual": {}}

    store.save_cursors(cursors, tmp_path)

    assert store.load_cursors(tmp_path) == cursors


def test_save_cursors_creates_state_dir(tmp_path: Path) -> None:
    """``.state/`` может ещё не существовать (свежий корпус) — писатель его создаёт."""
    store.save_cursors({"agora": {}}, tmp_path)

    assert store.cursors_path(tmp_path).parent == store.state_dir(tmp_path)
    assert store.cursors_path(tmp_path).exists()


def test_cursors_live_under_state_dir_of_default_sources() -> None:
    """Операционное состояние корпуса — в ``sources/.state/`` (2026-07-25), одним словом
    с пер-документным ``.state.yaml``; курсоры больше не валяются в корне корпуса."""
    assert store.state_dir() == store.STATE_DIR == schema.DEFAULT_SOURCES / ".state"
    assert store.CURSORS_PATH == store.STATE_DIR / "cursors.yaml"
    assert store.cursors_path() == store.CURSORS_PATH


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
