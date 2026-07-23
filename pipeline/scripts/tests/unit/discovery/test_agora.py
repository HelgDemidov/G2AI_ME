"""Тесты discovery/connectors/agora.py (spec discovery-agora).

Fetch/parse РАЗДЕЛЕНЫ (чартер, тест-принципы): эти тесты — чистый parse/config/cache
на синтетических фикстурах, БЕЗ сети. Живой смок (``discover.py discover --only agora
--dry-run`` на реальном дампе) — вне CI, спек §Тестовое покрытие.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml

from core import schema
from discovery import registry_store
from discovery.connectors import agora


# --- load_config / frontier_year / resolve_min_year ---


def test_load_config_reads_real_tracked_config() -> None:
    """pipeline/config/discovery_agora.yaml — настоящий трекаемый файл, не фикстура."""
    config = agora.load_config()
    assert config.enabled is True
    assert config.zenodo_doi == "10.5281/zenodo.13883066"
    assert config.non_us_include_all is True
    assert config.us_probe_limit == 50
    assert config.us_probe_min_year is None
    assert "agent" in config.us_probe_match_terms


def test_load_config_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "discovery_agora.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": False,
                "zenodo_doi": "10.5281/zenodo.999",
                "non_us": {"include_all": True},
                "us_axis_probe": {"limit": 10, "min_year": 2020, "match_terms": ["agent"]},
            }
        ),
        encoding="utf-8",
    )
    config = agora.load_config(path)
    assert config.enabled is False
    assert config.us_probe_limit == 10
    assert config.us_probe_min_year == 2020
    assert config.us_probe_match_terms == ("agent",)


def test_frontier_year_reads_real_tracked_triage_config() -> None:
    assert agora.frontier_year() == 2025


def test_resolve_min_year_override_wins(tmp_path: Path) -> None:
    config = agora.AgoraConfig(
        enabled=True, zenodo_doi="10.5281/zenodo.1", non_us_include_all=True,
        us_probe_limit=50, us_probe_min_year=2019, us_probe_match_terms=("agent",),
    )
    assert agora.resolve_min_year(config) == 2019


def test_resolve_min_year_null_reads_frontier_year() -> None:
    config = agora.AgoraConfig(
        enabled=True, zenodo_doi="10.5281/zenodo.1", non_us_include_all=True,
        us_probe_limit=50, us_probe_min_year=None, us_probe_match_terms=("agent",),
    )
    assert agora.resolve_min_year(config) == agora.frontier_year()


# --- _concept_recid / cursor_from_metadata / download_url_from_metadata ---


def test_concept_recid_extracts_from_doi() -> None:
    assert agora._concept_recid("10.5281/zenodo.13883066") == "13883066"


def test_concept_recid_rejects_malformed_doi() -> None:
    with pytest.raises(ValueError, match="recid"):
        agora._concept_recid("not-a-doi")


def _fake_zenodo_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": 21390882,
        "metadata": {"version": "1.31.0"},
        "files": [
            {
                "key": "agora.zip",
                "checksum": "md5:7a284f8b33f92f282d6e62e829c331a5",
                "links": {"self": "https://zenodo.org/api/records/21390882/files/agora.zip/content"},
            }
        ],
    }
    record.update(overrides)
    return record


def test_cursor_from_metadata_extracts_version_id_md5() -> None:
    cursor = agora.cursor_from_metadata(_fake_zenodo_record())
    assert cursor == {
        "zenodo_version": "1.31.0",
        "record_id": 21390882,
        "md5": "md5:7a284f8b33f92f282d6e62e829c331a5",
    }


def test_cursor_from_metadata_no_files_gives_none_md5() -> None:
    cursor = agora.cursor_from_metadata(_fake_zenodo_record(files=[]))
    assert cursor["md5"] is None


def test_download_url_from_metadata_extracts_content_link() -> None:
    url = agora.download_url_from_metadata(_fake_zenodo_record())
    assert url == "https://zenodo.org/api/records/21390882/files/agora.zip/content"


def test_download_url_from_metadata_no_files_raises() -> None:
    with pytest.raises(ValueError, match="файл"):
        agora.download_url_from_metadata(_fake_zenodo_record(files=[]))


# --- ingest_dump ---


def _build_fixture_zip(zip_path: Path) -> None:
    """Синтетический zip, имитирующий структуру реального дампа AGORA (CSV в agora/-подпапке).

    Полный набор колонок, которые читает гибрид-фильтр (§4/§5) — не минимальное
    подмножество: `discover_agora`-тесты прогоняют реальный SQL-запрос, не только parse.
    """
    documents_csv = (
        "AGORA ID,Official name,Casual name,Authority,Link to document,Most recent activity,"
        "Most recent activity date,Proposed date,Short summary,Tags,"
        "Summaries and tags may include unreviewed machine output\n"
        "1,Test Act,,Test Authority,https://ex.org/test-act,Enacted,"
        "2025-01-01,,A short summary,agent;autonomous,false\n"
    )
    authorities_csv = "Name,Jurisdiction,Parent authority\nTest Authority,United States,\n"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("agora/documents.csv", documents_csv)
        zf.writestr("agora/authorities.csv", authorities_csv)
        zf.writestr("agora/collections.csv", "Name,Description\n")


def test_ingest_dump_loads_documents_and_authorities(tmp_path: Path) -> None:
    zip_path = tmp_path / "agora-1.0.0.zip"
    _build_fixture_zip(zip_path)
    db_path = tmp_path / "registry.duckdb"

    agora.ingest_dump(zip_path, source_version="1.0.0", db_path=db_path)

    conn = registry_store.connect(db_path)
    docs = conn.execute("SELECT \"AGORA ID\", \"Official name\" FROM agora.documents_raw").fetchall()
    authorities = conn.execute("SELECT \"Name\", \"Jurisdiction\" FROM agora.authorities_raw").fetchall()
    conn.close()
    assert docs == [(1, "Test Act")]
    assert authorities == [("Test Authority", "United States")]


def test_ingest_dump_replaces_on_second_call_with_new_version(tmp_path: Path) -> None:
    zip_path = tmp_path / "agora-1.0.0.zip"
    _build_fixture_zip(zip_path)
    db_path = tmp_path / "registry.duckdb"
    agora.ingest_dump(zip_path, source_version="1.0.0", db_path=db_path)

    zip_path_v2 = tmp_path / "agora-1.1.0.zip"
    _build_fixture_zip(zip_path_v2)
    agora.ingest_dump(zip_path_v2, source_version="1.1.0", db_path=db_path)

    conn = registry_store.connect(db_path)
    version = conn.execute("SELECT DISTINCT _source_version FROM agora.documents_raw").fetchall()
    conn.close()
    assert version == [("1.1.0",)]  # старая версия не осталась (REPLACE, не UNION)


# --- гибрид-фильтр + маппинг (§4/§5) ---

_DOC_COLUMNS = (
    "AGORA ID,Official name,Casual name,Authority,Link to document,Most recent activity,"
    "Most recent activity date,Proposed date,Short summary,Tags,"
    "Summaries and tags may include unreviewed machine output"
)
_AUTH_COLUMNS = "Name,Jurisdiction,Parent authority"


def _doc_row(
    doc_id: int,
    *,
    title: str = "Doc",
    casual: str = "",
    authority: str = "Gov",
    url: str = "https://ex.org/d",
    activity: str = "Enacted",
    activity_date: str = "2026-01-01",
    proposed_date: str = "",
    summary: str = "A summary",
    tags: str = "",
    machine: str = "false",
) -> str:
    return (
        f"{doc_id},{title},{casual},{authority},{url},{activity},"
        f"{activity_date},{proposed_date},{summary},{tags},{machine}"
    )


def _build_filter_test_db(tmp_path: Path, doc_rows: list[str], auth_rows: list[str]) -> Path:
    docs_csv = tmp_path / "documents.csv"
    auth_csv = tmp_path / "authorities.csv"
    docs_csv.write_text(_DOC_COLUMNS + "\n" + "\n".join(doc_rows) + "\n", encoding="utf-8")
    auth_csv.write_text(_AUTH_COLUMNS + "\n" + "\n".join(auth_rows) + "\n", encoding="utf-8")

    db_path = tmp_path / "registry.duckdb"
    conn = registry_store.connect(db_path)
    registry_store.ingest_csv(
        conn, schema="agora", table="documents_raw", csv_path=docs_csv, source_version="1.0.0"
    )
    registry_store.ingest_csv(
        conn, schema="agora", table="authorities_raw", csv_path=auth_csv, source_version="1.0.0"
    )
    conn.close()
    return db_path


_MATCH_TERMS = ("agent", "autonomous", "agentic", "planning ability")


def test_hybrid_filter_all_non_us_pass_without_ranking(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[
            _doc_row(1, authority="China Authority", activity_date="2020-01-01", tags="unrelated"),
            _doc_row(2, authority="Empty Authority", activity_date="2020-01-01", tags="unrelated"),
        ],
        auth_rows=["China Authority,China,", "Empty Authority,,"],
    )
    conn = registry_store.connect(db_path)
    candidates, skipped = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert skipped == 0
    assert {c.native_id for c in candidates} == {"1", "2"}  # не-US проходит без фильтра по году/термину


def test_hybrid_filter_us_below_frontier_year_excluded(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[_doc_row(1, authority="US Gov", activity_date="2020-01-01", tags="agent")],
        auth_rows=["US Gov,United States,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert candidates == []


def test_hybrid_filter_us_recent_without_match_term_excluded(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[_doc_row(1, authority="US Gov", activity_date="2026-01-01", tags="unrelated topic")],
        auth_rows=["US Gov,United States,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert candidates == []


def test_hybrid_filter_us_recent_with_match_term_included_and_tagged(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[_doc_row(1, authority="US Gov", activity_date="2026-01-01", tags="agent;other")],
        auth_rows=["US Gov,United States,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert len(candidates) == 1
    assert any("us-axis-probe (matched: agent)" in t for t in candidates[0].native_tags or [])


def test_hybrid_filter_word_boundary_agency_does_not_match_agent(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[_doc_row(1, authority="US Gov", activity_date="2026-01-01", tags="agency management")],
        auth_rows=["US Gov,United States,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert candidates == []


def test_hybrid_filter_ranks_by_match_count_then_recency_and_respects_limit(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[
            _doc_row(1, authority="US Gov", activity_date="2025-01-01", tags="agent"),  # 1 match, older
            _doc_row(2, authority="US Gov", activity_date="2025-06-01", tags="agent;autonomous"),  # 2 matches
            _doc_row(3, authority="US Gov", activity_date="2026-01-01", tags="agent"),  # 1 match, newest
        ],
        auth_rows=["US Gov,United States,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=2
    )
    conn.close()
    assert [c.native_id for c in candidates] == ["2", "3"]  # 2 matches first, then newest 1-match


def test_hybrid_filter_machine_flag_gates_native_summary(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[
            _doc_row(1, authority="China A", activity_date="2020-01-01", summary="Clean summary", machine="false"),
            _doc_row(2, authority="China A", activity_date="2020-01-01", summary="Machine summary", machine="true"),
        ],
        auth_rows=["China A,China,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    by_id = {c.native_id: c for c in candidates}
    assert by_id["1"].native_summary == "Clean summary"
    assert by_id["2"].native_summary is None


def test_hybrid_filter_activity_status_tag_present(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[_doc_row(1, authority="China A", activity="Defunct", activity_date="2020-01-01")],
        auth_rows=["China A,China,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert "AGORA activity: Defunct" in (candidates[0].native_tags or [])


def test_hybrid_filter_invalid_source_url_skipped_not_crashed(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[
            _doc_row(1, authority="China A", url="not-a-url", activity_date="2020-01-01"),
            _doc_row(2, authority="China A", url="https://ex.org/ok", activity_date="2020-01-01"),
        ],
        auth_rows=["China A,China,"],
    )
    conn = registry_store.connect(db_path)
    candidates, skipped = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert skipped == 1
    assert [c.native_id for c in candidates] == ["2"]


def test_hybrid_filter_normalized_url_and_raw_hash_set(tmp_path: Path) -> None:
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[_doc_row(1, authority="China A", url="https://Ex.org/D/", activity_date="2020-01-01")],
        auth_rows=["China A,China,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    cand = candidates[0]
    assert cand.normalized_url == "https://ex.org/D"  # normalize_url lower-кейсит host, не path
    assert len(cand.raw_hash) == 64


def test_hybrid_filter_candidate_summary_max_truncated(tmp_path: Path) -> None:
    long_summary = "x" * (schema.CANDIDATE_SUMMARY_MAX + 200)
    db_path = _build_filter_test_db(
        tmp_path,
        doc_rows=[_doc_row(1, authority="China A", activity_date="2020-01-01", summary=long_summary)],
        auth_rows=["China A,China,"],
    )
    conn = registry_store.connect(db_path)
    candidates, _ = agora.select_and_map_candidates(
        conn, match_terms=_MATCH_TERMS, min_year=2025, limit=50
    )
    conn.close()
    assert len(candidates[0].native_summary or "") == schema.CANDIDATE_SUMMARY_MAX


# --- discover_agora / AgoraConnector ---


def _fake_config(**overrides: object) -> agora.AgoraConfig:
    base = dict(
        enabled=True,
        zenodo_doi="10.5281/zenodo.13883066",
        non_us_include_all=True,
        us_probe_limit=50,
        us_probe_min_year=2025,
        us_probe_match_terms=_MATCH_TERMS,
    )
    base.update(overrides)
    return agora.AgoraConfig(**base)  # type: ignore[arg-type]


def test_discover_agora_unchanged_version_is_noop(tmp_path: Path) -> None:
    cursor = {"zenodo_version": "1.31.0", "record_id": 21390882, "md5": "abc"}
    calls: list[str] = []

    def fake_fetch(doi: str) -> dict[str, object]:
        return _fake_zenodo_record()

    def fake_download(url: str, dest: Path) -> None:
        calls.append(url)

    result = agora.discover_agora(
        cursor,
        config=_fake_config(),
        fetch_metadata=fake_fetch,
        download=fake_download,
        cache_dir=tmp_path / "cache",
        db_path=tmp_path / "registry.duckdb",
    )
    assert result.candidates == []
    assert result.cursor == cursor
    assert result.diagnostics["status"] == "unchanged"
    assert calls == []  # fetch не тронут — сеть не участвует на неизменённой версии


def test_discover_agora_new_version_downloads_ingests_and_maps(tmp_path: Path) -> None:
    downloaded: list[Path] = []

    def fake_fetch(doi: str) -> dict[str, object]:
        return _fake_zenodo_record(metadata={"version": "1.31.0"})

    def fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _build_fixture_zip(dest)
        downloaded.append(dest)

    result = agora.discover_agora(
        None,
        config=_fake_config(),
        fetch_metadata=fake_fetch,
        download=fake_download,
        cache_dir=tmp_path / "cache",
        db_path=tmp_path / "registry.duckdb",
    )
    assert len(downloaded) == 1
    assert result.cursor["zenodo_version"] == "1.31.0"
    assert result.diagnostics["status"] == "fetched"
    assert len(result.candidates) == 1  # фикстура: 1 документ, US, matches "agent"/"autonomous"


def test_discover_agora_caches_zip_does_not_redownload_same_version(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_fetch(doi: str) -> dict[str, object]:
        return _fake_zenodo_record(metadata={"version": "1.31.0"})

    def fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _build_fixture_zip(dest)

    cache_dir = tmp_path / "cache"
    db_path = tmp_path / "registry.duckdb"
    agora.discover_agora(
        None, config=_fake_config(), fetch_metadata=fake_fetch, download=fake_download,
        cache_dir=cache_dir, db_path=db_path,
    )
    # курсор из ПРЕДЫДУЩЕГО прогона с ТОЙ ЖЕ версией, но zip уже лежит в кэше —
    # discover_agora не должен звать download второй раз (версия сравнивается ДО кэша,
    # но раз версия "изменилась" относительно cursor=None, качаем; проверяем, что при
    # повторном вызове с cursor=None и уже кэшированным zip download не дублируется)
    agora.discover_agora(
        None, config=_fake_config(), fetch_metadata=fake_fetch, download=fake_download,
        cache_dir=cache_dir, db_path=db_path,
    )
    assert len(calls) == 1  # второй вызов нашёл zip в кэше и не скачивал повторно


def test_agora_connector_implements_protocol() -> None:
    conn = agora.AgoraConnector(enabled=True)
    assert conn.id == "agora"
    assert conn.kind == schema.ConnectorKind.registry
    assert conn.enabled is True
