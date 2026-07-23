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
    """Синтетический zip, имитирующий структуру реального дампа AGORA (CSV в agora/-подпапке)."""
    documents_csv = (
        "AGORA ID,Official name,Authority,Most recent activity date,Tags,Short summary\n"
        "1,Test Act,Test Authority,2025-01-01,agent;autonomous,A short summary\n"
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
