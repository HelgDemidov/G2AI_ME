"""Тесты discovery/registry_store.py: DuckDB bronze-слой архетипа `registry` (spec discovery-agora §2).

Синтетический CSV, embedded DuckDB на tmp_path — без сети/сервера/модели, полностью герметично.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from discovery import registry_store


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_ingest_csv_creates_schema_and_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "docs.csv"
    _write_csv(csv_path, ["id,title", "1,Alpha", "2,Beta"])

    conn = registry_store.connect(tmp_path / "test.duckdb")
    registry_store.ingest_csv(
        conn, schema="agora", table="documents_raw", csv_path=csv_path, source_version="1.0.0"
    )
    rows = conn.execute("SELECT id, title FROM agora.documents_raw ORDER BY id").fetchall()
    assert rows == [(1, "Alpha"), (2, "Beta")]


def test_ingest_csv_stamps_provenance_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "docs.csv"
    _write_csv(csv_path, ["id,title", "1,Alpha"])

    conn = registry_store.connect(tmp_path / "test.duckdb")
    registry_store.ingest_csv(
        conn, schema="agora", table="documents_raw", csv_path=csv_path, source_version="1.31.0"
    )
    row = conn.execute(
        "SELECT _source_version, _ingested_at FROM agora.documents_raw LIMIT 1"
    ).fetchone()
    assert row is not None
    version, ingested_at = row
    assert version == "1.31.0"
    assert ingested_at is not None


def test_ingest_csv_replaces_not_appends_on_version_bump(tmp_path: Path) -> None:
    csv_path = tmp_path / "docs.csv"
    conn = registry_store.connect(tmp_path / "test.duckdb")

    _write_csv(csv_path, ["id,title", "1,Alpha"])
    registry_store.ingest_csv(
        conn, schema="agora", table="documents_raw", csv_path=csv_path, source_version="1.0.0"
    )
    assert conn.execute("SELECT count(*) FROM agora.documents_raw").fetchone() == (1,)

    _write_csv(csv_path, ["id,title", "2,Beta", "3,Gamma"])
    registry_store.ingest_csv(
        conn, schema="agora", table="documents_raw", csv_path=csv_path, source_version="1.1.0"
    )
    rows = conn.execute("SELECT id, title FROM agora.documents_raw ORDER BY id").fetchall()
    assert rows == [(2, "Beta"), (3, "Gamma")]  # старая строка id=1 не осталась (не UNION)


def test_ingest_csv_strips_bom_from_column_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "docs.csv"
    csv_path.write_bytes("﻿AGORA ID,Title\n1,Alpha\n".encode("utf-8"))

    conn = registry_store.connect(tmp_path / "test.duckdb")
    registry_store.ingest_csv(
        conn, schema="agora", table="documents_raw", csv_path=csv_path, source_version="1.0.0"
    )
    cols = [row[1] for row in conn.execute("PRAGMA table_info('agora.documents_raw')").fetchall()]
    assert cols[0] == "AGORA ID"  # DuckDB auto_detect снимает BOM сам, без ручного utf-8-sig


def test_ingest_csv_isolates_schemas_per_connector(tmp_path: Path) -> None:
    conn = registry_store.connect(tmp_path / "test.duckdb")
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    _write_csv(a_csv, ["id", "1"])
    _write_csv(b_csv, ["id", "9"])

    registry_store.ingest_csv(conn, schema="agora", table="docs", csv_path=a_csv, source_version="1")
    registry_store.ingest_csv(conn, schema="dpa", table="docs", csv_path=b_csv, source_version="1")

    assert conn.execute("SELECT id FROM agora.docs").fetchone() == (1,)
    assert conn.execute("SELECT id FROM dpa.docs").fetchone() == (9,)


@pytest.mark.parametrize("bad", ["agora; DROP TABLE x", "agora.docs", "1agora", ""])
def test_ingest_csv_rejects_unsafe_identifiers(tmp_path: Path, bad: str) -> None:
    csv_path = tmp_path / "docs.csv"
    _write_csv(csv_path, ["id", "1"])
    conn = registry_store.connect(tmp_path / "test.duckdb")

    with pytest.raises(ValueError, match="идентификатор"):
        registry_store.ingest_csv(
            conn, schema=bad, table="documents_raw", csv_path=csv_path, source_version="1.0.0"
        )


def test_connect_creates_missing_cache_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "registry.duckdb"
    conn = registry_store.connect(db_path)
    assert db_path.parent.is_dir()
    conn.execute("SELECT 1").fetchone()
