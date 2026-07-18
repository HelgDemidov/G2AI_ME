"""Смоук-тест CLI hsearch (spec analyze-retrieval §5) — fts-only путь (--backend none,
без модели), CI-safe."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hsearch import main
from index.chunking import Chunk
from index.corpus_index import _rebuild_facets, create_db, fts5_available, index_chunks
from core.schema import SourceRecord
from tests.support import valid_record

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")


def _build_db(path: Path) -> None:
    conn = create_db(path)
    index_chunks(conn, [
        Chunk("doc-a", 0, "agentic ai governance framework", 4, "Governance Chapter"),
        Chunk("doc-b", 0, "unrelated content here", 4),
    ])
    conn.close()


def test_hsearch_backend_none_finds_hit(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    assert main(["governance", "--db", str(db), "--backend", "none"]) == 0
    out = capsys.readouterr().out
    assert "doc-a" in out
    assert "Governance Chapter" in out


def test_hsearch_no_hits_reports_nothing_found(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    assert main(["zzznomatchword", "--db", str(db), "--backend", "none"]) == 0
    assert "ничего не найдено" in capsys.readouterr().out


def test_hsearch_missing_db_reports_error(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "absent.db"
    assert main(["query", "--db", str(db), "--backend", "none"]) == 2
    assert "не найден" in capsys.readouterr().err


def test_hsearch_default_backend_is_openrouter(tmp_path: Path, monkeypatch: Any) -> None:
    """API-first (spec embed-api-first §4): дефолт --backend = openrouter."""
    db = tmp_path / "c.db"
    _build_db(db)
    captured: dict[str, Any] = {}

    def fake_make_embedder(backend: str, model: Any) -> None:
        captured["backend"] = backend
        return None  # None -> честный FTS-only путь retrieve()

    monkeypatch.setattr("hsearch._make_embedder", fake_make_embedder)
    assert main(["governance", "--db", str(db)]) == 0
    assert captured["backend"] == "openrouter"


def test_hsearch_entity_filter_narrows_results(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    conn = create_db(db)
    rec_a = SourceRecord.model_validate({**valid_record(), "id": "sg-doc-2026", "entity_id": "sg"})
    rec_b = SourceRecord.model_validate({**valid_record(), "id": "ee-doc-2026", "entity_id": "ee"})
    _rebuild_facets(conn, [rec_a, rec_b])
    index_chunks(conn, [
        Chunk(rec_a.id, 0, "shared governance term", 4),
        Chunk(rec_b.id, 0, "shared governance term", 4),
    ])
    conn.close()

    assert main(["governance", "--db", str(db), "--backend", "none", "--entity", "sg"]) == 0
    out = capsys.readouterr().out
    assert rec_a.id in out
    assert rec_b.id not in out
