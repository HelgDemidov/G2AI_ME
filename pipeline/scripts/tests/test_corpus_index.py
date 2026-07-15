"""Тесты FTS5-индекса на синтетических чанках (без модели/токенизатора — CI-safe)."""
from __future__ import annotations

from pathlib import Path

import pytest

from chunking import Chunk
from corpus_index import create_db, fts5_available, fts_search, index_chunks

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite собран без FTS5")


def sample_chunks() -> list[Chunk]:
    return [
        Chunk("doc-a", 0, "agentic ai governance framework", 4),
        Chunk("doc-a", 1, "human oversight and accountability", 4),
        Chunk("doc-b", 0, "tool call verification testing", 4),
    ]


def test_build_and_search(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, sample_chunks())
    hits = fts_search(conn, "governance")
    assert len(hits) == 1
    assert hits[0].doc_id == "doc-a"
    assert hits[0].chunk_index == 0


def test_search_across_docs(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, sample_chunks())
    hits = fts_search(conn, "verification OR oversight")
    assert {h.doc_id for h in hits} == {"doc-a", "doc-b"}


def test_reindex_is_idempotent(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, sample_chunks())
    index_chunks(conn, sample_chunks())  # повторно — не должно дублироваться
    assert len(fts_search(conn, "governance")) == 1


def test_no_match_returns_empty(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, sample_chunks())
    assert fts_search(conn, "nonexistentword") == []


def test_snippet_highlights_match(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, sample_chunks())
    hits = fts_search(conn, "governance")
    assert "[" in hits[0].snippet and "]" in hits[0].snippet
