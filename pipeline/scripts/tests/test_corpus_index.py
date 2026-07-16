"""Тесты FTS5-индекса на синтетических чанках (без модели/токенизатора — CI-safe)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from chunking import Chunk
from corpus_index import corpus_fingerprint, create_db, fts5_available, fts_search, index_chunks, read_meta, write_meta
from test_schema import valid_record, write_doc

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


# --- index_meta: read_meta/write_meta ---


def test_write_read_meta_roundtrip(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    write_meta(conn, "corpus_fingerprint", "abc123")
    conn.commit()
    assert read_meta(conn, "corpus_fingerprint") == "abc123"


def test_read_meta_missing_key_returns_none(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    assert read_meta(conn, "nope") is None


def test_read_meta_on_connection_without_create_db(tmp_path: Path) -> None:
    """read_meta работает даже на «сыром» соединении, минуя create_db (defensive
    CREATE IF NOT EXISTS — как vector_store.ensure_schema)."""
    import sqlite3

    conn = sqlite3.connect(tmp_path / "raw.db")
    assert read_meta(conn, "corpus_fingerprint") is None


def test_write_meta_upserts(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    write_meta(conn, "k", "v1")
    write_meta(conn, "k", "v2")
    conn.commit()
    assert read_meta(conn, "k") == "v2"


# --- corpus_fingerprint (stat-based) ---


def _write_corpus_doc(
    root: Path, *, raw: bytes | None = b"pdf", md: str | None = "body text", **rec_over: Any
) -> Path:
    rec = valid_record()
    rec.update(rec_over)
    return write_doc(root, rec, raw=raw, md=md)


def test_corpus_fingerprint_deterministic(tmp_path: Path) -> None:
    _write_corpus_doc(tmp_path)
    assert corpus_fingerprint(tmp_path) == corpus_fingerprint(tmp_path)


def test_corpus_fingerprint_sensitive_to_mtime(tmp_path: Path) -> None:
    d = _write_corpus_doc(tmp_path)
    fp1 = corpus_fingerprint(tmp_path)
    time.sleep(0.01)
    (d / "doc.md").write_text("body text", encoding="utf-8")  # то же содержимое, новый mtime
    assert corpus_fingerprint(tmp_path) != fp1


def test_corpus_fingerprint_sensitive_to_size(tmp_path: Path) -> None:
    d = _write_corpus_doc(tmp_path)
    fp1 = corpus_fingerprint(tmp_path)
    (d / "doc.md").write_text("body text, changed and longer", encoding="utf-8")
    assert corpus_fingerprint(tmp_path) != fp1


def test_corpus_fingerprint_sensitive_to_doc_set(tmp_path: Path) -> None:
    _write_corpus_doc(tmp_path)
    fp1 = corpus_fingerprint(tmp_path)
    _write_corpus_doc(tmp_path, id="second-doc-2026", entity_id="se")
    assert corpus_fingerprint(tmp_path) != fp1


def test_corpus_fingerprint_skips_missing_doc_md(tmp_path: Path) -> None:
    """Запись без doc.md (ещё не сконвертирована) не влияет на отпечаток."""
    _write_corpus_doc(tmp_path)  # с doc.md
    fp_with_converted_only = corpus_fingerprint(tmp_path)
    _write_corpus_doc(tmp_path, id="no-md-doc-2026", entity_id="nm", md=None)  # без doc.md
    assert corpus_fingerprint(tmp_path) == fp_with_converted_only


def test_corpus_fingerprint_empty_corpus(tmp_path: Path) -> None:
    assert corpus_fingerprint(tmp_path) == corpus_fingerprint(tmp_path)  # не падает, детерминирован
