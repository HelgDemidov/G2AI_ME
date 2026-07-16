"""Тесты FTS5-индекса на синтетических чанках (без модели/токенизатора — CI-safe)."""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from chunking import Chunk
from corpus_index import (
    SCHEMA_VERSION,
    _cmd_search,
    content_hash,
    corpus_fingerprint,
    create_db,
    fts5_available,
    fts_search,
    index_chunks,
    read_meta,
    sanitize_fts_query,
    write_meta,
)
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


# --- sanitize_fts_query: экранирование пользовательского FTS5-запроса ---


def test_sanitize_hyphenated_term_quoted() -> None:
    assert sanitize_fts_query("state-as-mcp") == '"state-as-mcp"'


def test_sanitize_multiword_quotes_each_token() -> None:
    assert sanitize_fts_query("state as mcp") == '"state" "as" "mcp"'


def test_sanitize_doubles_internal_quotes() -> None:
    assert sanitize_fts_query('он сказал "привет"') == '"он" "сказал" """привет"""'


def test_sanitize_empty_string_does_not_crash() -> None:
    assert sanitize_fts_query("") == '""'
    assert sanitize_fts_query("   ") == '""'


def test_sanitize_colon_and_parens_quoted() -> None:
    assert sanitize_fts_query("AI:(pilot)") == '"AI:(pilot)"'


# --- fts_search / _cmd_search: краш-кейсы FTS5-синтаксиса ---


def test_fts_search_raw_hyphenated_query_raises_operational_error(tmp_path: Path) -> None:
    """Демонстрирует САМУ проблему: сырой (неэкранированный) запрос с дефисом —
    невалидный FTS5-синтаксис (дефис не входит в bareword-грамматику)."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, sample_chunks())
    with pytest.raises(sqlite3.OperationalError):
        fts_search(conn, "state-as-mcp")  # без sanitize_fts_query


def test_fts_search_sanitized_hyphenated_query_works(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "state as mcp architecture pattern", 5)])
    hits = fts_search(conn, sanitize_fts_query("state-as-mcp"))
    assert len(hits) == 1
    assert hits[0].doc_id == "doc-a"


def test_cmd_search_default_sanitizes_hyphenated_query(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "state as mcp architecture pattern", 5)])
    conn.close()
    args = argparse.Namespace(db=db, query="state-as-mcp", limit=10, raw=False)
    assert _cmd_search(args) == 0


def test_cmd_search_raw_syntax_error_reported_not_raised(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, sample_chunks())
    conn.close()
    args = argparse.Namespace(db=db, query="state-as-mcp", limit=10, raw=True)  # честный синтаксис — падает
    assert _cmd_search(args) == 2
    assert "некорректный" in capsys.readouterr().err


# --- content_hash: ключ содержимого чанка (spec index-incremental §1) ---


def test_content_hash_matches_sha256() -> None:
    import hashlib

    assert content_hash("hello") == hashlib.sha256(b"hello").hexdigest()
    assert len(content_hash("hello")) == 64  # полный sha256-hex, не усечённый


def test_content_hash_deterministic_and_distinguishing() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_index_chunks_stores_content_hash(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, sample_chunks())
    rows = conn.execute("SELECT text, content_hash FROM chunks ORDER BY chunk_id").fetchall()
    assert rows
    for text, ch in rows:
        assert ch == content_hash(text)


# --- схема v2 + миграция легаси-БД (spec index-incremental §1) ---


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _chunks_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}


def test_create_db_fresh_has_v2_schema(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "fresh.db")
    assert "content_hash" in _chunks_columns(conn)
    assert "doc_state" in _table_names(conn)
    assert read_meta(conn, "schema_version") == SCHEMA_VERSION
    conn.close()


def test_create_db_reopen_current_preserves_data(tmp_path: Path) -> None:
    """Повторное открытие уже-v2 БД НЕ считается легаси — данные не сносятся."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, sample_chunks())
    conn.close()
    conn2 = create_db(db)
    assert conn2.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == len(sample_chunks())
    conn2.close()


def _make_legacy_db(path: Path) -> None:
    """БД схемы v1: chunks без content_hash, vectors на chunk_id, без schema_version."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL, n_tokens INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5 (
            text, content='chunks', content_rowid='chunk_id', tokenize='unicode61'
        );
        CREATE TABLE vectors (
            chunk_id INTEGER NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL,
            PRIMARY KEY (chunk_id, model)
        );
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, n_tokens) VALUES ('old', 0, 'legacy', 1)"
    )
    conn.execute("INSERT INTO index_meta (key, value) VALUES ('corpus_fingerprint', 'old-fp')")
    conn.commit()
    conn.close()


def test_migration_legacy_db_recreates_derived_tables(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    _make_legacy_db(db)
    conn = create_db(db)  # обязан мигрировать
    assert "content_hash" in _chunks_columns(conn)
    assert "doc_state" in _table_names(conn)
    assert read_meta(conn, "schema_version") == SCHEMA_VERSION
    # легаси-данные снесены (chunks пуст, старый fingerprint очищен)
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert read_meta(conn, "corpus_fingerprint") is None
    conn.close()


def test_migration_then_index_works(tmp_path: Path) -> None:
    """После миграции индекс собирается и ищется штатно (content_hash NOT NULL не мешает)."""
    db = tmp_path / "legacy.db"
    _make_legacy_db(db)
    conn = create_db(db)
    index_chunks(conn, sample_chunks())
    hits = fts_search(conn, "governance")
    assert len(hits) == 1
    conn.close()
