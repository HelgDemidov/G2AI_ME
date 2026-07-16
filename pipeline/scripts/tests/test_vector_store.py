"""Тесты векторного хранилища и брутфорс-косинуса (синтетические векторы, без модели)."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bge_tokenizer import EMBED_MAX_TOKENS
from chunking import Chunk
from corpus_index import create_db, fts5_available, index_chunks, read_meta, write_meta
from embed import l2_normalize
from vector_store import (
    VectorsStaleError,
    _cmd_embed,
    check_chunk_budget,
    chunk_texts,
    load_vectors,
    semantic_search,
    store_vectors,
)

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")

FP = "test-corpus-fingerprint"  # фиксированный fingerprint для тестовых БД — store_vectors
# и semantic_search сверяют его между собой, а не пересчитывают из реального корпуса.


def _setup(tmp_path: Path) -> tuple[sqlite3.Connection, list[int]]:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-a", 0, "first", 1),
            Chunk("doc-a", 1, "second", 1),
            Chunk("doc-b", 0, "third", 1),
        ],
        corpus_fingerprint=FP,
    )
    ids, _ = chunk_texts(conn)
    return conn, ids


def test_store_and_load(tmp_path: Path) -> None:
    conn, ids = _setup(tmp_path)
    store_vectors(conn, ids, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    got_ids, mat = load_vectors(conn, "m")
    assert got_ids == ids
    assert mat.shape == (3, 3)


def test_semantic_search_nearest(tmp_path: Path) -> None:
    conn, ids = _setup(tmp_path)
    store_vectors(conn, ids, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)  # ближе всего к первому чанку
    hits = semantic_search(conn, query, "m", top_k=3)
    assert hits[0].chunk_id == ids[0]
    assert hits[0].doc_id == "doc-a"
    assert hits[0].chunk_index == 0
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_reembed_replaces(tmp_path: Path) -> None:
    conn, ids = _setup(tmp_path)
    vecs = l2_normalize(np.ones((3, 3), dtype=np.float32))
    store_vectors(conn, ids, vecs, "m")
    store_vectors(conn, ids, vecs, "m")  # повторно — не дублируется (PK)
    got_ids, _ = load_vectors(conn, "m")
    assert got_ids == ids


def test_absent_model_raises_vectors_stale(tmp_path: Path) -> None:
    """Модель, которую никогда не эмбеддили, — тот же честный отказ, что у
    устаревших векторов (spec index-consistency §3: «расхождение ИЛИ отсутствие»)."""
    conn, _ = _setup(tmp_path)
    with pytest.raises(VectorsStaleError):
        semantic_search(conn, np.zeros(3, dtype=np.float32), "absent", 5)


def test_semantic_search_raises_when_fingerprint_mismatches(tmp_path: Path) -> None:
    conn, ids = _setup(tmp_path)
    store_vectors(conn, ids, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    write_meta(conn, "corpus_fingerprint", "different-fingerprint-now")  # корпус «изменился»
    conn.commit()
    with pytest.raises(VectorsStaleError):
        semantic_search(conn, np.zeros(3, dtype=np.float32), "m", 5)


def test_semantic_search_raises_when_no_corpus_fingerprint_at_all(tmp_path: Path) -> None:
    """БД без единого записанного corpus_fingerprint — свежесть векторов неизвестна,
    безопаснее отказать, чем молча доверять."""
    conn = create_db(tmp_path / "no-fp.db")
    index_chunks(conn, [Chunk("doc-a", 0, "first", 1)])  # без corpus_fingerprint
    ids, _ = chunk_texts(conn)
    store_vectors(conn, ids, l2_normalize(np.eye(1, dtype=np.float32)), "m")
    assert read_meta(conn, "corpus_fingerprint") is None
    with pytest.raises(VectorsStaleError):
        semantic_search(conn, np.zeros(1, dtype=np.float32), "m", 5)


def test_store_vectors_stamps_fingerprint_matching_corpus(tmp_path: Path) -> None:
    conn, ids = _setup(tmp_path)
    store_vectors(conn, ids, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    assert read_meta(conn, "vectors_fingerprint:m") == FP


# --- check_chunk_budget: инвариант «чанк целиком видим обоим поискам» ---


def test_check_chunk_budget_passes_when_within_limit(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS)
    check_chunk_budget(conn)  # не должно бросать


def test_check_chunk_budget_passes_when_absent(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)])  # без chunk_max_tokens
    check_chunk_budget(conn)  # неизвестность — не повод отказывать


def test_check_chunk_budget_raises_when_exceeds_limit(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS * 2)
    with pytest.raises(ValueError, match=str(EMBED_MAX_TOKENS)):
        check_chunk_budget(conn)


def test_store_vectors_raises_when_chunk_budget_exceeded(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn, [Chunk("doc-a", 0, "x", 1)],
        corpus_fingerprint=FP, chunk_max_tokens=EMBED_MAX_TOKENS * 2,
    )
    with pytest.raises(ValueError):
        store_vectors(conn, [1], l2_normalize(np.eye(1, dtype=np.float32)), "m")


def test_cmd_embed_reports_budget_error_without_calling_embedder(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Гейт срабатывает ДО дорогого embedder.embed() — не тратить минуты ONNX впустую."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(
        conn, [Chunk("doc-a", 0, "x", 1)],
        corpus_fingerprint=FP, chunk_max_tokens=EMBED_MAX_TOKENS * 2,
    )
    conn.close()

    def fail_if_called(backend: str, model: str | None) -> Any:
        raise AssertionError("эмбеддер не должен вызываться — гейт обязан отсечь раньше")

    monkeypatch.setattr("vector_store._make_embedder", fail_if_called)

    args = argparse.Namespace(db=db, backend="bge", model=None)
    assert _cmd_embed(args) == 2
    assert str(EMBED_MAX_TOKENS) in capsys.readouterr().err


def test_l2_normalize_unit_and_zero() -> None:
    mat = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(mat)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # нулевая строка остаётся нулевой
