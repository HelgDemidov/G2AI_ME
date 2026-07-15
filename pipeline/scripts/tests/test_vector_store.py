"""Тесты векторного хранилища и брутфорс-косинуса (синтетические векторы, без модели)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from chunking import Chunk
from corpus_index import create_db, fts5_available, index_chunks
from embed import l2_normalize
from vector_store import chunk_texts, load_vectors, semantic_search, store_vectors

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")


def _setup(tmp_path: Path) -> tuple[sqlite3.Connection, list[int]]:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-a", 0, "first", 1),
            Chunk("doc-a", 1, "second", 1),
            Chunk("doc-b", 0, "third", 1),
        ],
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


def test_absent_model_empty(tmp_path: Path) -> None:
    conn, _ = _setup(tmp_path)
    assert semantic_search(conn, np.zeros(3, dtype=np.float32), "absent", 5) == []


def test_l2_normalize_unit_and_zero() -> None:
    mat = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(mat)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # нулевая строка остаётся нулевой
