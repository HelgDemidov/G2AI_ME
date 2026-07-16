"""Тесты векторного хранилища и брутфорс-косинуса (синтетические векторы, без модели).

Ключ вектора — content_hash чанка (spec index-incremental §3): пере-чанковка не
осиротит неизменившиеся векторы, вектор не может указать на чужой текст; эмбеддинг
инкрементален (только новые хэши), осиротевшие чистит gc_vectors."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bge_tokenizer import EMBED_MAX_TOKENS
from chunking import Chunk
from corpus_index import create_db, fts5_available, index_chunks
from embed import l2_normalize
from vector_store import (
    _cmd_embed,
    check_chunk_budget,
    chunk_hashes,
    gc_vectors,
    load_vectors,
    semantic_search,
    store_vectors,
    unembedded_count,
)

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")


def _setup(tmp_path: Path) -> sqlite3.Connection:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-a", 0, "first", 1),
            Chunk("doc-a", 1, "second", 1),
            Chunk("doc-b", 0, "third", 1),
        ],
    )
    return conn


def _vec_count(conn: sqlite3.Connection, model: str) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM vectors WHERE model=?", (model,)).fetchone()[0])


# --- store / load: ключ content_hash ---


def test_store_and_load(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    got, mat = load_vectors(conn, "m")
    assert set(got) == set(hashes)
    assert mat.shape == (3, 3)


def test_reembed_upserts_not_duplicates(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    vecs = l2_normalize(np.ones((3, 3), dtype=np.float32))
    store_vectors(conn, hashes, vecs, "m")
    store_vectors(conn, hashes, vecs, "m")  # повторно — upsert по (content_hash, model)
    assert _vec_count(conn, "m") == 3


def test_semantic_search_nearest(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, texts = chunk_hashes(conn)
    vecs = l2_normalize(np.eye(len(hashes), dtype=np.float32))
    store_vectors(conn, hashes, vecs, "m")
    ti = texts.index("second")  # целимся в конкретный текст, не в позицию хэша
    hits = semantic_search(conn, vecs[ti], "m", top_k=len(hashes))
    assert hits[0].text == "second"
    assert hits[0].doc_id == "doc-a"
    assert hits[0].chunk_index == 1
    assert hits[0].score >= hits[-1].score


def test_semantic_search_empty_when_no_vectors(tmp_path: Path) -> None:
    """Ни одного вектора модели → пустая выдача (не исключение): устаревших/неверных
    результатов не бывает по построению, отсутствие репортит CLI."""
    conn = _setup(tmp_path)
    assert semantic_search(conn, np.zeros(3, dtype=np.float32), "m", 5) == []


# --- boilerplate: общий хэш двух документов = один вектор, оба чанка находимы ---


def test_shared_hash_one_vector_both_chunks_found(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-a", 0, "boilerplate", 1),  # тот же текст…
            Chunk("doc-b", 0, "boilerplate", 1),  # …в другом документе
            Chunk("doc-b", 1, "unique", 1),
        ],
    )
    hashes, texts = chunk_hashes(conn)
    assert len(hashes) == 2  # дубль схлопнут: boilerplate + unique
    store_vectors(conn, hashes, l2_normalize(np.eye(2, dtype=np.float32)), "m")
    assert _vec_count(conn, "m") == 2  # один вектор на общий хэш, не два

    bi = texts.index("boilerplate")
    hits = semantic_search(conn, l2_normalize(np.eye(2, dtype=np.float32))[bi], "m", top_k=2)
    boiler = {(h.doc_id, h.chunk_index) for h in hits if h.text == "boilerplate"}
    assert boiler == {("doc-a", 0), ("doc-b", 0)}  # score роздан обоим носителям хэша


# --- инкрементальный отбор: только новые хэши ---


def test_chunk_hashes_pending_excludes_embedded(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    all_hashes, _ = chunk_hashes(conn)
    assert len(all_hashes) == 3
    store_vectors(conn, all_hashes[:2], l2_normalize(np.eye(2, dtype=np.float32)), "m")

    pending, _ = chunk_hashes(conn, not_embedded_for="m")
    assert set(pending) == {all_hashes[2]}  # только ещё не заэмбедженный
    other, _ = chunk_hashes(conn, not_embedded_for="other")
    assert set(other) == set(all_hashes)  # другая модель — всё заново


def test_chunk_hashes_pending_when_vectors_table_absent(tmp_path: Path) -> None:
    """Первый embed: таблицы vectors ещё нет — pending-отбор не должен падать
    (регресс real-run: подзапрос по vectors ронял `no such table`)."""
    conn = _setup(tmp_path)  # чанки есть, ни одного store_vectors → таблицы vectors нет
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'"
    ).fetchone() is None
    pending, _ = chunk_hashes(conn, not_embedded_for="m")
    all_hashes, _ = chunk_hashes(conn)
    assert set(pending) == set(all_hashes)  # ничего не заэмбеддено — все pending


def test_unembedded_count(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    assert unembedded_count(conn, "m") == 3
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes[:1], l2_normalize(np.eye(1, dtype=np.float32)), "m")
    assert unembedded_count(conn, "m") == 2


# --- GC осиротевших векторов ---


def test_gc_removes_orphaned_vectors(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    orphan = hashes[0]
    conn.execute("DELETE FROM chunks WHERE content_hash=?", (orphan,))  # текст исчез из корпуса
    conn.commit()

    removed = gc_vectors(conn, "m")
    assert removed == 1
    remaining, _ = load_vectors(conn, "m")
    assert orphan not in remaining
    assert len(remaining) == 2


def test_gc_keeps_vectors_still_referenced(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    assert gc_vectors(conn, "m") == 0  # все хэши ещё в chunks
    assert _vec_count(conn, "m") == 3


# --- check_chunk_budget: инвариант «чанк целиком видим обоим поискам» (index-consistency §6) ---


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
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS * 2)
    with pytest.raises(ValueError):
        store_vectors(conn, ["deadbeef"], l2_normalize(np.eye(1, dtype=np.float32)), "m")


def test_cmd_embed_reports_budget_error_without_calling_embedder(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Гейт срабатывает ДО дорогого embedder.embed() — не тратить минуты ONNX впустую."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS * 2)
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
