"""Тесты RRF-гибрида retrieve() — синтетические чанки, фейковый эмбеддер (CI-safe)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analyze.retrieve import RetrievalFilters, retrieve
from index.chunking import Chunk
from index.corpus_index import _rebuild_facets, content_hash, create_db, fts5_available, index_chunks
from index.embed import FloatArray, l2_normalize
from index.vector_store import chunk_hashes, store_vectors
from core.schema import SourceRecord
from tests.support import valid_record

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")

MODEL = "fake-model"


class FakeEmbedder:
    """Фейковый эмбеддер: любой запрос -> заданный вектор (детерминизм без модели)."""

    name = MODEL
    dim = 1

    def __init__(self, query_vec: FloatArray) -> None:
        self._vec = query_vec

    def embed(self, texts: list[str]) -> FloatArray:
        return np.vstack([self._vec for _ in texts])


def _store_orthonormal_vectors(
    conn: object, model: str = MODEL
) -> tuple[list[str], list[str], FloatArray]:
    """Заэмбеддить ВСЕ хэши корпуса ортонормированным базисом — косинус с любым
    базисным вектором точно определяет ранг (паттерн test_vector_store.py)."""
    hashes, texts = chunk_hashes(conn)  # type: ignore[arg-type]
    vecs = l2_normalize(np.eye(len(hashes), dtype=np.float32))
    store_vectors(conn, hashes, vecs, model)  # type: ignore[arg-type]
    return hashes, texts, vecs


# --- деградация без эмбеддера / пустая выдача ---


def test_fts_only_when_embedder_none(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [
        Chunk("doc-a", 0, "agentic governance framework", 4),
        Chunk("doc-b", 0, "unrelated content here", 4),
    ])
    results = retrieve(conn, "governance", None, k=10)
    assert len(results) == 1
    assert results[0].doc_id == "doc-a"
    assert results[0].fts_rank == 1
    assert results[0].vec_rank is None


def test_empty_both_channels_returns_empty_list(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "some text", 2)])
    assert retrieve(conn, "nonexistentword", None, k=10) == []


# --- RRF-математика на синтетических ранговых списках ---


def test_chunk_only_in_vector_channel_has_null_fts_rank(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "zzz completely unrelated terms", 5)])
    _hashes, _texts, vecs = _store_orthonormal_vectors(conn)
    embedder = FakeEmbedder(vecs[0])
    results = retrieve(conn, "nomatchword", embedder, k=10)
    assert len(results) == 1
    assert results[0].fts_rank is None
    assert results[0].vec_rank == 1


def test_rrf_score_combines_both_channel_ranks(tmp_path: Path) -> None:
    """Чанк на 1-м месте ОБОИХ каналов ранжируется выше чанка на 2-м месте обоих —
    сумма 1/(RRF_K+rank) по каналам, не голосование большинством."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [
        Chunk("doc-best", 0, "shared keyword appears here", 5),
        Chunk("doc-second", 0, "shared keyword only in text", 5),
    ])
    _hashes, texts, vecs = _store_orthonormal_vectors(conn)
    best_idx = texts.index("shared keyword appears here")
    embedder = FakeEmbedder(vecs[best_idx])

    results = retrieve(conn, "shared keyword", embedder, k=10)

    assert results[0].doc_id == "doc-best"
    assert results[0].fts_rank == 1 and results[0].vec_rank == 1
    assert results[1].doc_id == "doc-second"
    assert results[1].fts_rank == 2 and results[1].vec_rank == 2
    assert results[0].rrf_score > results[1].rrf_score


def test_tie_break_by_doc_id_when_scores_equal(tmp_path: Path) -> None:
    """Оба чанка на ранге 1 своего (единственного) канала -> равный rrf_score;
    тай-брейк — doc_id ASC (spec analyze-retrieval §4.4)."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [
        Chunk("doc-b", 0, "keyword appears only here", 4),
        Chunk("doc-a", 0, "totally different phrase", 4),
    ])
    # заэмбеддить ТОЛЬКО doc-a: doc-b виден исключительно FTS-каналом, doc-a —
    # исключительно векторным, оба на ранге 1 своего канала -> равный score.
    vec = l2_normalize(np.ones((1, 4), dtype=np.float32))
    store_vectors(conn, [content_hash("totally different phrase")], vec, MODEL)
    embedder = FakeEmbedder(vec[0])

    results = retrieve(conn, "keyword", embedder, k=10)

    assert [r.doc_id for r in results] == ["doc-a", "doc-b"]
    assert results[0].rrf_score == pytest.approx(results[1].rrf_score)


def test_results_sorted_score_desc_then_key_asc(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [
        Chunk("doc-a", 0, "keyword one", 2),
        Chunk("doc-b", 0, "keyword two", 2),
        Chunk("doc-c", 0, "keyword three", 2),
    ])
    results = retrieve(conn, "keyword", None, k=10)
    scores = [(r.rrf_score, r.doc_id, r.chunk_index) for r in results]
    assert scores == sorted(scores, key=lambda t: (-t[0], t[1], t[2]))


# --- фасетные фильтры (spec analyze-retrieval §4.1) ---


def _record(id_: str, entity_id: str, topics: list[str], axis: str, target_fit: str) -> SourceRecord:
    rec = valid_record()
    rec["id"] = id_
    rec["entity_id"] = entity_id
    rec["topics"] = topics
    rec["relevance"]["axis"] = axis
    rec["relevance"]["target_fit"] = target_fit
    return SourceRecord.model_validate(rec)


def test_filters_narrow_results_by_entity_topic_axis(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    rec_sg = _record("sg-doc-2026", "sg", ["agentic-ai"], "agentic_g2ai", "primary")
    rec_ee = _record("ee-doc-2026", "ee", ["digital-id"], "digital_sovereignty", "context")
    _rebuild_facets(conn, [rec_sg, rec_ee])
    index_chunks(conn, [
        Chunk(rec_sg.id, 0, "shared governance term", 4),
        Chunk(rec_ee.id, 0, "shared governance term", 4),
    ])

    unfiltered = retrieve(conn, "governance", None, k=10)
    assert {r.doc_id for r in unfiltered} == {rec_sg.id, rec_ee.id}

    by_entity = retrieve(conn, "governance", None, k=10, filters=RetrievalFilters(entity_id="sg"))
    assert {r.doc_id for r in by_entity} == {rec_sg.id}

    by_topic = retrieve(conn, "governance", None, k=10, filters=RetrievalFilters(topic="digital-id"))
    assert {r.doc_id for r in by_topic} == {rec_ee.id}

    by_axis = retrieve(conn, "governance", None, k=10, filters=RetrievalFilters(axis="agentic_g2ai"))
    assert {r.doc_id for r in by_axis} == {rec_sg.id}


def test_filters_empty_match_returns_empty_list(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    rec = _record("sg-doc-2026", "sg", ["agentic-ai"], "agentic_g2ai", "primary")
    _rebuild_facets(conn, [rec])
    index_chunks(conn, [Chunk(rec.id, 0, "governance text", 3)])
    results = retrieve(conn, "governance", None, k=10, filters=RetrievalFilters(entity_id="nowhere"))
    assert results == []
