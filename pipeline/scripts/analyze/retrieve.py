"""Единый retrieval-фасад: RRF-гибрид (FTS5 + вектор) с фасетными фильтрами.

Два несвязанных канала (``corpus_index.fts_search`` / ``vector_store.semantic_search``)
объединяются здесь в один ранжированный список чанков — Reciprocal Rank Fusion,
стандартный baseline гибридного поиска (spec analyze-retrieval §4). Фильтры по
метаданным (``doc_facets``/``topics_map``) резолвятся в множество ``doc_id`` ДО
похода в оба канала — оба канала фильтруют кандидатов на своей стороне, а не
постфактум (иначе top-N кандидатов каждого канала могли бы целиком не пройти фильтр).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from index.corpus_index import SearchHit, fts_search, sanitize_fts_query
from index.embed import Embedder
from index.vector_store import VecHit, semantic_search

RRF_K = 60  # решение чартера §8.3: константа, НЕ конфиг; тюнинг только после eval-данных
POOL = 50  # кандидатов на канал до слияния (практика 2026: top-50)


@dataclass(frozen=True)
class RetrievalFilters:
    entity_id: str | None = None
    doc_type: str | None = None
    authority: str | None = None
    topic: str | None = None  # membership в topics_map
    axis: str | None = None
    target_fit: str | None = None


@dataclass(frozen=True)
class ScoredChunk:
    doc_id: str
    chunk_index: int
    breadcrumb: str
    text: str
    rrf_score: float
    fts_rank: int | None  # 1-based ранг в канале; None — канал не нашёл
    vec_rank: int | None


def _resolve_filters(conn: sqlite3.Connection, filters: RetrievalFilters | None) -> set[str] | None:
    """SELECT doc_id из ``doc_facets`` [+ JOIN ``topics_map`` при ``topic``]; все
    условия — AND. ``None`` при пустых фильтрах (весь корпус доступен обоим каналам)."""
    if filters is None:
        return None
    conditions: list[str] = []
    params: list[str] = []
    for column, value in (
        ("entity_id", filters.entity_id),
        ("doc_type", filters.doc_type),
        ("authority", filters.authority),
        ("axis", filters.axis),
        ("target_fit", filters.target_fit),
    ):
        if value is not None:
            conditions.append(f"doc_facets.{column} = ?")
            params.append(value)
    sql = "SELECT DISTINCT doc_facets.doc_id FROM doc_facets"
    if filters.topic is not None:
        sql += " JOIN topics_map ON topics_map.doc_id = doc_facets.doc_id"
        conditions.append("topics_map.topic = ?")
        params.append(filters.topic)
    if not conditions:
        return None
    sql += " WHERE " + " AND ".join(conditions)
    rows = conn.execute(sql, params).fetchall()
    return {str(r[0]) for r in rows}


def _rank_map(keys: list[tuple[str, int]]) -> dict[tuple[str, int], int]:
    """1-based ранг по позиции в выдаче канала (порядок уже задан вызывающим каналом)."""
    return {key: i + 1 for i, key in enumerate(keys)}


def _chunk_lookup(
    conn: sqlite3.Connection, keys: list[tuple[str, int]]
) -> dict[tuple[str, int], tuple[str, str]]:
    """breadcrumb+text для итоговых (doc_id, chunk_index) — ОДИН SQL-запрос на весь
    итоговый список, не по хиту (spec analyze-retrieval §4.5)."""
    if not keys:
        return {}
    conditions = " OR ".join(["(doc_id = ? AND chunk_index = ?)"] * len(keys))
    params: list[object] = [v for doc_id, idx in keys for v in (doc_id, idx)]
    rows = conn.execute(
        f"SELECT doc_id, chunk_index, breadcrumb, text FROM chunks WHERE {conditions}", params
    ).fetchall()
    return {(str(r[0]), int(r[1])): (str(r[2]), str(r[3])) for r in rows}


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    embedder: Embedder | None,
    k: int = 20,
    filters: RetrievalFilters | None = None,
) -> list[ScoredChunk]:
    """RRF-гибрид FTS+вектор с опциональными фасетными фильтрами.

    ``embedder=None`` — честная деградация до FTS-only (``vec_rank`` всех результатов
    — ``None``): нужна CI/свежей БД без векторов и CLI ``--backend none``. Санитизация
    запроса — ВНУТРИ (фасад для аналитика; сырой FTS5-синтаксис остаётся у
    ``corpus_index.py search --raw``).
    """
    allowed = _resolve_filters(conn, filters)

    fts_hits: list[SearchHit] = fts_search(conn, sanitize_fts_query(query), POOL, allowed_doc_ids=allowed)
    vec_hits: list[VecHit] = []
    if embedder is not None:
        query_vec = embedder.embed([query], kind="query")[0]
        vec_hits = semantic_search(conn, query_vec, embedder.name, POOL, allowed_doc_ids=allowed)

    fts_rank = _rank_map([(h.doc_id, h.chunk_index) for h in fts_hits])
    vec_rank = _rank_map([(h.doc_id, h.chunk_index) for h in vec_hits])
    keys = set(fts_rank) | set(vec_rank)

    scored: list[tuple[float, str, int, int | None, int | None]] = []
    for doc_id, chunk_index in keys:
        fr = fts_rank.get((doc_id, chunk_index))
        vr = vec_rank.get((doc_id, chunk_index))
        score = (1.0 / (RRF_K + fr) if fr is not None else 0.0) + (
            1.0 / (RRF_K + vr) if vr is not None else 0.0
        )
        scored.append((score, doc_id, chunk_index, fr, vr))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))  # score DESC, (doc_id, chunk_index) ASC — детерминизм
    top = scored[:k]

    lookup = _chunk_lookup(conn, [(doc_id, idx) for _, doc_id, idx, _, _ in top])
    return [
        ScoredChunk(
            doc_id=doc_id,
            chunk_index=chunk_index,
            breadcrumb=lookup[(doc_id, chunk_index)][0],
            text=lookup[(doc_id, chunk_index)][1],
            rrf_score=score,
            fts_rank=fr,
            vec_rank=vr,
        )
        for score, doc_id, chunk_index, fr, vr in top
    ]
