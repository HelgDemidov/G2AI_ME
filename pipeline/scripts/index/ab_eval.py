"""A/B-харнесс качества retrieval: режимы каналов (fts/vector/hybrid) x модели.

На КОНТРОЛЬНЫХ запросах-перефразировках (часто без буквального совпадения слов, см.
``pipeline/config/eval_queries.yaml``) считает hit@k: попал ли в топ-k выдачи чанк,
содержащий ожидаемый термин.

ВНИМАНИЕ: на маленьком корпусе это смоук-сравнение плумбинга и первый сигнал, а не
строгий бенчмарк — становится показательным по мере роста корпуса. Векторные режимы
требуют .env с OPENROUTER_API_KEY (для эталона, опционально) и локально скачанный bge-m3.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analyze.retrieve import retrieve
from index.corpus_index import DEFAULT_DB, fts_search, sanitize_fts_query
from index.embed import Embedder, get_embedder
from core.env import REPO_ROOT, load_dotenv
from index.vector_store import check_chunk_budget, chunk_hashes, embed_and_store, semantic_search

DEFAULT_EVAL_QUERIES = REPO_ROOT / "pipeline" / "config" / "eval_queries.yaml"


@dataclass(frozen=True)
class ControlQuery:
    query: str
    expect: tuple[str, ...]  # любой из терминов в топ-чанке = попадание (регистронезависимо)


def load_eval_queries(path: Path) -> list[ControlQuery]:
    """Загрузить контрольные запросы из YAML (spec analyze-retrieval §6).

    Пустой/отсутствующий ключ ``queries`` или запись без ``query``/``expect`` (или
    с пустым ``expect``) -> понятная ``ValueError`` — молчаливый пропуск проверок
    хуже явного отказа."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("queries")
    if not items:
        raise ValueError(f"{path}: пустой или отсутствующий ключ 'queries'")
    queries: list[ControlQuery] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "query" not in item or "expect" not in item:
            raise ValueError(f"{path}: запись #{i} без обязательных полей query/expect: {item!r}")
        expect = item["expect"]
        if not expect:
            raise ValueError(f"{path}: запись #{i} с пустым expect: {item!r}")
        queries.append(ControlQuery(str(item["query"]), tuple(str(e) for e in expect)))
    return queries


@dataclass(frozen=True)
class QueryOutcome:
    query: str
    hit1: bool
    hitk: bool
    top_score: float


@dataclass(frozen=True)
class ModelResult:
    name: str  # напр. "fts" (модель-независим) или "bge-m3-onnx-int8 · hybrid"
    hit1_rate: float
    hitk_rate: float
    outcomes: list[QueryOutcome]


def hit_at_k(ranked_texts: list[str], expect: tuple[str, ...], k: int) -> bool:
    """Содержит ли какой-либо из топ-k чанков ожидаемый термин."""
    for text in ranked_texts[:k]:
        low = text.lower()
        if any(term in low for term in expect):
            return True
    return False


def _summarize(name: str, outcomes: list[QueryOutcome], n_queries: int) -> ModelResult:
    n = n_queries or 1
    return ModelResult(
        name,
        sum(o.hit1 for o in outcomes) / n,
        sum(o.hitk for o in outcomes) / n,
        outcomes,
    )


def evaluate_fts(conn: sqlite3.Connection, queries: list[ControlQuery], k: int = 3) -> ModelResult:
    """fts-режим: ранжированные ПОЛНЫЕ тексты (не snippet) из ``fts_search`` —
    модель-независим, вычисляется один раз вне зависимости от выбора эмбеддера."""
    outcomes: list[QueryOutcome] = []
    for cq in queries:
        hits = fts_search(conn, sanitize_fts_query(cq.query), k)
        ranked = []
        for h in hits:
            row = conn.execute(
                "SELECT text FROM chunks WHERE doc_id = ? AND chunk_index = ?",
                (h.doc_id, h.chunk_index),
            ).fetchone()
            ranked.append(str(row[0]) if row else "")
        outcomes.append(
            QueryOutcome(
                cq.query,
                hit_at_k(ranked, cq.expect, 1),
                hit_at_k(ranked, cq.expect, k),
                hits[0].rank if hits else 0.0,  # bm25: меньше = лучше (в отличие от cosine-строк ниже)
            )
        )
    return _summarize("fts", outcomes, len(queries))


def evaluate_vector(
    conn: sqlite3.Connection,
    embedder: Embedder,
    hashes: list[str],
    texts: list[str],
    queries: list[ControlQuery],
    k: int = 3,
) -> ModelResult:
    """vector-режим: доэмбеддивает переданные (инкрементальные — только НЕ
    заэмбедженные этой моделью, spec embed-local-swap §5) хэши батчами через
    ``embed_and_store`` — уже посчитанное в предыдущих прогонах A/B не считается
    заново, — затем ``semantic_search`` (по ВСЕМ векторам модели в БД, не только
    только что добавленным) на запрос."""
    embed_and_store(conn, embedder, hashes, texts)
    outcomes: list[QueryOutcome] = []
    for cq in queries:
        query_vec = embedder.embed([cq.query], kind="query")
        hits = semantic_search(conn, query_vec[0], embedder.name, k)
        ranked = [h.text for h in hits]
        outcomes.append(
            QueryOutcome(
                cq.query,
                hit_at_k(ranked, cq.expect, 1),
                hit_at_k(ranked, cq.expect, k),
                hits[0].score if hits else 0.0,
            )
        )
    return _summarize(f"{embedder.name} · vector", outcomes, len(queries))


def evaluate_hybrid(
    conn: sqlite3.Connection, embedder: Embedder, queries: list[ControlQuery], k: int = 3
) -> ModelResult:
    """hybrid-режим: ``retrieve()`` (RRF FTS+вектор). Предполагает, что векторы
    ``embedder`` УЖЕ сохранены (см. ``evaluate_vector`` — вызывается раньше в main)."""
    outcomes: list[QueryOutcome] = []
    for cq in queries:
        scored = retrieve(conn, cq.query, embedder, k)
        ranked = [c.text for c in scored]
        outcomes.append(
            QueryOutcome(
                cq.query,
                hit_at_k(ranked, cq.expect, 1),
                hit_at_k(ranked, cq.expect, k),
                scored[0].rrf_score if scored else 0.0,
            )
        )
    return _summarize(f"{embedder.name} · hybrid", outcomes, len(queries))


def _report(results: list[ModelResult], k: int, n_queries: int) -> None:
    print("=" * 72)
    print(f"A/B качества retrieval — hit@1 / hit@{k} на {n_queries} контрольных запросах")
    print("=" * 72)
    for res in results:
        print(f"\n### {res.name}   hit@1={res.hit1_rate:.0%}   hit@{k}={res.hitk_rate:.0%}")
        for out in res.outcomes:
            mark = "✓" if out.hit1 else ("~" if out.hitk else "✗")
            print(f"  {mark} [top={out.top_score:.3f}] {out.query}")
    print("\nЛегенда: ✓ ожидаемый термин в топ-1, ~ в топ-k, ✗ не найден.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B качества retrieval: режимы каналов x модели")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--eval-queries", type=Path, default=DEFAULT_EVAL_QUERIES)
    parser.add_argument(
        "--mode", choices=["fts", "vector", "hybrid", "all"], default="all",
        help="fts — модель-независим; vector/hybrid — требуют эмбеддер(ы)",
    )
    parser.add_argument(
        "--reference-model", default="google/gemini-embedding-001", help="эталон через OpenRouter"
    )
    parser.add_argument("--no-reference", action="store_true", help="только локальный bge-m3")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"нет БД {args.db} — сначала corpus_index.py build", file=sys.stderr)
        return 2
    try:
        queries = load_eval_queries(args.eval_queries)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    load_dotenv()
    conn = sqlite3.connect(args.db)
    modes = ["fts", "vector", "hybrid"] if args.mode == "all" else [args.mode]
    results: list[ModelResult] = []

    if "fts" in modes:
        results.append(evaluate_fts(conn, queries, args.k))

    if "vector" in modes or "hybrid" in modes:
        total_row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        if not total_row or not total_row[0]:
            print("нет чанков в БД", file=sys.stderr)
            conn.close()
            return 2
        embedders: list[Embedder] = [get_embedder("bge")]
        if not args.no_reference:
            try:
                embedders.append(get_embedder("openrouter", model=args.reference_model))
            except RuntimeError as exc:
                print(f"\nэталон пропущен: {exc}", file=sys.stderr)

        for embedder in embedders:
            # ПОСЛЕ создания эмбеддера (нужен embedder.max_tokens), по каждому
            # эмбеддеру отдельно — разные модели могут иметь разный бюджет
            # (spec embed-local-swap §4); намеренно без try/except — несовместимость
            # чанков с бюджетом сравниваемой модели должна остановить прогон, не спрятаться.
            check_chunk_budget(conn, embedder.max_tokens)
            # инкрементально ПО КАЖДОМУ эмбеддеру (spec embed-local-swap §5): разные
            # модели держат разные множества уже заэмбедженных хэшей; повторный A/B
            # той же моделью не пере-считает уже посчитанное.
            hashes, texts = chunk_hashes(conn, not_embedded_for=embedder.name)
            if "vector" in modes:
                results.append(evaluate_vector(conn, embedder, hashes, texts, queries, args.k))
            else:
                embed_and_store(conn, embedder, hashes, texts)  # hybrid тоже нужен
            if "hybrid" in modes:
                results.append(evaluate_hybrid(conn, embedder, queries, args.k))

    conn.close()
    _report(results, args.k, len(queries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
