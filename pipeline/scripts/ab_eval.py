"""A/B-харнесс качества эмбеддингов: локальный bge-m3 vs эталон (gemini через OpenRouter).

На КОНТРОЛЬНЫХ запросах-перефразировках (часто без буквального совпадения слов) считает
hit@k: попал ли в топ-k семантического поиска чанк, содержащий ожидаемый термин.
Сравнивает модели бок о бок.

ВНИМАНИЕ: на маленьком корпусе (пока 1 документ) это смоук-сравнение плумбинга и первый
сигнал, а не строгий бенчмарк — становится показательным по мере роста корпуса.
Требует .env с OPENROUTER_API_KEY (для эталона) и локально скачанный bge-m3.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from corpus_index import DEFAULT_DB
from embed import Embedder, get_embedder
from env import load_dotenv
from vector_store import chunk_hashes, semantic_search, store_vectors


@dataclass(frozen=True)
class ControlQuery:
    query: str
    expect: tuple[str, ...]  # любой из терминов в топ-чанке = попадание (регистронезависимо)


CONTROL_QUERIES: list[ControlQuery] = [
    ControlQuery(
        "who is accountable when an autonomous agent makes a harmful decision",
        ("account", "responsib"),
    ),
    ControlQuery("how should AI agents be tested before they are deployed", ("test",)),
    ControlQuery(
        "restricting which tools and permissions an agent is allowed to use",
        ("permission", "tool"),
    ),
    ControlQuery("keeping a human overseeing the agent's actions", ("oversight", "human")),
    ControlQuery("monitoring agent behaviour after it is deployed", ("monitor",)),
]


@dataclass(frozen=True)
class QueryOutcome:
    query: str
    hit1: bool
    hitk: bool
    top_score: float


@dataclass(frozen=True)
class ModelResult:
    name: str
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


def evaluate(
    conn: sqlite3.Connection,
    embedder: Embedder,
    hashes: list[str],
    texts: list[str],
    queries: list[ControlQuery],
    k: int = 3,
) -> ModelResult:
    # Бенч эмбеддит ПОЛНУЮ матрицу хэшей корпуса на модель (не инкремент — сравнение
    # моделей); store_vectors ключуется content_hash (spec index-incremental §3a).
    store_vectors(conn, hashes, embedder.embed(texts), embedder.name)
    outcomes: list[QueryOutcome] = []
    for cq in queries:
        query_vec = embedder.embed([cq.query])
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
    n = len(queries) or 1
    return ModelResult(
        embedder.name,
        sum(o.hit1 for o in outcomes) / n,
        sum(o.hitk for o in outcomes) / n,
        outcomes,
    )


def _report(results: list[ModelResult], k: int) -> None:
    print("=" * 72)
    print(f"A/B качества эмбеддингов — hit@1 / hit@{k} на {len(CONTROL_QUERIES)} контрольных запросах")
    print("=" * 72)
    for res in results:
        print(f"\n### {res.name}   hit@1={res.hit1_rate:.0%}   hit@{k}={res.hitk_rate:.0%}")
        for out in res.outcomes:
            mark = "✓" if out.hit1 else ("~" if out.hitk else "✗")
            print(f"  {mark} [top={out.top_score:.3f}] {out.query}")
    print("\nЛегенда: ✓ ожидаемый термин в топ-1, ~ в топ-k, ✗ не найден.")
    print("NB: корпус мал (1 документ) — это смоук-сравнение, не строгий бенчмарк.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B качества эмбеддингов bge-m3 vs эталон")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--reference-model", default="google/gemini-embedding-001", help="эталон через OpenRouter"
    )
    parser.add_argument("--no-reference", action="store_true", help="только локальный bge-m3")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"нет БД {args.db} — сначала corpus_index.py build", file=sys.stderr)
        return 2
    load_dotenv()
    conn = sqlite3.connect(args.db)
    hashes, texts = chunk_hashes(conn)  # уникальные хэши корпуса — полная матрица на модель
    if not hashes:
        print("нет чанков в БД", file=sys.stderr)
        return 2

    results: list[ModelResult] = []
    print(f"Локальный bge-m3: эмбеддинг {len(texts)} чанков + {len(CONTROL_QUERIES)} запросов…")
    results.append(evaluate(conn, get_embedder("bge"), hashes, texts, CONTROL_QUERIES, args.k))

    if not args.no_reference:
        try:
            ref = get_embedder("openrouter", model=args.reference_model)
        except RuntimeError as exc:
            print(f"\nэталон пропущен: {exc}", file=sys.stderr)
        else:
            print(f"Эталон {args.reference_model} через OpenRouter…")
            results.append(evaluate(conn, ref, hashes, texts, CONTROL_QUERIES, args.k))

    conn.close()
    _report(results, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
