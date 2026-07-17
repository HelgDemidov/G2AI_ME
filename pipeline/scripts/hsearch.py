"""CLI: гибридный поиск по корпусу (RRF FTS5+вектор, фасетные фильтры).

Тонкий верхнеуровневый вход поверх ``analyze.retrieve.retrieve()`` (spec
analyze-retrieval §5, чартер analyze §6). ``--backend none`` — FTS-only без модели.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from analyze.retrieve import RetrievalFilters, retrieve
from core.env import load_dotenv
from index.corpus_index import DEFAULT_DB
from index.embed import Embedder, get_embedder
from index.vector_store import unembedded_count


def _make_embedder(backend: str, model: str | None) -> Embedder | None:
    if backend == "none":
        return None
    load_dotenv()
    if backend == "openrouter":
        return get_embedder("openrouter", **({"model": model} if model else {}))
    return get_embedder("bge")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Гибридный поиск по корпусу G2AI (RRF: FTS5 + вектор)")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=20)
    parser.add_argument("--entity", dest="entity_id", help="фильтр по entity_id")
    parser.add_argument("--topic", help="фильтр по topics (topics_map)")
    parser.add_argument("--doc-type", dest="doc_type", help="фильтр по doc_type")
    parser.add_argument("--authority", help="фильтр по authority")
    parser.add_argument("--axis", help="фильтр по оси relevance (agentic_g2ai|digital_sovereignty)")
    parser.add_argument("--tier", dest="target_fit", help="фильтр по target_fit (primary|context|background)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--backend", choices=["bge", "openrouter", "none"], default="bge")
    parser.add_argument("--model", default=None, help="имя модели для openrouter")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"индекс не найден: {args.db} (сначала corpus_index.py build)", file=sys.stderr)
        return 2

    embedder = _make_embedder(args.backend, args.model)
    filters = RetrievalFilters(
        entity_id=args.entity_id,
        doc_type=args.doc_type,
        authority=args.authority,
        topic=args.topic,
        axis=args.axis,
        target_fit=args.target_fit,
    )

    conn = sqlite3.connect(args.db)
    try:
        results = retrieve(conn, args.query, embedder, k=args.k, filters=filters)
    except sqlite3.OperationalError as exc:
        print(f"ошибка поиска: {exc}", file=sys.stderr)
        conn.close()
        return 2

    if embedder is not None:
        missing = unembedded_count(conn, embedder.name)
        if missing:
            print(
                f"⚠ {missing} чанков ещё без векторов {embedder.name} — результат неполон, "
                f"прогоните vector_store.py embed-corpus",
                file=sys.stderr,
            )
    conn.close()

    if not results:
        print("ничего не найдено")
        return 0
    for r in results:
        preview = r.text[:120].replace("\n", " ")
        crumb = f" · {r.breadcrumb}" if r.breadcrumb else ""
        print(f"[{r.rrf_score:.4f}] {r.doc_id} #{r.chunk_index}{crumb}: {preview}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
