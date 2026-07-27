"""CLI: гибридный поиск по корпусу (RRF FTS5+вектор, фасетные фильтры).

Тонкий верхнеуровневый вход поверх ``analyze.retrieve.retrieve()`` (spec
analyze-retrieval §5, чартер analyze §6). ``--backend none`` — FTS-only без модели.

``--sources`` (knowledge-hardening §9) — advisory-сверка ``corpus_fingerprint``:
предупреждает, если индекс собран до последней правки корпуса (напр. новое ребро
``supersedes``), не блокирует поиск.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from analyze.retrieve import RetrievalFilters, retrieve
from core.env import load_dotenv
from core.schema import DEFAULT_SOURCES
from index.corpus_index import DEFAULT_DB, corpus_fingerprint, read_meta
from index.embed import DEFAULT_BACKEND, Embedder, get_embedder
from index.vector_store import unembedded_count


def _make_embedder(backend: str, model: str | None) -> Embedder | None:
    if backend == "none":
        return None
    load_dotenv()
    if backend == "openrouter":
        return get_embedder("openrouter", **({"model": model} if model else {}))
    return get_embedder("bge")


def _warn_if_stale(conn: sqlite3.Connection, sources_root: Path) -> None:
    """Advisory-предупреждение о протухшем индексе (knowledge-hardening §9 — шов
    аудита между графом и retrieval): граф читает ``meta.yaml`` живьём при каждом
    вызове, ``doc_facets`` — только при прогоне индексации. Правка ребра
    ``supersedes`` без прогона ``run_pipeline`` оставляла ``retrieve()`` молча
    фильтровать по старому множеству — сверка дешёвая (``corpus_fingerprint`` —
    только ``stat()``) и НЕ блокирует поиск, только предупреждает.

    Несуществующий корень (запуск вне репо/без локального корпуса) — сверить
    нечего, молча пропускаем, а не отказываем."""
    if not sources_root.exists():
        return
    stored = read_meta(conn, "corpus_fingerprint")
    if stored is not None and stored != corpus_fingerprint(sources_root):
        print(
            "⚠ индекс отстаёт от корпуса (corpus_fingerprint не совпадает) — "
            "прогоните run_pipeline.py, иначе фасетные фильтры видят устаревшее состояние",
            file=sys.stderr,
        )


def _provenance_notes(conn: sqlite3.Connection, doc_ids: set[str]) -> dict[str, str]:
    """``doc_id -> короткая пометка происхождения/качества`` из ``doc_facets``
    (spec convert-knowledge-seam-hardening §3).

    Сигналы acquire/convert (fidelity добычи, находки recheck, lint-дефекты) до этого
    спека умирали в ``.state.yaml``: документ из Wayback-снимка был в выдаче неотличим
    от живого оригинала, а скан с непогашенным расхождением чисел — от чистого.
    Легаси-БД без колонок — пустой словарь (поиск не отказывает)."""
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    try:
        rows = conn.execute(
            f"SELECT doc_id, fidelity, quality_flags FROM doc_facets WHERE doc_id IN ({placeholders})",
            sorted(doc_ids),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    notes: dict[str, str] = {}
    for doc_id, fidelity, quality_flags in rows:
        parts = [p for p in (str(quality_flags or ""), ) if p]
        if fidelity == "archived_snapshot":
            parts.insert(0, "archived_snapshot")
        if parts:
            notes[str(doc_id)] = ";".join(parts)
    return notes


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
    parser.add_argument(
        "--include-superseded", action="store_true",
        help="не скрывать заменённые редакции (по умолчанию выдаются только действующие; "
             "флаг нужен для исследования истории документа)",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--sources", type=Path, default=DEFAULT_SOURCES,
        help="корень корпуса — для staleness-сверки индекса (knowledge-hardening §9)",
    )
    parser.add_argument(
        "--backend", choices=["bge", "openrouter", "none"], default=DEFAULT_BACKEND,
        help="openrouter — production-дефолт; bge — локальный фолбэк; none — FTS-only (офлайн)",
    )
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
        include_superseded=args.include_superseded,
    )

    conn = sqlite3.connect(args.db)
    _warn_if_stale(conn, args.sources)
    try:
        results = retrieve(conn, args.query, embedder, k=args.k, filters=filters)
    except sqlite3.OperationalError as exc:
        print(f"ошибка поиска: {exc}", file=sys.stderr)
        conn.close()
        return 2
    notes = _provenance_notes(conn, {r.doc_id for r in results})

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
        # Провенанс чанка (spec convert-knowledge-seam-hardening §2): машинная
        # реконструкция фигуры выглядит как обычная проза документа — до этой пометки
        # аналитик не мог отличить её в выдаче вообще ничем.
        provenance = " · ⚠ reconstruction" if r.reconstruction else ""
        note = f" · ⚑ {notes[r.doc_id]}" if r.doc_id in notes else ""
        print(f"[{r.rrf_score:.4f}] {r.doc_id} #{r.chunk_index}{crumb}{provenance}{note}: {preview}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
