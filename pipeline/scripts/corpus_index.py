"""Единый SQLite-индекс корпуса: канонические чанки + полнотекстовый поиск FTS5.

Схема (одна БД, векторный слой Фазы 3 c6 добавит таблицу сюда же):
  chunks(chunk_id, doc_id, chunk_index, text, n_tokens)
  chunks_fts — внешне-контентная FTS5 над chunks.text, tokenize=unicode61 (многоязычно).

CLI: собрать индекс из ``doc.md`` корпуса (записи — обход ``sources/**/meta.yaml``,
пути выводятся из папок-документов) и/или искать.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from bge_tokenizer import token_counter
from chunking import Chunk, TokenCounter, chunk_text, strip_frontmatter
from schema import load_records, md_file
from validate_sources import DEFAULT_SOURCES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "pipeline" / "index" / "corpus.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY,
    doc_id      TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    n_tokens    INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='unicode61'
);
"""


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    chunk_index: int
    rank: float
    snippet: str


def fts5_available() -> bool:
    """Проверить, что sqlite3 текущего Python собран с FTS5."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
    except sqlite3.OperationalError:
        return False
    return True


def create_db(db_path: Path) -> sqlite3.Connection:
    """Открыть/создать БД со схемой."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def index_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Полная переиндексация: заменить содержимое и перестроить FTS (идемпотентно)."""
    conn.execute("DELETE FROM chunks")
    conn.executemany(
        "INSERT INTO chunks (doc_id, chunk_index, text, n_tokens) VALUES (?, ?, ?, ?)",
        [(c.doc_id, c.index, c.text, c.n_tokens) for c in chunks],
    )
    conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")
    conn.commit()


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[SearchHit]:
    """Полнотекстовый поиск (FTS5 MATCH), ранжирование bm25 (меньше = лучше)."""
    cur = conn.execute(
        "SELECT c.doc_id, c.chunk_index, bm25(chunks_fts) AS rank, "
        "snippet(chunks_fts, 0, '[', ']', '…', 12) AS snip "
        "FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (query, limit),
    )
    return [SearchHit(str(r[0]), int(r[1]), float(r[2]), str(r[3])) for r in cur.fetchall()]


def chunks_from_corpus(
    sources_root: Path,
    count_tokens: TokenCounter,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Собрать канонические чанки всех doc.md корпуса (пути выводятся из папки-документа)."""
    chunks: list[Chunk] = []
    for rec in load_records(sources_root):
        md = md_file(rec, sources_root)
        if not md.exists():
            print(f"  пропуск {rec.id}: нет файла {md}", file=sys.stderr)
            continue
        text = strip_frontmatter(md.read_text(encoding="utf-8"))
        chunks.extend(chunk_text(text, count_tokens, max_tokens, doc_id=rec.id))
    return chunks


def _cmd_build(args: argparse.Namespace) -> int:
    if not fts5_available():
        print("SQLite без поддержки FTS5 — индекс не построить", file=sys.stderr)
        return 3
    chunks = chunks_from_corpus(args.sources, token_counter(), args.max_tokens)
    conn = create_db(args.db)
    index_chunks(conn, chunks)
    n_docs = len({c.doc_id for c in chunks})
    print(f"Проиндексировано: {len(chunks)} чанков из {n_docs} документов -> {args.db}")
    conn.close()
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"индекс не найден: {args.db} (сначала build)", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    hits = fts_search(conn, args.query, args.limit)
    conn.close()
    if not hits:
        print("ничего не найдено")
        return 0
    for hit in hits:
        print(f"[{hit.rank:+.2f}] {hit.doc_id} #{hit.chunk_index}: {hit.snippet}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FTS5-индекс корпуса G2AI")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"путь к БД ({DEFAULT_DB})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="построить индекс из .md корпуса")
    p_build.add_argument("sources", nargs="?", type=Path, default=DEFAULT_SOURCES)
    p_build.add_argument("--max-tokens", type=int, default=512)
    p_build.set_defaults(func=_cmd_build)

    p_search = sub.add_parser("search", help="полнотекстовый поиск")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
