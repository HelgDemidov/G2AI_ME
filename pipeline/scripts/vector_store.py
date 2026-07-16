"""Хранение эмбеддингов в corpus.db + семантический поиск (брутфорс-косинус, numpy).

Векторы L2-нормализованы -> косинус = скалярное произведение. На масштабе корпуса
(10-30 тыс. чанков) брутфорс в numpy — миллисекунды, отдельная векторная БД не нужна.
Таблица vectors живёт в той же БД, что и chunks/chunks_fts (Фаза 3 c5).

CLI:
  embed-corpus [--backend bge|openrouter] — заэмбеддить чанки корпуса и сохранить.
  vsearch <запрос> [--backend ...]        — семантический поиск.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from corpus_index import DEFAULT_DB, read_meta, write_meta
from embed import Embedder, FloatArray, get_embedder
from env import load_dotenv

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id INTEGER NOT NULL,
    model    TEXT    NOT NULL,
    vec      BLOB    NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
"""


@dataclass(frozen=True)
class VecHit:
    chunk_id: int
    doc_id: str
    chunk_index: int
    score: float
    text: str


class VectorsStaleError(RuntimeError):
    """Векторы данной модели отсутствуют или не соответствуют текущему индексу —
    чанки пересобирались после последнего ``embed-corpus`` (``corpus_index.
    index_chunks`` дропает таблицу ``vectors`` при пересборке, см. spec
    index-consistency §3: старый ``chunk_id`` иначе молча указывал бы на чужой
    текст нового поколения). Честный отказ вместо правдоподобно неверных результатов.
    """


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def store_vectors(
    conn: sqlite3.Connection, chunk_ids: list[int], vectors: FloatArray, model: str
) -> None:
    """Полная замена векторов данной модели (идемпотентно).

    Штампует ``vectors_fingerprint:<model>`` = текущий ``corpus_fingerprint`` этой
    же БД (``index_meta``, пишет ``corpus_index.index_chunks``) — ``semantic_search``
    сверяет его перед поиском. Если ``corpus_fingerprint`` в БД ещё не установлен
    (индекс собран без него) — штамп не пишется: без опорного отпечатка свежесть
    неизвестна, безопаснее считать векторы непроверенными, чем молча доверять им.
    """
    ensure_schema(conn)
    conn.execute("DELETE FROM vectors WHERE model = ?", (model,))
    conn.executemany(
        "INSERT INTO vectors (chunk_id, model, vec) VALUES (?, ?, ?)",
        [
            (int(cid), model, vectors[i].astype(np.float32).tobytes())
            for i, cid in enumerate(chunk_ids)
        ],
    )
    fp = read_meta(conn, "corpus_fingerprint")
    if fp is not None:
        write_meta(conn, f"vectors_fingerprint:{model}", fp)
    conn.commit()


def load_vectors(conn: sqlite3.Connection, model: str) -> tuple[list[int], FloatArray]:
    ensure_schema(conn)  # поиск может идти до первого эмбеддинга — таблицы может не быть
    rows = conn.execute(
        "SELECT chunk_id, vec FROM vectors WHERE model = ? ORDER BY chunk_id", (model,)
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    ids = [int(r[0]) for r in rows]
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).astype(np.float32)
    return ids, mat


def semantic_search(
    conn: sqlite3.Connection, query_vec: FloatArray, model: str, top_k: int = 10
) -> list[VecHit]:
    """Семантический поиск. Перед поиском сверяет ``vectors_fingerprint:<model>``
    с текущим ``corpus_fingerprint`` — расхождение ИЛИ отсутствие (никогда не
    эмбеддили эту модель, либо чанки пересобирались после последнего
    ``embed-corpus``) кидают ``VectorsStaleError``, а не молча возвращают
    правдоподобно неверные (или пустые) результаты."""
    current_fp = read_meta(conn, "corpus_fingerprint")
    vectors_fp = read_meta(conn, f"vectors_fingerprint:{model}")
    if current_fp is None or vectors_fp != current_fp:
        raise VectorsStaleError(
            f"векторы модели {model!r} отсутствуют или устарели для текущего индекса — "
            "прогоните run_pipeline --embed (или vector_store.py embed-corpus)"
        )
    ids, mat = load_vectors(conn, model)
    if not ids:
        return []
    query = query_vec.reshape(-1).astype(np.float32)
    scores = mat @ query
    order = np.argsort(-scores)[:top_k]
    hits: list[VecHit] = []
    for idx in order:
        cid = ids[int(idx)]
        row = conn.execute(
            "SELECT doc_id, chunk_index, text FROM chunks WHERE chunk_id = ?", (cid,)
        ).fetchone()
        if row is None:
            continue
        hits.append(VecHit(cid, str(row[0]), int(row[1]), float(scores[int(idx)]), str(row[2])))
    return hits


def chunk_texts(conn: sqlite3.Connection) -> tuple[list[int], list[str]]:
    rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    return [int(r[0]) for r in rows], [str(r[1]) for r in rows]


def _make_embedder(backend: str, model: str | None) -> Embedder:
    load_dotenv()
    if backend == "openrouter":
        kwargs = {"model": model} if model else {}
        return get_embedder("openrouter", **kwargs)
    return get_embedder("bge")


def _cmd_embed(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"нет БД {args.db} — сначала corpus_index.py build", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    ids, texts = chunk_texts(conn)
    if not ids:
        print("нет чанков в БД", file=sys.stderr)
        return 2
    embedder = _make_embedder(args.backend, args.model)
    vectors = embedder.embed(texts)
    store_vectors(conn, ids, vectors, embedder.name)
    conn.close()
    print(f"Векторов: {len(ids)} x {embedder.dim} ({embedder.name}) -> {args.db}")
    return 0


def _cmd_vsearch(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"нет БД {args.db}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    embedder = _make_embedder(args.backend, args.model)
    query_vec = embedder.embed([args.query])
    try:
        hits = semantic_search(conn, query_vec[0], embedder.name, args.limit)
    except VectorsStaleError as exc:
        print(str(exc), file=sys.stderr)
        conn.close()
        return 2
    conn.close()
    if not hits:
        print("ничего не найдено")
        return 0
    for hit in hits:
        preview = hit.text[:120].replace("\n", " ")
        print(f"[{hit.score:.3f}] {hit.doc_id} #{hit.chunk_index}: {preview}…")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=DEFAULT_DB)
    common.add_argument("--backend", choices=["bge", "openrouter"], default="bge")
    common.add_argument("--model", default=None, help="имя модели для openrouter")

    parser = argparse.ArgumentParser(description="Векторный слой корпуса G2AI (эмбеддинги + поиск)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_embed = sub.add_parser(
        "embed-corpus", parents=[common], help="заэмбеддить чанки и сохранить векторы"
    )
    p_embed.set_defaults(func=_cmd_embed)

    p_search = sub.add_parser("vsearch", parents=[common], help="семантический поиск")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=_cmd_vsearch)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
