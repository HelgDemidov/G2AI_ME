"""Хранение эмбеддингов в corpus.db + семантический поиск (брутфорс-косинус, numpy).

Векторы L2-нормализованы -> косинус = скалярное произведение. На масштабе корпуса
(10-30 тыс. чанков) брутфорс в numpy — миллисекунды, отдельная векторная БД не нужна.
Таблица vectors живёт в той же БД, что и chunks/chunks_fts.

Ключ вектора — ``content_hash`` чанка (sha256 текста), НЕ эфемерный ``chunk_id``:
пере-чанковка не осиротит неизменившиеся векторы, а вектор физически не может
указать на чужой текст (spec index-incremental §3). Эмбеддинг инкрементален —
эмбеддятся только НОВЫЕ хэши (``chunk_hashes(not_embedded_for=…)``), осиротевшие
подчищает ``gc_vectors``. Fingerprint-сверка векторов (index-consistency стоп-гэп)
упразднена — устаревших результатов не бывает по построению.

CLI:
  embed-corpus [--backend bge|openrouter] — доэмбеддить новые чанки корпуса + GC.
  vsearch <запрос> [--backend ...]        — семантический поиск.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from index.bge_tokenizer import EMBED_MAX_TOKENS
from index.chunking import embed_input
from index.corpus_index import DEFAULT_DB, read_meta
from index.embed import Embedder, FloatArray, get_embedder
from core.env import load_dotenv

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    vec          BLOB NOT NULL,
    PRIMARY KEY (content_hash, model)
);
"""


@dataclass(frozen=True)
class VecHit:
    chunk_id: int
    doc_id: str
    chunk_index: int
    score: float
    text: str
    breadcrumb: str


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def check_chunk_budget(conn: sqlite3.Connection) -> None:
    """Гейт: чанки индекса не должны быть крупнее бюджета эмбеддера
    (``EMBED_MAX_TOKENS``) — иначе ``OnnxBgeEmbedder`` молча truncate'ит каждый
    чанк до своего лимита, и вектор представлял бы только префикс: инвариант
    «канонический чанк целиком видим и FTS, и векторному поиску» перестал бы
    быть проверяемым, оставаясь лишь подразумеваемым. Отсутствие
    ``chunk_max_tokens`` в ``index_meta`` (индекс собран без него) не гейтится —
    неизвестность не повод отказывать.
    """
    chunk_max_str = read_meta(conn, "chunk_max_tokens")
    if chunk_max_str is not None and int(chunk_max_str) > EMBED_MAX_TOKENS:
        raise ValueError(
            f"индекс собран с чанками {chunk_max_str} > лимита эмбеддера {EMBED_MAX_TOKENS}: "
            f"векторы представляли бы префиксы; пересоберите с --max-tokens {EMBED_MAX_TOKENS}"
        )


def store_vectors(
    conn: sqlite3.Connection, content_hashes: list[str], vectors: FloatArray, model: str
) -> None:
    """Идемпотентный upsert векторов по ключу ``(content_hash, model)``.

    Тупой писатель: НЕ решает, какие хэши эмбеддить (это ``chunk_hashes`` —
    инкрементальный отбор для production, полная матрица для ab_eval), и НЕ штампует
    fingerprint. Ключ — СОДЕРЖИМОЕ чанка (sha256), поэтому вектор физически не может
    указать на чужой текст: класс дефекта «векторы врут» (spec index-consistency
    §0.1) устранён по построению, а не сверкой отпечатков (упразднена).
    """
    check_chunk_budget(conn)
    ensure_schema(conn)
    conn.executemany(
        "INSERT INTO vectors (content_hash, model, vec) VALUES (?, ?, ?) "
        "ON CONFLICT(content_hash, model) DO UPDATE SET vec = excluded.vec",
        [(h, model, vectors[i].astype(np.float32).tobytes()) for i, h in enumerate(content_hashes)],
    )
    conn.commit()


def load_vectors(conn: sqlite3.Connection, model: str) -> tuple[list[str], FloatArray]:
    ensure_schema(conn)  # поиск может идти до первого эмбеддинга — таблицы может не быть
    rows = conn.execute(
        "SELECT content_hash, vec FROM vectors WHERE model = ? ORDER BY content_hash", (model,)
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    hashes = [str(r[0]) for r in rows]
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).astype(np.float32)
    return hashes, mat


def semantic_search(
    conn: sqlite3.Connection,
    query_vec: FloatArray,
    model: str,
    top_k: int = 10,
    *,
    allowed_doc_ids: set[str] | None = None,
) -> list[VecHit]:
    """Семантический поиск по векторам модели (брутфорс-косинус). Вектор ключуется
    ``content_hash`` — join к ``chunks`` раздаёт score ВСЕМ носителям хэша (общий
    boilerplate-чанк находим в каждом документе-носителе). Fingerprint-гейт УПРАЗДНЁН:
    вектор не может указать на чужой текст (ключ — содержимое), устаревших
    результатов не бывает; неполноту (часть чанков ещё не заэмбеддена) репортит
    CLI по ``unembedded_count``.

    ``allowed_doc_ids`` — опциональный фасетный фильтр (``retrieve()``, spec
    analyze-retrieval §3.2): носители вне множества отбрасываются на этапе раздачи
    хэша носителям; если у хэша не осталось носителей — хэш пропускается БЕЗ учёта в
    бюджете ``top_k``, следующий по score хэш добирает бюджет (итерация по полному
    ``order`` до наполнения — НЕ срез ``[:top_k]`` до фильтра, иначе легитимные хиты
    терялись бы за отфильтрованными). ``None`` — прежнее поведение (бюджет — по
    числу рассмотренных хэшей, как раньше)."""
    hashes, mat = load_vectors(conn, model)
    if not hashes:
        return []
    query = query_vec.reshape(-1).astype(np.float32)
    scores = mat @ query
    order = np.argsort(-scores)
    hits: list[VecHit] = []
    contributed = 0
    for idx in order:
        if contributed >= top_k:
            break
        score = float(scores[int(idx)])
        rows = conn.execute(
            "SELECT chunk_id, doc_id, chunk_index, text, breadcrumb FROM chunks WHERE content_hash = ? "
            "ORDER BY doc_id, chunk_index",
            (hashes[int(idx)],),
        ).fetchall()
        if allowed_doc_ids is not None:
            rows = [r for r in rows if str(r[1]) in allowed_doc_ids]
        if not rows:
            continue  # хэш без носителей внутри фильтра — не считается в бюджет top_k
        contributed += 1
        hits.extend(
            VecHit(int(r[0]), str(r[1]), int(r[2]), score, str(r[3]), str(r[4])) for r in rows
        )
    return hits


def chunk_hashes(
    conn: sqlite3.Connection, *, not_embedded_for: str | None = None
) -> tuple[list[str], list[str]]:
    """Уникальные ``(content_hash, embed_input(breadcrumb, text))`` чанков корпуса —
    эмбеддер видит breadcrumb-контекст (spec analyze-retrieval §3.1), не голый text.
    ``not_embedded_for=<model>`` — только хэши, ещё НЕ заэмбедженные этой моделью
    (инкрементальный отбор: правка одного документа эмбеддит лишь его новые хэши,
    дубли boilerplate — один раз); ``None`` — все (ab_eval: полная матрица на модель).
    ``content_hash`` NOT NULL, так что ``NOT IN`` без NULL-ловушки."""
    if not_embedded_for is None:
        rows = conn.execute(
            "SELECT DISTINCT content_hash, breadcrumb, text FROM chunks ORDER BY content_hash"
        ).fetchall()
    else:
        ensure_schema(conn)  # первый embed: таблицы vectors ещё нет — подзапрос иначе падает
        rows = conn.execute(
            "SELECT DISTINCT content_hash, breadcrumb, text FROM chunks "
            "WHERE content_hash NOT IN (SELECT content_hash FROM vectors WHERE model = ?) "
            "ORDER BY content_hash",
            (not_embedded_for,),
        ).fetchall()
    hashes = [str(r[0]) for r in rows]
    texts = [embed_input(str(r[1]), str(r[2])) for r in rows]
    return hashes, texts


def gc_vectors(conn: sqlite3.Connection, model: str) -> int:
    """Удалить осиротевшие векторы модели — те, чей ``content_hash`` больше не
    встречается ни в одном чанке (документ изменён/удалён). Возвращает число
    удалённых; держит ``vectors`` в согласии с текущим поколением чанков."""
    ensure_schema(conn)
    cur = conn.execute(
        "DELETE FROM vectors WHERE model = ? "
        "AND content_hash NOT IN (SELECT DISTINCT content_hash FROM chunks)",
        (model,),
    )
    conn.commit()
    return int(cur.rowcount)


def unembedded_count(conn: sqlite3.Connection, model: str) -> int:
    """Сколько уникальных хэшей чанков ещё без вектора данной модели (missing-репорт
    vsearch)."""
    ensure_schema(conn)
    row = conn.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT content_hash FROM chunks "
        "WHERE content_hash NOT IN (SELECT content_hash FROM vectors WHERE model = ?))",
        (model,),
    ).fetchone()
    return int(row[0]) if row else 0


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
    total_row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    if not total_row or not total_row[0]:
        print("нет чанков в БД", file=sys.stderr)
        conn.close()
        return 2
    try:
        check_chunk_budget(conn)  # до дорогого embedder.embed() — не тратить минуты ONNX впустую
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        conn.close()
        return 2
    embedder = _make_embedder(args.backend, args.model)
    hashes, texts = chunk_hashes(conn, not_embedded_for=embedder.name)  # только НОВЫЕ хэши
    if hashes:
        store_vectors(conn, hashes, embedder.embed(texts), embedder.name)
    removed = gc_vectors(conn, embedder.name)
    conn.close()
    print(f"Векторы {embedder.name}: +{len(hashes)} новых, GC удалил {removed} -> {args.db}")
    return 0


def _cmd_vsearch(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"нет БД {args.db}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    embedder = _make_embedder(args.backend, args.model)
    query_vec = embedder.embed([args.query])
    hits = semantic_search(conn, query_vec[0], embedder.name, args.limit)
    has_vectors = (
        conn.execute("SELECT 1 FROM vectors WHERE model = ? LIMIT 1", (embedder.name,)).fetchone()
        is not None
    )
    missing = unembedded_count(conn, embedder.name)
    conn.close()
    if not has_vectors:  # ни одного вектора модели — искать нечем (честный отказ)
        print(
            f"векторы модели {embedder.name} отсутствуют — прогоните "
            f"vector_store.py embed-corpus (или run_pipeline --embed)",
            file=sys.stderr,
        )
        return 2
    if missing:  # часть чанков ещё не заэмбеддена — выдача неполна
        print(
            f"⚠ {missing} чанков ещё без векторов {embedder.name} — результат неполон, "
            f"прогоните embed-corpus",
            file=sys.stderr,
        )
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
