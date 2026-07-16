"""Единый SQLite-индекс корпуса: канонические чанки + полнотекстовый поиск FTS5.

Схема (одна БД, векторный слой в vector_store.py):
  chunks(chunk_id, doc_id, chunk_index, text, n_tokens, content_hash)
    content_hash = sha256(text) — стабильный «адрес» СОДЕРЖИМОГО чанка: к нему
    привязан вектор (vector_store), поэтому пере-чанковка не осиротит эмбеддинги
    неизменившихся чанков, а старый вектор физически не может указать на чужой текст.
  chunks_fts — внешне-контентная FTS5 над chunks.text, tokenize=unicode61 (многоязычно).
  doc_state(doc_id, fingerprint) — per-doc отпечаток проиндексированного doc.md;
    инкрементальная переиндексация трогает только изменившиеся документы.
  index_meta(key, value) — ключ-значение: corpus_fingerprint/chunk_max_tokens/
    schema_version (пишет этот модуль).

CLI: собрать индекс из ``doc.md`` корпуса (записи — обход ``sources/**/meta.yaml``,
пути выводятся из папок-документов) и/или искать.
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from bge_tokenizer import EMBED_MAX_TOKENS, token_counter
from chunking import Chunk, TokenCounter, chunk_text, strip_frontmatter
from env import REPO_ROOT
from schema import DEFAULT_SOURCES, load_records, md_file

DEFAULT_DB = REPO_ROOT / "pipeline" / "index" / "corpus.db"

# Версия схемы производного слоя. Инкремент = несовместимая форма таблиц; открытие
# старой БД (create_db) пересоздаёт производные таблицы с нуля (артефакт производный,
# цена нулевая на текущем корпусе). v1 = vectors на chunk_id, chunks без content_hash;
# v2 = content_hash в chunks + doc_state + vectors на content_hash (spec index-incremental).
SCHEMA_VERSION = "2"

# Производные (пересоздаваемые) таблицы — дропаются при миграции легаси-БД. Порядок
# важен: FTS5-таблица внешнего контента дропается ДО своей content-таблицы chunks.
_DERIVED_TABLES = ("chunks_fts", "chunks", "doc_state", "vectors")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY,
    doc_id       TEXT    NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT    NOT NULL,
    n_tokens     INTEGER NOT NULL,
    content_hash TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc  ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS doc_state (
    doc_id      TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_META_SCHEMA = "CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"


def content_hash(text: str) -> str:
    """sha256-hex текста чанка (полный, 64 hex): стабильный ключ содержимого для
    векторного слоя. Коллизии на 10–30 тыс. чанков исключены практикой; усечение
    отвергнуто (spec index-incremental) — экономия нулевая, рассуждать о коллизиях
    не хочется вовсе."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Прочитать значение из ``index_meta`` (``None`` — ключ/таблица отсутствуют).

    Создаёт таблицу defensively (как ``vector_store.ensure_schema``) — вызывающая
    сторона может подключиться к БД напрямую (``sqlite3.connect``), минуя ``create_db``.
    """
    conn.execute(_META_SCHEMA)
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row is not None else None


def write_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Записать (upsert) значение в ``index_meta``. Не коммитит — часть вызывающей
    транзакции (см. ``index_chunks``: fingerprint пишется атомарно с чанками)."""
    conn.execute(_META_SCHEMA)
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _doc_fingerprint(md: Path) -> str:
    """Per-doc отпечаток ``doc.md`` (только ``stat``): ``"<size>:<mtime_ns>"``.
    Единица инкрементальной переиндексации — хранится в ``doc_state``."""
    st = md.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _fingerprint_from_parts(parts: list[str]) -> str:
    """sha256 отсортированных строк — общий хэш для ``corpus_fingerprint`` и его
    инкрементального пересчёта (``index_corpus_incremental``): формат обязан
    совпадать, поэтому считается ОДНОЙ функцией."""
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


def corpus_fingerprint(sources_root: Path) -> str:
    """Дешёвый (только ``stat``, без чтения содержимого) отпечаток состояния корпуса:
    sha256 отсортированных ``"<id>:<size>:<mtime_ns>"`` по всем СУЩЕСТВУЮЩИМ ``doc.md``.

    Документы без ``doc.md`` не входят — их появление (после конвертации) меняет
    отпечаток и триггерит пересборку индекса. На ~200 документах — ~200 ``stat()``,
    миллисекунды. Остаётся быстрым глобальным гейтом «есть ли работа вообще»
    (реконсиляция run_pipeline); тонкую per-doc инкрементальность даёт ``doc_state``.
    """
    parts = [
        f"{rec.id}:{_doc_fingerprint(md)}"
        for rec in load_records(sources_root)
        if (md := md_file(rec, sources_root)).exists()
    ]
    return _fingerprint_from_parts(parts)


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


def _is_legacy_schema(conn: sqlite3.Connection) -> bool:
    """БД требует миграции (пересоздания производных таблиц)? Легаси, если таблица
    ``chunks`` существует, но без колонки ``content_hash`` ИЛИ ``schema_version`` в
    ``index_meta`` не совпадает с текущей. Свежая БД (нет ``chunks``) — не легаси."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    if not cols:
        return False
    if "content_hash" not in cols:
        return True
    return read_meta(conn, "schema_version") != SCHEMA_VERSION


def _reset_derived_tables(conn: sqlite3.Connection) -> None:
    """Снести производные таблицы старого поколения (миграция). Артефакт производный —
    следующий rebuild соберёт заново; отпечатки/бюджет прошлого поколения невалидны."""
    conn.execute("DROP TABLE IF EXISTS chunks_fts")  # внешний контент — до content-таблицы
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS doc_state")
    conn.execute("DROP TABLE IF EXISTS vectors")
    conn.execute(_META_SCHEMA)
    conn.execute("DELETE FROM index_meta")
    conn.commit()


def create_db(db_path: Path) -> sqlite3.Connection:
    """Открыть/создать БД со схемой; мигрировать легаси-форму (пересоздать производные
    таблицы). ``schema_version`` штампуется в ``index_meta`` — детект будущих миграций."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    if _is_legacy_schema(conn):
        _reset_derived_tables(conn)
    conn.executescript(_SCHEMA)
    write_meta(conn, "schema_version", SCHEMA_VERSION)
    conn.commit()
    return conn


def index_chunks(
    conn: sqlite3.Connection,
    chunks: list[Chunk],
    *,
    corpus_fingerprint: str | None = None,
    chunk_max_tokens: int | None = None,
) -> None:
    """Полная переиндексация: заменить содержимое и перестроить FTS (идемпотентно).

    Векторы (``vector_store``) НЕ трогаются: они ключуются ``content_hash``, а не
    ``chunk_id``, поэтому переживают пересборку — неизменившийся текст даёт тот же
    хэш и тот же валидный вектор; осиротевшие (текст исчез) подчистит ``gc_vectors``
    при следующем ``embed-corpus``. Стоп-гэп «DROP vectors при любой пересборке»
    (index-consistency §3) упразднён — вектор физически не может указать на чужой
    текст (spec index-incremental §3).

    ``corpus_fingerprint``/``chunk_max_tokens``, если переданы, пишутся в
    ``index_meta`` АТОМАРНО с чанками — в одном ``conn.commit()``. На этом
    полагается реконсиляция пересборки в ``run_pipeline``: крах между шагами
    оставляет старый (или отсутствующий) отпечаток, следующий прогон честно
    пересоберёт — самовосстановление по построению, без отдельного флага/статуса.
    ``chunk_max_tokens`` — бюджет, с которым собраны чанки; ``vector_store``
    сверяет его с лимитом эмбеддера перед ``embed-corpus`` (см. spec
    index-consistency §6: инвариант «чанк целиком видим обоим поискам»).
    """
    conn.execute(_META_SCHEMA)  # defensive — index_meta может не существовать без create_db
    conn.execute("DELETE FROM chunks")
    conn.executemany(
        "INSERT INTO chunks (doc_id, chunk_index, text, n_tokens, content_hash) "
        "VALUES (?, ?, ?, ?, ?)",
        [(c.doc_id, c.index, c.text, c.n_tokens, content_hash(c.text)) for c in chunks],
    )
    conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")
    if corpus_fingerprint is not None:
        write_meta(conn, "corpus_fingerprint", corpus_fingerprint)
    if chunk_max_tokens is not None:
        write_meta(conn, "chunk_max_tokens", str(chunk_max_tokens))
    conn.commit()


def _delete_doc_chunks(conn: sqlite3.Connection, doc_id: str) -> None:
    """Удалить чанки документа из ``chunks`` И из внешне-контентного FTS-индекса.

    FTS5 external content не самосинхронизируется (триггеров нет): удаление строки —
    ТОЛЬКО через ``INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', …)``
    с ТЕМ ЖЕ текстом, что был проиндексирован, иначе в индексе остаются висячие
    постинги (верифицировано по sqlite.org/fts5). Текст берём прямо из ``chunks`` — он
    побайтово совпадает со вставленным (``_insert_doc_chunks`` пишет FTS тем же
    ``c.text``). Порядок принципиален: FTS-delete КАЖДОЙ строки ДО ``DELETE FROM
    chunks`` — после удаления строки её текст для delete уже не прочитать.
    """
    old = conn.execute("SELECT chunk_id, text FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    for chunk_id, text in old:
        conn.execute(
            "INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', ?, ?)",
            (chunk_id, text),
        )
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))


def _insert_doc_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Вставить чанки документа в ``chunks`` + вручную синхронизировать FTS.
    ``chunk_id`` присваивается rowid'ом при INSERT — берём ``lastrowid`` и пишем в FTS
    ТОТ ЖЕ текст (инвариант для последующего ``_delete_doc_chunks``)."""
    for c in chunks:
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, n_tokens, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (c.doc_id, c.index, c.text, c.n_tokens, content_hash(c.text)),
        )
        conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, c.text))


def _rebuild_doc_state(conn: sqlite3.Connection, sources_root: Path) -> None:
    """Переписать ``doc_state`` под текущий корпус (после полного rebuild): иначе
    следующий инкремент счёл бы все документы «изменившимися» (пустой doc_state) и
    пере-чанковал бы весь корпус повторно."""
    conn.execute("DELETE FROM doc_state")
    conn.executemany(
        "INSERT INTO doc_state (doc_id, fingerprint) VALUES (?, ?)",
        [
            (rec.id, _doc_fingerprint(md))
            for rec in load_records(sources_root)
            if (md := md_file(rec, sources_root)).exists()
        ],
    )
    conn.commit()


def index_corpus_incremental(
    conn: sqlite3.Connection,
    sources_root: Path,
    count_tokens: TokenCounter,
    max_tokens: int,
) -> tuple[int, int]:
    """Инкрементальная переиндексация: пере-чанкуются (дорогая bge-токенизация) и
    пере-индексируются в FTS ТОЛЬКО документы, чей ``doc.md`` изменился (fingerprint
    разошёлся с ``doc_state``) или исчез из корпуса. Возвращает ``(изменено, удалено)``.

    Одна транзакция на прогон (``with conn``): краш откатывает к консистентному
    прошлому поколению — ``doc_state`` не обгоняет ``chunks``. DDL/миграция здесь
    ОТСУТСТВУЮТ (они в ``create_db``): питоновский ``sqlite3`` неявно коммитит на DDL
    и порвал бы атомарность. ``corpus_fingerprint``/``chunk_max_tokens`` пишутся той
    же транзакцией — глобальный гейт реконсиляции остаётся согласован с чанками.
    """
    current: dict[str, tuple[str, Path]] = {}
    for rec in load_records(sources_root):
        md = md_file(rec, sources_root)
        if md.exists():
            current[rec.id] = (_doc_fingerprint(md), md)
    stored = {
        str(r[0]): str(r[1])
        for r in conn.execute("SELECT doc_id, fingerprint FROM doc_state").fetchall()
    }

    changed = [doc_id for doc_id, (fp, _) in current.items() if stored.get(doc_id) != fp]
    vanished = [doc_id for doc_id in stored if doc_id not in current]
    corpus_fp = _fingerprint_from_parts([f"{d}:{fp}" for d, (fp, _) in current.items()])

    with conn:  # атомарно: либо всё новое поколение, либо ничего
        for doc_id in (*changed, *vanished):
            _delete_doc_chunks(conn, doc_id)
        for doc_id in changed:
            fp, md = current[doc_id]
            text = strip_frontmatter(md.read_text(encoding="utf-8"))
            _insert_doc_chunks(conn, chunk_text(text, count_tokens, max_tokens, doc_id=doc_id))
            conn.execute(
                "INSERT INTO doc_state (doc_id, fingerprint) VALUES (?, ?) "
                "ON CONFLICT(doc_id) DO UPDATE SET fingerprint = excluded.fingerprint",
                (doc_id, fp),
            )
        for doc_id in vanished:
            conn.execute("DELETE FROM doc_state WHERE doc_id = ?", (doc_id,))
        write_meta(conn, "corpus_fingerprint", corpus_fp)
        write_meta(conn, "chunk_max_tokens", str(max_tokens))
    return len(changed), len(vanished)


def index_corpus(
    conn: sqlite3.Connection,
    sources_root: Path,
    count_tokens: TokenCounter,
    max_tokens: int,
    *,
    force: bool = False,
) -> str:
    """Единая точка индексации. Полный rebuild (эталонный ``index_chunks`` +
    ``_rebuild_doc_state``) при ``force`` или смене ``chunk_max_tokens`` (границы
    чанков изменились → пере-чанковать весь корпус); иначе — инкремент. Свежая/только
    что мигрированная БД (пустой ``doc_state``) строится инкрементальным путём: все
    документы «изменились», FTS чист — висячих постингов нет. Возвращает статус."""
    stored_max = read_meta(conn, "chunk_max_tokens")
    full = force or (stored_max is not None and int(stored_max) != max_tokens)
    if full:
        chunks = chunks_from_corpus(sources_root, count_tokens, max_tokens)
        index_chunks(
            conn, chunks,
            corpus_fingerprint=corpus_fingerprint(sources_root),
            chunk_max_tokens=max_tokens,
        )
        _rebuild_doc_state(conn, sources_root)
        return f"полная пересборка: {len(chunks)} чанков"
    changed, vanished = index_corpus_incremental(conn, sources_root, count_tokens, max_tokens)
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    total = int(row[0]) if row else 0
    return f"инкремент: изменено {changed}, удалено {vanished}; всего {total} чанков"


def sanitize_fts_query(q: str) -> str:
    """Безопасно превратить произвольную пользовательскую строку в FTS5 MATCH-запрос.

    По грамматике FTS5 bareword — буквы/цифры/подчёркивание/не-ASCII; всё прочее
    (``-``, ``:``, ``(``, ``"``, …) — синтаксис. Дефис в barewords не входит: без
    экранирования запрос вида ``state-as-mcp`` — синтаксическая ошибка, не фраза
    (верифицировано по sqlite.org/fts5.html). Каждый токен оборачивается в
    двойные кавычки (внутренние — удваиваются, SQL-style), implicit-AND
    многословного запроса сохраняется.
    """
    tokens = q.split()
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens) or '""'


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[SearchHit]:
    """Полнотекстовый поиск (FTS5 MATCH), ранжирование bm25 (меньше = лучше).

    ``query`` — уже готовая MATCH-строка (санитизация — на границе пользовательского
    ввода, см. ``sanitize_fts_query``/CLI ``--raw``; API-функция принимает и честный
    FTS5-синтаксис — NEAR/колонки понадобятся будущему analyze-слою).
    """
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
    max_tokens: int = EMBED_MAX_TOKENS,
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
    conn = create_db(args.db)
    status = index_corpus(conn, args.sources, token_counter(), args.max_tokens, force=args.force)
    conn.close()
    print(f"{status} -> {args.db}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"индекс не найден: {args.db} (сначала build)", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    query = args.query if args.raw else sanitize_fts_query(args.query)
    try:
        hits = fts_search(conn, query, args.limit)
    except sqlite3.OperationalError as exc:
        print(f"некорректный FTS5-запрос: {exc}", file=sys.stderr)
        conn.close()
        return 2
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

    p_build = sub.add_parser("build", help="построить/обновить индекс из .md корпуса (инкрементально)")
    p_build.add_argument("sources", nargs="?", type=Path, default=DEFAULT_SOURCES)
    p_build.add_argument("--max-tokens", type=int, default=EMBED_MAX_TOKENS)
    p_build.add_argument(
        "--force", action="store_true",
        help="полная пересборка вместо инкремента по изменённым doc.md",
    )
    p_build.set_defaults(func=_cmd_build)

    p_search = sub.add_parser("search", help="полнотекстовый поиск")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument(
        "--raw", action="store_true",
        help="не экранировать запрос — честный FTS5-синтаксис (NEAR, колонки, ...)",
    )
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
