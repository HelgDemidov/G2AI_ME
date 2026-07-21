"""Тесты векторного хранилища и брутфорс-косинуса (синтетические векторы, без модели).

Ключ вектора — content_hash чанка (spec index-incremental §3): пере-чанковка не
осиротит неизменившиеся векторы, вектор не может указать на чужой текст; эмбеддинг
инкрементален (только новые хэши), осиротевшие чистит gc_vectors."""
from __future__ import annotations

import argparse
import sqlite3
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.schema import SourceRecord
from index.bge_tokenizer import EMBED_MAX_TOKENS
from index.chunking import Chunk
from index.corpus_index import create_db, fts5_available, index_chunks
from index.embed import FloatArray, l2_normalize
from index.vector_store import (
    _cmd_embed,
    check_chunk_budget,
    chunk_hashes,
    confidential_doc_ids,
    embed_and_store,
    gc_vectors,
    load_vectors,
    semantic_search,
    store_vectors,
    unembedded_count,
)
from tests.support import valid_record

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")


def _setup(tmp_path: Path) -> sqlite3.Connection:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-a", 0, "first", 1),
            Chunk("doc-a", 1, "second", 1),
            Chunk("doc-b", 0, "third", 1),
        ],
    )
    return conn


def _vec_count(conn: sqlite3.Connection, model: str) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM vectors WHERE model=?", (model,)).fetchone()[0])


# --- store / load: ключ content_hash ---


def test_store_and_load(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    got, mat = load_vectors(conn, "m")
    assert set(got) == set(hashes)
    assert mat.shape == (3, 3)


def test_reembed_upserts_not_duplicates(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    vecs = l2_normalize(np.ones((3, 3), dtype=np.float32))
    store_vectors(conn, hashes, vecs, "m")
    store_vectors(conn, hashes, vecs, "m")  # повторно — upsert по (content_hash, model)
    assert _vec_count(conn, "m") == 3


# --- load_vectors: потоковая сборка матрицы (бэклог §16) ---


def _load_peak_ratio(conn: sqlite3.Connection, model: str) -> tuple[float, FloatArray]:
    """Пик аллокаций внутри load_vectors, нормированный на размер итоговой матрицы."""
    tracemalloc.start()
    try:
        _, mat = load_vectors(conn, model)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / mat.nbytes, mat


def test_load_vectors_does_not_hold_extra_full_copies(tmp_path: Path) -> None:
    """Регрессия: прежняя форма (fetchall + vstack + избыточный astype) держала в пике
    ТРИ копии данных — замерено 3.07x. Потоковая сборка даёт ~1x. Порог 2.0x ловит
    возврат любой одной лишней полной копии, оставляя запас на служебные аллокации."""
    conn = create_db(tmp_path / "c.db")
    n, dim = 2000, 512
    store_vectors(
        conn,
        [f"h{i:05d}" for i in range(n)],
        l2_normalize(np.random.default_rng(0).random((n, dim), dtype=np.float32)),
        "m",
    )
    ratio, mat = _load_peak_ratio(conn, "m")
    assert mat.nbytes == n * dim * 4  # матрица действительно крупная — порог осмыслен
    assert ratio < 2.0, f"пик {ratio:.2f}x от матрицы — вернулась лишняя полная копия"


def test_load_vectors_roundtrip_preserves_values_dtype_and_hash_order(tmp_path: Path) -> None:
    """Потоковая запись в предвыделенный буфер не меняет контракт: float32, writable,
    строка i соответствует hashes[i], порядок — по content_hash (ORDER BY)."""
    conn = create_db(tmp_path / "c.db")
    vecs = l2_normalize(np.eye(4, dtype=np.float32))
    stored = ["h-c", "h-a", "h-d", "h-b"]  # намеренно НЕ отсортированы при записи
    store_vectors(conn, stored, vecs, "m")
    hashes, mat = load_vectors(conn, "m")

    assert hashes == sorted(stored)
    assert mat.dtype == np.dtype(np.float32)
    assert mat.flags.writeable
    assert mat.shape == (4, 4)
    for i, h in enumerate(hashes):
        assert np.array_equal(mat[i], vecs[stored.index(h)])  # строка i <-> hashes[i]


def test_load_vectors_single_row_allocates_from_first_row_width(tmp_path: Path) -> None:
    """Ширина буфера берётся из ПЕРВОЙ строки — вырожденный случай n=1 не особый."""
    conn = create_db(tmp_path / "c.db")
    store_vectors(conn, ["only"], l2_normalize(np.ones((1, 7), dtype=np.float32)), "m")
    hashes, mat = load_vectors(conn, "m")
    assert hashes == ["only"] and mat.shape == (1, 7)


def test_load_vectors_absent_model_returns_empty(tmp_path: Path) -> None:
    """Модели нет -> COUNT=0 -> ранний выход без обращения к курсору."""
    conn = create_db(tmp_path / "c.db")
    store_vectors(conn, ["h1"], l2_normalize(np.ones((1, 3), dtype=np.float32)), "m")
    hashes, mat = load_vectors(conn, "other-model")
    assert hashes == [] and mat.shape == (0, 0)
    assert mat.dtype == np.dtype(np.float32)


def test_load_vectors_raises_on_mixed_dimensions(tmp_path: Path) -> None:
    """Документированное поведение: вектор чужой размерности роняет загрузку, а не
    просачивается молча (порядок по content_hash -> 'a' задаёт ширину, 'b' её ломает)."""
    conn = create_db(tmp_path / "c.db")
    store_vectors(conn, ["a"], l2_normalize(np.ones((1, 4), dtype=np.float32)), "m")
    store_vectors(conn, ["b"], l2_normalize(np.ones((1, 8), dtype=np.float32)), "m")
    with pytest.raises(ValueError):
        load_vectors(conn, "m")


def test_semantic_search_nearest(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, texts = chunk_hashes(conn)
    vecs = l2_normalize(np.eye(len(hashes), dtype=np.float32))
    store_vectors(conn, hashes, vecs, "m")
    ti = texts.index("second")  # целимся в конкретный текст, не в позицию хэша
    hits = semantic_search(conn, vecs[ti], "m", top_k=len(hashes))
    assert hits[0].text == "second"
    assert hits[0].doc_id == "doc-a"
    assert hits[0].chunk_index == 1
    assert hits[0].score >= hits[-1].score


def test_semantic_search_empty_when_no_vectors(tmp_path: Path) -> None:
    """Ни одного вектора модели → пустая выдача (не исключение): устаревших/неверных
    результатов не бывает по построению, отсутствие репортит CLI."""
    conn = _setup(tmp_path)
    assert semantic_search(conn, np.zeros(3, dtype=np.float32), "m", 5) == []


# --- embed_input: эмбеддер видит breadcrumb-контекст (spec analyze-retrieval §3.1) ---


def test_chunk_hashes_applies_embed_input_with_breadcrumb(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "body text", 2, "H1 › H2")])
    _, texts = chunk_hashes(conn)
    assert texts == ["H1 › H2\nbody text"]


def test_chunk_hashes_no_breadcrumb_returns_plain_text(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    _, texts = chunk_hashes(conn)
    assert set(texts) == {"first", "second", "third"}


# --- semantic_search: allowed_doc_ids фасетный фильтр (spec analyze-retrieval §3.2) ---


def test_semantic_search_allowed_doc_ids_drops_excluded_carriers(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-a", 0, "boilerplate", 1),
            Chunk("doc-b", 0, "boilerplate", 1),
        ],
    )
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(1, dtype=np.float32)), "m")
    hits = semantic_search(
        conn, l2_normalize(np.eye(1, dtype=np.float32))[0], "m", top_k=10, allowed_doc_ids={"doc-a"}
    )
    assert {h.doc_id for h in hits} == {"doc-a"}


def test_semantic_search_backfills_top_k_past_filtered_hash(tmp_path: Path) -> None:
    """Хэш без носителей внутри фильтра НЕ занимает место бюджета top_k — следующий
    по score хэш добирает недостающий слот (spec analyze-retrieval §3.2)."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-excluded", 0, "best match", 2),
            Chunk("doc-allowed", 0, "second best", 2),
        ],
    )
    hashes, texts = chunk_hashes(conn)
    # вектор "best match" ближе к запросу, чем "second best" — обычный (0), (1) порядок eye
    vecs = l2_normalize(np.eye(len(hashes), dtype=np.float32))
    store_vectors(conn, hashes, vecs, "m")
    query = vecs[texts.index("best match")]
    hits = semantic_search(conn, query, "m", top_k=1, allowed_doc_ids={"doc-allowed"})
    assert len(hits) == 1
    assert hits[0].doc_id == "doc-allowed"


def test_semantic_search_none_allowed_doc_ids_matches_prior_behavior(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, texts = chunk_hashes(conn)
    vecs = l2_normalize(np.eye(len(hashes), dtype=np.float32))
    store_vectors(conn, hashes, vecs, "m")
    ti = texts.index("second")
    hits_unfiltered = semantic_search(conn, vecs[ti], "m", top_k=len(hashes))
    hits_none = semantic_search(conn, vecs[ti], "m", top_k=len(hashes), allowed_doc_ids=None)
    assert hits_unfiltered == hits_none


def test_vec_hit_carries_breadcrumb(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "body", 1, "Chapter One")])
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(1, dtype=np.float32)), "m")
    hits = semantic_search(conn, l2_normalize(np.eye(1, dtype=np.float32))[0], "m", top_k=1)
    assert hits[0].breadcrumb == "Chapter One"


# --- boilerplate: общий хэш двух документов = один вектор, оба чанка находимы ---


def test_shared_hash_one_vector_both_chunks_found(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-a", 0, "boilerplate", 1),  # тот же текст…
            Chunk("doc-b", 0, "boilerplate", 1),  # …в другом документе
            Chunk("doc-b", 1, "unique", 1),
        ],
    )
    hashes, texts = chunk_hashes(conn)
    assert len(hashes) == 2  # дубль схлопнут: boilerplate + unique
    store_vectors(conn, hashes, l2_normalize(np.eye(2, dtype=np.float32)), "m")
    assert _vec_count(conn, "m") == 2  # один вектор на общий хэш, не два

    bi = texts.index("boilerplate")
    hits = semantic_search(conn, l2_normalize(np.eye(2, dtype=np.float32))[bi], "m", top_k=2)
    boiler = {(h.doc_id, h.chunk_index) for h in hits if h.text == "boilerplate"}
    assert boiler == {("doc-a", 0), ("doc-b", 0)}  # score роздан обоим носителям хэша


# --- инкрементальный отбор: только новые хэши ---


def test_chunk_hashes_pending_excludes_embedded(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    all_hashes, _ = chunk_hashes(conn)
    assert len(all_hashes) == 3
    store_vectors(conn, all_hashes[:2], l2_normalize(np.eye(2, dtype=np.float32)), "m")

    pending, _ = chunk_hashes(conn, not_embedded_for="m")
    assert set(pending) == {all_hashes[2]}  # только ещё не заэмбедженный
    other, _ = chunk_hashes(conn, not_embedded_for="other")
    assert set(other) == set(all_hashes)  # другая модель — всё заново


def test_chunk_hashes_pending_when_vectors_table_absent(tmp_path: Path) -> None:
    """Первый embed: таблицы vectors ещё нет — pending-отбор не должен падать
    (регресс real-run: подзапрос по vectors ронял `no such table`)."""
    conn = _setup(tmp_path)  # чанки есть, ни одного store_vectors → таблицы vectors нет
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'"
    ).fetchone() is None
    pending, _ = chunk_hashes(conn, not_embedded_for="m")
    all_hashes, _ = chunk_hashes(conn)
    assert set(pending) == set(all_hashes)  # ничего не заэмбеддено — все pending


def test_unembedded_count(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    assert unembedded_count(conn, "m") == 3
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes[:1], l2_normalize(np.eye(1, dtype=np.float32)), "m")
    assert unembedded_count(conn, "m") == 2


# --- GC осиротевших векторов ---


def test_gc_removes_orphaned_vectors(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    orphan = hashes[0]
    conn.execute("DELETE FROM chunks WHERE content_hash=?", (orphan,))  # текст исчез из корпуса
    conn.commit()

    removed = gc_vectors(conn, "m")
    assert removed == 1
    remaining, _ = load_vectors(conn, "m")
    assert orphan not in remaining
    assert len(remaining) == 2


def test_gc_keeps_vectors_still_referenced(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(3, dtype=np.float32)), "m")
    assert gc_vectors(conn, "m") == 0  # все хэши ещё в chunks
    assert _vec_count(conn, "m") == 3


# --- sensitivity-гейт: confidential-документы не эмбеддятся облаком (spec embed-api-first §3) ---


def _record(id_: str, *, sensitivity: str = "normal") -> SourceRecord:
    rec_dict = valid_record()
    rec_dict["id"] = id_
    rec_dict["sensitivity"] = sensitivity
    return SourceRecord.model_validate(rec_dict)


def test_confidential_doc_ids_reads_from_doc_facets(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [Chunk("doc-conf", 0, "x", 1), Chunk("doc-pub", 0, "y", 1)],
        records=[_record("doc-conf", sensitivity="confidential"), _record("doc-pub")],
    )
    assert confidential_doc_ids(conn) == {"doc-conf"}


def test_confidential_doc_ids_empty_on_legacy_db_without_column(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "legacy.db")
    conn.execute("CREATE TABLE doc_facets (doc_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO doc_facets (doc_id) VALUES ('x')")
    conn.commit()
    assert confidential_doc_ids(conn) == set()


def test_chunk_hashes_exclude_all_carriers_semantics(tmp_path: Path) -> None:
    """spec embed-api-first §3.2: hash A (носители confidential+public) допущен, hash B
    (только confidential) исключён, hash C (только public) допущен."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(
        conn,
        [
            Chunk("doc-conf", 0, "shared boilerplate", 2),      # hash A: носитель confidential
            Chunk("doc-pub", 0, "shared boilerplate", 2),       # hash A: носитель public (тот же текст)
            Chunk("doc-conf", 1, "confidential only text", 3),  # hash B: только confidential
            Chunk("doc-pub", 1, "public only text", 3),         # hash C: только public
        ],
        records=[_record("doc-conf", sensitivity="confidential"), _record("doc-pub")],
    )
    confidential = confidential_doc_ids(conn)
    assert confidential == {"doc-conf"}

    hashes, texts = chunk_hashes(conn, exclude_all_carriers_in=confidential)
    assert set(texts) == {"shared boilerplate", "public only text"}  # hash B исключён


def test_chunk_hashes_exclude_all_carriers_empty_set_is_noop(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    all_hashes, _ = chunk_hashes(conn)
    filtered, _ = chunk_hashes(conn, exclude_all_carriers_in=set())
    assert set(filtered) == set(all_hashes)


class _CloudFakeEmbedder:
    name = "cloud-model"
    dim = 1
    max_tokens: int | None = None

    def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
        return l2_normalize(np.ones((len(texts), self.dim), dtype=np.float32))


def test_cmd_embed_openrouter_skips_confidential_only_chunks_and_reports(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(
        conn,
        [Chunk("doc-conf", 0, "confidential only text", 3), Chunk("doc-pub", 0, "public text", 2)],
        records=[_record("doc-conf", sensitivity="confidential"), _record("doc-pub")],
    )
    conn.close()

    monkeypatch.setattr("index.vector_store._make_embedder", lambda backend, model: _CloudFakeEmbedder())

    args = argparse.Namespace(db=db, backend="openrouter", model=None)
    assert _cmd_embed(args) == 0
    out = capsys.readouterr().out
    assert "1 чанков только-confidential" in out
    assert "--backend bge" in out

    conn2 = sqlite3.connect(db)
    assert _vec_count(conn2, "cloud-model") == 1  # только публичный чанк заэмбеджен


def test_cli_default_backend_is_openrouter(tmp_path: Path, monkeypatch: Any) -> None:
    """API-first (spec embed-api-first §4): дефолт --backend CLI = openrouter."""
    from index.vector_store import main as vs_main

    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)])
    conn.close()
    captured: dict[str, Any] = {}

    def fake_make_embedder(backend: str, model: Any) -> Any:
        captured["backend"] = backend
        return _CloudFakeEmbedder()

    monkeypatch.setattr("index.vector_store._make_embedder", fake_make_embedder)
    assert vs_main(["embed-corpus", "--db", str(db)]) == 0
    assert captured["backend"] == "openrouter"


def test_cmd_embed_bge_backend_ignores_sensitivity_gate(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Локальный бэкенд эмбеддит confidential-чанки как обычно — гейт применим только
    к облачному пути (данные и так не покидают машину)."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(
        conn,
        [Chunk("doc-conf", 0, "confidential only text", 3)],
        records=[_record("doc-conf", sensitivity="confidential")],
    )
    conn.close()

    fake = _CloudFakeEmbedder()
    fake.name = "local-model"
    monkeypatch.setattr("index.vector_store._make_embedder", lambda backend, model: fake)

    args = argparse.Namespace(db=db, backend="bge", model=None)
    assert _cmd_embed(args) == 0
    out = capsys.readouterr().out
    assert "только-confidential" not in out

    conn2 = sqlite3.connect(db)
    assert _vec_count(conn2, "local-model") == 1


# --- check_chunk_budget: инвариант «чанк целиком видим обоим поискам» (index-consistency §6,
# сигнатура — embedder_max явным параметром вместо константы, spec embed-local-swap §4) ---


def test_check_chunk_budget_passes_when_within_limit(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS)
    check_chunk_budget(conn, EMBED_MAX_TOKENS)  # не должно бросать


def test_check_chunk_budget_passes_when_absent(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)])  # без chunk_max_tokens
    check_chunk_budget(conn, EMBED_MAX_TOKENS)  # неизвестность — не повод отказывать


def test_check_chunk_budget_raises_when_exceeds_limit(tmp_path: Path) -> None:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS * 2)
    with pytest.raises(ValueError, match=str(EMBED_MAX_TOKENS)):
        check_chunk_budget(conn, EMBED_MAX_TOKENS)


def test_check_chunk_budget_none_max_never_raises(tmp_path: Path) -> None:
    """embedder_max=None — гейт неприменим (облачный эмбеддер без фиксированного
    лимита) — не бросает ДАЖЕ при огромном chunk_max_tokens (spec embed-local-swap §4)."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS * 100)
    check_chunk_budget(conn, None)  # не должно бросать


def test_store_vectors_does_not_gate_chunk_budget(tmp_path: Path) -> None:
    """store_vectors — тупой писатель: НЕ вызывает check_chunk_budget (ответственность
    вызывающего, он знает embedder.max_tokens — spec embed-local-swap §4). Раньше
    store_vectors сам бросал ValueError на превышении бюджета — это поведение снято."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS * 2)
    store_vectors(conn, ["deadbeef"], l2_normalize(np.eye(1, dtype=np.float32)), "m")  # не бросает


def test_cmd_embed_reports_budget_error_without_calling_embed(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Гейт срабатывает ПОСЛЕ создания эмбеддера (нужен embedder.max_tokens), но ДО
    дорогого embedder.embed() — не тратить минуты инференса впустую (spec
    embed-local-swap §4)."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "x", 1)], chunk_max_tokens=EMBED_MAX_TOKENS * 2)
    conn.close()

    class _FakeBudgetEmbedder:
        name = "fake"
        dim = 1
        max_tokens = EMBED_MAX_TOKENS

        def embed(self, texts: list[str], *, kind: str = "doc") -> Any:
            raise AssertionError("embed() не должен вызываться — гейт обязан отсечь раньше")

    monkeypatch.setattr(
        "index.vector_store._make_embedder", lambda backend, model: _FakeBudgetEmbedder()
    )

    args = argparse.Namespace(db=db, backend="bge", model=None)
    assert _cmd_embed(args) == 2
    assert str(EMBED_MAX_TOKENS) in capsys.readouterr().err


def test_l2_normalize_unit_and_zero() -> None:
    mat = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(mat)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # нулевая строка остаётся нулевой


# --- embed_and_store: чекпоинтинг батчами (spec embed-local-swap §5, закрывает бэклог §11) ---


class _CountingEmbedder:
    name = "counting"
    dim = 1
    max_tokens: int | None = None

    def __init__(self) -> None:
        self.calls: list[int] = []  # длины батчей, переданных в embed()

    def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
        self.calls.append(len(texts))
        return l2_normalize(np.ones((len(texts), self.dim), dtype=np.float32))


def _big_corpus(tmp_path: Path, n: int) -> sqlite3.Connection:
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk(f"doc-{i}", 0, f"unique text {i}", 1) for i in range(n)])
    return conn


def test_embed_and_store_batches_calls_and_stores_all(tmp_path: Path) -> None:
    conn = _big_corpus(tmp_path, 150)
    hashes, texts = chunk_hashes(conn)
    embedder = _CountingEmbedder()
    total = embed_and_store(conn, embedder, hashes, texts, batch=64)
    assert total == 150
    assert embedder.calls == [64, 64, 22]  # ровно 3 вызова embed(), не один на весь корпус
    assert _vec_count(conn, "counting") == 150


def test_embed_and_store_empty_hashes_is_noop(tmp_path: Path) -> None:
    conn = _setup(tmp_path)
    embedder = _CountingEmbedder()
    assert embed_and_store(conn, embedder, [], []) == 0
    assert embedder.calls == []


def test_embed_and_store_checkpoints_partial_progress_on_failure(tmp_path: Path) -> None:
    """Фейк падает на 2-м батче — 1-й батч (64 хэша) уже физически в БД: обрыв
    (kill/OOM/закрытие терминала) теряет МАКСИМУМ один батч, не весь прогон."""
    conn = _big_corpus(tmp_path, 150)
    hashes, texts = chunk_hashes(conn)

    class _FailsOnSecondCall:
        name = "flaky"
        dim = 1
        max_tokens: int | None = None

        def __init__(self) -> None:
            self.n_calls = 0

        def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
            self.n_calls += 1
            if self.n_calls == 2:
                raise RuntimeError("обрыв (симуляция kill/OOM)")
            return l2_normalize(np.ones((len(texts), self.dim), dtype=np.float32))

    with pytest.raises(RuntimeError):
        embed_and_store(conn, _FailsOnSecondCall(), hashes, texts, batch=64)
    assert _vec_count(conn, "flaky") == 64  # ровно 1-й батч сохранён, ничего не потеряно


# --- _make_embedder: реальная диспетчеризация backend -> Embedder (не мок) ---


def test_make_embedder_openrouter_dispatches_to_openrouter_embedder(monkeypatch: Any) -> None:
    from index.vector_store import _make_embedder

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    embedder = _make_embedder("openrouter", None)
    assert embedder.name.startswith("google/gemini-embedding-001")


def test_make_embedder_openrouter_passes_model_override(monkeypatch: Any) -> None:
    from index.vector_store import _make_embedder

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    embedder = _make_embedder("openrouter", "custom/model")
    assert embedder.name.startswith("custom/model")


def test_make_embedder_bge_dispatches_to_get_embedder_bge(monkeypatch: Any) -> None:
    """Диспетчеризация в get_embedder("bge") без конструирования реальной модели —
    OnnxBgeEmbedder подменён, тест проверяет ТОЛЬКО маршрутизацию _make_embedder."""
    from index import vector_store as vs_module

    calls: list[str] = []

    def fake_get_embedder(backend: str, **kw: Any) -> Any:
        calls.append(backend)
        return object()

    monkeypatch.setattr(vs_module, "get_embedder", fake_get_embedder)
    vs_module._make_embedder("bge", None)
    assert calls == ["bge"]


# --- _cmd_embed: no-db / no-chunks guards ---


def test_cmd_embed_no_db_returns_error(tmp_path: Path, capsys: Any) -> None:
    args = argparse.Namespace(db=tmp_path / "nope.db", backend="openrouter", model=None)
    assert _cmd_embed(args) == 2
    assert "нет БД" in capsys.readouterr().err


def test_cmd_embed_no_chunks_returns_error(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    create_db(db)  # таблица chunks создана, но пуста
    args = argparse.Namespace(db=db, backend="openrouter", model=None)
    assert _cmd_embed(args) == 2
    assert "нет чанков" in capsys.readouterr().err


# --- _cmd_vsearch: CLI (коды возврата, вывод) ---


def test_cmd_vsearch_no_db_returns_error(tmp_path: Path, capsys: Any) -> None:
    from index.vector_store import _cmd_vsearch

    args = argparse.Namespace(db=tmp_path / "nope.db", backend="openrouter", model=None, query="q", limit=10)
    assert _cmd_vsearch(args) == 2
    assert "нет БД" in capsys.readouterr().err


def test_cmd_vsearch_no_vectors_for_model_returns_error(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    from index.vector_store import _cmd_vsearch

    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "text", 1)])
    conn.close()

    monkeypatch.setattr("index.vector_store._make_embedder", lambda backend, model: _CloudFakeEmbedder())
    args = argparse.Namespace(db=db, backend="openrouter", model=None, query="q", limit=10)
    assert _cmd_vsearch(args) == 2
    assert "векторы модели" in capsys.readouterr().err


def test_cmd_vsearch_warns_when_chunks_unembedded_but_succeeds(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    from index.vector_store import _cmd_vsearch

    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "first", 1), Chunk("doc-b", 0, "second", 1)])
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes[:1], l2_normalize(np.eye(1, dtype=np.float32)), "cloud-model")
    conn.close()

    monkeypatch.setattr("index.vector_store._make_embedder", lambda backend, model: _CloudFakeEmbedder())
    args = argparse.Namespace(db=db, backend="openrouter", model=None, query="q", limit=10)
    assert _cmd_vsearch(args) == 0
    assert "1 чанков ещё без векторов" in capsys.readouterr().err


def test_cmd_vsearch_no_hits_with_orphaned_vectors_reports_nothing_found(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """has_vectors=True (таблица vectors непуста), но соответствующий чанк уже удалён из
    chunks — semantic_search честно не находит носителей, CLI репортит "ничего не найдено"."""
    from index.vector_store import _cmd_vsearch

    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "text", 1)])
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(1, dtype=np.float32)), "cloud-model")
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()

    monkeypatch.setattr("index.vector_store._make_embedder", lambda backend, model: _CloudFakeEmbedder())
    args = argparse.Namespace(db=db, backend="openrouter", model=None, query="q", limit=10)
    assert _cmd_vsearch(args) == 0
    assert "ничего не найдено" in capsys.readouterr().out


def test_cmd_vsearch_prints_hits_with_score_and_preview(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    from index.vector_store import _cmd_vsearch

    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "hello world", 2)])
    hashes, _ = chunk_hashes(conn)
    store_vectors(conn, hashes, l2_normalize(np.eye(1, dtype=np.float32)), "cloud-model")
    conn.close()

    monkeypatch.setattr("index.vector_store._make_embedder", lambda backend, model: _CloudFakeEmbedder())
    args = argparse.Namespace(db=db, backend="openrouter", model=None, query="q", limit=10)
    assert _cmd_vsearch(args) == 0
    out = capsys.readouterr().out
    assert "doc-a" in out
    assert "hello world" in out


def test_embed_and_store_restart_after_failure_completes_without_duplicates(tmp_path: Path) -> None:
    """Рестарт (новый процесс = новый экземпляр эмбеддера) добирает остаток через
    ``chunk_hashes(not_embedded_for=...)`` без повторного счёта/дублей."""
    conn = _big_corpus(tmp_path, 150)
    hashes, texts = chunk_hashes(conn)

    class _FailsOnSecondCall:
        name = "flaky2"
        dim = 1
        max_tokens: int | None = None

        def __init__(self) -> None:
            self.n_calls = 0

        def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
            self.n_calls += 1
            if self.n_calls == 2:
                raise RuntimeError("обрыв")
            return l2_normalize(np.ones((len(texts), self.dim), dtype=np.float32))

    class _WorkingEmbedder:
        name = "flaky2"  # тот же model-неймспейс — рестарт видит уже сохранённый 1-й батч
        dim = 1
        max_tokens: int | None = None

        def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
            return l2_normalize(np.ones((len(texts), self.dim), dtype=np.float32))

    with pytest.raises(RuntimeError):
        embed_and_store(conn, _FailsOnSecondCall(), hashes, texts, batch=64)
    assert _vec_count(conn, "flaky2") == 64

    pending_hashes, pending_texts = chunk_hashes(conn, not_embedded_for="flaky2")
    assert len(pending_hashes) == 86  # 150 - 64, без дублей уже сохранённых
    embed_and_store(conn, _WorkingEmbedder(), pending_hashes, pending_texts, batch=64)
    assert _vec_count(conn, "flaky2") == 150
