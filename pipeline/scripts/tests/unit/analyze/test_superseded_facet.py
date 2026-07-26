"""Фасет `superseded` и его фильтр в retrieval — spec graph-v2 §2.

Заменённая редакция закона в top-k evidence-пака — тихая содержательная ошибка, поэтому
дефолт исключающий, а история — по явному флагу. Ключевая регрессия здесь: пока в корпусе
НЕТ рёбер supersedes, выдача обязана не меняться ни на бит.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from analyze.retrieve import RetrievalFilters, retrieve
from core.schema import SourceRecord
from index.chunking import Chunk
from index.corpus_index import _rebuild_facets, create_db, fts5_available, index_chunks
from tests.support import valid_record

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")


def _rec(doc_id: str, relations: list[dict[str, str]] | None = None) -> SourceRecord:
    data = valid_record()
    data["id"] = doc_id
    data["relations"] = relations or []
    return SourceRecord.model_validate(data)


def _corpus(tmp_path: Path, records: list[SourceRecord]) -> object:
    """Индекс с одним чанком на документ; текст общий, чтобы FTS находил всех."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk(r.id, 0, "agentic governance framework", 4, "") for r in records])
    _rebuild_facets(conn, records)
    conn.commit()
    return conn


def _found(conn: object, **kw: object) -> list[str]:
    filters = RetrievalFilters(**kw) if kw else None  # type: ignore[arg-type]
    return sorted({h.doc_id for h in retrieve(conn, "agentic", None, k=20, filters=filters)})  # type: ignore[arg-type]


def _pair() -> list[SourceRecord]:
    return [
        _rec("me-law-2024"),
        _rec("me-law-2025", [{"type": "supersedes", "target": "me-law-2024"}]),
    ]


# --- фасет считается кросс-документно ---


def test_facet_marks_predecessor_only(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, _pair())
    rows = dict(conn.execute("SELECT doc_id, superseded FROM doc_facets").fetchall())  # type: ignore[attr-defined]
    assert rows == {"me-law-2024": 1, "me-law-2025": 0}


def test_facet_reads_reverse_edge(tmp_path: Path) -> None:
    """`superseded_by` у предшественника — та же связь; фасет обязан видеть оба
    направления, потому что считает их общий `schema.superseded_ids`."""
    records = [
        _rec("me-law-2024", [{"type": "superseded_by", "target": "me-law-2025"}]),
        _rec("me-law-2025"),
    ]
    conn = _corpus(tmp_path, records)
    rows = dict(conn.execute("SELECT doc_id, superseded FROM doc_facets").fetchall())  # type: ignore[attr-defined]
    assert rows == {"me-law-2024": 1, "me-law-2025": 0}


# --- фильтр retrieval ---


def test_superseded_hidden_by_default(tmp_path: Path) -> None:
    assert _found(_corpus(tmp_path, _pair()), entity_id="sg") == ["me-law-2025"]


def test_superseded_hidden_when_filters_omitted(tmp_path: Path) -> None:
    """РЕГРЕССИЯ контракта дефолта (§2): `retrieve(...)` БЕЗ filters — самый частый путь
    вызова, и раньше он обходил фасетный слой целиком. Дефолт в dataclass сам по себе
    здесь не сработал бы: ошибка была бы тише всего именно на этом пути."""
    assert _found(_corpus(tmp_path, _pair())) == ["me-law-2025"]


def test_include_superseded_shows_history(tmp_path: Path) -> None:
    assert _found(_corpus(tmp_path, _pair()), include_superseded=True) == [
        "me-law-2024", "me-law-2025",
    ]


def test_superseded_gate_composes_with_other_filters(tmp_path: Path) -> None:
    """Гейт ВЫЧИТАЕТ из множества явных фильтров, а не заменяет его."""
    records = _pair() + [_rec("ee-other-2025")]
    conn = _corpus(tmp_path, records)
    assert _found(conn, entity_id="sg") == ["ee-other-2025", "me-law-2025"]


def test_no_supersedes_edges_is_bit_for_bit_noop(tmp_path: Path) -> None:
    """Пока данных нет — поведение прежнее: гейт не превращает `allowed=None` в
    конкретное множество и не может обнулить выдачу."""
    records = [_rec("me-law-2024"), _rec("me-law-2025")]
    conn = _corpus(tmp_path, records)
    assert _found(conn) == ["me-law-2024", "me-law-2025"]


def test_gate_survives_unpopulated_facets(tmp_path: Path) -> None:
    """Индекс собран без записей (фасеты пусты) — выдача не должна схлопнуться в ноль:
    именно поэтому «вселенная поиска» берётся из chunks, а не из doc_facets."""
    conn = create_db(tmp_path / "c.db")
    index_chunks(conn, [Chunk("doc-a", 0, "agentic governance framework", 4, "")])
    conn.commit()
    assert _found(conn) == ["doc-a"]


def test_gate_tolerates_legacy_db_without_column(tmp_path: Path) -> None:
    """БД, собранная до graph-v2: колонки нет — прежнее поведение, не краш."""
    conn = _corpus(tmp_path, _pair())
    conn.execute("ALTER TABLE doc_facets DROP COLUMN superseded")  # type: ignore[attr-defined]
    conn.commit()  # type: ignore[attr-defined]
    assert _found(conn) == ["me-law-2024", "me-law-2025"]
