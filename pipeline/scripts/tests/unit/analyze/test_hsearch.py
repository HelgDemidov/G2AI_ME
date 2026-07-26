"""Смоук-тест CLI hsearch (spec analyze-retrieval §5) — fts-only путь (--backend none,
без модели), CI-safe."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from hsearch import main
from index.chunking import Chunk
from index.corpus_index import (
    _rebuild_facets,
    corpus_fingerprint,
    create_db,
    fts5_available,
    index_chunks,
    write_meta,
)
from core.schema import SourceRecord
from tests.support import valid_record, write_doc

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")


def _build_db(path: Path) -> None:
    conn = create_db(path)
    index_chunks(conn, [
        Chunk("doc-a", 0, "agentic ai governance framework", 4, "Governance Chapter"),
        Chunk("doc-b", 0, "unrelated content here", 4),
    ])
    conn.close()


def test_hsearch_backend_none_finds_hit(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    assert main(["governance", "--db", str(db), "--backend", "none"]) == 0
    out = capsys.readouterr().out
    assert "doc-a" in out
    assert "Governance Chapter" in out


def test_hsearch_no_hits_reports_nothing_found(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    assert main(["zzznomatchword", "--db", str(db), "--backend", "none"]) == 0
    assert "ничего не найдено" in capsys.readouterr().out


def test_hsearch_missing_db_reports_error(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "absent.db"
    assert main(["query", "--db", str(db), "--backend", "none"]) == 2
    assert "не найден" in capsys.readouterr().err


def test_hsearch_default_backend_is_openrouter(tmp_path: Path, monkeypatch: Any) -> None:
    """API-first (spec embed-api-first §4): дефолт --backend = openrouter."""
    db = tmp_path / "c.db"
    _build_db(db)
    captured: dict[str, Any] = {}

    def fake_make_embedder(backend: str, model: Any) -> None:
        captured["backend"] = backend
        return None  # None -> честный FTS-only путь retrieve()

    monkeypatch.setattr("hsearch._make_embedder", fake_make_embedder)
    assert main(["governance", "--db", str(db)]) == 0
    assert captured["backend"] == "openrouter"


def test_hsearch_entity_filter_narrows_results(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    conn = create_db(db)
    rec_a = SourceRecord.model_validate({**valid_record(), "id": "sg-doc-2026", "entity_id": "sg"})
    rec_b = SourceRecord.model_validate({**valid_record(), "id": "ee-doc-2026", "entity_id": "ee"})
    _rebuild_facets(conn, [rec_a, rec_b])
    index_chunks(conn, [
        Chunk(rec_a.id, 0, "shared governance term", 4),
        Chunk(rec_b.id, 0, "shared governance term", 4),
    ])
    conn.close()

    assert main(["governance", "--db", str(db), "--backend", "none", "--entity", "sg"]) == 0
    out = capsys.readouterr().out
    assert rec_a.id in out
    assert rec_b.id not in out


# --- staleness-предупреждение: индекс отстаёт от корпуса (knowledge-hardening §9) ---


def test_hsearch_warns_when_index_is_stale(tmp_path: Path, capsys: Any) -> None:
    """Шов аудита между графом и retrieval: доc_facets обновляются только прогоном
    индексации, граф читает meta.yaml живьём — правка корпуса без run_pipeline
    оставляла retrieve() молча фильтровать по старому множеству."""
    sources = tmp_path / "sources"
    write_doc(sources, {**valid_record(), "id": "doc-a"}, md="agentic ai governance framework")

    db = tmp_path / "c.db"
    _build_db(db)
    conn = sqlite3.connect(db)
    write_meta(conn, "corpus_fingerprint", "stale-value-does-not-match-real-corpus")
    conn.commit()
    conn.close()

    assert main(["governance", "--db", str(db), "--sources", str(sources), "--backend", "none"]) == 0
    assert "отстаёт от корпуса" in capsys.readouterr().err


def test_hsearch_no_warning_when_fingerprint_matches(tmp_path: Path, capsys: Any) -> None:
    sources = tmp_path / "sources"
    write_doc(sources, {**valid_record(), "id": "doc-a"}, md="agentic ai governance framework")
    fp = corpus_fingerprint(sources)

    db = tmp_path / "c.db"
    _build_db(db)
    conn = sqlite3.connect(db)
    write_meta(conn, "corpus_fingerprint", fp)
    conn.commit()
    conn.close()

    assert main(["governance", "--db", str(db), "--sources", str(sources), "--backend", "none"]) == 0
    assert "отстаёт" not in capsys.readouterr().err


def test_hsearch_no_warning_when_sources_root_missing(tmp_path: Path, capsys: Any) -> None:
    """Несуществующий корень (запуск вне репо/без локального корпуса) — сверить
    нечего, advisory-проверка молча пропускается, а не отказывает."""
    db = tmp_path / "c.db"
    _build_db(db)
    conn = sqlite3.connect(db)
    write_meta(conn, "corpus_fingerprint", "anything")
    conn.commit()
    conn.close()

    missing = tmp_path / "no-such-sources"
    assert main(["governance", "--db", str(db), "--sources", str(missing), "--backend", "none"]) == 0
    assert "отстаёт" not in capsys.readouterr().err


def test_hsearch_no_warning_when_index_meta_has_no_fingerprint(tmp_path: Path, capsys: Any) -> None:
    """Регресс: БД без штампа corpus_fingerprint (собрана раньше этого спека или
    без записей) — молчание, а не ложное предупреждение."""
    sources = tmp_path / "sources"
    write_doc(sources, {**valid_record(), "id": "doc-a"}, md="text")
    db = tmp_path / "c.db"
    _build_db(db)  # без write_meta corpus_fingerprint

    assert main(["governance", "--db", str(db), "--sources", str(sources), "--backend", "none"]) == 0
    assert "отстаёт" not in capsys.readouterr().err
