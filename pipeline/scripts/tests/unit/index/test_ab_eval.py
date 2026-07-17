"""Тесты логики A/B-харнесса (hit@k, загрузчик YAML, диспетчер --mode) — без модели/
сети, CI-safe (эмбеддер — фейк, реальный OpenRouter/bge-m3 не задействован)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from index.ab_eval import DEFAULT_EVAL_QUERIES, hit_at_k, load_eval_queries, main
from index.chunking import Chunk
from index.corpus_index import create_db, fts5_available, index_chunks
from index.embed import FloatArray

pytestmark = pytest.mark.skipif(not fts5_available(), reason="sqlite без FTS5")


def test_hit_in_top1() -> None:
    assert hit_at_k(["about ACCOUNTABILITY here", "other"], ("account",), 1) is True


def test_case_insensitive() -> None:
    assert hit_at_k(["Human OVERSIGHT matters"], ("oversight",), 1) is True


def test_not_top1_but_topk() -> None:
    ranked = ["irrelevant chunk", "text about monitoring agents"]
    assert hit_at_k(ranked, ("monitor",), 1) is False
    assert hit_at_k(ranked, ("monitor",), 3) is True


def test_miss() -> None:
    assert hit_at_k(["a", "b", "c"], ("zzz",), 3) is False


# --- load_eval_queries: валидация YAML (spec analyze-retrieval §6) ---


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_load_eval_queries_valid(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "q.yaml", {
        "queries": [
            {"query": "a", "expect": ["x"]},
            {"query": "b", "expect": ["y", "z"], "note": "опционально"},
        ]
    })
    qs = load_eval_queries(p)
    assert len(qs) == 2
    assert qs[0].query == "a"
    assert qs[0].expect == ("x",)
    assert qs[1].expect == ("y", "z")


def test_load_eval_queries_empty_list_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "q.yaml", {"queries": []})
    with pytest.raises(ValueError, match="queries"):
        load_eval_queries(p)


def test_load_eval_queries_missing_key_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "q.yaml", {"other": 1})
    with pytest.raises(ValueError):
        load_eval_queries(p)


def test_load_eval_queries_missing_field_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "q.yaml", {"queries": [{"query": "a"}]})
    with pytest.raises(ValueError):
        load_eval_queries(p)


def test_load_eval_queries_empty_expect_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "q.yaml", {"queries": [{"query": "a", "expect": []}]})
    with pytest.raises(ValueError):
        load_eval_queries(p)


def test_default_eval_queries_file_loads() -> None:
    """Реальный pipeline/config/eval_queries.yaml (тела документов не нужны — валиден в CI)."""
    qs = load_eval_queries(DEFAULT_EVAL_QUERIES)
    assert len(qs) >= 3
    for cq in qs:
        assert cq.query.strip()
        assert cq.expect


# --- main(): диспетчер --mode (fts — без модели; vector/hybrid — фейковый эмбеддер) ---


class _FakeEmbedder:
    name = "fake-model"
    dim = 2
    max_tokens = None

    def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
        return np.ones((len(texts), self.dim), dtype=np.float32)


class _FakeEmbedderSmallBudget:
    """max_tokens намеренно меньше chunk_max_tokens корпуса — гейт обязан отсечь
    ДО embed() (spec embed-local-swap §4)."""

    name = "fake-small-budget"
    dim = 2
    max_tokens = 10

    def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
        raise AssertionError("embed() не должен вызываться — гейт обязан отсечь раньше")


def _build_db(path: Path) -> None:
    conn = create_db(path)
    index_chunks(conn, [Chunk("doc-a", 0, "accountability and responsibility text", 5)])
    conn.close()


def _write_queries(path: Path) -> Path:
    return _write_yaml(path, {"queries": [{"query": "accountability", "expect": ["account"]}]})


def test_main_mode_fts_needs_no_embedder(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    q = _write_queries(tmp_path / "q.yaml")
    assert main(["--db", str(db), "--eval-queries", str(q), "--mode", "fts", "--k", "1"]) == 0
    out = capsys.readouterr().out
    assert "fts" in out
    assert "hit@1=100%" in out


def test_main_mode_vector_dispatches_via_get_embedder(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    q = _write_queries(tmp_path / "q.yaml")
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: _FakeEmbedder())
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = ["--db", str(db), "--eval-queries", str(q), "--mode", "vector", "--k", "1", "--no-reference"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "fake-model · vector" in out


def test_main_mode_hybrid_dispatches_via_get_embedder(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    q = _write_queries(tmp_path / "q.yaml")
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: _FakeEmbedder())
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = ["--db", str(db), "--eval-queries", str(q), "--mode", "hybrid", "--k", "1", "--no-reference"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "fake-model · hybrid" in out


def test_main_mode_all_reports_all_three(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    q = _write_queries(tmp_path / "q.yaml")
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: _FakeEmbedder())
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = ["--db", str(db), "--eval-queries", str(q), "--mode", "all", "--k", "1", "--no-reference"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "### fts" in out
    assert "fake-model · vector" in out
    assert "fake-model · hybrid" in out


def test_main_vector_mode_propagates_chunk_budget_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """check_chunk_budget вызывается ПО КАЖДОМУ эмбеддеру в цикле (spec
    embed-local-swap §4) — несовместимость чанков с бюджетом модели останавливает
    прогон (без try/except: это не должно прятаться)."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(conn, [Chunk("doc-a", 0, "accountability text", 5)], chunk_max_tokens=100)
    conn.close()
    q = _write_queries(tmp_path / "q.yaml")
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: _FakeEmbedderSmallBudget())
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = ["--db", str(db), "--eval-queries", str(q), "--mode", "vector", "--k", "1", "--no-reference"]
    with pytest.raises(ValueError, match="100"):
        main(argv)


def test_main_missing_db_reports_error(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "absent.db"
    q = _write_queries(tmp_path / "q.yaml")
    assert main(["--db", str(db), "--eval-queries", str(q)]) == 2
    assert "нет БД" in capsys.readouterr().err


def test_main_invalid_eval_queries_reports_error(tmp_path: Path, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    q = _write_yaml(tmp_path / "bad.yaml", {"queries": []})
    assert main(["--db", str(db), "--eval-queries", str(q), "--mode", "fts"]) == 2
    assert "queries" in capsys.readouterr().err
