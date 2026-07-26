"""Тесты логики A/B-харнесса (hit@k, загрузчик YAML, диспетчер --mode) — без модели/
сети, CI-safe (эмбеддер — фейк, реальный OpenRouter/bge-m3 не задействован)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from core.schema import SourceRecord
from index.ab_eval import (
    DEFAULT_EVAL_QUERIES,
    PRECISION_AT_K,
    ModelResult,
    QueryOutcome,
    _report,
    hit_at_k,
    load_eval_queries,
    main,
    parse_backends,
    precision_at_k,
    reciprocal_rank,
)
from index.chunking import Chunk
from index.corpus_index import create_db, fts5_available, index_chunks
from index.embed import FloatArray, OpenRouterEmbedder
from tests.support import valid_record

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


# --- reciprocal_rank / precision_at_k (бэклог §17, eval-precision-metrics) ---


def test_reciprocal_rank_top1() -> None:
    assert reciprocal_rank(["about ACCOUNTABILITY here", "other"], ("account",)) == 1.0


def test_reciprocal_rank_third_position() -> None:
    ranked = ["irrelevant", "also irrelevant", "text about monitoring agents"]
    assert reciprocal_rank(ranked, ("monitor",)) == pytest.approx(1 / 3)


def test_reciprocal_rank_miss_is_zero() -> None:
    assert reciprocal_rank(["a", "b"], ("zzz",)) == 0.0


def test_precision_at_k_counts_matches_in_window() -> None:
    ranked = ["monitor here", "irrelevant", "monitor there", "irrelevant", "irrelevant"]
    assert precision_at_k(ranked, ("monitor",), 5) == pytest.approx(2 / 5)


def test_precision_at_k_smaller_window_than_k_divides_by_actual_count() -> None:
    """Индекс отдал меньше кандидатов, чем k — делим на реальное число, не на k
    формально (иначе тонкий корпус штрафуется за нехватку кандидатов)."""
    assert precision_at_k(["monitor here"], ("monitor",), 5) == 1.0


def test_precision_at_k_empty_ranked_is_zero() -> None:
    assert precision_at_k([], ("monitor",), 5) == 0.0


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


# --- lang: per-language eval (spec embed-api-first §5) ---


def test_load_eval_queries_lang_field_parsed(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "q.yaml", {"queries": [{"query": "a", "expect": ["x"], "lang": "cnr"}]})
    qs = load_eval_queries(p)
    assert qs[0].lang == "cnr"


def test_load_eval_queries_lang_defaults_to_en(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "q.yaml", {"queries": [{"query": "a", "expect": ["x"]}]})
    qs = load_eval_queries(p)
    assert qs[0].lang == "en"


def test_default_eval_queries_has_cnr_and_et_lang_tags() -> None:
    qs = load_eval_queries(DEFAULT_EVAL_QUERIES)
    langs = {cq.lang for cq in qs}
    assert {"cnr", "et"} <= langs


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


# --- parse_backends: comma-список bge|openrouter:<model> (spec embed-api-first §4) ---


class _NamedFakeEmbedder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dim = 2
        self.max_tokens: int | None = None

    def embed(self, texts: list[str], *, kind: str = "doc") -> FloatArray:
        return np.ones((len(texts), self.dim), dtype=np.float32)


def _fake_get_embedder(backend: str, **kw: Any) -> Any:
    if backend == "bge":
        return _NamedFakeEmbedder("bge-fake")
    return _NamedFakeEmbedder(f"or-fake:{kw.get('model')}")


def test_parse_backends_bge_and_openrouter_correct_names(monkeypatch: Any) -> None:
    monkeypatch.setattr("index.ab_eval.get_embedder", _fake_get_embedder)
    embedders = parse_backends("bge,openrouter:qwen/qwen3-embedding-8b")
    assert [e.name for e in embedders] == ["bge-fake", "or-fake:qwen/qwen3-embedding-8b"]


def test_parse_backends_openrouter_without_model_raises() -> None:
    with pytest.raises(ValueError, match="openrouter"):
        parse_backends("openrouter:")


def test_parse_backends_unknown_token_raises() -> None:
    with pytest.raises(ValueError, match="неизвестный бэкенд"):
        parse_backends("nonsense")


def test_parse_backends_model_with_colon_suffix_splits_on_first_colon(monkeypatch: Any) -> None:
    """OpenRouter ':free'-варианты содержат двоеточие В ИМЕНИ модели — сплит только
    по первому ':' после 'openrouter'."""
    captured: dict[str, Any] = {}

    def fake_get_embedder(backend: str, **kw: Any) -> Any:
        captured["model"] = kw.get("model")
        return _NamedFakeEmbedder("x")

    monkeypatch.setattr("index.ab_eval.get_embedder", fake_get_embedder)
    parse_backends("openrouter:nvidia/nemotron-3-embed-1b:free")
    assert captured["model"] == "nvidia/nemotron-3-embed-1b:free"


def test_main_backends_flag_uses_multiple_embedders(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    db = tmp_path / "c.db"
    _build_db(db)
    q = _write_queries(tmp_path / "q.yaml")
    monkeypatch.setattr("index.ab_eval.get_embedder", _fake_get_embedder)
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = [
        "--db", str(db), "--eval-queries", str(q), "--mode", "vector", "--k", "1",
        "--backends", "bge,openrouter:some-model", "--no-reference",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "bge-fake · vector" in out
    assert "or-fake:some-model · vector" in out


def test_main_default_backends_matches_prior_single_bge_behavior(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Без --backends дефолт "bge" даёт ровно тот же единственный эмбеддер, что и
    старое захардкоженное поведение — обратная совместимость."""
    db = tmp_path / "c.db"
    _build_db(db)
    q = _write_queries(tmp_path / "q.yaml")
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: _FakeEmbedder())
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = ["--db", str(db), "--eval-queries", str(q), "--mode", "vector", "--k", "1", "--no-reference"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("· vector") == 1


# --- _report: per-language разбивка (spec embed-api-first §5) ---


def test_report_prints_per_language_breakdown_when_multilingual(capsys: Any) -> None:
    outcomes = [
        QueryOutcome("q1", True, True, 1.0, 1.0, 1.0, "en"),
        QueryOutcome("q2", False, True, 0.5, 0.4, 0.5, "cnr"),
        QueryOutcome("q3", False, False, 0.0, 0.0, 0.1, "cnr"),
    ]
    res = ModelResult("fake · vector", 1 / 3, 2 / 3, 0.5, 0.4, outcomes)
    _report([res], k=3, n_queries=3)
    out = capsys.readouterr().out
    assert "cnr: hit@1=0% hit@3=50% MRR=0.250 precision@5=20%" in out
    assert "en: hit@1=100% hit@3=100% MRR=1.000 precision@5=100%" in out


def test_report_omits_per_language_breakdown_when_monolingual(capsys: Any) -> None:
    outcomes = [QueryOutcome("q1", True, True, 1.0, 1.0, 1.0, "en")]
    res = ModelResult("fake · vector", 1.0, 1.0, 1.0, 1.0, outcomes)
    _report([res], k=1, n_queries=1)
    out = capsys.readouterr().out
    assert "en:" not in out


def test_report_prints_mrr_and_precision5_in_header(capsys: Any) -> None:
    outcomes = [QueryOutcome("q1", True, True, 1.0, 0.6, 1.0, "en")]
    res = ModelResult("fake · vector", 1.0, 1.0, 1.0, 0.6, outcomes)
    _report([res], k=3, n_queries=1)
    out = capsys.readouterr().out
    assert f"MRR=1.000   precision@{PRECISION_AT_K}=60%" in out


# --- sensitivity-гейт: третья дверь облачного эмбеддинга (knowledge-hardening §5) ---


def _sensitive_record(id_: str, *, sensitivity: str = "normal") -> SourceRecord:
    rec_dict = valid_record()
    rec_dict["id"] = id_
    rec_dict["sensitivity"] = sensitivity
    return SourceRecord.model_validate(rec_dict)


def _vec_count(conn: sqlite3.Connection, model: str) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM vectors WHERE model=?", (model,)).fetchone()[0])


def test_main_vector_mode_applies_sensitivity_gate_to_real_openrouter_embedder(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """ab_eval была ЕДИНСТВЕННОЙ из трёх дверей облачного эмбеддинга без sensitivity-
    гейта (vector_store embed-corpus и run_pipeline --embed его уже применяли).
    Гейт различает бэкенды через ``isinstance(embedder, OpenRouterEmbedder)`` — ab_eval
    сравнивает гетерогенный список уже сконструированных эмбеддеров, а не один backend
    по строке, поэтому фейк, не наследующий реальный класс, гейт корректно НЕ триггерит
    (см. другие тесты этого файла с ``_FakeEmbedder``) — здесь нужен настоящий экземпляр."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(
        conn,
        [Chunk("doc-conf", 0, "confidential only text", 3), Chunk("doc-pub", 0, "public text", 2)],
        records=[
            _sensitive_record("doc-conf", sensitivity="confidential"),
            _sensitive_record("doc-pub"),
        ],
    )
    conn.close()
    q = _write_queries(tmp_path / "q.yaml")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    real = OpenRouterEmbedder(model="cloud-model", dims=2)
    monkeypatch.setattr(
        OpenRouterEmbedder, "_request", lambda self, batch: [[1.0, 2.0] for _ in batch]
    )
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: real)
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = [
        "--db", str(db), "--eval-queries", str(q), "--mode", "vector", "--k", "1",
        "--no-reference", "--backends", "openrouter:cloud-model",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "1 чанков только-confidential" in out

    conn2 = sqlite3.connect(db)
    assert _vec_count(conn2, real.name) == 1  # только публичный чанк заэмбеджен облаком


def _mark_for(out: str, section_header: str, query_text: str) -> str:
    """Символ-маркер (✓/~/✗) строки данного запроса внутри секции отчёта."""
    start = out.index(section_header)
    end = out.index("\n\n", start) if "\n\n" in out[start:] else len(out)
    for line in out[start:end].splitlines():
        stripped = line.strip()
        if stripped.endswith(query_text):
            return stripped[0]
    raise AssertionError(f"запрос {query_text!r} не найден в секции {section_header!r}")


def test_main_all_modes_exclude_superseded_document_uniformly(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """С первым ребром supersedes fts/vector/hybrid обязаны согласованно исключать
    предшественника (knowledge-hardening §6) — до фикса evaluate_vector/evaluate_fts
    не гейтились вовсе, и с этим ребром режимы начали бы сравнивать разные вселенные
    документов, тихо переставая быть сравнением одного и того же."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    old = valid_record() | {"id": "me-law-2024"}
    new = valid_record() | {
        "id": "me-law-2025",
        "relations": [{"type": "supersedes", "target": "me-law-2024"}],
    }
    index_chunks(
        conn,
        [
            Chunk("me-law-2024", 0, "legacy predecessor unique-term-alpha text", 6),
            Chunk("me-law-2025", 0, "current successor unique-term-beta text", 6),
        ],
        records=[SourceRecord.model_validate(old), SourceRecord.model_validate(new)],
    )
    conn.close()
    q = _write_yaml(
        tmp_path / "q.yaml",
        {
            "queries": [
                {"query": "unique-term-alpha", "expect": ["unique-term-alpha"]},
                {"query": "unique-term-beta", "expect": ["unique-term-beta"]},
            ]
        },
    )
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: _FakeEmbedder())
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = ["--db", str(db), "--eval-queries", str(q), "--mode", "all", "--k", "1", "--no-reference"]
    assert main(argv) == 0
    out = capsys.readouterr().out

    for header in ("### fts", "fake-model · vector", "fake-model · hybrid"):
        assert _mark_for(out, header, "unique-term-alpha") == "✗"  # предшественник исключён
        assert _mark_for(out, header, "unique-term-beta") == "✓"   # преемник виден


def test_main_vector_mode_fake_embedder_bypasses_sensitivity_gate(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Регресс: локальный/фейковый эмбеддер (не OpenRouterEmbedder) НЕ триггерит гейт —
    поведение существующих тестов ``_FakeEmbedder`` не должно измениться."""
    db = tmp_path / "c.db"
    conn = create_db(db)
    index_chunks(
        conn,
        [Chunk("doc-conf", 0, "confidential only text", 3)],
        records=[_sensitive_record("doc-conf", sensitivity="confidential")],
    )
    conn.close()
    q = _write_queries(tmp_path / "q.yaml")
    monkeypatch.setattr("index.ab_eval.get_embedder", lambda backend, **kw: _FakeEmbedder())
    monkeypatch.setattr("index.ab_eval.load_dotenv", lambda: None)

    argv = ["--db", str(db), "--eval-queries", str(q), "--mode", "vector", "--k", "1", "--no-reference"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "только-confidential" not in out

    conn2 = sqlite3.connect(db)
    assert _vec_count(conn2, "fake-model") == 1  # confidential-чанк заэмбеджен как обычно
