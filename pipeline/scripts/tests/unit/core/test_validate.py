"""Тесты валидатора корпуса: словари, уникальность id, relations, инварианты папок (corpus-layout-v2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.schema import VOCAB_DIR
from tests.support import valid_record, write_doc
from core.validate_sources import main, validate_sources


def _errors(root: Path, vocab_dir: Path = VOCAB_DIR) -> list[str]:
    """validate_sources() теперь возвращает (errors, records) — этим тестам
    нужны только errors; records проверяются отдельно (см. ниже)."""
    errors, _records = validate_sources(root, vocab_dir)
    return errors


def test_valid_doc_passes(tmp_path: Path) -> None:
    write_doc(tmp_path, valid_record())
    assert _errors(tmp_path) == []


def test_empty_corpus_valid(tmp_path: Path) -> None:
    assert _errors(tmp_path) == []


def test_duplicate_id(tmp_path: Path) -> None:
    a = valid_record()
    a["entity_id"] = "e1"
    b = valid_record()
    b["entity_id"] = "e2"  # тот же id, другая сущность (папка)
    write_doc(tmp_path, a)
    write_doc(tmp_path, b)
    assert any("дубль id" in e for e in _errors(tmp_path))


def test_topic_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["topics"] = ["not-a-real-topic"]
    write_doc(tmp_path, rec)
    assert any("topic 'not-a-real-topic' вне словаря" in e for e in _errors(tmp_path))


def test_g2ai_pattern_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["g2ai_pattern"] = ["invented-pattern"]
    write_doc(tmp_path, rec)
    assert any("g2ai_pattern 'invented-pattern' вне словаря" in e for e in _errors(tmp_path))


def test_doc_type_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["doc_type"] = "not-a-real-doc-type"
    write_doc(tmp_path, rec)
    assert any("doc_type 'not-a-real-doc-type' вне словаря" in e for e in _errors(tmp_path))


def test_authority_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["authority"] = "not-a-real-authority"
    write_doc(tmp_path, rec)
    assert any("authority 'not-a-real-authority' вне словаря" in e for e in _errors(tmp_path))


def test_axis_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["relevance"]["axis"] = "economy"
    write_doc(tmp_path, rec)
    errors = _errors(tmp_path)
    assert len(errors) == 1
    assert "relevance.axis" in errors[0] and "вне словаря" in errors[0]


def test_axis_valid_passes(tmp_path: Path) -> None:
    for axis in ("agentic_g2ai", "digital_sovereignty"):
        rec = valid_record()
        rec["relevance"]["axis"] = axis
        write_doc(tmp_path, rec)
        assert _errors(tmp_path) == []


def test_missing_relevance_does_not_crash_axis_check(tmp_path: Path) -> None:
    rec = valid_record()
    rec["relevance"] = None
    write_doc(tmp_path, rec)
    errors = _errors(tmp_path)
    assert any("отсутствует relevance" in e for e in errors)
    assert not any("axis" in e for e in errors)


def test_dangling_relation(tmp_path: Path) -> None:
    rec = valid_record()
    rec["relations"] = [{"type": "implements", "target": "eu-ec-ai-act-2024"}]
    write_doc(tmp_path, rec)
    assert any("неизвестный id 'eu-ec-ai-act-2024'" in e for e in _errors(tmp_path))


def test_valid_relation_between_records(tmp_path: Path) -> None:
    a = valid_record()
    a["relations"] = [{"type": "references", "target": "eu-ec-ai-act-2024"}]
    b = valid_record()
    b["id"] = "eu-ec-ai-act-2024"
    b["entity_id"] = "eu"
    write_doc(tmp_path, a)
    write_doc(tmp_path, b)
    assert _errors(tmp_path) == []


def test_structural_error_reported(tmp_path: Path) -> None:
    rec = valid_record()
    rec["language"] = "english"  # не проходит ни ISO 639-1, ни 639-3 по форме
    write_doc(tmp_path, rec)
    assert any("language" in e for e in _errors(tmp_path))


def test_folder_invariant_violation(tmp_path: Path) -> None:
    rec = valid_record()
    d = tmp_path / "intl-xperience" / "WRONG" / rec["id"]  # папка сущности != entity_id
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text(yaml.safe_dump(rec, allow_unicode=True), encoding="utf-8")
    assert any("entity_id" in e for e in _errors(tmp_path))


def test_missing_relevance_rejected(tmp_path: Path) -> None:
    rec = valid_record()
    del rec["relevance"]
    write_doc(tmp_path, rec)
    assert any("relevance" in e for e in _errors(tmp_path))


def test_national_entity_id_iso2_form_accepted(tmp_path: Path) -> None:
    rec = valid_record()  # geo_scope=national, entity_id='sg' — уже форма iso2
    write_doc(tmp_path, rec)
    assert _errors(tmp_path) == []


def test_national_entity_id_non_iso2_rejected(tmp_path: Path) -> None:
    rec = valid_record()
    rec["entity_id"] = "singapore"
    write_doc(tmp_path, rec)
    errors = _errors(tmp_path)
    assert any(
        "geo_scope=national требует entity_id формы iso2" in e and "singapore" in e for e in errors
    )


def test_non_national_entity_id_not_gated(tmp_path: Path) -> None:
    rec = valid_record()
    rec["geo_scope"] = "international"
    rec["entity_id"] = "oecd"
    write_doc(tmp_path, rec)
    assert _errors(tmp_path) == []


def test_validate_sources_returns_parsed_records_sorted_by_id(tmp_path: Path) -> None:
    """records — второй элемент кортежа: успешно распарсенные записи, отсортированные
    по id (тот же порядок, что даёт schema.load_records) — используются вызывающей
    стороной (run_pipeline/build_graph) вместо повторного обхода дерева."""
    a = valid_record()
    a["id"], a["entity_id"] = "zz-later-doc-2026", "zz"
    b = valid_record()
    b["id"], b["entity_id"] = "aa-earlier-doc-2026", "aa"
    write_doc(tmp_path, a)
    write_doc(tmp_path, b)
    errors, records = validate_sources(tmp_path, VOCAB_DIR)
    assert errors == []
    assert [r.id for r in records] == ["aa-earlier-doc-2026", "zz-later-doc-2026"]


def test_validate_sources_records_present_even_with_errors(tmp_path: Path) -> None:
    rec = valid_record()
    del rec["relevance"]
    write_doc(tmp_path, rec)
    errors, records = validate_sources(tmp_path, VOCAB_DIR)
    assert errors  # структурно распарсилась, но relevance отсутствует
    assert len(records) == 1  # структурно валидная запись всё равно возвращается


# --- YAML-синтаксическая ошибка (except yaml.YAMLError) ---


def test_malformed_yaml_reported(tmp_path: Path) -> None:
    d = tmp_path / "intl-xperience" / "sg" / "bad-doc-2026"
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text("id: [unterminated flow sequence\n", encoding="utf-8")
    errors = _errors(tmp_path)
    assert any("YAML" in e for e in errors)


def test_malformed_yaml_does_not_abort_scan_of_other_docs(tmp_path: Path) -> None:
    """Одна битая meta.yaml не должна ронять валидацию остального корпуса (continue после
    YAMLError, аналог изоляции отказа документа в run_pipeline)."""
    bad_dir = tmp_path / "intl-xperience" / "sg" / "bad-doc-2026"
    bad_dir.mkdir(parents=True)
    (bad_dir / "meta.yaml").write_text("id: [unterminated\n", encoding="utf-8")
    write_doc(tmp_path, valid_record())

    errors, records = validate_sources(tmp_path, VOCAB_DIR)
    assert any("YAML" in e for e in errors)
    assert len(records) == 1  # валидный документ всё же распарсен и возвращён


# --- main(): CLI (коды возврата, stdout/stderr) ---


def test_main_valid_corpus_returns_zero_and_prints_ok(tmp_path: Path, capsys: Any) -> None:
    write_doc(tmp_path, valid_record())
    assert main([str(tmp_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_main_invalid_corpus_returns_one_and_prints_errors(tmp_path: Path, capsys: Any) -> None:
    rec = valid_record()
    rec["topics"] = ["not-a-real-topic"]
    write_doc(tmp_path, rec)
    assert main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "вне словаря" in err
    assert "ошибок" in err


def test_main_default_sources_arg_is_default_sources_constant(monkeypatch: Any) -> None:
    """Без позиционного аргумента используется core.schema.DEFAULT_SOURCES (argparse default),
    не хардкод CLI-обвязки."""
    captured: dict[str, Any] = {}

    def fake_validate(sources_path: Path, vocab_dir: Path) -> tuple[list[str], list[Any]]:
        captured["sources_path"] = sources_path
        return [], []

    monkeypatch.setattr("core.validate_sources.validate_sources", fake_validate)
    assert main([]) == 0
    from core.schema import DEFAULT_SOURCES

    assert captured["sources_path"] == DEFAULT_SOURCES
