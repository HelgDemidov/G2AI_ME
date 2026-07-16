"""Тесты валидатора корпуса: словари, уникальность id, relations, инварианты папок (corpus-layout-v2)."""
from __future__ import annotations

from pathlib import Path

import yaml

from core.schema import VOCAB_DIR
from tests.support import valid_record, write_doc
from core.validate_sources import validate_sources


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
