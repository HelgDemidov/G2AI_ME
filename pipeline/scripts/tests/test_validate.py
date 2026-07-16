"""Тесты валидатора корпуса: словари, уникальность id, relations, инварианты папок (corpus-layout-v2)."""
from __future__ import annotations

from pathlib import Path

import yaml

from schema import VOCAB_DIR
from test_schema import valid_record, write_doc
from validate_sources import validate_sources


def test_valid_doc_passes(tmp_path: Path) -> None:
    write_doc(tmp_path, valid_record())
    assert validate_sources(tmp_path, VOCAB_DIR) == []


def test_empty_corpus_valid(tmp_path: Path) -> None:
    assert validate_sources(tmp_path, VOCAB_DIR) == []


def test_duplicate_id(tmp_path: Path) -> None:
    a = valid_record()
    a["entity_id"] = "e1"
    b = valid_record()
    b["entity_id"] = "e2"  # тот же id, другая сущность (папка)
    write_doc(tmp_path, a)
    write_doc(tmp_path, b)
    assert any("дубль id" in e for e in validate_sources(tmp_path, VOCAB_DIR))


def test_topic_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["topics"] = ["not-a-real-topic"]
    write_doc(tmp_path, rec)
    assert any("topic 'not-a-real-topic' вне словаря" in e for e in validate_sources(tmp_path, VOCAB_DIR))


def test_g2ai_pattern_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["g2ai_pattern"] = ["invented-pattern"]
    write_doc(tmp_path, rec)
    assert any("g2ai_pattern 'invented-pattern' вне словаря" in e for e in validate_sources(tmp_path, VOCAB_DIR))


def test_dangling_relation(tmp_path: Path) -> None:
    rec = valid_record()
    rec["relations"] = [{"type": "implements", "target": "eu-ec-ai-act-2024"}]
    write_doc(tmp_path, rec)
    assert any("неизвестный id 'eu-ec-ai-act-2024'" in e for e in validate_sources(tmp_path, VOCAB_DIR))


def test_valid_relation_between_records(tmp_path: Path) -> None:
    a = valid_record()
    a["relations"] = [{"type": "references", "target": "eu-ec-ai-act-2024"}]
    b = valid_record()
    b["id"] = "eu-ec-ai-act-2024"
    b["entity_id"] = "eu"
    write_doc(tmp_path, a)
    write_doc(tmp_path, b)
    assert validate_sources(tmp_path, VOCAB_DIR) == []


def test_structural_error_reported(tmp_path: Path) -> None:
    rec = valid_record()
    rec["language"] = "eng"  # не ISO 639-1
    write_doc(tmp_path, rec)
    assert any("language" in e for e in validate_sources(tmp_path, VOCAB_DIR))


def test_folder_invariant_violation(tmp_path: Path) -> None:
    rec = valid_record()
    d = tmp_path / "intl-xperience" / "WRONG" / rec["id"]  # папка сущности != entity_id
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text(yaml.safe_dump(rec, allow_unicode=True), encoding="utf-8")
    assert any("entity_id" in e for e in validate_sources(tmp_path, VOCAB_DIR))


def test_missing_relevance_rejected(tmp_path: Path) -> None:
    rec = valid_record()
    del rec["relevance"]
    write_doc(tmp_path, rec)
    assert any("relevance" in e for e in validate_sources(tmp_path, VOCAB_DIR))
