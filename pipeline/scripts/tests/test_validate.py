"""Тесты валидатора реестра: словари, уникальность id, ссылочная целостность relations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from schema import VOCAB_DIR
from test_schema import valid_record
from validate_sources import validate_sources


def _write(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(records, allow_unicode=True), encoding="utf-8")
    return path


def test_valid_list_passes(tmp_path: Path) -> None:
    errors = validate_sources(_write(tmp_path, [valid_record()]), VOCAB_DIR)
    assert errors == []


def test_duplicate_id(tmp_path: Path) -> None:
    errors = validate_sources(_write(tmp_path, [valid_record(), valid_record()]), VOCAB_DIR)
    assert any("дубль id" in e for e in errors)


def test_topic_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["topics"] = ["not-a-real-topic"]
    errors = validate_sources(_write(tmp_path, [rec]), VOCAB_DIR)
    assert any("topic 'not-a-real-topic' вне словаря" in e for e in errors)


def test_g2ai_pattern_not_in_vocab(tmp_path: Path) -> None:
    rec = valid_record()
    rec["g2ai_pattern"] = ["invented-pattern"]
    errors = validate_sources(_write(tmp_path, [rec]), VOCAB_DIR)
    assert any("g2ai_pattern 'invented-pattern' вне словаря" in e for e in errors)


def test_dangling_relation(tmp_path: Path) -> None:
    rec = valid_record()
    rec["relations"] = [{"type": "implements", "target": "eu-ec-ai-act-2024"}]
    errors = validate_sources(_write(tmp_path, [rec]), VOCAB_DIR)
    assert any("неизвестный id 'eu-ec-ai-act-2024'" in e for e in errors)


def test_valid_relation_between_records(tmp_path: Path) -> None:
    a = valid_record()
    b = valid_record()
    b["id"] = "eu-ec-ai-act-2024"
    a["relations"] = [{"type": "references", "target": "eu-ec-ai-act-2024"}]
    errors = validate_sources(_write(tmp_path, [a, b]), VOCAB_DIR)
    assert errors == []


def test_structural_error_reported(tmp_path: Path) -> None:
    rec = valid_record()
    rec["id"] = "BAD ID"
    errors = validate_sources(_write(tmp_path, [rec]), VOCAB_DIR)
    assert any("запись #0" in e and "id" in e for e in errors)


def test_missing_relevance_rejected(tmp_path: Path) -> None:
    rec = valid_record()
    del rec["relevance"]
    errors = validate_sources(_write(tmp_path, [rec]), VOCAB_DIR)
    assert any("relevance" in e for e in errors)


def test_top_level_not_list(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text("foo: bar\n", encoding="utf-8")
    errors = validate_sources(path, VOCAB_DIR)
    assert any("верхний уровень" in e for e in errors)
