"""Тесты pydantic-схемы записи sources.yaml и рендера frontmatter."""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from schema import AcquisitionMethod, Fidelity, Sensitivity, SourceRecord, load_vocab, render_frontmatter


def valid_record() -> dict[str, Any]:
    """Минимально валидная запись (термины — из реальных словарей pipeline/vocab/)."""
    return {
        "id": "sg-imda-mgf-agentic-2026",
        "title": "Model AI Governance Framework for Agentic AI",
        "issuer": "Infocomm Media Development Authority (IMDA)",
        "issuer_type": "government",
        "country": "Singapore",
        "country_iso2": "sg",
        "geo_scope": "national",
        "language": "en",
        "dates": {"published": "2026-05-20", "retrieved": "2026-07-15"},
        "doc_type": "framework",
        "authority": "soft_law",
        "topics": ["ai-governance", "agentic-ai"],
        "g2ai_pattern": ["agent-governance-framework"],
        "source_url": "https://example.org/doc.pdf",
        "status": "verified",
    }


def test_valid_record_parses() -> None:
    rec = SourceRecord.model_validate(valid_record())
    assert rec.id == "sg-imda-mgf-agentic-2026"
    assert rec.dates.published is not None
    assert rec.translation_status.value == "not_started"  # значение по умолчанию


def test_extra_field_forbidden() -> None:
    data = valid_record()
    data["unknown_field"] = "x"
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(data)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("id", "SG_Bad_ID"),          # не kebab-slug
        ("id", "singleword"),          # один сегмент
        ("country_iso2", "SGP"),       # не 2 строчные буквы
        ("language", "eng"),           # не ISO 639-1
        ("sha256", "xyz"),             # не 64 hex
        ("issuer_type", "ministry"),   # вне enum
        ("status", "done"),            # вне enum Status
        ("source_url", "ftp://x/y"),   # не http(s)
        ("acquisition_method", "torrent"),  # вне enum AcquisitionMethod
        ("fidelity", "trustme"),            # вне enum Fidelity
        ("sensitivity", "top_secret"),       # вне enum Sensitivity
        ("official_alt_url", "ftp://x/y"),   # не http(s)
    ],
)
def test_bad_field_rejected(field: str, bad: str) -> None:
    data = valid_record()
    data[field] = bad
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(data)


def test_missing_required_rejected() -> None:
    data = valid_record()
    del data["issuer"]
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(data)


def test_render_frontmatter() -> None:
    rec = SourceRecord.model_validate(valid_record())
    fm = render_frontmatter(rec)
    assert fm.startswith("---\n")
    assert fm.rstrip().endswith("---")
    assert "id: sg-imda-mgf-agentic-2026" in fm
    assert "published: '2026-05-20'" in fm or "published: 2026-05-20" in fm


def test_acquisition_fields_default() -> None:
    """Обратная совместимость: запись без новых полей (как реальная запись SG) парсится."""
    rec = SourceRecord.model_validate(valid_record())
    assert rec.acquisition_method is None
    assert rec.acquisition_checked is None
    assert rec.fidelity is None
    assert rec.retrieved_snapshot_date is None
    assert rec.official_alt_url is None
    assert rec.sensitivity == Sensitivity.normal


def test_acquisition_fields_parse() -> None:
    data = valid_record()
    data["acquisition_method"] = "manual"
    data["acquisition_checked"] = "2026-07-15"
    data["fidelity"] = "archived_snapshot"
    data["retrieved_snapshot_date"] = "2026-01-24"
    data["official_alt_url"] = "https://example.org/alt.pdf"
    data["sensitivity"] = "confidential"
    rec = SourceRecord.model_validate(data)
    assert rec.acquisition_method == AcquisitionMethod.manual
    assert rec.fidelity == Fidelity.archived_snapshot
    assert rec.sensitivity == Sensitivity.confidential
    assert rec.official_alt_url == "https://example.org/alt.pdf"


def test_load_real_vocab_nonempty() -> None:
    for name in ("doc_types", "authority", "topics", "g2ai_patterns"):
        terms = load_vocab(name)
        assert terms, f"словарь {name} пуст"
        assert all(isinstance(t, str) for t in terms)
