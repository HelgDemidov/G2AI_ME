"""Тесты pydantic-схемы записи sources.yaml и рендера frontmatter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from schema import (
    AcquisitionMethod,
    AssessedStage,
    Axis,
    CandidateRecord,
    ConnectorKind,
    Fidelity,
    GeoScope,
    IssuerType,
    OperationalState,
    Relevance,
    Rights,
    Sensitivity,
    SourceRecord,
    TargetFit,
    Track,
    TranslationStatus,
    load_candidates,
    load_state,
    load_vocab,
    promote_candidate,
    render_frontmatter,
    save_state,
)


def valid_record() -> dict[str, Any]:
    """Минимально валидная запись (термины — из реальных словарей pipeline/vocab/)."""
    return {
        "id": "sg-imda-mgf-agentic-2026",
        "entity_id": "sg",
        "track": "intl-xperience",
        "title": "Model AI Governance Framework for Agentic AI",
        "issuer": "Infocomm Media Development Authority (IMDA)",
        "issuer_type": "government",
        "geo_scope": "national",
        "language": "en",
        "dates": {"published": "2026-05-20", "retrieved": "2026-07-15"},
        "doc_type": "framework",
        "authority": "soft_law",
        "topics": ["ai-governance", "agentic-ai"],
        "g2ai_pattern": ["agent-governance-framework"],
        "source_url": "https://example.org/doc.pdf",
        "relevance": {
            "target_fit": "primary",
            "axis": "agentic_g2ai",
            "assessed_stage": "confirmed",
            "rationale": "эталонный агентный G2AI-документ",
            "assessed_date": "2026-07-15",
        },
    }


def write_doc(
    root: Path,
    rec: dict[str, Any],
    *,
    raw: bytes | None = None,
    md: str | None = None,
    state: dict[str, Any] | None = None,
) -> Path:
    """Создать папку-документ sources/<track>/<entity>/<id>/ + meta.yaml (+ raw.pdf/doc.md/.state.yaml)."""
    import yaml as _yaml

    d = root / rec["track"] / rec["entity_id"] / rec["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.yaml").write_text(_yaml.safe_dump(rec, allow_unicode=True), encoding="utf-8")
    if raw is not None:
        (d / "raw.pdf").write_bytes(raw)
    if md is not None:
        (d / "doc.md").write_text(md, encoding="utf-8")
    if state is not None:
        (d / ".state.yaml").write_text(_yaml.safe_dump(state, allow_unicode=True), encoding="utf-8")
    return d


def test_valid_record_parses() -> None:
    rec = SourceRecord.model_validate(valid_record())
    assert rec.id == "sg-imda-mgf-agentic-2026"
    assert rec.entity_id == "sg" and rec.track.value == "intl-xperience"
    assert rec.dates.published is not None


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
        ("language", "eng"),           # не ISO 639-1
        ("issuer_type", "ministry"),   # вне enum
        ("source_url", "ftp://x/y"),   # не http(s)
        ("sensitivity", "top_secret"),       # вне enum Sensitivity
        ("official_alt_url", "ftp://x/y"),   # не http(s)
        ("rights", "gpl"),                   # вне enum Rights
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


def test_curated_provenance_fields() -> None:
    """Оставшиеся в meta.yaml provenance-поля: official_alt_url/sensitivity/rights (acquisition -> .state.yaml)."""
    rec = SourceRecord.model_validate(valid_record())
    assert rec.official_alt_url is None
    assert rec.sensitivity == Sensitivity.normal
    assert rec.rights == Rights.unknown
    data = valid_record()
    data["official_alt_url"] = "https://example.org/alt.pdf"
    data["sensitivity"] = "confidential"
    rec2 = SourceRecord.model_validate(data)
    assert rec2.official_alt_url == "https://example.org/alt.pdf"
    assert rec2.sensitivity == Sensitivity.confidential


def test_rights_default() -> None:
    """Обратная совместимость: запись без rights дефолтит unknown."""
    rec = SourceRecord.model_validate(valid_record())
    assert rec.rights == Rights.unknown


def test_rights_parse() -> None:
    data = valid_record()
    data["rights"] = "cc-by"
    rec = SourceRecord.model_validate(data)
    assert rec.rights == Rights.cc_by


def test_relevance_parse() -> None:
    rec = SourceRecord.model_validate(valid_record())
    assert rec.relevance is not None
    assert rec.relevance.target_fit == TargetFit.primary
    assert rec.relevance.axis == Axis.agentic_g2ai
    assert rec.relevance.assessed_stage == AssessedStage.confirmed


def test_relevance_default_none() -> None:
    """Обратная совместимость: запись без relevance парсится (Optional на pydantic-уровне)."""
    data = valid_record()
    del data["relevance"]
    rec = SourceRecord.model_validate(data)
    assert rec.relevance is None
    assert rec.in_force is None


@pytest.mark.parametrize("drop", ["rationale", "assessed_date"])
def test_relevance_missing_required_rejected(drop: str) -> None:
    data = valid_record()
    del data["relevance"][drop]
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(data)


@pytest.mark.parametrize(
    "field,bad",
    [("target_fit", "core"), ("axis", "economy"), ("assessed_stage", "final")],
)
def test_relevance_bad_enum_rejected(field: str, bad: str) -> None:
    data = valid_record()
    data["relevance"][field] = bad
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(data)


@pytest.mark.parametrize("value", [True, False, None])
def test_in_force_parse(value: bool | None) -> None:
    data = valid_record()
    data["in_force"] = value
    rec = SourceRecord.model_validate(data)
    assert rec.in_force is value


def test_triage_config_wellformed() -> None:
    """pipeline/config/triage.yaml — валидный YAML с целочисленным frontier_year."""
    import yaml as _yaml

    from schema import VOCAB_DIR

    config_path = VOCAB_DIR.parent / "config" / "triage.yaml"
    data = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data["frontier_year"], int)


def valid_candidate() -> dict[str, Any]:
    """Минимально валидный CandidateRecord (только обязательные поля добычи)."""
    return {
        "connector_id": "agora",
        "connector_kind": "registry",
        "retrieved_at": "2026-07-15",
        "source_ref": "zenodo:10.5281/zenodo.13883066#row-42",
        "raw_hash": "deadbeef",
    }


def test_candidate_minimal() -> None:
    cand = CandidateRecord.model_validate(valid_candidate())
    assert cand.connector_kind == ConnectorKind.registry
    assert cand.source_url is None  # best-effort библиография опускаема
    assert cand.native_tags == []


def test_candidate_permissive_extra_allowed() -> None:
    data = valid_candidate()
    data["some_connector_specific_field"] = {"nested": 1}
    cand = CandidateRecord.model_validate(data)  # extra="allow" не падает
    assert cand.connector_id == "agora"


def test_candidate_full_bibliography() -> None:
    data = valid_candidate()
    data.update(
        title="X",
        issuer="Y",
        source_url="https://example.org/a.pdf",
        language="en",
        rights="cc-by",
        sensitivity="confidential",
        native_tags=["risk", "governance"],
    )
    cand = CandidateRecord.model_validate(data)
    assert cand.source_url == "https://example.org/a.pdf"
    assert cand.rights == Rights.cc_by
    assert cand.sensitivity == Sensitivity.confidential
    assert cand.native_tags == ["risk", "governance"]


@pytest.mark.parametrize(
    "field,bad",
    [("connector_kind", "spider"), ("source_url", "ftp://x/y"), ("rights", "gpl")],
)
def test_candidate_bad_field_rejected(field: str, bad: str) -> None:
    data = valid_candidate()
    data[field] = bad
    with pytest.raises(ValidationError):
        CandidateRecord.model_validate(data)


def test_candidate_missing_required_rejected() -> None:
    data = valid_candidate()
    del data["connector_id"]
    with pytest.raises(ValidationError):
        CandidateRecord.model_validate(data)


def test_load_candidates(tmp_path: Path) -> None:
    import yaml as _yaml

    path = tmp_path / "candidates.yaml"
    path.write_text(_yaml.safe_dump([valid_candidate()], allow_unicode=True), encoding="utf-8")
    cands = load_candidates(path)
    assert len(cands) == 1
    assert cands[0].connector_id == "agora"


def test_load_candidates_empty(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    path.write_text("# пусто\n", encoding="utf-8")
    assert load_candidates(path) == []


def _relevance() -> Relevance:
    return Relevance.model_validate(
        {
            "target_fit": "primary",
            "axis": "agentic_g2ai",
            "assessed_stage": "triage",
            "rationale": "ok",
            "assessed_date": "2026-07-15",
        }
    )


def test_promote_candidate_success() -> None:
    data = valid_candidate()
    data.update(
        title="Doc",
        issuer="Gov",
        language="en",
        source_url="https://ex.org/d.pdf",
        doc_date="2026-03-01",
        rights="cc-by",
        sensitivity="confidential",
    )
    cand = CandidateRecord.model_validate(data)
    rec = promote_candidate(
        cand,
        id="ae-cabinet-agentic-2026",
        entity_id="ae",
        track=Track.intl_xperience,
        issuer_type=IssuerType.government,
        geo_scope=GeoScope.national,
        doc_type="framework",
        authority="soft_law",
        relevance=_relevance(),
    )
    assert isinstance(rec, SourceRecord)
    assert rec.id == "ae-cabinet-agentic-2026"
    assert rec.entity_id == "ae" and rec.track == Track.intl_xperience
    assert rec.source_url == "https://ex.org/d.pdf"
    assert rec.dates.published is not None
    assert rec.rights == Rights.cc_by  # перенесён с кандидата
    assert rec.sensitivity == Sensitivity.confidential
    assert rec.relevance is not None and rec.relevance.target_fit == TargetFit.primary


def test_promote_candidate_missing_source_url_raises() -> None:
    cand = CandidateRecord.model_validate(
        {**valid_candidate(), "title": "Doc", "issuer": "Gov", "language": "en"}  # без source_url
    )
    with pytest.raises(ValueError, match="source_url"):
        promote_candidate(
            cand,
            id="x-y-2026",
            entity_id="xx",
            track=Track.intl_xperience,
            issuer_type=IssuerType.government,
            geo_scope=GeoScope.national,
            doc_type="framework",
            authority="soft_law",
            relevance=_relevance(),
        )


def test_operational_state_default() -> None:
    st = OperationalState()
    assert st.sha256 is None
    assert st.acquisition_method is None
    assert st.translation_status == TranslationStatus.not_started


def test_operational_state_parse() -> None:
    st = OperationalState.model_validate(
        {
            "sha256": "a" * 64,
            "acquisition_method": "direct",
            "fidelity": "live",
            "acquisition_checked": "2026-07-16",
        }
    )
    assert st.acquisition_method == AcquisitionMethod.direct
    assert st.fidelity == Fidelity.live


@pytest.mark.parametrize(
    "field,bad",
    [("sha256", "xyz"), ("acquisition_method", "torrent"), ("fidelity", "nope")],
)
def test_operational_state_bad_rejected(field: str, bad: str) -> None:
    with pytest.raises(ValidationError):
        OperationalState.model_validate({field: bad})


def test_load_state_missing(tmp_path: Path) -> None:
    assert load_state(tmp_path / "nope.state.yaml") == OperationalState()


def test_save_load_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / ".state.yaml"
    st = OperationalState.model_validate(
        {"sha256": "b" * 64, "acquisition_method": "archive", "fidelity": "archived_snapshot"}
    )
    save_state(path, st)
    assert load_state(path) == st


def test_load_real_vocab_nonempty() -> None:
    for name in ("doc_types", "authority", "topics", "g2ai_patterns"):
        terms = load_vocab(name)
        assert terms, f"словарь {name} пуст"
        assert all(isinstance(t, str) for t in terms)
