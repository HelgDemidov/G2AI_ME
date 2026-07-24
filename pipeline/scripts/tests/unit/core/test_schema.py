"""Тесты pydantic-схем корпуса (meta.yaml / .state.yaml / candidates.yaml) и рендера frontmatter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.support import valid_record, write_doc

import re

import yaml

from core.schema import (
    ENTITY_PATTERN,
    VOCAB_DIR,
    AcquisitionMethod,
    AssessedStage,
    CandidateRecord,
    Fidelity,
    GeoScope,
    IssuerType,
    OperationalState,
    Relation,
    RelationType,
    Relevance,
    Rights,
    Sensitivity,
    SourceFormat,
    SourceRecord,
    TargetFit,
    Track,
    TranslationStatus,
    check_layout,
    doc_dir,
    load_candidates,
    load_records,
    load_state,
    load_vocab,
    promote_candidate,
    raw_file,
    raw_target,
    render_frontmatter,
    save_record,
    save_state,
)


def test_valid_record_parses() -> None:
    rec = SourceRecord.model_validate(valid_record())
    assert rec.id == "sg-imda-mgf-agentic-2026"
    assert rec.entity_id == "sg" and rec.track.value == "intl-xperience"
    assert rec.dates.published is not None


@pytest.mark.parametrize("code", ["en", "cnr"])
def test_language_accepts_iso_639_1_and_639_3(code: str) -> None:
    """639-1 (2 буквы, 'en') и 639-3 там, где 639-1 нет (черногорский 'cnr') — оба валидны."""
    data = valid_record()
    data["language"] = code
    rec = SourceRecord.model_validate(data)
    assert rec.language == code


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
        ("language", "x"),             # короче 2 букв
        ("language", "engl"),          # длиннее 3 букв
        ("language", "CNR"),           # не lowercase
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


def test_source_format_default_pdf() -> None:
    """Обратная совместимость: существующие meta.yaml без source_format остаются валидными."""
    rec = SourceRecord.model_validate(valid_record())
    assert rec.source_format == SourceFormat.pdf


def test_source_format_html_parse() -> None:
    data = valid_record()
    data["source_format"] = "html"
    rec = SourceRecord.model_validate(data)
    assert rec.source_format == SourceFormat.html


def test_source_format_docx_parse() -> None:
    data = valid_record()
    data["source_format"] = "docx"
    rec = SourceRecord.model_validate(data)
    assert rec.source_format == SourceFormat.docx


def test_source_format_xlsx_parse() -> None:
    data = valid_record()
    data["source_format"] = "xlsx"
    rec = SourceRecord.model_validate(data)
    assert rec.source_format == SourceFormat.xlsx


def test_source_format_bad_rejected() -> None:
    data = valid_record()
    data["source_format"] = "odt"  # не в enum (legacy-формат — сознательно вне скоупа)
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(data)


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
    assert rec.relevance.axis == "agentic_g2ai"
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
    [("target_fit", "core"), ("assessed_stage", "final")],
)
def test_relevance_bad_enum_rejected(field: str, bad: str) -> None:
    data = valid_record()
    data["relevance"][field] = bad
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(data)


def test_relevance_axis_accepts_any_nonempty_string() -> None:
    """Ось — словарь (validate_sources.py), не enum: pydantic принимает любую непустую строку."""
    data = valid_record()
    data["relevance"]["axis"] = "economy"
    rec = SourceRecord.model_validate(data)
    assert rec.relevance is not None
    assert rec.relevance.axis == "economy"

    data["relevance"]["axis"] = ""
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

    from core.schema import VOCAB_DIR

    config_path = VOCAB_DIR.parent / "config" / "triage.yaml"
    data = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data["frontier_year"], int)


def valid_candidate() -> dict[str, Any]:
    """Минимально валидный CandidateRecord (только обязательные поля добычи)."""
    return {
        "connector_id": "agora",
        "retrieved_at": "2026-07-15",
        "raw_hash": "deadbeef",
    }


def test_candidate_minimal() -> None:
    cand = CandidateRecord.model_validate(valid_candidate())
    assert cand.connector_id == "agora"
    assert cand.source_url is None  # best-effort библиография опускаема
    assert cand.native_tags is None  # None, не [] — пустой список не пишется в YAML


def test_candidate_legacy_fields_absorbed_as_extras() -> None:
    """Слим 2026-07-21: connector_kind/source_ref/reported_status убраны из модели,
    старые candidates.yaml с ними остаются загружаемыми (extra="allow")."""
    data = valid_candidate()
    data.update(
        connector_kind="registry",
        source_ref="zenodo:10.5281/zenodo.13883066#row-42",
        reported_status="in_force",
    )
    cand = CandidateRecord.model_validate(data)
    assert cand.connector_id == "agora"


def test_candidate_native_summary_hard_cap() -> None:
    from core.schema import CANDIDATE_SUMMARY_MAX

    data = valid_candidate()
    data["native_summary"] = "x" * (CANDIDATE_SUMMARY_MAX + 1)
    with pytest.raises(ValidationError):
        CandidateRecord.model_validate(data)
    data["native_summary"] = "x" * CANDIDATE_SUMMARY_MAX
    assert CandidateRecord.model_validate(data).native_summary is not None


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
    [("source_url", "ftp://x/y"), ("rights", "gpl"), ("sensitivity", "top_secret")],
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
    assert rec.source_format == SourceFormat.pdf  # дефолт, если не передан явно


def test_promote_candidate_source_format_passthrough() -> None:
    data = valid_candidate()
    data.update(title="Doc", issuer="EU", language="en", source_url="https://ex.org/d.html")
    cand = CandidateRecord.model_validate(data)
    rec = promote_candidate(
        cand,
        id="eu-doc-2024",
        entity_id="eu",
        track=Track.intl_xperience,
        issuer_type=IssuerType.igo,
        geo_scope=GeoScope.regional,
        doc_type="legislation",
        authority="binding_law",
        relevance=_relevance(),
        source_format=SourceFormat.html,
    )
    assert rec.source_format == SourceFormat.html


def test_promote_candidate_v2_analytics_fields_populated() -> None:
    data = valid_candidate()
    data.update(title="Doc", issuer="Gov", language="en", source_url="https://ex.org/d.pdf")
    cand = CandidateRecord.model_validate(data)
    rec = promote_candidate(
        cand,
        id="me-example-strategy-2026",
        entity_id="me",
        track=Track.montenegro,
        issuer_type=IssuerType.government,
        geo_scope=GeoScope.national,
        doc_type="strategy",
        authority="official",
        relevance=_relevance(),
        topics=["ai-governance"],
        g2ai_pattern=["agent-governance-framework"],
        summary="2-3 sentences EN",
        relations=[Relation(type=RelationType.implements, target="eu-ai-act-2024")],
    )
    assert rec.topics == ["ai-governance"]
    assert rec.g2ai_pattern == ["agent-governance-framework"]
    assert rec.summary == "2-3 sentences EN"
    assert rec.relations == [Relation(type=RelationType.implements, target="eu-ai-act-2024")]


def test_promote_candidate_v2_fields_default_empty_when_omitted() -> None:
    """Обратная совместимость: без v2-аргументов — прежние пустые дефолты (батч-каналы)."""
    data = valid_candidate()
    data.update(title="Doc", issuer="Gov", language="en", source_url="https://ex.org/d.pdf")
    cand = CandidateRecord.model_validate(data)
    rec = promote_candidate(
        cand,
        id="me-example-strategy-2026",
        entity_id="me",
        track=Track.montenegro,
        issuer_type=IssuerType.government,
        geo_scope=GeoScope.national,
        doc_type="strategy",
        authority="official",
        relevance=_relevance(),
    )
    assert rec.topics == []
    assert rec.g2ai_pattern == []
    assert rec.summary is None
    assert rec.relations == []


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


def test_promote_candidate_language_override_success() -> None:
    """spec discovery-agora §7: registry-каналы (AGORA) не дают язык — триаж подаёт
    его через override при admit, симметрично doc_type/authority."""
    data = valid_candidate()
    data.update(title="Doc", issuer="Gov", source_url="https://ex.org/d.pdf")  # без language
    cand = CandidateRecord.model_validate(data)
    assert cand.language is None
    rec = promote_candidate(
        cand,
        id="us-cabinet-agentic-2026",
        entity_id="us",
        track=Track.intl_xperience,
        issuer_type=IssuerType.government,
        geo_scope=GeoScope.national,
        doc_type="framework",
        authority="soft_law",
        relevance=_relevance(),
        language="en",
    )
    assert rec.language == "en"


def test_promote_candidate_language_override_none_and_candidate_none_raises() -> None:
    """Без override И без языка у кандидата — прежнее поведение (ValueError), не тихий None."""
    data = valid_candidate()
    data.update(title="Doc", issuer="Gov", source_url="https://ex.org/d.pdf")  # без language
    cand = CandidateRecord.model_validate(data)
    with pytest.raises(ValueError, match="language"):
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


def test_promote_candidate_language_on_candidate_wins_without_override() -> None:
    """manual-кандидат с языком (inject) продолжает работать без override — обратная совместимость."""
    data = valid_candidate()
    data.update(title="Doc", issuer="Gov", language="cnr", source_url="https://ex.org/d.pdf")
    cand = CandidateRecord.model_validate(data)
    rec = promote_candidate(
        cand,
        id="me-example-2026",
        entity_id="me",
        track=Track.montenegro,
        issuer_type=IssuerType.government,
        geo_scope=GeoScope.national,
        doc_type="framework",
        authority="soft_law",
        relevance=_relevance(),
    )
    assert rec.language == "cnr"


def test_operational_state_default() -> None:
    st = OperationalState()
    assert st.sha256 is None
    assert st.acquisition_method is None
    assert st.translation_status == TranslationStatus.not_started
    assert st.lint_defects == []


def test_operational_state_legacy_yaml_without_lint_defects_still_valid() -> None:
    """C1 (spec convert-hardening): старые .state.yaml, записанные ДО появления
    поля, остаются валидны — Field с default, не required."""
    st = OperationalState.model_validate({"sha256": "a" * 64, "acquisition_method": "direct"})
    assert st.lint_defects == []


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


def test_save_state_uses_fsio_atomic_write(tmp_path: Path, monkeypatch: Any) -> None:
    """Мигрировано на fsio.atomic_write_text — единая staging-политика."""
    from core import fsio

    calls: list[Path] = []
    real = fsio.atomic_write_text

    def spy(target: Path, text: str) -> None:
        calls.append(target)
        real(target, text)

    monkeypatch.setattr("core.schema.fsio.atomic_write_text", spy)
    path = tmp_path / ".state.yaml"
    save_state(path, OperationalState(sha256="a" * 64))
    assert calls == [path]
    assert load_state(path).sha256 == "a" * 64


def test_load_real_vocab_nonempty() -> None:
    for name in ("doc_types", "authority", "topics", "g2ai_patterns", "axes"):
        terms = load_vocab(name)
        assert terms, f"словарь {name} пуст"
        assert all(isinstance(t, str) for t in terms)


def test_raw_target_default_ext(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    assert raw_target(rec, tmp_path) == doc_dir(rec, tmp_path) / "raw.pdf"


def test_raw_target_custom_ext(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    assert raw_target(rec, tmp_path, ext="html") == doc_dir(rec, tmp_path) / "raw.html"


def test_raw_target_does_not_require_existing_file(tmp_path: Path) -> None:
    """В отличие от raw_file (читает существующий), raw_target — чисто путь-конструктор."""
    rec = SourceRecord.model_validate(valid_record())
    target = raw_target(rec, tmp_path)
    assert not target.exists()
    assert raw_file(rec, tmp_path) is None  # папка ещё пуста


# --- check_layout: единый источник инвариантов раскладки (raise в load_records, collect в validate) ---


def test_check_layout_clean_returns_empty(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    meta_path = tmp_path / rec.track.value / rec.entity_id / rec.id / "meta.yaml"
    assert check_layout(meta_path, rec, set()) == []


def test_check_layout_folder_name_mismatch(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    meta_path = tmp_path / rec.track.value / rec.entity_id / "wrong-folder-name" / "meta.yaml"
    errors = check_layout(meta_path, rec, set())
    assert any("!= id" in e for e in errors)


def test_check_layout_entity_folder_mismatch(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    meta_path = tmp_path / rec.track.value / "wrong-entity" / rec.id / "meta.yaml"
    errors = check_layout(meta_path, rec, set())
    assert any("entity_id" in e for e in errors)


def test_check_layout_track_folder_mismatch(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    meta_path = tmp_path / "wrong-track" / rec.entity_id / rec.id / "meta.yaml"
    errors = check_layout(meta_path, rec, set())
    assert any("track" in e for e in errors)


def test_check_layout_duplicate_id(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    meta_path = tmp_path / rec.track.value / rec.entity_id / rec.id / "meta.yaml"
    errors = check_layout(meta_path, rec, {rec.id})  # id уже "видели"
    assert any("дубль id" in e for e in errors)


def test_check_layout_does_not_mutate_seen_ids(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    meta_path = tmp_path / rec.track.value / rec.entity_id / rec.id / "meta.yaml"
    seen: set[str] = set()
    check_layout(meta_path, rec, seen)
    assert seen == set()  # регистрация id — забота вызывающей стороны, не check_layout


def test_load_records_collects_documents_sorted_by_id(tmp_path: Path) -> None:
    a = valid_record()
    a["id"], a["entity_id"] = "zz-later-doc-2026", "zz"
    b = valid_record()
    b["id"], b["entity_id"] = "aa-earlier-doc-2026", "aa"
    write_doc(tmp_path, a)
    write_doc(tmp_path, b)
    records = load_records(tmp_path)
    assert [r.id for r in records] == ["aa-earlier-doc-2026", "zz-later-doc-2026"]


def test_load_records_raises_on_folder_invariant_violation(tmp_path: Path) -> None:
    rec = valid_record()
    d = tmp_path / rec["track"] / "WRONG" / rec["id"]  # папка сущности != entity_id
    d.mkdir(parents=True)
    import yaml as _yaml

    (d / "meta.yaml").write_text(_yaml.safe_dump(rec, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="entity_id"):
        load_records(tmp_path)


def test_load_records_raises_on_duplicate_id(tmp_path: Path) -> None:
    a = valid_record()
    a["entity_id"] = "e1"
    b = valid_record()
    b["entity_id"] = "e2"  # тот же id, другая сущность (папка)
    write_doc(tmp_path, a)
    write_doc(tmp_path, b)
    with pytest.raises(ValueError, match="дубль id"):
        load_records(tmp_path)


# --- save_record: писатель meta.yaml (spec discovery-manual §1) ---


def test_save_record_writes_valid_meta_yaml_roundtrip(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    path = save_record(rec, tmp_path)
    assert path == doc_dir(rec, tmp_path) / "meta.yaml"
    assert path.exists()
    loaded = load_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == rec


def test_save_record_creates_missing_doc_dir(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    assert not doc_dir(rec, tmp_path).exists()
    save_record(rec, tmp_path)
    assert doc_dir(rec, tmp_path).is_dir()


def test_save_record_existing_file_raises(tmp_path: Path) -> None:
    rec = SourceRecord.model_validate(valid_record())
    save_record(rec, tmp_path)
    with pytest.raises(ValueError, match="уже существует"):
        save_record(rec, tmp_path)


def test_save_record_roundtrip_preserves_relations_and_analytics(tmp_path: Path) -> None:
    data = valid_record()
    data["relations"] = [{"type": "implements", "target": "eu-ai-act-2024"}]
    rec = SourceRecord.model_validate(data)
    save_record(rec, tmp_path)
    loaded = load_records(tmp_path)[0]
    assert loaded.relations == [Relation(type=RelationType.implements, target="eu-ai-act-2024")]
    assert loaded.topics == rec.topics
    assert loaded.g2ai_pattern == rec.g2ai_pattern


def test_track_tech_standards_accepted_with_international_geo_scope(tmp_path: Path) -> None:
    """Спек aiforgood-standards §2: tech-standards требует geo_scope=international,
    entity_id — слаг организации (не iso2) — не задевает entity_id==iso2-гейт."""
    data = valid_record()
    data["track"] = "tech-standards"
    data["entity_id"] = "itu-t"
    data["geo_scope"] = "international"
    rec = SourceRecord.model_validate(data)
    assert rec.track == Track.tech_standards
    save_record(rec, tmp_path)
    assert doc_dir(rec, tmp_path).is_dir()


# --- vocab_standards_bodies.yaml (спек aiforgood-standards §3) ---

STANDARDS_BODIES_PATH = VOCAB_DIR / "vocab_standards_bodies.yaml"
_VALID_KINDS = {"international", "national", "sectoral"}


def _load_standards_bodies() -> dict[str, Any]:
    data: Any = yaml.safe_load(STANDARDS_BODIES_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_standards_bodies_entity_ids_match_entity_pattern() -> None:
    bodies = _load_standards_bodies()
    assert bodies, "справочник не должен быть пуст"
    for entity_id in bodies:
        assert re.fullmatch(ENTITY_PATTERN, entity_id), entity_id


def test_standards_bodies_kind_and_full_name_present() -> None:
    bodies = _load_standards_bodies()
    for entity_id, entry in bodies.items():
        assert entry["kind"] in _VALID_KINDS, f"{entity_id}: неизвестный kind {entry.get('kind')!r}"
        assert entry["full_name"], f"{entity_id}: пустой full_name"


def test_standards_bodies_covers_spec_organizations() -> None:
    """Спек §3 перечисляет 6 организаций (itu-t/itu-r/ietf/u4ssc/etsi/tta) — регресс-гвард
    против случайного удаления записи, которую коннектор (коммит 3) рассчитывает найти."""
    bodies = _load_standards_bodies()
    assert set(bodies) == {"itu-t", "itu-r", "ietf", "u4ssc", "etsi", "tta"}
