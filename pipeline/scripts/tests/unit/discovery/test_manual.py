"""Тесты discovery/manual.py: inject/worksheet (spec discovery-manual §2-3)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from tests.support import valid_record

from core import schema
from discovery import manual, store


def test_raw_hash_for_manual_deterministic() -> None:
    h1 = manual.raw_hash_for_manual("https://ex.org/a", "Title", dt.date(2026, 1, 1))
    h2 = manual.raw_hash_for_manual("https://ex.org/a", "Title", dt.date(2026, 1, 1))
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest


def test_raw_hash_for_manual_differs_on_input_change() -> None:
    h1 = manual.raw_hash_for_manual("https://ex.org/a", "Title", None)
    h2 = manual.raw_hash_for_manual("https://ex.org/a", "Other Title", None)
    assert h1 != h2


def test_inject_minimal_adds_candidate(tmp_path: Path) -> None:
    cand, is_new = manual.inject(
        url="https://gov.example.org/strategy.pdf",
        title="National AI Strategy",
        issuer="Ministry of Digital Affairs",
        language="en",
        root=tmp_path,
    )
    assert is_new
    assert cand.connector_kind == schema.ConnectorKind.manual
    assert cand.connector_id == "manual"
    loaded = store.load(tmp_path / "candidates.yaml")
    assert len(loaded) == 1
    assert loaded[0].raw_hash == cand.raw_hash


def test_inject_directed_search_requires_campaign_and_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="campaign"):
        manual.inject(
            url="https://gov.example.org/a.pdf",
            title="T",
            issuer="I",
            language="en",
            kind=schema.ConnectorKind.directed_search,
            query="ai strategy",
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="query"):
        manual.inject(
            url="https://gov.example.org/a.pdf",
            title="T",
            issuer="I",
            language="en",
            kind=schema.ConnectorKind.directed_search,
            campaign="small-states-2026",
            root=tmp_path,
        )


def test_inject_directed_search_sets_provenance(tmp_path: Path) -> None:
    cand, is_new = manual.inject(
        url="https://gov.example.org/a.pdf",
        title="T",
        issuer="I",
        language="en",
        kind=schema.ConnectorKind.directed_search,
        campaign="small-states-2026",
        query="national ai strategy small state",
        root=tmp_path,
    )
    assert is_new
    assert cand.connector_id == "search:small-states-2026"
    assert cand.source_ref == "national ai strategy small state"
    assert cand.matched_query == "national ai strategy small state"


def test_inject_duplicate_url_is_noop(tmp_path: Path) -> None:
    manual.inject(
        url="https://gov.example.org/a.pdf", title="T", issuer="I", language="en", root=tmp_path
    )
    cand2, is_new2 = manual.inject(
        url="https://gov.example.org/a.pdf", title="T", issuer="I", language="en", root=tmp_path
    )
    assert is_new2 is False
    assert len(store.load(tmp_path / "candidates.yaml")) == 1


def test_inject_duplicate_of_rejected_reports_reason(tmp_path: Path) -> None:
    cand, _ = manual.inject(
        url="https://gov.example.org/a.pdf", title="T", issuer="I", language="en", root=tmp_path
    )
    all_cands = store.load(tmp_path / "candidates.yaml")
    all_cands[0].rejected_reason = "вне обеих осей"
    store.save(all_cands, tmp_path / "candidates.yaml")

    cand2, is_new2 = manual.inject(
        url="https://gov.example.org/a.pdf", title="T", issuer="I", language="en", root=tmp_path
    )
    assert is_new2 is False
    assert cand2.rejected_reason == "вне обеих осей"


def test_inject_normalizes_url_for_dedup(tmp_path: Path) -> None:
    manual.inject(
        url="https://gov.example.org/a.pdf/",
        title="T",
        issuer="I",
        language="en",
        root=tmp_path,
    )
    _, is_new2 = manual.inject(
        url="http://gov.example.org/a.pdf",  # http vs https, trailing slash — тот же документ
        title="T",
        issuer="I",
        language="en",
        root=tmp_path,
    )
    assert is_new2 is False


# --- pending_candidates / render_worksheet (spec §3) ---


def _candidate(**overrides: object) -> schema.CandidateRecord:
    data: dict[str, object] = {
        "connector_id": "manual",
        "connector_kind": "manual",
        "retrieved_at": "2026-07-21",
        "source_ref": "https://gov.example.org/a.pdf",
        "raw_hash": "a" * 64,
        "title": "T",
        "issuer": "I",
        "language": "en",
        "source_url": "https://gov.example.org/a.pdf",
    }
    data.update(overrides)
    return schema.CandidateRecord.model_validate(data)


def test_pending_candidates_includes_fresh_unrejected() -> None:
    cand = _candidate()
    assert manual.pending_candidates([cand], []) == [cand]


def test_pending_candidates_excludes_rejected() -> None:
    cand = _candidate(rejected_reason="вне обеих осей")
    assert manual.pending_candidates([cand], []) == []


def test_pending_candidates_excludes_already_registered_by_url() -> None:
    cand = _candidate(
        normalized_url="https://gov.example.org/a.pdf",
        source_url="https://gov.example.org/a.pdf",
    )
    rec_data = valid_record()
    rec_data["source_url"] = "https://gov.example.org/a.pdf"
    rec = schema.SourceRecord.model_validate(rec_data)
    assert manual.pending_candidates([cand], [rec]) == []


def test_pending_candidates_normalizes_url_before_comparing() -> None:
    cand = _candidate(source_url="http://gov.example.org/a.pdf/", normalized_url=None)
    rec_data = valid_record()
    rec_data["source_url"] = "https://gov.example.org/a.pdf"  # https, без trailing slash
    rec = schema.SourceRecord.model_validate(rec_data)
    assert manual.pending_candidates([cand], [rec]) == []


def test_pending_candidates_without_url_stays_pending() -> None:
    cand = _candidate(source_url=None, normalized_url=None, content_hash="deadbeef")
    assert manual.pending_candidates([cand], []) == [cand]


def test_render_worksheet_includes_header_and_row() -> None:
    cand = _candidate(jurisdiction="me", doc_date="2026-03-01", native_tags=["ai-governance"])
    text = manual.render_worksheet([cand])
    assert "raw_hash" in text and "relations" in text and "source_format" in text
    assert cand.raw_hash[:12] in text
    assert "me" in text
    assert "2026-03-01" in text
    assert "ai-governance" in text


def test_render_worksheet_empty_pending_still_has_header() -> None:
    text = manual.render_worksheet([])
    assert "Триаж-worksheet" in text
    assert "raw_hash" in text


# --- apply_decisions (spec §4) ---


def _admit_decision(raw_hash: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "raw_hash": raw_hash,
        "action": "admit",
        "id": "me-example-strategy-2026",
        "entity_id": "me",
        "track": "montenegro",
        "issuer_type": "government",
        "geo_scope": "national",
        "doc_type": "strategy",
        "authority": "official",
        "relevance": {
            "target_fit": "primary",
            "axis": "agentic_g2ai",
            "assessed_stage": "triage",
            "rationale": "matches axis",
            "assessed_date": "2026-07-21",
        },
    }
    data.update(overrides)
    return data


def test_apply_reject_sets_rejected_reason(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="a" * 64)
    store.save([cand], tmp_path / "candidates.yaml")

    summary = manual.apply_decisions(
        [{"raw_hash": "a" * 64, "action": "reject", "reason": "вне обеих осей"}], root=tmp_path
    )
    assert summary.errors == []
    reloaded = store.load(tmp_path / "candidates.yaml")
    assert reloaded[0].rejected_reason == "вне обеих осей"


def test_apply_reject_does_not_overwrite_existing_reason(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="a" * 64, rejected_reason="первая причина")
    store.save([cand], tmp_path / "candidates.yaml")

    summary = manual.apply_decisions(
        [{"raw_hash": "a" * 64, "action": "reject", "reason": "новая причина"}], root=tmp_path
    )
    assert summary.errors == []
    reloaded = store.load(tmp_path / "candidates.yaml")
    assert reloaded[0].rejected_reason == "первая причина"


def test_apply_admit_creates_meta_yaml_at_correct_path(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path / "candidates.yaml")

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert summary.errors == []
    meta_path = tmp_path / "montenegro" / "me" / "me-example-strategy-2026" / "meta.yaml"
    assert meta_path.exists()
    records = schema.load_records(tmp_path)
    assert len(records) == 1 and records[0].id == "me-example-strategy-2026"


def test_apply_admit_does_not_touch_candidate_in_store(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path / "candidates.yaml")

    manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    reloaded = store.load(tmp_path / "candidates.yaml")
    assert len(reloaded) == 1
    assert reloaded[0].rejected_reason is None  # кандидат — аудит-след, apply его не трогает


def test_apply_admit_v2_fields_reach_meta_yaml(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path / "candidates.yaml")

    decision = _admit_decision(
        "b" * 64,
        topics=["ai-governance"],
        g2ai_pattern=["agent-governance-framework"],
        summary="short EN summary",
        relations=[{"type": "implements", "target": "eu-ai-act-2024"}],
    )
    manual.apply_decisions([decision], root=tmp_path)
    rec = schema.load_records(tmp_path)[0]
    assert rec.topics == ["ai-governance"]
    assert rec.summary == "short EN summary"
    assert rec.relations[0].target == "eu-ai-act-2024"


def test_apply_incomplete_admit_reports_error_rest_of_batch_applied(tmp_path: Path) -> None:
    good = _candidate(raw_hash="b" * 64)
    bad = _candidate(raw_hash="c" * 64, source_ref="https://gov.example.org/bad.pdf")
    store.save([good, bad], tmp_path / "candidates.yaml")

    incomplete = _admit_decision("c" * 64)
    del incomplete["relevance"]
    summary = manual.apply_decisions([_admit_decision("b" * 64), incomplete], root=tmp_path)

    assert len(summary.errors) == 1
    assert summary.errors[0].raw_hash == "c" * 64
    assert len(schema.load_records(tmp_path)) == 1  # хороший применился, плохой — нет


def test_apply_ambiguous_raw_hash_prefix_reports_error(tmp_path: Path) -> None:
    cand1 = _candidate(raw_hash="a" * 64, source_ref="https://gov.example.org/1.pdf")
    cand2 = _candidate(raw_hash="a" * 63 + "b", source_ref="https://gov.example.org/2.pdf")
    store.save([cand1, cand2], tmp_path / "candidates.yaml")

    summary = manual.apply_decisions(
        [{"raw_hash": "a" * 12, "action": "reject", "reason": "x"}], root=tmp_path
    )
    assert len(summary.errors) == 1
    assert "неоднозначен" in summary.errors[0].detail


def test_apply_unknown_raw_hash_reports_error(tmp_path: Path) -> None:
    summary = manual.apply_decisions(
        [{"raw_hash": "d" * 64, "action": "reject", "reason": "x"}], root=tmp_path
    )
    assert len(summary.errors) == 1


def test_apply_dry_run_does_not_write(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path / "candidates.yaml")

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path, dry_run=True)
    assert summary.dry_run is True
    assert summary.errors == []
    assert schema.load_records(tmp_path) == []
    meta_path = tmp_path / "montenegro" / "me" / "me-example-strategy-2026" / "meta.yaml"
    assert not meta_path.exists()


def test_apply_dry_run_reject_does_not_write(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="a" * 64)
    store.save([cand], tmp_path / "candidates.yaml")

    manual.apply_decisions(
        [{"raw_hash": "a" * 64, "action": "reject", "reason": "x"}], root=tmp_path, dry_run=True
    )
    assert store.load(tmp_path / "candidates.yaml")[0].rejected_reason is None


def test_resolve_candidate_rejects_short_prefix() -> None:
    with pytest.raises(ValueError, match=">=12"):
        manual._resolve_candidate("a" * 8, [_candidate(raw_hash="a" * 64)])
