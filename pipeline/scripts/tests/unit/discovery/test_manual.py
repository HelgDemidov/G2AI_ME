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
