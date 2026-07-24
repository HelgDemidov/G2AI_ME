"""Тесты discovery/connectors/snowball.py — маппинг RawLink -> CandidateRecord, pre-сигнал
matched_vocab_tags, source_filter/url_filter, document_fingerprint (spec discovery-snowball
§3/§4, коммит 4)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from core import schema
from discovery.connectors.snowball import (
    RawLink,
    SourceFilter,
    UrlFilter,
    _load_vocab_terms,
    apply_source_filter,
    document_fingerprint,
    is_url_filtered,
    map_link,
    match_vocab_tags,
)
from tests.support import valid_record, write_doc

# --- map_link ---


def _source_record() -> schema.SourceRecord:
    return schema.SourceRecord.model_validate(valid_record())


def test_map_link_pdf_fields() -> None:
    link = RawLink(url="https://ai.gov.eg/strategy.pdf", anchor="Egypt AI Strategy", page_number=12)
    cand = map_link(link, source_record=_source_record(), location_kind="pdf", vocab_terms=[])
    assert cand.title == "Egypt AI Strategy"
    assert cand.source_url == "https://ai.gov.eg/strategy.pdf"
    assert cand.native_summary == "Egypt AI Strategy"
    assert cand.native_id == f"{_source_record().id}#p12"
    assert cand.native_tags is not None
    assert "domain: ai.gov.eg" in cand.native_tags
    assert f"source: {_source_record().id}" in cand.native_tags
    assert cand.connector_id == "snowball"
    assert cand.normalized_url is not None


def test_map_link_html_and_md_native_id_forms() -> None:
    rec = _source_record()
    html_cand = map_link(
        RawLink(url="https://example.org/doc", anchor="x"),
        source_record=rec,
        location_kind="html",
        vocab_terms=[],
    )
    md_cand = map_link(
        RawLink(url="https://example.org/doc2", anchor="y"),
        source_record=rec,
        location_kind="md",
        vocab_terms=[],
    )
    assert html_cand.native_id == f"{rec.id}#html"
    assert md_cand.native_id == f"{rec.id}#md"


def test_map_link_fallback_title_when_anchor_empty() -> None:
    link = RawLink(url="https://example.org/reports/annual-2025.pdf", anchor="")
    cand = map_link(link, source_record=_source_record(), location_kind="html", vocab_terms=[])
    assert cand.title == "annual-2025.pdf"
    assert cand.native_summary is None


def test_map_link_ocr_text_url_flag_adds_native_tag() -> None:
    link = RawLink(url="https://example.org/x", anchor="anchor", ocr_text_url=True)
    cand = map_link(link, source_record=_source_record(), location_kind="md", vocab_terms=[])
    assert cand.native_tags is not None
    assert "ocr-text-url" in cand.native_tags


def test_map_link_raw_hash_deterministic_on_same_url() -> None:
    link1 = RawLink(url="https://example.org/x", anchor="a")
    link2 = RawLink(url="https://example.org/x", anchor="different anchor text")
    rec = _source_record()
    c1 = map_link(link1, source_record=rec, location_kind="pdf", vocab_terms=[])
    c2 = map_link(link2, source_record=rec, location_kind="html", vocab_terms=[])
    assert c1.raw_hash == c2.raw_hash  # ключ по normalized_url, не по anchor/kind


def test_map_link_matched_vocab_tags_none_when_no_match() -> None:
    link = RawLink(url="https://example.org/x", anchor="completely unrelated text")
    vocab_terms = [("agentic-ai", "agentic ai")]
    cand = map_link(link, source_record=_source_record(), location_kind="pdf", vocab_terms=vocab_terms)
    assert cand.matched_vocab_tags is None


def test_map_link_matched_vocab_tags_populated_on_match() -> None:
    link = RawLink(url="https://example.org/x", anchor="A report on Agentic AI governance")
    vocab_terms = [("agentic-ai", "agentic ai"), ("e-government", "e government")]
    cand = map_link(link, source_record=_source_record(), location_kind="pdf", vocab_terms=vocab_terms)
    assert cand.matched_vocab_tags == ["agentic-ai"]


# --- match_vocab_tags / _load_vocab_terms (реальные трекаемые словари) ---


def test_load_vocab_terms_reads_real_tracked_files() -> None:
    terms = _load_vocab_terms()
    keys = {key for key, _ in terms}
    assert "agentic-ai" in keys
    assert "e-government" in keys


def test_match_vocab_tags_case_insensitive() -> None:
    terms = [("agentic-ai", "agentic ai")]
    assert match_vocab_tags("This is about AGENTIC AI systems.", terms) == ["agentic-ai"]


# --- apply_source_filter ---


def _rec(id_: str, *, track: str = "montenegro", target_fit: str = "primary") -> schema.SourceRecord:
    data = valid_record()
    data["id"] = id_
    data["track"] = track
    data["entity_id"] = "me" if track == "montenegro" else "sg"
    data["relevance"]["target_fit"] = target_fit
    return schema.SourceRecord.model_validate(data)


def test_apply_source_filter_empty_is_permissive() -> None:
    records = [_rec("a-doc-one"), _rec("b-doc-two")]
    empty = SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=())
    assert apply_source_filter(records, empty) == records


def test_apply_source_filter_tracks() -> None:
    a = _rec("a-doc-one", track="montenegro")
    b = _rec("b-doc-two", track="intl-xperience")
    sf = SourceFilter(tracks=("montenegro",), target_fit=(), include_doc_ids=(), exclude_doc_ids=())
    assert apply_source_filter([a, b], sf) == [a]


def test_apply_source_filter_target_fit() -> None:
    a = _rec("a-doc-one", target_fit="primary")
    b = _rec("b-doc-two", target_fit="background")
    sf = SourceFilter(tracks=(), target_fit=("primary",), include_doc_ids=(), exclude_doc_ids=())
    assert apply_source_filter([a, b], sf) == [a]


def test_apply_source_filter_include_doc_ids_is_allowlist() -> None:
    a, b = _rec("a-doc-one"), _rec("b-doc-two")
    sf = SourceFilter(tracks=(), target_fit=(), include_doc_ids=("a-doc-one",), exclude_doc_ids=())
    assert apply_source_filter([a, b], sf) == [a]


def test_apply_source_filter_exclude_doc_ids() -> None:
    a, b = _rec("a-doc-one"), _rec("b-doc-two")
    sf = SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=("b-doc-two",))
    assert apply_source_filter([a, b], sf) == [a]


def test_apply_source_filter_include_and_exclude_intersecting() -> None:
    """include+exclude одновременно на пересекающемся множестве — exclude выигрывает
    (применяется вторым, после include-allowlist)."""
    a, b = _rec("a-doc-one"), _rec("b-doc-two")
    sf = SourceFilter(
        tracks=(), target_fit=(), include_doc_ids=("a-doc-one", "b-doc-two"), exclude_doc_ids=("b-doc-two",)
    )
    assert apply_source_filter([a, b], sf) == [a]


# --- is_url_filtered ---


def test_is_url_filtered_exact_domain() -> None:
    uf = UrlFilter(exclude_domains=("blog.example.com",), exclude_url_substrings=())
    assert is_url_filtered("https://blog.example.com/post", uf)
    assert not is_url_filtered("https://gov.example.com/doc", uf)


def test_is_url_filtered_subdomain_match() -> None:
    uf = UrlFilter(exclude_domains=("example.com",), exclude_url_substrings=())
    assert is_url_filtered("https://blog.example.com/post", uf)


def test_is_url_filtered_substring() -> None:
    uf = UrlFilter(exclude_domains=(), exclude_url_substrings=("/press-release/",))
    assert is_url_filtered("https://gov.example.com/press-release/2025", uf)
    assert not is_url_filtered("https://gov.example.com/law/2025", uf)


def test_is_url_filtered_empty_filters_reject_nothing() -> None:
    uf = UrlFilter(exclude_domains=(), exclude_url_substrings=())
    assert not is_url_filtered("https://anything.example.com/x", uf)


# --- document_fingerprint ---


def test_document_fingerprint_missing_state_and_md_uses_dash_literal(tmp_path: Path) -> None:
    rec = _rec("a-doc-one")
    write_doc(tmp_path, valid_record() | {"id": "a-doc-one", "entity_id": "me", "track": "montenegro"})
    fp = document_fingerprint(rec, tmp_path)
    expected = hashlib.sha256(b"-|-").hexdigest()
    assert fp == expected


def test_document_fingerprint_changes_with_raw_sha(tmp_path: Path) -> None:
    rec = _rec("a-doc-one")
    data = valid_record() | {"id": "a-doc-one", "entity_id": "me", "track": "montenegro"}
    write_doc(tmp_path, data, md="hello", state={"sha256": "a" * 64})
    fp1 = document_fingerprint(rec, tmp_path)
    write_doc(tmp_path, data, md="hello", state={"sha256": "b" * 64})
    fp2 = document_fingerprint(rec, tmp_path)
    assert fp1 != fp2


def test_document_fingerprint_changes_with_doc_md_content(tmp_path: Path) -> None:
    rec = _rec("a-doc-one")
    data = valid_record() | {"id": "a-doc-one", "entity_id": "me", "track": "montenegro"}
    write_doc(tmp_path, data, md="version one", state={"sha256": "a" * 64})
    fp1 = document_fingerprint(rec, tmp_path)
    write_doc(tmp_path, data, md="version two", state={"sha256": "a" * 64})
    fp2 = document_fingerprint(rec, tmp_path)
    assert fp1 != fp2
