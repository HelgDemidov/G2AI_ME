"""Тесты discovery/dedup.py: normalize_url/normalized_title + кросс-коннекторный merge
(spec discovery-core §3)."""
from __future__ import annotations

import datetime as dt

from core import schema
from discovery.dedup import dedup, normalize_url, normalized_title


def _candidate(**overrides: object) -> schema.CandidateRecord:
    fields: dict[str, object] = {
        "connector_id": "manual",
        "connector_kind": schema.ConnectorKind.manual,
        "retrieved_at": dt.date(2026, 7, 21),
        "source_ref": "https://example.gov/doc",
        "raw_hash": "h0",
    }
    fields.update(overrides)
    return schema.CandidateRecord.model_validate(fields)


# --- normalize_url --------------------------------------------------------------


def test_normalize_url_scheme_ignored() -> None:
    assert normalize_url("http://example.gov/doc") == normalize_url("https://example.gov/doc")


def test_normalize_url_trailing_slash_ignored() -> None:
    assert normalize_url("https://example.gov/doc/") == normalize_url("https://example.gov/doc")


def test_normalize_url_root_trailing_slash_ignored() -> None:
    assert normalize_url("https://example.gov/") == normalize_url("https://example.gov")


def test_normalize_url_fragment_stripped() -> None:
    assert normalize_url("https://example.gov/doc#section-2") == normalize_url("https://example.gov/doc")


def test_normalize_url_host_case_insensitive() -> None:
    assert normalize_url("https://EXAMPLE.gov/doc") == normalize_url("https://example.gov/doc")


def test_normalize_url_path_case_preserved() -> None:
    """Только host lower-кейсится (spec §3) — путь регистрозависим (реальные серверы такие)."""
    assert normalize_url("https://example.gov/Doc") != normalize_url("https://example.gov/doc")


def test_normalize_url_query_preserved() -> None:
    assert normalize_url("https://example.gov/doc?id=1") != normalize_url("https://example.gov/doc?id=2")


# --- normalized_title -------------------------------------------------------------


def test_normalized_title_case_insensitive() -> None:
    assert normalized_title("AI Governance Framework") == normalized_title("ai governance framework")


def test_normalized_title_punctuation_and_whitespace_ignored() -> None:
    assert normalized_title("AI-Act, 2026.") == normalized_title("AI   Act 2026")


def test_normalized_title_diacritics_preserved_as_letters() -> None:
    """Балканская диакритика (č/š/đ) не отбрасывается — иначе Član/Odjeljak слились бы с шумом."""
    assert normalized_title("Član 1") != normalized_title("Clan 1")
    assert normalized_title("Član 1") == normalized_title("ČLAN, 1")


# --- dedup ------------------------------------------------------------------------


def test_dedup_no_duplicates_passthrough() -> None:
    a = _candidate(source_ref="a", raw_hash="ha", title="Doc A", issuer="Gov")
    b = _candidate(source_ref="b", raw_hash="hb", title="Doc B", issuer="Gov")
    fresh, absorbed = dedup([a, b], existing=[])
    assert fresh == [a, b]
    assert absorbed == 0


def test_dedup_matches_by_normalized_url_against_existing() -> None:
    existing_cand = _candidate(
        connector_id="agora", source_ref="a", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    new_cand = _candidate(
        connector_id="manual", source_ref="b", raw_hash="hb",
        normalized_url=normalize_url("http://EXAMPLE.gov/doc/"),
    )
    fresh, absorbed = dedup([new_cand], existing=[existing_cand])
    assert fresh == []
    assert absorbed == 1
    assert existing_cand.merged_connector_ids == ["manual"]  # type: ignore[attr-defined]


def test_dedup_matches_by_issuer_title_date_when_no_url_key() -> None:
    existing_cand = _candidate(
        connector_id="agora", source_ref="a", raw_hash="ha",
        title="AI Governance Framework", issuer="MinDigital", doc_date=dt.date(2026, 1, 1),
    )
    new_cand = _candidate(
        connector_id="manual", source_ref="b", raw_hash="hb",
        title="ai-governance framework", issuer="MinDigital", doc_date=dt.date(2026, 1, 1),
    )
    fresh, absorbed = dedup([new_cand], existing=[existing_cand])
    assert fresh == []
    assert absorbed == 1


def test_dedup_rejected_existing_not_resurrected() -> None:
    """Отклонённый триажем кандидат (rejected_reason) не должен ре-инжектиться как свежий."""
    rejected = _candidate(
        connector_id="agora", source_ref="a", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
        rejected_reason="вне обеих осей",
    )
    new_cand = _candidate(
        connector_id="manual", source_ref="b", raw_hash="hb",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    fresh, absorbed = dedup([new_cand], existing=[rejected])
    assert fresh == []
    assert absorbed == 1
    assert rejected.rejected_reason == "вне обеих осей"  # неприкосновенно


def test_dedup_never_overwrites_existing_fields() -> None:
    existing_cand = _candidate(source_ref="a", raw_hash="ha", title="Original Title", issuer="Gov")
    dup = _candidate(
        connector_id="directed_search", connector_kind=schema.ConnectorKind.directed_search,
        source_ref="b", raw_hash="hb", title="original title", issuer="Gov",
    )
    dedup([dup], existing=[existing_cand])
    assert existing_cand.title == "Original Title"


def test_dedup_within_new_batch_first_wins() -> None:
    a = _candidate(connector_id="manual", source_ref="a", raw_hash="ha", title="Same Doc", issuer="Gov")
    b = _candidate(
        connector_id="directed_search", connector_kind=schema.ConnectorKind.directed_search,
        source_ref="b", raw_hash="hb", title="same doc", issuer="Gov",
    )
    fresh, absorbed = dedup([a, b], existing=[])
    assert fresh == [a]
    assert absorbed == 1
    assert a.merged_connector_ids == ["directed_search"]  # type: ignore[attr-defined]


def test_dedup_same_connector_rediscovery_does_not_self_reference() -> None:
    """Тот же коннектор повторно нашёл тот же URL — не плодим self-referential provenance."""
    existing_cand = _candidate(
        connector_id="agora", source_ref="a", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    dup_same_connector = _candidate(
        connector_id="agora", source_ref="b", raw_hash="hb",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    fresh, absorbed = dedup([dup_same_connector], existing=[existing_cand])
    assert fresh == []
    assert absorbed == 1
    assert getattr(existing_cand, "merged_connector_ids", None) is None


def test_dedup_content_hash_fallback_when_no_url_or_title() -> None:
    existing_cand = _candidate(source_ref="a", raw_hash="ha", content_hash="deadbeef")
    dup = _candidate(connector_id="agora", source_ref="b", raw_hash="hb", content_hash="deadbeef")
    fresh, absorbed = dedup([dup], existing=[existing_cand])
    assert fresh == []
    assert absorbed == 1
