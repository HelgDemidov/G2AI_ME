"""Тесты discovery/dedup.py: normalize_url/normalized_title + кросс-коннекторный merge
(spec discovery-core §3)."""
from __future__ import annotations

import datetime as dt

from core import schema
from discovery.dedup import dedup, format_hint_from_url, normalize_url, normalized_title


def _candidate(**overrides: object) -> schema.CandidateRecord:
    fields: dict[str, object] = {
        "connector_id": "manual",
        "retrieved_at": dt.date(2026, 7, 21),
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


# --- format_hint_from_url (spec discovery-acquire-seam-hardening §8, Г7) ----------


def test_format_hint_from_url_pdf() -> None:
    assert format_hint_from_url("https://gov.example.org/law.pdf") == schema.SourceFormat.pdf


def test_format_hint_from_url_docx() -> None:
    assert format_hint_from_url("https://gov.example.org/law.docx") == schema.SourceFormat.docx


def test_format_hint_from_url_xlsx() -> None:
    assert format_hint_from_url("https://gov.example.org/data.xlsx") == schema.SourceFormat.xlsx


def test_format_hint_from_url_html() -> None:
    assert format_hint_from_url("https://gov.example.org/law.html") == schema.SourceFormat.html


def test_format_hint_from_url_htm() -> None:
    assert format_hint_from_url("https://gov.example.org/law.htm") == schema.SourceFormat.html


def test_format_hint_from_url_case_insensitive_extension() -> None:
    assert format_hint_from_url("https://gov.example.org/LAW.PDF") == schema.SourceFormat.pdf


def test_format_hint_from_url_query_string_not_sniffed() -> None:
    """Точность важнее покрытия: лгущая подсказка (``?format=pdf`` у лендинга)
    опаснее отсутствующей — query/фрагмент НЕ смотрим."""
    assert format_hint_from_url("https://gov.example.org/view?format=pdf") is None


def test_format_hint_from_url_no_extension_is_none() -> None:
    assert format_hint_from_url("https://gov.example.org/document/view/8842") is None


def test_format_hint_from_url_unknown_extension_is_none() -> None:
    assert format_hint_from_url("https://gov.example.org/page.aspx") is None


# --- dedup ------------------------------------------------------------------------


def test_dedup_no_duplicates_passthrough() -> None:
    a = _candidate(raw_hash="ha", title="Doc A", issuer="Gov")
    b = _candidate(raw_hash="hb", title="Doc B", issuer="Gov")
    outcome = dedup([a, b], existing=[])
    assert outcome.fresh == [a, b]
    assert outcome.absorbed == 0


def test_dedup_matches_by_normalized_url_against_existing() -> None:
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    new_cand = _candidate(
        connector_id="manual", raw_hash="hb",
        normalized_url=normalize_url("http://EXAMPLE.gov/doc/"),
    )
    outcome = dedup([new_cand], existing=[existing_cand])
    assert outcome.fresh == []
    assert outcome.absorbed == 1
    assert existing_cand.merged_connector_ids == ["manual"]  # type: ignore[attr-defined]


def test_dedup_matches_by_issuer_title_date_when_no_url_key() -> None:
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        title="AI Governance Framework", issuer="MinDigital", doc_date=dt.date(2026, 1, 1),
    )
    new_cand = _candidate(
        connector_id="manual", raw_hash="hb",
        title="ai-governance framework", issuer="MinDigital", doc_date=dt.date(2026, 1, 1),
    )
    outcome = dedup([new_cand], existing=[existing_cand])
    assert outcome.fresh == []
    assert outcome.absorbed == 1


def test_dedup_rejected_existing_not_resurrected() -> None:
    """Отклонённый триажем кандидат (rejected_reason) не должен ре-инжектиться как свежий."""
    rejected = _candidate(
        connector_id="agora", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
        rejected_reason="вне обеих осей",
    )
    new_cand = _candidate(
        connector_id="manual", raw_hash="hb",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    outcome = dedup([new_cand], existing=[rejected])
    assert outcome.fresh == []
    assert outcome.absorbed == 1
    assert rejected.rejected_reason == "вне обеих осей"  # неприкосновенно


def test_dedup_never_overwrites_existing_fields() -> None:
    existing_cand = _candidate(raw_hash="ha", title="Original Title", issuer="Gov")
    dup = _candidate(
        connector_id="search:test",
        raw_hash="hb", title="original title", issuer="Gov",
    )
    dedup([dup], existing=[existing_cand])
    assert existing_cand.title == "Original Title"


def test_dedup_within_new_batch_first_wins() -> None:
    a = _candidate(connector_id="manual", raw_hash="ha", title="Same Doc", issuer="Gov")
    b = _candidate(
        connector_id="search:test",
        raw_hash="hb", title="same doc", issuer="Gov",
    )
    outcome = dedup([a, b], existing=[])
    assert outcome.fresh == [a]
    assert outcome.absorbed == 1
    assert a.merged_connector_ids == ["search:test"]  # type: ignore[attr-defined]


def test_dedup_same_connector_rediscovery_does_not_self_reference() -> None:
    """Тот же коннектор повторно нашёл тот же URL — не плодим self-referential provenance."""
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    dup_same_connector = _candidate(
        connector_id="agora", raw_hash="hb",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    outcome = dedup([dup_same_connector], existing=[existing_cand])
    assert outcome.fresh == []
    assert outcome.absorbed == 1
    assert getattr(existing_cand, "merged_connector_ids", None) is None


def test_legacy_content_hash_key_is_absorbed_as_extra_and_ignored() -> None:
    """spec triage-intake-hardening §4: поле снято со схемы (писателей не было ни
    одного), но старые шарды с ключом ``content_hash`` обязаны грузиться — ``extra=
    "allow"``. Дедупом значение больше не участвует: два кандидата с одинаковым
    легаси-хэшем и без общих url/issuer+title остаются РАЗНЫМИ."""
    existing_cand = _candidate(raw_hash="ha", content_hash="deadbeef")
    dup = _candidate(connector_id="agora", raw_hash="hb", content_hash="deadbeef")
    assert existing_cand.content_hash == "deadbeef"  # type: ignore[attr-defined]  # extra="allow"
    outcome = dedup([dup], existing=[existing_cand])
    assert outcome.fresh == [dup]
    assert outcome.absorbed == 0


# --- DedupOutcome.absorptions / alternate_source_urls (spec discovery-acquire-
# seam-hardening §5, Г4) ---


def test_dedup_absorptions_pairs_dup_with_real_absorber() -> None:
    """``absorptions`` несёт пары (дубль, поглотитель) — честный ответ вызывающей
    стороне (``inject``), а не только счётчик."""
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    dup = _candidate(
        connector_id="manual", raw_hash="hb",
        normalized_url=normalize_url("https://example.gov/doc"),
    )
    outcome = dedup([dup], existing=[existing_cand])
    assert outcome.absorptions == [(dup, existing_cand)]


def test_merge_provenance_accumulates_alternate_source_url_on_mismatch() -> None:
    """Живой сценарий (Г4): зеркало WAF-заблокированного первоисточника поглощается
    стратегией issuer+title+date — рабочий URL зеркала не должен теряться."""
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        title="AI Governance Framework", issuer="MinDigital", doc_date=dt.date(2026, 1, 1),
        source_url="https://blocked.gov/doc.pdf",
        normalized_url=normalize_url("https://blocked.gov/doc.pdf"),
    )
    mirror = _candidate(
        connector_id="manual", raw_hash="hb",
        title="ai-governance framework", issuer="MinDigital", doc_date=dt.date(2026, 1, 1),
        source_url="https://mirror.example.org/doc.pdf",
        normalized_url=normalize_url("https://mirror.example.org/doc.pdf"),
    )
    dedup([mirror], existing=[existing_cand])
    assert existing_cand.alternate_source_urls == ["https://mirror.example.org/doc.pdf"]  # type: ignore[attr-defined]


def test_merge_provenance_no_alternate_when_url_matches() -> None:
    """Поглощение по самому URL (стратегия 1) — дубль и поглотитель уже несут ОДИН
    URL, копить его же в alternate_source_urls незачем."""
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        normalized_url=normalize_url("https://example.gov/doc"),
        source_url="https://example.gov/doc",
    )
    dup = _candidate(
        connector_id="manual", raw_hash="hb",
        normalized_url=normalize_url("http://EXAMPLE.gov/doc/"),
        source_url="http://EXAMPLE.gov/doc/",
    )
    dedup([dup], existing=[existing_cand])
    assert getattr(existing_cand, "alternate_source_urls", None) is None


def test_merge_provenance_deduplicates_repeated_alternate_url() -> None:
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        title="Doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://blocked.gov/doc.pdf",
        normalized_url=normalize_url("https://blocked.gov/doc.pdf"),
    )
    mirror_a = _candidate(
        connector_id="manual", raw_hash="hb",
        title="doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://mirror.example.org/doc.pdf",
        normalized_url=normalize_url("https://mirror.example.org/doc.pdf"),
    )
    mirror_b = _candidate(
        connector_id="search:x", raw_hash="hc",
        title="doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://mirror.example.org/doc.pdf",
        normalized_url=normalize_url("https://mirror.example.org/doc.pdf"),
    )
    dedup([mirror_a, mirror_b], existing=[existing_cand])
    assert existing_cand.alternate_source_urls == ["https://mirror.example.org/doc.pdf"]  # type: ignore[attr-defined]


def test_dedup_candidate_without_any_key_is_never_absorbed() -> None:
    """Кандидат без ОБОИХ ключей (ни URL, ни пары issuer+title) не имеет идентичности
    для dedup — приходит «свежим» даже против побитово такого же.

    Свойство ИСХОДНОГО дизайна (прежний линейный ``_find_match`` возвращал
    None на тех же входах), не следствие индексации — найдено property-тестом
    (``test_dedup_properties``) и запинено здесь, чтобы поведение было видимым. Схема
    такое допускает (всё, кроме connector_id/retrieved_at/raw_hash, опционально), но
    реальные каналы его не порождают: ``inject`` требует url+title, registry-коннекторы
    дают native-URL. ``raw_hash`` (идентичность для worksheet/apply) СТРАТЕГИЕЙ dedup
    сознательно не является — набор ключей задан чартером §4.4.
    """
    keyless = _candidate(raw_hash="ha")
    twin = _candidate(connector_id="agora", raw_hash="hb")

    outcome = dedup([twin], existing=[keyless])

    assert outcome.fresh == [twin]
    assert outcome.absorbed == 0
