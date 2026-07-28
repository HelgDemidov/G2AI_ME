"""Тесты discovery/dedup.py: normalize_url/normalized_title + кросс-коннекторный merge
(spec discovery-core §3)."""
from __future__ import annotations

import ast
import datetime as dt

from core import schema
from core.env import REPO_ROOT
from discovery.dedup import (
    _MERGE_EXEMPT,
    dedup,
    format_hint_from_url,
    normalize_url,
    normalized_title,
)


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


def test_dedup_matches_by_url_against_existing() -> None:
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        source_url="https://example.gov/doc",
    )
    new_cand = _candidate(
        connector_id="manual", raw_hash="hb",
        source_url="http://EXAMPLE.gov/doc/",
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
        source_url="https://example.gov/doc",
        rejected_reason="вне обеих осей",
    )
    new_cand = _candidate(
        connector_id="manual", raw_hash="hb",
        source_url="https://example.gov/doc",
    )
    outcome = dedup([new_cand], existing=[rejected])
    assert outcome.fresh == []
    assert outcome.absorbed == 1
    assert rejected.rejected_reason == "вне обеих осей"  # неприкосновенно


def test_dedup_foreign_source_never_overwrites_existing_fields() -> None:
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
        source_url="https://example.gov/doc",
    )
    dup_same_connector = _candidate(
        connector_id="agora", raw_hash="hb",
        source_url="https://example.gov/doc",
    )
    outcome = dedup([dup_same_connector], existing=[existing_cand])
    assert outcome.fresh == []
    assert outcome.absorbed == 1
    assert getattr(existing_cand, "merged_connector_ids", None) is None


def test_legacy_keys_load_as_extra_and_are_ignored() -> None:
    """Старые шарды с ключами ``content_hash``/``normalized_url`` обязаны грузиться
    (``extra="allow"``), но дедупом их значения не участвуют: ключ URL считается на лету
    из ``source_url``, а идентичность записи в источнике — из ``connector_id``/``native_id``."""
    existing_cand = _candidate(
        raw_hash="ha", content_hash="deadbeef", normalized_url="https://example.gov/doc"
    )
    dup = _candidate(
        connector_id="agora", raw_hash="hb",
        content_hash="deadbeef", normalized_url="https://example.gov/doc",
    )
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
        source_url="https://example.gov/doc",
    )
    dup = _candidate(
        connector_id="manual", raw_hash="hb",
        source_url="https://example.gov/doc",
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
    )
    mirror = _candidate(
        connector_id="manual", raw_hash="hb",
        title="ai-governance framework", issuer="MinDigital", doc_date=dt.date(2026, 1, 1),
        source_url="https://mirror.example.org/doc.pdf",
    )
    dedup([mirror], existing=[existing_cand])
    assert existing_cand.alternate_source_urls == ["https://mirror.example.org/doc.pdf"]  # type: ignore[attr-defined]


def test_merge_provenance_no_alternate_when_url_matches() -> None:
    """Поглощение по самому URL (стратегия 1) — дубль и поглотитель уже несут ОДИН
    URL, копить его же в alternate_source_urls незачем."""
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        source_url="https://example.gov/doc",
    )
    dup = _candidate(
        connector_id="manual", raw_hash="hb",
        source_url="http://EXAMPLE.gov/doc/",
    )
    dedup([dup], existing=[existing_cand])
    assert getattr(existing_cand, "alternate_source_urls", None) is None


def test_merge_provenance_deduplicates_repeated_alternate_url() -> None:
    existing_cand = _candidate(
        connector_id="agora", raw_hash="ha",
        title="Doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://blocked.gov/doc.pdf",
    )
    mirror_a = _candidate(
        connector_id="manual", raw_hash="hb",
        title="doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://mirror.example.org/doc.pdf",
    )
    mirror_b = _candidate(
        connector_id="search:x", raw_hash="hc",
        title="doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://mirror.example.org/doc.pdf",
    )
    dedup([mirror_a, mirror_b], existing=[existing_cand])
    assert existing_cand.alternate_source_urls == ["https://mirror.example.org/doc.pdf"]  # type: ignore[attr-defined]


# --- недостоверный URL и идентичность записи в источнике (spec candidate-identity-
# hardening §2) ---


def test_suspect_url_does_not_collapse_different_documents() -> None:
    """Замер на боевом снапшоте OECD: один ``website`` у записей с РАЗНЫМИ заголовками —
    23 группы, 25 кандидатов поглощалось до триажа. Метка ``suspect`` снимает адрес с
    роли ключа, и документы доходят до очереди."""
    norway = _candidate(
        connector_id="oecd", raw_hash="ha", native_id="1",
        title="Joint AI plan", issuer="Norway",
        source_url="https://oecd.example/shared", url_provenance="suspect",
    )
    ukraine = _candidate(
        connector_id="oecd", raw_hash="hb", native_id="2",
        title="Operational Plan for the WINWIN Strategy", issuer="Ukraine",
        source_url="https://oecd.example/shared", url_provenance="suspect",
    )

    outcome = dedup([norway, ukraine], existing=[])

    assert outcome.fresh == [norway, ukraine]
    assert outcome.absorbed == 0


def test_suspect_url_candidates_do_not_double_on_rerun() -> None:
    """Снятие ключа не должно оборачиваться задвоением: те же записи на следующем
    харвесте ловятся парой issuer+title."""
    first = _candidate(
        connector_id="oecd", raw_hash="ha", native_id="1",
        title="Joint AI plan", issuer="Norway",
        source_url="https://oecd.example/shared", url_provenance="suspect",
    )
    again = first.model_copy(deep=True, update={"raw_hash": "hb"})

    outcome = dedup([again], existing=[first])

    assert outcome.fresh == []
    assert outcome.absorbed == 1


def test_keyless_candidate_absorbed_by_source_identity() -> None:
    """§23 бэклога: кандидат без обоих ключей раньше приходил «свежим» каждый прогон.
    Идентичность записи в источнике (``connector_id`` + ``native_id``) закрывает класс."""
    existing_cand = _candidate(connector_id="oecd", raw_hash="ha", native_id="42")
    same_record_again = _candidate(connector_id="oecd", raw_hash="hb", native_id="42")

    outcome = dedup([same_record_again], existing=[existing_cand])

    assert outcome.fresh == []
    assert outcome.absorbed == 1


def test_keyless_source_identity_survives_record_edit() -> None:
    """Ключ — ``native_id``, а не ``raw_hash``: правка записи в источнике (``updatedAt``
    и т.п.) меняет дайджест описания, но не идентификатор источника. Замер: на
    ``raw_hash`` совпадение обнулялось 749/749."""
    before = _candidate(connector_id="oecd", raw_hash="ha", native_id="42")
    after_edit = _candidate(connector_id="oecd", raw_hash="hz", native_id="42")

    assert dedup([after_edit], existing=[before]).absorbed == 1


def test_keyless_candidates_from_different_sources_stay_distinct() -> None:
    existing_cand = _candidate(connector_id="oecd", raw_hash="ha", native_id="42")
    other_source = _candidate(connector_id="agora", raw_hash="hb", native_id="42")

    outcome = dedup([other_source], existing=[existing_cand])

    assert outcome.fresh == [other_source]
    assert outcome.absorbed == 0


def test_source_identity_is_not_a_fallback_for_missed_url() -> None:
    """⚠ Живой репро при прототипировании: у snowball ``native_id`` — «документ#канал»
    (227 кандидатов на 55 значений), поэтому стратегия 3 применяется ТОЛЬКО к кандидату
    без обоих ключей. Кандидат с достоверным URL, который ни с чем не совпал, — НОВЫЙ
    документ; фолбэк схлопнул бы разные ссылки одного документа в одну."""
    first_link = _candidate(
        connector_id="snowball", raw_hash="ha", native_id="me-law-2025#p3",
        source_url="https://a.example/one.pdf",
    )
    second_link = _candidate(
        connector_id="snowball", raw_hash="hb", native_id="me-law-2025#p3",
        source_url="https://b.example/two.pdf",
    )

    outcome = dedup([second_link], existing=[first_link])

    assert outcome.fresh == [second_link]
    assert second_link.source_url == "https://b.example/two.pdf"


# --- слияние по отношению источников (spec candidate-identity-hardening §3) ---


def test_merge_own_source_updates_what_it_produced() -> None:
    """Переобнаружение своим источником авторитетнее прежнего снимка: обновляется всё,
    что источник произвёл, включая ПОНИЖЕНИЕ доверия к адресу."""
    existing_cand = _candidate(
        connector_id="oecd", raw_hash="ha", native_id="42",
        title="Draft strategy", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://oecd.example/doc", url_provenance="stated",
        native_summary="old abstract",
    )
    refreshed = _candidate(
        connector_id="oecd", raw_hash="hb", native_id="42",
        title="Draft strategy", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        source_url="https://oecd.example/doc", url_provenance="suspect",
        native_summary="new abstract",
    )

    dedup([refreshed], existing=[existing_cand])

    assert existing_cand.native_summary == "new abstract"
    assert existing_cand.url_provenance is schema.UrlProvenance.suspect


def test_merge_foreign_source_fills_empty_only() -> None:
    existing_cand = _candidate(
        connector_id="oecd", raw_hash="ha",
        title="Doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        native_summary="own abstract", language=None,
    )
    other = _candidate(
        connector_id="agora", raw_hash="hb",
        title="doc", issuer="Gov", doc_date=dt.date(2026, 1, 1),
        native_summary="foreign abstract", language="en",
    )

    dedup([other], existing=[existing_cand])

    assert existing_cand.native_summary == "own abstract"  # заполненное не переписывается
    assert existing_cand.language == "en"  # пустое заполняется
    assert existing_cand.merged_connector_ids == ["agora"]  # type: ignore[attr-defined]


def test_merge_own_source_does_not_erase_field_it_stopped_producing() -> None:
    """``None`` у дубля означает «источник этого не сказал», а не «значения нет»."""
    existing_cand = _candidate(
        connector_id="oecd", raw_hash="ha", native_id="42", matched_query="ai strategy"
    )
    quiet = _candidate(connector_id="oecd", raw_hash="hb", native_id="42")

    dedup([quiet], existing=[existing_cand])

    assert existing_cand.matched_query == "ai strategy"


def test_merge_never_carries_curated_state_from_dup() -> None:
    """``_MERGE_EXEMPT``: без него дубль с ``rejected_reason`` пометил бы поглотителя
    отклонённым (найдено property-тестом). В проде такой вход сегодня невозможен —
    правило обязано держаться и на входе, который так не выглядит."""
    existing_cand = _candidate(connector_id="oecd", raw_hash="ha", native_id="42")
    poisoned = _candidate(
        connector_id="oecd", raw_hash="hb", native_id="42",
        rejected_reason="вне обеих осей", rejected_kind="unacquirable",
        admitted_as="me-other-doc-2026", probe_finding="blocked: WAF",
    )

    dedup([poisoned], existing=[existing_cand])

    assert existing_cand.rejected_reason is None
    assert existing_cand.rejected_kind is None
    assert existing_cand.admitted_as is None
    assert existing_cand.probe_finding is None
    assert existing_cand.raw_hash == "ha"  # идентичность записи тоже неприкосновенна


def test_merge_own_source_move_keeps_previous_address() -> None:
    """Свой источник перенёс документ: новый адрес становится основным, прежний уходит
    в ``alternate_source_urls`` — вытесняемый адрес не теряется ни в одном направлении."""
    existing_cand = _candidate(
        connector_id="oecd", raw_hash="ha", native_id="42",
        title="Doc", issuer="Gov", source_url="https://old.example/doc",
    )
    moved = _candidate(
        connector_id="oecd", raw_hash="hb", native_id="42",
        title="Doc", issuer="Gov", source_url="https://new.example/doc",
    )

    dedup([moved], existing=[existing_cand])

    assert existing_cand.source_url == "https://new.example/doc"
    assert existing_cand.alternate_source_urls == ["https://old.example/doc"]  # type: ignore[attr-defined]


def test_merge_filling_empty_url_is_not_an_alternate() -> None:
    """Поглотитель без адреса: поле просто заполняется, вытеснять нечего."""
    existing_cand = _candidate(
        connector_id="oecd", raw_hash="ha", title="Doc", issuer="Gov", source_url=None
    )
    with_url = _candidate(
        connector_id="agora", raw_hash="hb", title="doc", issuer="Gov",
        source_url="https://found.example/doc",
    )

    dedup([with_url], existing=[existing_cand])

    assert existing_cand.source_url == "https://found.example/doc"
    assert getattr(existing_cand, "alternate_source_urls", None) is None


def test_dedup_exempt_matches_unproduced_fields() -> None:
    """AST-гейт: поле ``CandidateRecord``, которого не выставляет НИ ОДИН производитель
    кандидатов, ОБЯЗАНО быть в ``_MERGE_EXEMPT``.

    Иначе слияние протащило бы курируемое/машинное состояние из дубля в поглотителя, и
    заметить это можно было бы только на боевых данных. Список — то, чего этот слой в
    остальном избегает (природа поля из типа не выводится), поэтому его состав проверяется
    гейтом, а не дисциплиной."""
    sources = [REPO_ROOT / "pipeline" / "scripts" / "discovery" / "manual.py"]
    connectors_dir = REPO_ROOT / "pipeline" / "scripts" / "discovery" / "connectors"
    sources += sorted(p for p in connectors_dir.glob("*.py") if p.name != "__init__.py")

    produced: set[str] = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "CandidateRecord":
                produced.update(kw.arg for kw in node.keywords if kw.arg)
            elif name == "model_validate" and node.args and isinstance(node.args[0], ast.Dict):
                for key in node.args[0].keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        produced.add(key.value)

    assert "title" in produced and "source_url" in produced  # AST-сборка вообще сработала
    unproduced = set(schema.CandidateRecord.model_fields) - produced
    assert unproduced <= set(_MERGE_EXEMPT), (
        f"поля без производителя, не защищённые _MERGE_EXEMPT: {sorted(unproduced - set(_MERGE_EXEMPT))}"
    )
