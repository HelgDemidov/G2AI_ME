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
    assert cand.connector_id == "manual"  # архетип канала — грамматика id, отдельного поля нет
    loaded = store.load(tmp_path)
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
    assert cand.matched_query == "national ai strategy small state"


def test_inject_duplicate_url_is_noop(tmp_path: Path) -> None:
    manual.inject(
        url="https://gov.example.org/a.pdf", title="T", issuer="I", language="en", root=tmp_path
    )
    cand2, is_new2 = manual.inject(
        url="https://gov.example.org/a.pdf", title="T", issuer="I", language="en", root=tmp_path
    )
    assert is_new2 is False
    assert len(store.load(tmp_path)) == 1


def test_inject_duplicate_of_rejected_reports_reason(tmp_path: Path) -> None:
    cand, _ = manual.inject(
        url="https://gov.example.org/a.pdf", title="T", issuer="I", language="en", root=tmp_path
    )
    all_cands = store.load(tmp_path)
    all_cands[0].rejected_reason = "вне обеих осей"
    store.save(all_cands, tmp_path)

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


def test_inject_mirror_absorbed_by_strategy_two_reports_real_absorber_and_reason(
    tmp_path: Path,
) -> None:
    """Регресс репро D аудита (spec discovery-acquire-seam-hardening §5, Г4): дубль
    поглощён стратегией 2 (issuer+title+date), а НЕ по URL-паре — старый код искал
    поглотителя ТОЛЬКО по ``(normalized_url, supersedes)``, находил ``None`` и
    возвращал СВЕЖЕГО кандидата; куратор видел «уже есть» без причины отказа, даже
    не узнавая, что попал в отклонённого.

    Оба inject идут КАНАЛОМ ``manual``, то есть «свой источник» (spec candidate-identity-
    hardening §3): куратор только что назвал рабочее зеркало, оно и становится адресом
    кандидата, а заблокированный первоисточник уходит в ``alternate_source_urls`` — не
    теряется нигде (Г4). Побочно это чинит недобываемого по-настоящему: популяция (c)
    recheck пробует ``source_url``, и теперь она пробует живой адрес, а не тот, о который
    уже разбилась лестница."""
    manual.inject(
        url="https://blocked.gov/law.pdf", title="Registration Law",
        issuer="Ministry", language="en", date=dt.date(2026, 1, 1), root=tmp_path,
    )
    all_cands = store.load(tmp_path)
    all_cands[0].rejected_reason = "WAF blocks every rung"
    all_cands[0].rejected_kind = schema.RejectionKind.unacquirable
    store.save(all_cands, tmp_path)

    mirror_cand, is_new = manual.inject(
        url="https://mirror.example.org/law.pdf", title="registration law",
        issuer="Ministry", language="en", date=dt.date(2026, 1, 1), root=tmp_path,
    )

    assert is_new is False
    assert mirror_cand.rejected_reason == "WAF blocks every rung"  # реальный поглотитель, не свежий
    assert mirror_cand.raw_hash == all_cands[0].raw_hash
    absorber = store.load(tmp_path)[0]
    assert absorber.source_url == "https://mirror.example.org/law.pdf"
    assert absorber.alternate_source_urls == ["https://blocked.gov/law.pdf"]  # type: ignore[attr-defined]
    assert absorber.rejected_kind is schema.RejectionKind.unacquirable  # отказ не тронут


# --- pending_candidates / render_worksheet (spec §3) ---


def _candidate(**overrides: object) -> schema.CandidateRecord:
    data: dict[str, object] = {
        "connector_id": "manual",
        "retrieved_at": "2026-07-21",
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


def _record(**overrides: object) -> schema.SourceRecord:
    data = valid_record()
    data.update(overrides)
    return schema.SourceRecord.model_validate(data)


def test_pending_candidates_excludes_admitted_by_doc_id() -> None:
    rec = _record()
    cand = _candidate(admitted_as=rec.id)
    assert manual.pending_candidates([cand], [rec]) == []


def test_pending_candidates_url_match_alone_does_not_hide() -> None:
    """spec candidate-identity-hardening §1: реконсиляция идёт ПО ШТАМПУ, не по адресу.

    Осознанная смена семантики: раньше кандидат гасился совпадением ``source_url`` с
    любой записью реестра. Теперь совпадение адреса само по себе ничего не значит —
    гасит только собственный промоушен. Замер на боевом store: расхождение затрагивало
    ровно 6 кандидатов, и все шесть были теми самыми промоутнутыми (§5 спека штампует
    их разово).
    """
    rec = _record(source_url="https://gov.example.org/a.pdf")
    cand = _candidate(source_url="https://gov.example.org/a.pdf")
    assert manual.pending_candidates([cand], [rec]) == [cand]


def test_pending_candidates_stale_stamp_returns_candidate() -> None:
    """Самовосстановление: удалили папку документа (документированный путь исправления
    ошибки допуска) -> id исчез из реестра -> кандидат вернулся в очередь."""
    cand = _candidate(admitted_as="me-deleted-doc-2026")
    assert manual.pending_candidates([cand], []) == [cand]


def test_admitted_ids_is_shared_reconciliation_primitive() -> None:
    """``admitted_ids`` — публичный примитив, общий для ОБЕИХ очередей слоя: расхождения
    двух копий реконсиляции (класс дефекта, чиненный PR #54) физически негде завести."""
    rec = _record()
    assert manual.admitted_ids([rec]) == {rec.id}
    admitted = _candidate(admitted_as=rec.id, rejected_kind="unacquirable", rejected_reason="WAF")
    assert manual.pending_candidates([admitted], [rec]) == []
    assert manual.unacquirable_candidates([admitted], [rec]) == []


def test_render_worksheet_includes_header_and_row() -> None:
    cand = _candidate(jurisdiction="me", doc_date="2026-03-01", native_tags=["ai-governance"])
    text = manual.render_worksheet([cand])
    assert "raw_hash" in text and "relations" in text and "source_format" in text
    assert cand.raw_hash[:12] in text
    assert "me" in text
    assert "2026-03-01" in text
    assert "ai-governance" in text


def test_render_worksheet_header_carries_decision_format_conventions() -> None:
    """Шапка — самодостаточный формат решений: id первым, action последним, контент EN,
    rationale = только факторы релевантности (не пересказ summary)."""
    text = manual.render_worksheet([])
    assert "- id: me-example-strategy-2026" in text  # id — первый ключ примера
    assert "action: admit" in text and "action: reject" in text
    assert "АНГЛИЙСКИЙ" in text
    assert "rationale" in text and "summary" in text


def test_render_worksheet_empty_pending_still_has_header() -> None:
    text = manual.render_worksheet([])
    assert "Триаж-worksheet" in text
    assert "raw_hash" in text


# --- missing-колонка + total-аннотация (spec triage-intake-hardening §1/§2) ---


def test_missing_conditional_fields_flags_absent_fields() -> None:
    cand = _candidate(issuer=None, language=None)
    assert manual._missing_conditional_fields(cand) == ["issuer", "language"]


def test_missing_conditional_fields_empty_when_all_present() -> None:
    cand = _candidate()
    assert manual._missing_conditional_fields(cand) == []


def test_missing_conditional_fields_decision_override_satisfies() -> None:
    """Override в решении закрывает недостачу — та же семантика, что admit-дверь."""
    cand = _candidate(issuer=None)
    assert manual._missing_conditional_fields(cand, {"issuer": "Overridden"}) == []


def test_render_worksheet_missing_column_shows_absent_fields() -> None:
    cand = _candidate(issuer=None, language=None)
    text = manual.render_worksheet([cand])
    row = next(line for line in text.splitlines() if cand.raw_hash[:12] in line)
    assert row.rstrip().endswith("issuer, language |")


def test_render_worksheet_missing_column_empty_when_all_present() -> None:
    cand = _candidate()
    text = manual.render_worksheet([cand])
    row = next(line for line in text.splitlines() if cand.raw_hash[:12] in line)
    assert row.rstrip().endswith("|  |")  # пустая ячейка missing


def test_render_worksheet_no_total_line_without_total() -> None:
    text = manual.render_worksheet([_candidate()])
    assert "Показано" not in text


def test_render_worksheet_total_line_appears_when_truncated() -> None:
    text = manual.render_worksheet([_candidate()], total=42)
    assert "Показано 1 из 42 ждущих." in text


def test_render_worksheet_no_total_line_when_nothing_truncated() -> None:
    """Регресс дефекта, найденного `/review` PR #56: строка печаталась при ЛЮБОМ отборе,
    поэтому `--connector oecd` без `--limit` давал «Показано 717 из 717 ждущих» — шум,
    от которого строку перестают читать ровно тогда, когда она наконец что-то значит.
    Прежний тест проверял только положительный случай (1 из 42) и дефект не ловил."""
    text = manual.render_worksheet([_candidate()], total=1)
    assert "Показано" not in text


def test_render_worksheet_truncated_unacquirable_section_is_announced() -> None:
    """Усечение ЛЮБОЙ секции обязано быть объявлено — молчаливое читалось бы как
    «это вся очередь»."""
    unacq = _candidate(raw_hash="c" * 64, rejected_reason="WAF", rejected_kind="unacquirable")
    text = manual.render_worksheet([_candidate()], [unacq], total=1, unacquirable_total=9)
    assert "Показано 1 из 9 недобываемых." in text
    assert text.count("Показано") == 1  # ждущие не усечены — про них строки нет


def test_render_worksheet_announces_both_sections_when_both_truncated() -> None:
    unacq = _candidate(raw_hash="c" * 64, rejected_reason="WAF", rejected_kind="unacquirable")
    text = manual.render_worksheet([_candidate()], [unacq], total=5, unacquirable_total=9)
    assert "Показано 1 из 5 ждущих." in text
    assert "Показано 1 из 9 недобываемых." in text


# --- worksheet_payload / --format json (spec triage-intake-hardening §2) ---


def test_worksheet_payload_contract_vocab_matches_live_source() -> None:
    """`vocab`/`defaults` эмитируются из живых источников истины, не литералом —
    иначе дрейф между JSON-контрактом и `vocab_axes.yaml`/картой authority."""
    payload = manual.worksheet_payload([])
    assert payload["contract"]["vocab"]["axes"] == sorted(schema.load_vocab("axes"))
    assert payload["contract"]["defaults"]["authority_by_doc_type"] == dict(
        manual._AUTHORITY_BY_DOC_TYPE
    )


def test_worksheet_payload_shown_and_total() -> None:
    payload = manual.worksheet_payload([_candidate()], total=42)
    assert payload["shown"] == 1
    assert payload["pending_total"] == 42


def test_worksheet_payload_total_none_defaults_to_shown() -> None:
    payload = manual.worksheet_payload([_candidate(), _candidate(raw_hash="b" * 64)])
    assert payload["pending_total"] == payload["shown"] == 2


def test_worksheet_payload_candidate_carries_missing_and_full_raw_hash() -> None:
    cand = _candidate(issuer=None)
    payload = manual.worksheet_payload([cand])
    row = payload["candidates"][0]
    assert row["raw_hash"] == cand.raw_hash  # полный хэш, не усечённый (машинный формат)
    assert row["missing"] == ["issuer"]


def test_worksheet_payload_unacquirable_included() -> None:
    unacq = _candidate(rejected_reason="WAF", rejected_kind="unacquirable")
    payload = manual.worksheet_payload([], [unacq])
    assert len(payload["unacquirable"]) == 1
    assert payload["unacquirable"][0]["raw_hash"] == unacq.raw_hash


def test_worksheet_md_and_json_share_same_raw_hash_set() -> None:
    """Оба формата строятся из одного worksheet_payload — набор кандидатов не может
    разъехаться между `--format md` и `--format json`."""
    pending = [_candidate(), _candidate(raw_hash="b" * 64, issuer=None)]
    unacq = [_candidate(raw_hash="c" * 64, rejected_reason="WAF", rejected_kind="unacquirable")]

    md_text = manual.render_worksheet(pending, unacq)
    payload = manual.worksheet_payload(pending, unacq)

    md_hashes = {c.raw_hash[:12] for c in pending} | {c.raw_hash[:12] for c in unacq}
    json_hashes = {row["raw_hash"][:12] for row in payload["candidates"]} | {
        row["raw_hash"][:12] for row in payload["unacquirable"]
    }
    assert json_hashes == md_hashes
    assert all(h in md_text for h in md_hashes)


# --- экранирование `|` в ячейках worksheet (spec discovery-acquire-seam-hardening §12) ---


def test_render_worksheet_escapes_pipe_in_pending_row() -> None:
    """Недоверенный title/issuer (реестры, анкоры снежного кома) может нести `|`
    естественно — без экранирования строка таблицы рвётся по колонкам."""
    cand = _candidate(title="AI Act | Draft", issuer="Ministry | Agency")
    text = manual.render_worksheet([cand])
    lines = [
        line for line in text.splitlines()
        if line.startswith("|") and cand.raw_hash[:12] in line
    ]
    assert len(lines) == 1
    # число колонок сохранено: экранированный `|` не создаёт лишних разделителей
    assert lines[0].count(" | ") == 11  # 12 колонок таблицы pending (+ doc_year, 2026-07-28)


def test_render_worksheet_escapes_pipe_in_unacquirable_row() -> None:
    cand = _candidate(
        title="AI Act | Draft",
        rejected_reason="WAF", rejected_kind="unacquirable",
        probe_finding="acquirable: HTTP 200 | ok",
    )
    text = manual.render_worksheet([], [cand])
    lines = [
        line for line in text.splitlines()
        if line.startswith("|") and cand.raw_hash[:12] in line
    ]
    assert len(lines) == 1
    assert lines[0].count(" | ") == 6  # 7 колонок таблицы unacquirable


def test_render_worksheet_flattens_newline_in_cell() -> None:
    """Найдено живьём на боевом store (2026-07-27): два кандидата ``aiforgood`` несут
    ``\\n`` внутри ``title`` — титул стандарта ITU-T, перенесённый в исходном каталоге.
    Строка таблицы разваливалась на ДВЕ физические, что хуже неэкранированного ``|``:
    тот добавляет колонку, а разрыв строки делает из хвоста ячейки псевдо-строку.
    Экранировать перевод строки в GFM-таблице нечем — ячейка однострочна по грамматике."""
    cand = _candidate(title="ITU-T FG-AI4A WG Roadmap:\nStandardization gaps\r\nand roadmap")
    text = manual.render_worksheet([cand])

    rows = [line for line in text.splitlines() if line.startswith("| ") and line.endswith(" |")]
    body = [line for line in rows if cand.raw_hash[:12] in line]
    assert len(body) == 1  # одна физическая строка, а не три
    assert body[0].count(" | ") == 11  # 12 колонок таблицы pending (+ doc_year, 2026-07-28)
    assert "Standardization gaps" in body[0]  # текст не потерян, только схлопнут


# --- authority-map ⊆ vocab_doc_types (spec discovery-acquire-seam-hardening §12) ---


def test_authority_by_doc_type_is_subset_of_doc_type_vocab() -> None:
    """Дрейф при переименовании термина словаря ловится тестом, а не молчаливым
    «нет дефолта» — полного покрытия словаря НЕ требует (честная деградация
    «задайте authority явно» остаётся штатной для новых терминов)."""
    assert set(manual._AUTHORITY_BY_DOC_TYPE) <= schema.load_vocab("doc_types")


# --- apply_decisions (spec §4) ---


def _admit_decision(raw_hash: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "raw_hash": raw_hash,
        "action": "admit",
        "id": "me-example-strategy-2026",
        "entity_id": "me",
        "track": "target-entity",
        "issuer_type": "government",
        "geo_scope": "national",
        "doc_type": "national_strategy",
        "authority": "soft_law",
        "admission": {
            "axis": "agentic_g2ai",
            "rationale": "matches axis",
        },
    }
    data.update(overrides)
    return data


def test_apply_reject_sets_rejected_reason(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="a" * 64)
    store.save([cand], tmp_path)

    summary = manual.apply_decisions(
        [{"raw_hash": "a" * 64, "action": "reject", "reason": "вне обеих осей"}], root=tmp_path
    )
    assert summary.errors == []
    reloaded = store.load(tmp_path)
    assert reloaded[0].rejected_reason == "вне обеих осей"


def test_apply_reject_does_not_overwrite_existing_reason(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="a" * 64, rejected_reason="первая причина")
    store.save([cand], tmp_path)

    summary = manual.apply_decisions(
        [{"raw_hash": "a" * 64, "action": "reject", "reason": "новая причина"}], root=tmp_path
    )
    assert summary.errors == []
    reloaded = store.load(tmp_path)
    assert reloaded[0].rejected_reason == "первая причина"


def test_apply_admit_creates_meta_yaml_at_correct_path(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert summary.errors == []
    meta_path = tmp_path / "target-entity" / "me" / "me-example-strategy-2026" / "meta.yaml"
    assert meta_path.exists()
    records = schema.load_records(tmp_path)
    assert len(records) == 1 and records[0].id == "me-example-strategy-2026"


def test_apply_admit_stamps_candidate_with_doc_id(tmp_path: Path) -> None:
    """Штамп идентичности (spec candidate-identity-hardening §1): единственная мутация
    кандидата при admit — ``admitted_as``, и именно она уводит его из очереди."""
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)

    manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    reloaded = store.load(tmp_path)
    assert len(reloaded) == 1
    assert reloaded[0].admitted_as == "me-example-strategy-2026"
    assert reloaded[0].rejected_reason is None  # отказной след apply не трогает
    assert manual.pending_candidates(reloaded, schema.load_records(tmp_path)) == []


def test_apply_admit_v2_fields_reach_meta_yaml(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)

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


def test_apply_admit_language_override_reaches_meta_yaml(tmp_path: Path) -> None:
    """spec discovery-agora §7: registry-кандидат без language (AGORA) промоутится, если
    decisions.yaml несёт language — иначе promote_candidate отказал бы (тест ниже)."""
    cand = _candidate(raw_hash="b" * 64, language=None)
    store.save([cand], tmp_path)

    decision = _admit_decision("b" * 64, language="en")
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert summary.errors == []
    rec = schema.load_records(tmp_path)[0]
    assert rec.language == "en"


def test_apply_admit_without_language_override_and_candidate_without_language_errors(
    tmp_path: Path,
) -> None:
    cand = _candidate(raw_hash="b" * 64, language=None)
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert len(summary.errors) == 1
    assert "language" in summary.errors[0].detail


@pytest.mark.parametrize(
    "field_name,override_value",
    [
        ("title", "Overridden Title"),
        ("issuer", "Overridden Gov"),
        ("source_url", "https://ex.org/override.pdf"),
    ],
)
def test_apply_admit_field_override_reaches_meta_yaml(
    tmp_path: Path, field_name: str, override_value: str
) -> None:
    """spec triage-intake-hardening §1: title/issuer/source_url override той же формы, что
    language выше — кандидат без поля промоутится, если decision несёт override."""
    cand = _candidate(raw_hash="b" * 64, **{field_name: None})
    store.save([cand], tmp_path)

    decision = _admit_decision("b" * 64, **{field_name: override_value})
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert summary.errors == []
    rec = schema.load_records(tmp_path)[0]
    assert getattr(rec, field_name) == override_value


@pytest.mark.parametrize("field_name", ["title", "issuer", "source_url"])
def test_apply_admit_without_field_override_and_candidate_without_field_errors(
    tmp_path: Path, field_name: str
) -> None:
    """Явная ошибка решения называет ИМЕННО недостающее поле и ключ, который его чинит —
    не общий ValueError из глубины promote_candidate."""
    cand = _candidate(raw_hash="b" * 64, **{field_name: None})
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert len(summary.errors) == 1
    assert f"`{field_name}`" in summary.errors[0].detail
    assert f"`{field_name}:`" in summary.errors[0].detail


# --- title_provenance: derived требует явного title (spec triage-intake-hardening §3) ---


def test_missing_flags_derived_title_despite_non_empty_value() -> None:
    """Заголовок ЕСТЬ, но это реконструкция из позиционного артефакта — в реестре он
    станет меткой узла графа, поэтому считается непригодным наравне с отсутствующим."""
    cand = _candidate(title="e Model AI Governance F ramework", title_provenance="derived")
    assert manual._missing_conditional_fields(cand) == ["title"]


def test_missing_does_not_flag_stated_or_legacy_title() -> None:
    assert manual._missing_conditional_fields(_candidate(title_provenance="stated")) == []
    assert manual._missing_conditional_fields(_candidate()) == []  # легаси: провенанс None


def test_apply_admit_derived_title_without_override_errors(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64, title="s AccelerateEstonia,", title_provenance="derived")
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert len(summary.errors) == 1
    detail = summary.errors[0].detail
    assert "derived" in detail and "`title:`" in detail
    assert "s AccelerateEstonia," in detail  # видно, ЧТО именно забраковано


def test_apply_admit_derived_title_with_override_succeeds(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64, title="s AccelerateEstonia,", title_provenance="derived")
    store.save([cand], tmp_path)

    decision = _admit_decision("b" * 64, title="Accelerate Estonia Programme")
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert summary.errors == []
    assert schema.load_records(tmp_path)[0].title == "Accelerate Estonia Programme"


def test_apply_admit_suspect_url_without_override_errors(tmp_path: Path) -> None:
    """spec triage-intake-hardening §6: адрес ЕСТЬ и выглядит рабочим, но источник отдал
    его нескольким разным документам — добыча по нему скачает чужой документ."""
    cand = _candidate(
        raw_hash="b" * 64,
        source_url="https://www.helsedirektoratet.no/rapporter/joint-ai-plan",
        url_provenance="suspect",
    )
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert len(summary.errors) == 1
    detail = summary.errors[0].detail
    assert "suspect" in detail and "`source_url:`" in detail
    assert "helsedirektoratet.no" in detail  # видно, ЧТО именно забраковано


def test_apply_admit_suspect_url_with_override_succeeds(tmp_path: Path) -> None:
    cand = _candidate(
        raw_hash="b" * 64,
        source_url="https://www.helsedirektoratet.no/rapporter/joint-ai-plan",
        url_provenance="suspect",
    )
    store.save([cand], tmp_path)

    decision = _admit_decision("b" * 64, source_url="https://sanidad.gob.es/estrategia-ia.pdf")
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert summary.errors == []
    assert schema.load_records(tmp_path)[0].source_url == "https://sanidad.gob.es/estrategia-ia.pdf"


def test_apply_incomplete_admit_reports_error_rest_of_batch_applied(tmp_path: Path) -> None:
    good = _candidate(raw_hash="b" * 64)
    bad = _candidate(raw_hash="c" * 64)
    store.save([good, bad], tmp_path)

    incomplete = _admit_decision("c" * 64)
    del incomplete["admission"]
    summary = manual.apply_decisions([_admit_decision("b" * 64), incomplete], root=tmp_path)

    assert len(summary.errors) == 1
    assert summary.errors[0].raw_hash == "c" * 64
    assert len(schema.load_records(tmp_path)) == 1  # хороший применился, плохой — нет


def test_apply_ambiguous_raw_hash_prefix_reports_error(tmp_path: Path) -> None:
    cand1 = _candidate(raw_hash="a" * 64)
    cand2 = _candidate(raw_hash="a" * 63 + "b")
    store.save([cand1, cand2], tmp_path)

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
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path, dry_run=True)
    assert summary.dry_run is True
    assert summary.errors == []
    assert schema.load_records(tmp_path) == []
    meta_path = tmp_path / "target-entity" / "me" / "me-example-strategy-2026" / "meta.yaml"
    assert not meta_path.exists()
    assert store.load(tmp_path)[0].admitted_as is None  # план не штампует


def test_apply_dry_run_reject_does_not_write(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="a" * 64)
    store.save([cand], tmp_path)

    manual.apply_decisions(
        [{"raw_hash": "a" * 64, "action": "reject", "reason": "x"}], root=tmp_path, dry_run=True
    )
    assert store.load(tmp_path)[0].rejected_reason is None


def test_resolve_candidate_rejects_short_prefix() -> None:
    with pytest.raises(ValueError, match=">=12"):
        manual._resolve_candidate("a" * 8, [_candidate(raw_hash="a" * 64)])


# --- дефолты authority/track в admit-решении (ревью 2026-07-21) ---


def _admit_no_defaults(raw_hash: str) -> dict[str, object]:
    """admit-решение БЕЗ authority/track — оба должны вывестись дефолтами."""
    d = _admit_decision(raw_hash)
    del d["authority"]
    del d["track"]
    return d


@pytest.mark.parametrize(
    "doc_type,expected_authority",
    [
        ("legislation", "binding_law"),
        ("regulation", "regulation"),
        ("report", "report"),
        ("academic_paper", "report"),
        ("guidance", "soft_law"),
        ("framework", "soft_law"),
        ("national_strategy", "soft_law"),
        ("technical_standard", "voluntary_standard"),
    ],
)
def test_apply_admit_authority_defaults_from_doc_type(
    tmp_path: Path, doc_type: str, expected_authority: str
) -> None:
    cand = _candidate(raw_hash="b" * 64, jurisdiction="me")
    store.save([cand], tmp_path)

    decision = _admit_no_defaults("b" * 64)
    decision["doc_type"] = doc_type
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert summary.errors == []
    rec = schema.load_records(tmp_path)[0]
    assert rec.authority == expected_authority


def test_apply_admit_track_defaults_me_jurisdiction_to_target_entity(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64, jurisdiction="me")
    store.save([cand], tmp_path)

    manual.apply_decisions([_admit_no_defaults("b" * 64)], root=tmp_path)
    assert schema.load_records(tmp_path)[0].track == schema.Track.target_entity


# --- source_format резолюция (spec discovery-acquire-seam-hardening §8, Г7) ------


def test_apply_admit_source_format_defaults_to_pdf_without_hint(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)
    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert summary.errors == []
    assert schema.load_records(tmp_path)[0].source_format == schema.SourceFormat.pdf


def test_apply_admit_source_format_defaults_from_candidate_hint(tmp_path: Path) -> None:
    """Подсказка кандидата замещает молчаливый дефолт "pdf", когда решение не
    указывает формат явно — эхо в сводке, той же механикой, что authority/track."""
    cand = _candidate(raw_hash="b" * 64, native_format_hint="html")
    store.save([cand], tmp_path)
    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert summary.errors == []
    assert schema.load_records(tmp_path)[0].source_format == schema.SourceFormat.html
    assert any("source_format=html" in o.detail for o in summary.outcomes)


def test_apply_admit_explicit_source_format_wins_over_hint(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64, native_format_hint="html")
    store.save([cand], tmp_path)
    decision = _admit_decision("b" * 64, source_format="docx")
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert summary.errors == []
    assert schema.load_records(tmp_path)[0].source_format == schema.SourceFormat.docx


# --- official_alt_url через admit-решение (spec discovery-acquire-seam-hardening §9, Г13) ---


def test_apply_admit_official_alt_url_reaches_meta_yaml(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)

    decision = _admit_decision("b" * 64, official_alt_url="https://mirror.example.org/doc.pdf")
    summary = manual.apply_decisions([decision], root=tmp_path)

    assert summary.errors == []
    assert schema.load_records(tmp_path)[0].official_alt_url == "https://mirror.example.org/doc.pdf"


def test_apply_admit_invalid_official_alt_url_fails_without_aborting_batch(tmp_path: Path) -> None:
    bad_cand = _candidate(raw_hash="b" * 64)
    ok_cand = _candidate(raw_hash="c" * 64)
    store.save([bad_cand, ok_cand], tmp_path)

    bad_decision = _admit_decision("b" * 64, official_alt_url="not-a-url")
    ok_decision = _admit_decision(
        "c" * 64, id="me-example-strategy-2027", official_alt_url="https://mirror.example.org/doc.pdf"
    )
    summary = manual.apply_decisions([bad_decision, ok_decision], root=tmp_path)

    assert len(summary.errors) == 1
    records = {r.id: r for r in schema.load_records(tmp_path)}
    assert "me-example-strategy-2026" not in records  # плохое решение не применилось
    assert records["me-example-strategy-2027"].official_alt_url == "https://mirror.example.org/doc.pdf"


def test_apply_admit_without_official_alt_url_is_prior_behavior(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)
    manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert schema.load_records(tmp_path)[0].official_alt_url is None


def test_render_worksheet_includes_format_hint_column() -> None:
    cand = _candidate(native_format_hint="html")
    text = manual.render_worksheet([cand])
    assert "format_hint" in text
    assert "| html |" in text


def test_load_target_entity_jurisdictions_reads_real_tracked_config() -> None:
    """pipeline/config/target_entities.yaml — настоящий трекаемый файл, не фикстура."""
    assert manual.load_target_entity_jurisdictions() == ("me",)


def test_load_target_entity_jurisdictions_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "target_entities.yaml"
    path.write_text("jurisdictions: [xx, yy]\n", encoding="utf-8")
    assert manual.load_target_entity_jurisdictions(path) == ("xx", "yy")


def test_default_track_not_hardcoded_me_falls_through_when_absent_from_config() -> None:
    """Решение куратора 2026-07-25: список конфигурируем, jurisdiction=='me' САМ ПО СЕБЕ
    в коде больше ничего не значит — только присутствие в конфиге. Инъекция другого
    списка (без 'me') доказывает отсутствие хардкода: 'me' без совпадения в списке
    падает в intl-xperience, как любая другая юрисдикция."""
    track = manual._default_track(
        "me", schema.IssuerType.government, target_entity_jurisdictions=("xx",)
    )
    assert track == schema.Track.intl_xperience


def test_default_track_uses_injected_jurisdiction_list() -> None:
    """Симметрично: юрисдикция из ИНЪЕЦИРОВАННОГО (не реального) списка триггерит
    target_entity — подтверждает, что список читается конфигурируемо, не завязан на 'me'."""
    track = manual._default_track(
        "xx", schema.IssuerType.government, target_entity_jurisdictions=("xx",)
    )
    assert track == schema.Track.target_entity


def test_apply_admit_track_defaults_think_tank_to_research_papers(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64, jurisdiction=None)
    store.save([cand], tmp_path)

    decision = _admit_no_defaults("b" * 64)
    decision.update(id="oi-example-report-2026", entity_id="oi", issuer_type="think_tank",
                    geo_scope="global", doc_type="report")
    manual.apply_decisions([decision], root=tmp_path)
    assert schema.load_records(tmp_path)[0].track == schema.Track.research_papers


def test_apply_admit_track_defaults_otherwise_to_intl_xperience(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64, jurisdiction="sg")
    store.save([cand], tmp_path)

    decision = _admit_no_defaults("b" * 64)
    decision.update(id="sg-example-framework-2026", entity_id="sg", doc_type="framework")
    manual.apply_decisions([decision], root=tmp_path)
    assert schema.load_records(tmp_path)[0].track == schema.Track.intl_xperience


def test_apply_admit_explicit_values_override_defaults(tmp_path: Path) -> None:
    """Явные authority/track всегда побеждают дефолт (кейс draft!)."""
    cand = _candidate(raw_hash="b" * 64, jurisdiction="me")
    store.save([cand], tmp_path)

    decision = _admit_no_defaults("b" * 64)
    decision["doc_type"] = "legislation"
    decision["authority"] = "draft"  # проект закона: жанр legislation, силы ещё нет
    manual.apply_decisions([decision], root=tmp_path)
    rec = schema.load_records(tmp_path)[0]
    assert rec.authority == "draft"


def test_apply_admit_unknown_doc_type_without_authority_errors(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)

    decision = _admit_no_defaults("b" * 64)
    decision["doc_type"] = "novel_genre"  # органически новый термин, карты дефолтов ещё нет
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert len(summary.errors) == 1
    assert "нет дефолта" in summary.errors[0].detail


def test_apply_admit_hidden_fields_annotation_ignored(tmp_path: Path) -> None:
    """hidden_fields — аннотация для человека, apply её не читает и не падает."""
    cand = _candidate(raw_hash="b" * 64, jurisdiction="me")
    store.save([cand], tmp_path)

    decision = _admit_no_defaults("b" * 64)
    decision["hidden_fields"] = ["authority", "track"]
    summary = manual.apply_decisions([decision], root=tmp_path)
    assert summary.errors == []
    assert len(schema.load_records(tmp_path)) == 1


def test_apply_admit_outcome_echoes_defaults(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64, jurisdiction="me")
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_no_defaults("b" * 64)], root=tmp_path)
    detail = summary.outcomes[0].detail
    assert "по дефолту" in detail
    assert "authority=soft_law" in detail
    assert "track=target-entity" in detail


def test_apply_admit_no_echo_when_all_explicit(tmp_path: Path) -> None:
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)
    assert "по дефолту" not in summary.outcomes[0].detail


def test_apply_rejects_decision_without_action_or_raw_hash(tmp_path: Path) -> None:
    """Мусорное решение (нет raw_hash / неизвестный action) — ошибка ЭТОГО решения,
    остальной батч применяется (изоляция отказов per-решение, spec §4)."""
    cand = _candidate(raw_hash="b" * 64)
    store.save([cand], tmp_path)

    summary = manual.apply_decisions(
        [
            {"action": "admit"},  # без raw_hash
            {"raw_hash": "b" * 64, "action": "postpone"},  # неизвестный action
            _admit_decision("b" * 64),
        ],
        root=tmp_path,
    )

    assert len(summary.errors) == 2
    assert all("raw_hash обязателен" in e.detail for e in summary.errors)
    assert len(schema.load_records(tmp_path)) == 1  # валидное решение отработало


def test_apply_isolates_existing_meta_conflict(tmp_path: Path) -> None:
    """Промоушен в уже занятый id (перезапись курируемого meta.yaml запрещена) — ошибка
    решения, не краш батча. Живой путь для редакций: коллизия id при admit новой
    редакции ловится здесь, а не порчей существующей записи."""
    first, second = _candidate(raw_hash="b" * 64), _candidate(raw_hash="c" * 64)
    store.save([first, second], tmp_path)
    manual.apply_decisions([_admit_decision("b" * 64)], root=tmp_path)  # id занят

    summary = manual.apply_decisions([_admit_decision("c" * 64)], root=tmp_path)  # тот же id

    assert len(summary.errors) == 1
    assert "уже существует" in summary.errors[0].detail
    assert len(schema.load_records(tmp_path)) == 1  # исходная запись не перезаписана
    # штамп ставится ТОЛЬКО после реально записанной meta.yaml: отказавший кандидат
    # остаётся ждущим, а не помечается допущенным в никуда
    assert {c.raw_hash[:1]: c.admitted_as for c in store.load(tmp_path)} == {
        "b": "me-example-strategy-2026",
        "c": None,
    }


def test_render_worksheet_header_documents_hidden_fields() -> None:
    text = manual.render_worksheet([])
    assert "hidden_fields" in text
    assert "binding_law" in text  # карта дефолтов authority видна куратору в шапке
