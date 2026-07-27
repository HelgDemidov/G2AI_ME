"""Recheck-контур — spec post-acquisition-lifecycle §1–§3.

Классификатор проверяется как ЧИСТАЯ функция от фикстурных ответов (сеть не нужна),
ротация и прогон — на tmp-корпусе с замоканным ``probe_url``.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from acquire import acquisition, recheck
from core import schema
from tests.support import valid_record, write_doc

TODAY = dt.date(2026, 7, 25)


def _classified(
    outcome: acquisition.AcquisitionOutcome = acquisition.AcquisitionOutcome.ok,
    *,
    reason: str = "valid PDF",
    etag: str | None = None,
    last_modified: str | None = None,
    status: int | None = 200,
) -> acquisition.ClassifiedResponse:
    return acquisition.ClassifiedResponse(outcome, status, reason, etag=etag, last_modified=last_modified)


def _probe(**kw: Any) -> recheck.ProbeOutcome:
    return recheck.ProbeOutcome(not_modified=False, classified=_classified(**kw))


def _state(**kw: Any) -> schema.OperationalState:
    base: dict[str, Any] = {"sha256": "a" * 64, "fidelity": "live", "acquisition_method": "direct"}
    base.update(kw)
    return schema.OperationalState.model_validate(base)


def _classify(state: schema.OperationalState, probe: recheck.ProbeOutcome, **kw: Any) -> recheck.Verdict:
    return recheck.classify_recheck(state, schema.SourceFormat.pdf, probe, **kw)


# --- §2: базовые исходы ---


def test_304_is_clean_and_confirms_validator() -> None:
    verdict = _classify(_state(etag='"v1"', etag_confirms=1), recheck.ProbeOutcome(True, None))
    assert verdict.finding is None
    assert verdict.etag_confirms == 2  # подтверждение стабильности — только так оно и копится


def test_changed_etag_is_drift() -> None:
    state = _state(etag='"v1"', etag_confirms=3)
    verdict = _classify(state, _probe(etag='"v2"'))
    assert verdict.finding is not None and verdict.finding.startswith("drift:")
    assert verdict.etag == '"v2"'          # состояние догоняет реальность
    assert verdict.etag_confirms == 0      # у нового валидатора подтверждений ещё нет


def test_matching_etag_is_clean() -> None:
    verdict = _classify(_state(etag='"v1"', etag_confirms=0), _probe(etag='"v1"'))
    assert verdict.finding is None and verdict.etag_confirms == 1


@pytest.mark.parametrize(
    "outcome,reason",
    [
        (acquisition.AcquisitionOutcome.dead, "HTTP 404"),
        (acquisition.AcquisitionOutcome.dead, "curl exit 6: host unreachable (DNS/connect)"),
    ],
)
def test_dead_url_is_link_rot(outcome: acquisition.AcquisitionOutcome, reason: str) -> None:
    verdict = _classify(_state(etag='"v1"'), _probe(outcome=outcome, reason=reason, status=None))
    assert verdict.finding is not None and verdict.finding.startswith("link-rot:")


def test_waf_challenge_bumps_without_finding() -> None:
    """Блок — известное состояние КАНАЛА, не событие документа: сказать про сам документ
    нечего, поэтому честное «непроверяемо», а не ложный drift."""
    state = _state(etag='"v1"', etag_confirms=2)
    verdict = _classify(
        state,
        _probe(outcome=acquisition.AcquisitionOutcome.blocked, reason="WAF challenge signature detected"),
    )
    assert verdict.finding is None
    assert verdict.etag == '"v1"' and verdict.etag_confirms == 2  # валидаторы не трогаем
    assert "unverifiable" in verdict.note


def test_bootstrap_validators_on_legacy_record() -> None:
    """Первый recheck по записи, добытой ДО этого спека, бесплатно её вооружает —
    заголовки уже пришли с ok-классифицированным ответом."""
    verdict = _classify(_state(), _probe(etag='"fresh"', last_modified="Wed, 21 Oct 2026 07:28:00 GMT"))
    assert verdict.finding is None
    assert verdict.etag == '"fresh"' and verdict.etag_confirms == 0
    assert "забутстраплены" in verdict.note


def test_no_validators_anywhere_is_honest_noop() -> None:
    verdict = _classify(_state(), _probe())
    assert verdict.finding is None and verdict.etag is None
    assert "--recheck-deep" in verdict.note


def test_last_modified_used_when_etag_absent() -> None:
    state = _state(http_last_modified="Wed, 21 Oct 2026 07:28:00 GMT", etag_confirms=2)
    verdict = _classify(state, _probe(last_modified="Thu, 22 Oct 2026 09:00:00 GMT"))
    assert verdict.finding is not None and "Last-Modified" in verdict.finding


def test_vanished_validator_does_not_wipe_stored_one() -> None:
    """Сервер перестал отдавать ETag — сравнивать не с чем, но забыть свой валидатор
    значило бы навсегда потерять способность видеть дрейф этого документа."""
    verdict = _classify(_state(etag='"v1"', etag_confirms=2), _probe())
    assert verdict.finding is None and verdict.etag == '"v1"' and verdict.etag_confirms == 2


# --- §2: правило стабильности html ---


def test_html_drift_suppressed_until_validator_proves_stable() -> None:
    """Валидаторы гос-порталов волатильны (динамические токены): без порога первый же
    прогон выдал бы шторм ложных drift по всему html-слою корпуса."""
    state = _state(etag='"v1"', etag_confirms=1)
    verdict = recheck.classify_recheck(state, schema.SourceFormat.html, _probe(etag='"v2"'))
    assert verdict.finding is None
    assert verdict.etag == '"v2"' and verdict.etag_confirms == 0
    assert "нестабилен" in verdict.note


def test_html_drift_fires_once_validator_is_stable() -> None:
    state = _state(etag='"v1"', etag_confirms=recheck.HTML_STABLE_CONFIRMS)
    verdict = recheck.classify_recheck(state, schema.SourceFormat.html, _probe(etag='"v2"'))
    assert verdict.finding is not None and verdict.finding.startswith("drift:")


def test_pdf_drift_is_not_gated_by_confirms() -> None:
    """Порог — исключительно html-лекарство; для pdf смена ETag сигнал сразу."""
    verdict = _classify(_state(etag='"v1"', etag_confirms=0), _probe(etag='"v2"'))
    assert verdict.finding is not None


# --- §2: семантика fidelity ---


def test_archived_snapshot_dead_is_expected_not_finding() -> None:
    """source_url был подтверждённо мёртв ещё при добыче: finding означал бы вечный
    link-rot-шум на каждой ротации по всем архивным записям."""
    state = _state(fidelity="archived_snapshot")
    verdict = _classify(state, _probe(outcome=acquisition.AcquisitionOutcome.dead, reason="HTTP 404"))
    assert verdict.finding is None and "ожидаемо" in verdict.note


def test_archived_snapshot_alive_is_resurrected() -> None:
    verdict = _classify(_state(fidelity="archived_snapshot"), _probe())
    assert verdict.finding is not None and verdict.finding.startswith("resurrected:")


def test_manual_record_bootstraps_validators_when_channel_unblocks() -> None:
    """У manual-записи валидаторов нет по построению (гейт §2). Живой ok-ответ значит,
    что канал разблокировался — будущие проверки станут содержательными."""
    verdict = _classify(_state(fidelity="manual", acquisition_method="manual"), _probe(etag='"now"'))
    assert verdict.finding is None and verdict.etag == '"now"'


# --- finding не гасится чистым исходом ---


def test_clean_probe_does_not_clear_existing_finding() -> None:
    """Флаг ставит машина, снимает — человек (передобычей). Иначе непонятый дрейф
    «выздоравливал» бы сам собой ровно после того, как мы обновили у себя валидатор."""
    state = _state(etag='"v2"', recheck_finding="drift: ETag \"v1\" -> \"v2\"")
    verdict = _classify(state, _probe(etag='"v2"'))
    assert verdict.finding == 'drift: ETag "v1" -> "v2"'


def test_apply_verdict_always_bumps_cursor() -> None:
    """Курсор и finding независимы: иначе документ с непогашенным флагом навсегда
    остался бы «самым давно не проверенным» и пере-probe'ился каждым прогоном."""
    state = _state(recheck_finding="link-rot: HTTP 404", acquisition_checked="2026-01-01")
    recheck.apply_verdict(state, _classify(state, _probe(outcome=acquisition.AcquisitionOutcome.dead)), TODAY)
    assert state.acquisition_checked == TODAY


# --- §2: глубокая сверка ---


def test_deep_matching_digest_is_clean() -> None:
    probe = recheck.ProbeOutcome(False, _classified(), digest="d" * 64)
    verdict = _classify(_state(), probe, deep=True, deep_baseline="d" * 64)
    assert verdict.finding is None


def test_deep_differing_digest_is_drift() -> None:
    probe = recheck.ProbeOutcome(False, _classified(), digest="e" * 64)
    verdict = _classify(_state(), probe, deep=True, deep_baseline="d" * 64)
    assert verdict.finding is not None and verdict.finding.startswith("drift: дайджест")


def test_deep_without_baseline_is_unverifiable_not_drift() -> None:
    """У скана, нормализованного OCR до появления original_sha256, издательских байт
    на диске нет — честное «непроверяемо» вместо ложного drift на каждом прогоне."""
    probe = recheck.ProbeOutcome(False, _classified(), digest="e" * 64)
    verdict = _classify(_state(), probe, deep=True, deep_baseline=None)
    assert verdict.finding is None and "unverifiable" in verdict.note


def test_deep_baseline_prefers_original_sha256(tmp_path: Path) -> None:
    rec = schema.SourceRecord.model_validate(valid_record())
    state = _state(original_sha256="b" * 64)
    assert recheck.deep_baseline(rec, tmp_path, state) == "b" * 64


def test_deep_baseline_falls_back_to_sha_for_born_digital(tmp_path: Path, monkeypatch: Any) -> None:
    rec = schema.SourceRecord.model_validate(valid_record())
    write_doc(tmp_path, valid_record(), raw=b"%PDF-1.4 born digital")
    monkeypatch.setattr("core.pdfmeta.was_ocr_normalized", lambda raw: False)
    assert recheck.deep_baseline(rec, tmp_path, _state()) == "a" * 64


def test_deep_baseline_is_none_for_ocr_normalized_scan(tmp_path: Path, monkeypatch: Any) -> None:
    """sha256 такого raw описывает файл ПОСЛЕ вшивания текст-слоя — сравнивать ответ
    издателя с ним значило бы объявлять drift на каждом прогоне."""
    rec = schema.SourceRecord.model_validate(valid_record())
    write_doc(tmp_path, valid_record(), raw=b"%PDF-1.4 scanned")
    monkeypatch.setattr("core.pdfmeta.was_ocr_normalized", lambda raw: True)
    assert recheck.deep_baseline(rec, tmp_path, _state()) is None


def test_deep_baseline_skips_ocr_check_for_non_pdf(tmp_path: Path) -> None:
    """OCR-путь существует только для PDF — открывать html/docx через pdfplumber
    незачем (и падало бы: живой урок scan_fallback_counts на raw.html)."""
    rec = schema.SourceRecord.model_validate({**valid_record(), "source_format": "html"})
    assert recheck.deep_baseline(rec, tmp_path, _state()) == "a" * 64


def test_deep_baseline_without_raw_falls_back_to_sha(tmp_path: Path) -> None:
    rec = schema.SourceRecord.model_validate(valid_record())
    assert recheck.deep_baseline(rec, tmp_path, _state()) == "a" * 64


def test_deep_baseline_is_none_when_origin_undecidable(tmp_path: Path) -> None:
    """Не смогли определить происхождение raw (битый/нечитаемый PDF) — «непроверяемо»,
    а не тихое допущение «born-digital»: ложный drift хуже честного пробела."""
    rec = schema.SourceRecord.model_validate(valid_record())
    write_doc(tmp_path, valid_record(), raw=b"not a pdf at all")
    assert recheck.deep_baseline(rec, tmp_path, _state()) is None


# --- probe_url: сборка условного запроса (сеть замокана на уровне fetch_raw) ---


def _fake_fetch_raw(
    monkeypatch: Any, response: acquisition.RawResponse
) -> list[dict[str, str] | None]:
    """Перехватить extra_headers, с которыми ушёл запрос."""
    sent: list[dict[str, str] | None] = []

    def fake(url: str, dest: Path, **kw: Any) -> acquisition.RawResponse:
        sent.append(kw.get("extra_headers"))
        return response

    monkeypatch.setattr(acquisition, "fetch_raw", fake)
    return sent


def test_probe_url_sends_both_conditional_headers(monkeypatch: Any) -> None:
    sent = _fake_fetch_raw(monkeypatch, acquisition.RawResponse(200, "HTTP/1.1 200 OK\r\n", b"%PDF body"))
    recheck.probe_url(
        "https://example.gov/d.pdf", user_agent="ua", expected=schema.SourceFormat.pdf,
        etag='"v1"', http_last_modified="Wed, 21 Oct 2026 07:28:00 GMT",
    )
    assert sent[0] == {
        "If-None-Match": '"v1"',
        "If-Modified-Since": "Wed, 21 Oct 2026 07:28:00 GMT",
    }


def test_probe_url_unconditional_when_asked(monkeypatch: Any) -> None:
    """Глубокая сверка и популяции (b)/(c) обязаны получить ТЕЛО, а не 304."""
    sent = _fake_fetch_raw(monkeypatch, acquisition.RawResponse(200, "HTTP/1.1 200 OK\r\n", b"%PDF"))
    recheck.probe_url(
        "https://example.gov/d.pdf", user_agent="ua", expected=schema.SourceFormat.pdf,
        etag='"v1"', conditional=False,
    )
    assert sent[0] == {}


def test_probe_url_recognises_304(monkeypatch: Any) -> None:
    _fake_fetch_raw(monkeypatch, acquisition.RawResponse(304, "HTTP/1.1 304 Not Modified\r\n", b""))
    probe = recheck.probe_url("https://e.gov/d.pdf", user_agent="ua", expected=schema.SourceFormat.pdf)
    assert probe.not_modified is True and probe.classified is None


def test_probe_url_maps_unreachable_to_dead(monkeypatch: Any) -> None:
    _fake_fetch_raw(
        monkeypatch,
        acquisition.RawResponse(None, "", b"", "curl exit 6: host unreachable (DNS/connect)"),
    )
    probe = recheck.probe_url("https://gone.gov/d.pdf", user_agent="ua", expected=schema.SourceFormat.pdf)
    assert probe.classified is not None
    assert probe.classified.outcome is acquisition.AcquisitionOutcome.dead


def test_probe_url_computes_digest_for_deep_compare(monkeypatch: Any) -> None:
    import hashlib

    body = b"%PDF-1.4 content"
    _fake_fetch_raw(monkeypatch, acquisition.RawResponse(200, "HTTP/1.1 200 OK\r\n", body))
    probe = recheck.probe_url("https://e.gov/d.pdf", user_agent="ua", expected=schema.SourceFormat.pdf)
    assert probe.digest == hashlib.sha256(body).hexdigest()


def test_probe_url_uses_format_agnostic_classifier_when_expected_none(monkeypatch: Any) -> None:
    """Популяция (c): у кандидата формата нет — вопрос «не закрыт ли канал», а не
    «тот ли это документ»."""
    _fake_fetch_raw(
        monkeypatch, acquisition.RawResponse(200, "HTTP/1.1 200 OK\r\n", b"<html>real page</html>")
    )
    probe = recheck.probe_url("https://e.gov/page", user_agent="ua", expected=None)
    assert probe.classified is not None
    assert probe.classified.outcome is acquisition.AcquisitionOutcome.ok


# --- §2: выбор URL проверки ---


def test_probe_url_follows_acquisition_method() -> None:
    rec = schema.SourceRecord.model_validate(
        {**valid_record(), "official_alt_url": "https://mirror.example/doc.pdf"}
    )
    assert recheck.probe_url_for(rec, _state()) == rec.source_url
    assert (
        recheck.probe_url_for(rec, _state(acquisition_method="official_alt"))
        == "https://mirror.example/doc.pdf"
    )


# --- §1: ротация ---


def _spy_snapshot(monkeypatch: Any) -> list[str]:
    """Перехватить SavePageNow (autouse-заглушка герметичности возвращает True без записи)."""
    calls: list[str] = []

    def fake(url: str, **kw: Any) -> bool:
        calls.append(url)
        return True

    monkeypatch.setattr(acquisition, "request_snapshot", fake)
    return calls


def _doc(root: Path, doc_id: str, *, state: dict[str, Any] | None = None, raw: bytes | None = b"%PDF") -> None:
    data = valid_record()
    data["id"] = doc_id
    write_doc(root, data, raw=raw, state=state)


def test_due_records_puts_never_checked_first(tmp_path: Path) -> None:
    _doc(tmp_path, "aa-checked-2026", state={"acquisition_checked": "2026-07-20"})
    _doc(tmp_path, "bb-never-2026", state={"sha256": "a" * 64})
    records = schema.load_records(tmp_path)
    with_raw, _ = recheck.due_records(records, tmp_path, limit=10)
    assert [r.id for r in with_raw] == ["bb-never-2026", "aa-checked-2026"]


def test_due_records_oldest_first_and_limited(tmp_path: Path) -> None:
    for doc_id, day in (("aa-doc-2026", "2026-07-24"), ("bb-doc-2026", "2026-07-01"), ("cc-doc-2026", "2026-07-10")):
        _doc(tmp_path, doc_id, state={"acquisition_checked": day})
    records = schema.load_records(tmp_path)
    with_raw, _ = recheck.due_records(records, tmp_path, limit=2)
    assert [r.id for r in with_raw] == ["bb-doc-2026", "cc-doc-2026"]


def test_due_records_excludes_superseded(tmp_path: Path) -> None:
    """Дрейф заменённой редакции — ожидаемое состояние (издатель работает над
    действующей), не сигнал. Множество считает общий schema.superseded_ids."""
    old = valid_record() | {"id": "me-law-2025"}
    new = valid_record() | {
        "id": "me-law-2026",
        "relations": [{"type": "supersedes", "target": "me-law-2025"}],
    }
    write_doc(tmp_path, old, raw=b"%PDF")
    write_doc(tmp_path, new, raw=b"%PDF")
    records = schema.load_records(tmp_path)
    with_raw, _ = recheck.due_records(records, tmp_path, limit=10)
    assert [r.id for r in with_raw] == ["me-law-2026"]


def test_due_records_splits_unacquired_population(tmp_path: Path) -> None:
    _doc(tmp_path, "aa-have-2026", state={"acquisition_checked": "2026-07-20"})
    _doc(tmp_path, "bb-missing-2026", raw=None, state={"acquisition_failed": "2026-07-19"})
    _doc(tmp_path, "cc-untouched-2026", raw=None)  # ни raw, ни провала — не наша забота
    records = schema.load_records(tmp_path)
    with_raw, without_raw = recheck.due_records(records, tmp_path, limit=10)
    assert [r.id for r in with_raw] == ["aa-have-2026"]
    assert [r.id for r in without_raw] == ["bb-missing-2026"]


def test_due_records_rotation_prefers_probe_checked_over_acquisition_failed(tmp_path: Path) -> None:
    """spec discovery-acquire-seam-hardening §3, Г2: ротация (b) сортирует по
    ``acquisition_probe_checked or acquisition_failed`` — легаси-запись без нового
    поля фолбэчит на прежний курсор."""
    _doc(tmp_path, "aa-legacy-2026", raw=None, state={"acquisition_failed": "2026-07-01"})
    _doc(
        tmp_path, "bb-recent-probe-2026", raw=None,
        state={"acquisition_failed": "2026-06-01", "acquisition_probe_checked": "2026-07-20"},
    )
    records = schema.load_records(tmp_path)
    _, without_raw = recheck.due_records(records, tmp_path, limit=10)
    assert [r.id for r in without_raw] == ["aa-legacy-2026", "bb-recent-probe-2026"]


def test_due_records_keeps_confidential_in_rotation(tmp_path: Path) -> None:
    """Условный запрос идёт к тому же официальному источнику, что и добыча: третьих
    сторон в нём нет (в отличие от SavePageNow, который гейтится)."""
    data = valid_record() | {"id": "me-secret-2026", "sensitivity": "confidential"}
    write_doc(tmp_path, data, raw=b"%PDF")
    records = schema.load_records(tmp_path)
    with_raw, _ = recheck.due_records(records, tmp_path, limit=10)
    assert [r.id for r in with_raw] == ["me-secret-2026"]


# --- §3: прогон, изоляция отказов, инвариант неприкосновенности ---


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    """Байты meta.yaml и raw.* всего дерева — инвариант §0 проверяется буквально."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and (p.name == "meta.yaml" or p.name.startswith("raw."))
    }


def test_run_recheck_never_touches_meta_or_raw(tmp_path: Path, monkeypatch: Any) -> None:
    _doc(tmp_path, "aa-doc-2026", state={"etag": '"v1"'})
    monkeypatch.setattr(recheck, "probe_url", lambda *a, **kw: _probe(etag='"v2"'))
    before = _tree_snapshot(tmp_path)

    recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    assert _tree_snapshot(tmp_path) == before


def test_run_recheck_writes_finding_and_bumps_cursor(tmp_path: Path, monkeypatch: Any) -> None:
    _doc(tmp_path, "aa-doc-2026", state={"etag": '"v1"', "etag_confirms": 3, "fidelity": "live"})
    monkeypatch.setattr(recheck, "probe_url", lambda *a, **kw: _probe(etag='"v2"'))

    summary = recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    assert len(summary.findings) == 1
    rec = schema.load_records(tmp_path)[0]
    st = schema.load_state(schema.state_file(rec, tmp_path))
    assert st.recheck_finding is not None and st.recheck_finding.startswith("drift:")
    assert st.acquisition_checked == TODAY


def test_run_recheck_isolates_document_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """Упавший документ остаётся с ПРЕЖНИМ курсором — следующий прогон возьмёт его снова."""
    _doc(tmp_path, "aa-boom-2026", state={"acquisition_checked": "2026-07-01"})
    _doc(tmp_path, "bb-ok-2026", state={"acquisition_checked": "2026-07-02"})

    def flaky(url: str, **kw: Any) -> recheck.ProbeOutcome:
        raise RuntimeError("сеть отвалилась")

    calls = {"n": 0}

    def probe(url: str, **kw: Any) -> recheck.ProbeOutcome:
        calls["n"] += 1
        if calls["n"] == 1:
            return flaky(url, **kw)
        return _probe(etag='"x"')

    monkeypatch.setattr(recheck, "probe_url", probe)
    summary = recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    assert len(summary.errors) == 1 and len(summary.items) == 2
    boom = next(r for r in schema.load_records(tmp_path) if r.id == "aa-boom-2026")
    assert schema.load_state(schema.state_file(boom, tmp_path)).acquisition_checked == dt.date(2026, 7, 1)


def test_due_records_skips_document_with_ambiguous_raw(tmp_path: Path, caplog: Any) -> None:
    """Папка с двумя raw.* — проблема раскладки конкретного документа, не повод рвать
    прогон контура: он громко пропускает её и идёт дальше."""
    _doc(tmp_path, "aa-broken-2026")
    (tmp_path / "intl-xperience" / "sg" / "aa-broken-2026" / "raw.html").write_bytes(b"<html/>")
    _doc(tmp_path, "bb-fine-2026")

    with caplog.at_level("WARNING", logger="recheck"):
        with_raw, _ = recheck.due_records(schema.load_records(tmp_path), tmp_path, limit=10)

    assert [r.id for r in with_raw] == ["bb-fine-2026"]
    assert "aa-broken-2026" in caplog.text


def test_run_recheck_isolates_unacquired_population_failure(tmp_path: Path, monkeypatch: Any) -> None:
    _doc(tmp_path, "aa-blocked-2026", raw=None, state={"acquisition_failed": "2026-07-01"})

    def boom(*a: Any, **kw: Any) -> recheck.ProbeOutcome:
        raise RuntimeError("сеть отвалилась")

    monkeypatch.setattr(recheck, "probe_url", boom)
    summary = recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    assert len(summary.errors) == 1
    rec = schema.load_records(tmp_path)[0]
    st = schema.load_state(schema.state_file(rec, tmp_path))
    assert st.acquisition_failed == dt.date(2026, 7, 1)  # курсор не сдвинут


def test_report_marks_failures_and_returns_nonzero(tmp_path: Path, caplog: Any) -> None:
    """Ненулевой код — только при отказе САМОГО прогона (сеть/ФС), не при findings."""
    summary = recheck.RecheckSummary(
        [recheck.RecheckItem("aa-ok-2026", "304 — не изменялся"),
         recheck.RecheckItem("bb-boom-2026", "", error="сеть отвалилась")]
    )
    with caplog.at_level("INFO", logger="recheck"):
        rc = recheck.report(summary)
    assert rc == 1
    assert "aa-ok-2026" in caplog.text


def test_report_returns_zero_when_only_findings(tmp_path: Path, monkeypatch: Any) -> None:
    """Findings — НЕ ошибки: нормальная работа контура не имеет права красить прогон."""
    _doc(tmp_path, "aa-doc-2026", state={"etag": '"v1"', "etag_confirms": 3})
    monkeypatch.setattr(recheck, "probe_url", lambda *a, **kw: _probe(etag='"v2"'))
    summary = recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)
    assert summary.findings and recheck.report(summary) == 0


def test_drift_triggers_snapshot_once(tmp_path: Path, monkeypatch: Any) -> None:
    """§4: снять ИЗМЕНИВШУЮСЯ редакцию, пока она жива. Повторное подтверждение того же
    finding в Wayback уже не стреляет — иначе каждая ротация била бы заново."""
    _doc(tmp_path, "aa-doc-2026", state={"etag": '"v1"', "etag_confirms": 3})
    monkeypatch.setattr(recheck, "probe_url", lambda *a, **kw: _probe(etag='"v2"'))
    calls = _spy_snapshot(monkeypatch)

    records = schema.load_records(tmp_path)
    recheck.run_recheck(records, tmp_path, user_agent="ua", today=TODAY)
    recheck.run_recheck(records, tmp_path, user_agent="ua", today=TODAY)

    assert len(calls) == 1


def test_drift_snapshot_targets_official_alt_when_acquired_via_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """spec acquire-convert-seam-hardening §7, В9-код: если добыча шла через
    official_alt (валидаторы сняты с НЕГО — probe_url_for это отражает), дрейф
    наблюдён на official_alt_url — SPN обязан снять ТУ же редакцию, не голый
    rec.source_url, о котором мы вообще ничего не знаем в этом сценарии."""
    data = valid_record() | {
        "id": "aa-alt-2026", "official_alt_url": "https://mirror.example.org/doc.pdf",
    }
    write_doc(
        tmp_path, data, raw=b"%PDF",
        state={"etag": '"v1"', "etag_confirms": 3, "acquisition_method": "official_alt"},
    )
    monkeypatch.setattr(recheck, "probe_url", lambda *a, **kw: _probe(etag='"v2"'))
    calls = _spy_snapshot(monkeypatch)

    recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    assert calls == ["https://mirror.example.org/doc.pdf"]


def test_confidential_drift_does_not_snapshot(tmp_path: Path, monkeypatch: Any) -> None:
    data = valid_record() | {"id": "me-secret-2026", "sensitivity": "confidential"}
    write_doc(tmp_path, data, raw=b"%PDF", state={"etag": '"v1"', "etag_confirms": 3, "fidelity": "live"})
    monkeypatch.setattr(recheck, "probe_url", lambda *a, **kw: _probe(etag='"v2"'))
    calls = _spy_snapshot(monkeypatch)

    recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    assert calls == []


def test_unacquired_population_clears_backoff_when_source_opens(tmp_path: Path, monkeypatch: Any) -> None:
    """Контур только СНИМАЕТ backoff; добирает документ ближайший штатный прогон —
    одна дверь к добыче, а не две. Успех бампает и курсор ротации `acquisition_probe_checked`."""
    _doc(tmp_path, "aa-blocked-2026", raw=None, state={"acquisition_failed": "2026-07-01",
                                                       "acquisition_failure_reason": "direct blocked"})
    monkeypatch.setattr(recheck, "probe_url", lambda *a, **kw: _probe())

    recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    rec = schema.load_records(tmp_path)[0]
    st = schema.load_state(schema.state_file(rec, tmp_path))
    assert st.acquisition_failed is None and st.acquisition_failure_reason is None
    assert st.acquisition_probe_checked == TODAY


def test_unacquired_population_does_not_extend_backoff_when_still_blocked(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Регресс репро B аудита (spec discovery-acquire-seam-hardening §3, Г2): probe
    заведомо слабее полной лестницы (один GET по source_url, без official_alt/
    browser/archive) и не имеет права переустанавливать якорь backoff — иначе окно
    не истекает никогда при регулярном --recheck. Курсор ротации сдвигается,
    якорь — нет; причина всё же обновляется свежей (полезна сводке)."""
    _doc(
        tmp_path, "aa-blocked-2026", raw=None,
        state={"acquisition_failed": "2026-07-01", "acquisition_failure_reason": "old reason"},
    )
    monkeypatch.setattr(
        recheck, "probe_url",
        lambda *a, **kw: _probe(outcome=acquisition.AcquisitionOutcome.blocked, reason="WAF challenge"),
    )

    recheck.run_recheck(schema.load_records(tmp_path), tmp_path, user_agent="ua", today=TODAY)

    rec = schema.load_records(tmp_path)[0]
    st = schema.load_state(schema.state_file(rec, tmp_path))
    assert st.acquisition_failed == dt.date(2026, 7, 1)  # якорь backoff НЕ переустановлен
    assert st.acquisition_failure_reason == "WAF challenge"  # причина всё же свежая
    assert st.acquisition_probe_checked == TODAY  # курсор ротации сдвинут
