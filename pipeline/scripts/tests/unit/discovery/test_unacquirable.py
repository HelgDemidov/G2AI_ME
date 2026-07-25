"""Типизация отказов и реанимация — spec post-acquisition-lifecycle §5.

«Не нужен» и «нужен, но недобываем» становятся разными состояниями с разной судьбой:
первое терминально, второе — очередь ожидания обстоятельств (секция worksheet,
популяция (c) recheck, действие ``revive``).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from acquire import acquisition, recheck
from core import schema
from discovery import manual, store

TODAY = dt.date(2026, 7, 25)


def _cand(raw_hash: str, **kw: Any) -> schema.CandidateRecord:
    base: dict[str, Any] = {
        "connector_id": "manual",
        "retrieved_at": "2026-07-20",
        "raw_hash": raw_hash,
        "title": f"Doc {raw_hash[:4]}",
        "issuer": "Ministry",
        "source_url": f"https://example.gov/{raw_hash[:4]}.pdf",
        "normalized_url": f"https://example.gov/{raw_hash[:4]}.pdf",
    }
    base.update(kw)
    return schema.CandidateRecord.model_validate(base)


# --- apply: reject_kind ---


def test_apply_maps_reject_kind(tmp_path: Path) -> None:
    store.save([_cand("a" * 64)], tmp_path)
    summary = manual.apply_decisions(
        [{"raw_hash": "a" * 12, "reason": "WAF blocks every rung", "reject_kind": "unacquirable",
          "action": "reject"}],
        root=tmp_path,
    )
    assert summary.errors == []
    saved = store.load(tmp_path)[0]
    assert saved.rejected_kind is schema.RejectionKind.unacquirable
    assert saved.rejected_reason == "WAF blocks every rung"


def test_apply_reject_without_kind_keeps_none(tmp_path: Path) -> None:
    """Опущенный ключ = легаси-семантика «содержательно не подходит»; дефолт в store
    не пишется — иначе каждая отказная запись потолстела бы на строку ни за чем."""
    store.save([_cand("b" * 64)], tmp_path)
    manual.apply_decisions(
        [{"raw_hash": "b" * 12, "reason": "off-axis", "action": "reject"}], root=tmp_path
    )
    assert store.load(tmp_path)[0].rejected_kind is None


def test_apply_rejects_unknown_reject_kind(tmp_path: Path) -> None:
    """Закрытое множество: свободная строка сломала бы маршрутизацию worksheet/recheck
    молча. Ошибка ОДНОГО решения не рвёт батч — она в summary.errors."""
    store.save([_cand("c" * 64)], tmp_path)
    summary = manual.apply_decisions(
        [{"raw_hash": "c" * 12, "reason": "x", "reject_kind": "maybe", "action": "reject"}],
        root=tmp_path,
    )
    assert len(summary.errors) == 1
    assert store.load(tmp_path)[0].rejected_reason is None  # решение не применилось


# --- apply: revive ---


def test_apply_revive_returns_candidate_to_pending(tmp_path: Path) -> None:
    """Реанимация — ЯВНОЕ решение человека; невоскрешение через dedup при этом не
    ослабляется (кандидат возвращается в ждущие штатной реконсиляцией)."""
    cand = _cand(
        "d" * 64, rejected_reason="WAF", rejected_kind="unacquirable",
        probe_checked="2026-07-24", probe_finding="acquirable: HTTP 200, тело 40000 Б",
    )
    store.save([cand], tmp_path)

    summary = manual.apply_decisions([{"raw_hash": "d" * 12, "action": "revive"}], root=tmp_path)

    assert summary.errors == []
    revived = store.load(tmp_path)[0]
    assert revived.rejected_reason is None and revived.rejected_kind is None
    assert revived.probe_checked is None and revived.probe_finding is None
    assert manual.pending_candidates([revived], []) == [revived]


def test_apply_revive_dry_run_writes_nothing(tmp_path: Path) -> None:
    store.save([_cand("e" * 64, rejected_reason="WAF", rejected_kind="unacquirable")], tmp_path)
    summary = manual.apply_decisions(
        [{"raw_hash": "e" * 12, "action": "revive"}], root=tmp_path, dry_run=True
    )
    assert summary.errors == []
    assert store.load(tmp_path)[0].rejected_reason == "WAF"


def test_apply_revive_on_active_candidate_is_noop(tmp_path: Path) -> None:
    store.save([_cand("f" * 64)], tmp_path)
    summary = manual.apply_decisions([{"raw_hash": "f" * 12, "action": "revive"}], root=tmp_path)
    assert summary.errors == []
    assert "не был отклонён" in summary.outcomes[0].detail


def test_apply_rejects_unknown_action(tmp_path: Path) -> None:
    store.save([_cand("1" * 64)], tmp_path)
    summary = manual.apply_decisions([{"raw_hash": "1" * 12, "action": "resurrect"}], root=tmp_path)
    assert len(summary.errors) == 1


# --- worksheet: отдельная секция ---


def test_unacquirable_candidates_selected_and_ordered() -> None:
    fresh = _cand("2" * 64, rejected_reason="WAF", rejected_kind="unacquirable", probe_checked="2026-07-24")
    never = _cand("3" * 64, rejected_reason="WAF", rejected_kind="unacquirable")
    plain = _cand("4" * 64, rejected_reason="off-axis")
    due = manual.unacquirable_candidates([fresh, never, plain])
    assert [c.raw_hash for c in due] == [never.raw_hash, fresh.raw_hash]  # непробованные первыми


def test_worksheet_renders_unacquirable_section() -> None:
    cand = _cand(
        "5" * 64, rejected_reason="WAF", rejected_kind="unacquirable",
        probe_checked="2026-07-24", probe_finding="acquirable: HTTP 200, тело 40000 Б",
    )
    text = manual.render_worksheet([], [cand])
    assert "Недобываемые — ждут смены обстоятельств" in text
    assert "acquirable: HTTP 200" in text
    assert "revive" in text


def test_worksheet_omits_section_when_nothing_waits() -> None:
    """Шум в типовом прогоне не нужен: пустая очередь — пустой вывод."""
    assert "Недобываемые" not in manual.render_worksheet([], [])


def test_unacquirable_stays_out_of_pending() -> None:
    """Секции не пересекаются: недобываемый отклонён, значит в ждущих его нет."""
    cand = _cand("6" * 64, rejected_reason="WAF", rejected_kind="unacquirable")
    assert manual.pending_candidates([cand], []) == []


# --- популяция (c) recheck ---


def test_due_candidates_only_unacquirable_with_url() -> None:
    ok = _cand("7" * 64, rejected_reason="WAF", rejected_kind="unacquirable")
    irrelevant = _cand("8" * 64, rejected_reason="off-axis")
    no_url = _cand("9" * 64, rejected_reason="WAF", rejected_kind="unacquirable", source_url=None)
    assert recheck.due_candidates([ok, irrelevant, no_url], limit=10) == [ok]


def test_probe_marks_candidate_acquirable(monkeypatch: Any) -> None:
    """Формат-агностичная классификация (§2): у кандидата source_format ещё нет, и с
    наивным `expected=pdf` любой оживший HTML-источник навсегда остался бы blocked."""
    cand = _cand("a1" + "0" * 62, rejected_reason="WAF", rejected_kind="unacquirable")
    monkeypatch.setattr(
        recheck, "probe_url",
        lambda *a, **kw: recheck.ProbeOutcome(
            False, acquisition.ClassifiedResponse(acquisition.AcquisitionOutcome.ok, 200, "HTTP 200, тело 40000 Б")
        ),
    )
    summary = recheck.run_recheck([], Path("/nonexistent"), user_agent="ua", today=TODAY, candidates=[cand])

    assert cand.probe_checked == TODAY
    assert cand.probe_finding is not None and cand.probe_finding.startswith("acquirable:")
    assert summary.candidates_changed is True
    assert len(summary.findings) == 1  # только открывшийся канал попадает в findings


def test_probe_marks_candidate_still_blocked(monkeypatch: Any) -> None:
    cand = _cand("b1" + "0" * 62, rejected_reason="WAF", rejected_kind="unacquirable")
    monkeypatch.setattr(
        recheck, "probe_url",
        lambda *a, **kw: recheck.ProbeOutcome(
            False,
            acquisition.ClassifiedResponse(
                acquisition.AcquisitionOutcome.blocked, 403, "WAF challenge signature detected"
            ),
        ),
    )
    summary = recheck.run_recheck([], Path("/nonexistent"), user_agent="ua", today=TODAY, candidates=[cand])

    assert cand.probe_finding is not None and cand.probe_finding.startswith("blocked:")
    assert summary.findings == []  # нечего делать человеку — не шумим


def test_probe_isolates_candidate_failure(monkeypatch: Any) -> None:
    cand = _cand("c1" + "0" * 62, rejected_reason="WAF", rejected_kind="unacquirable")

    def boom(*a: Any, **kw: Any) -> recheck.ProbeOutcome:
        raise RuntimeError("сеть отвалилась")

    monkeypatch.setattr(recheck, "probe_url", boom)
    summary = recheck.run_recheck([], Path("/nonexistent"), user_agent="ua", today=TODAY, candidates=[cand])

    assert len(summary.errors) == 1
    assert cand.probe_checked is None  # курсор не сдвинут — следующий прогон возьмёт снова


# --- формат-агностичный классификатор (§2) ---


def test_classify_probe_html_body_is_acquirable() -> None:
    result = acquisition.classify_probe(b"<html>real content</html>", "HTTP/1.1 200 OK\r\n")
    assert result.outcome is acquisition.AcquisitionOutcome.ok


def test_classify_probe_challenge_beats_status() -> None:
    """Заглушка WAF сама приходит как 200 — порядок проверок это учитывает."""
    result = acquisition.classify_probe(b"<html>Attention Required</html>", "HTTP/1.1 200 OK\r\n")
    assert result.outcome is acquisition.AcquisitionOutcome.blocked


def test_classify_probe_404_is_dead() -> None:
    result = acquisition.classify_probe(b"", "HTTP/1.1 404 Not Found\r\n")
    assert result.outcome is acquisition.AcquisitionOutcome.dead


def test_classify_probe_empty_200_is_blocked() -> None:
    result = acquisition.classify_probe(b"", "HTTP/1.1 200 OK\r\n")
    assert result.outcome is acquisition.AcquisitionOutcome.blocked
