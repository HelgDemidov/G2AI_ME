"""Схема «контура времени» — spec post-acquisition-lifecycle §5/§7.

Новые поля ``OperationalState``/``CandidateRecord``, enum ``RejectionKind`` и общий
хелпер ``superseded_ids``. Слим (удаление ``in_force``/``dates.last_checked``) живёт в
``test_schema.py`` рядом с остальными проверками курируемой записи.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core.schema import (
    CandidateRecord,
    OperationalState,
    Relation,
    RejectionKind,
    RelationType,
    SourceRecord,
    load_state,
    save_state,
    superseded_ids,
)
from tests.support import valid_record


def _record(doc_id: str, relations: list[Relation] | None = None) -> SourceRecord:
    data = valid_record()
    data["id"] = doc_id
    rec = SourceRecord.model_validate(data)
    if relations:
        rec.relations = relations
    return rec


# --- OperationalState: новые поля контура времени (§7) ---


def test_legacy_state_without_lifecycle_fields_stays_valid() -> None:
    """Обратная совместимость: .state.yaml, написанный ДО этого спека, парсится как есть.

    Это условие смысла всего слима: старые сайдкары не мигрируются вовсе (в отличие от
    курируемых meta.yaml), поэтому каждое новое поле обязано иметь дефолт."""
    state = OperationalState.model_validate(
        {"sha256": "a" * 64, "acquisition_method": "direct", "fidelity": "live"}
    )
    assert state.etag is None
    assert state.http_last_modified is None
    assert state.etag_confirms == 0
    assert state.recheck_finding is None
    assert state.snapshot_requested is None
    assert state.acquisition_failed is None
    assert state.acquisition_failure_reason is None


def test_lifecycle_fields_roundtrip_through_state_file(tmp_path: Path) -> None:
    """Поля переживают save/load: сайдкар пишется ``exclude_none``, но 0-й счётчик
    (не None!) обязан сохраниться — иначе «подтверждений не было» и «поле потеряно»
    стали бы неотличимы после первой же перезаписи."""
    path = tmp_path / ".state.yaml"
    state = OperationalState(
        etag='W/"abc"',
        http_last_modified="Wed, 21 Oct 2026 07:28:00 GMT",
        etag_confirms=0,
        recheck_finding="drift: ETag изменился",
        snapshot_requested=dt.date(2026, 7, 25),
        acquisition_failed=dt.date(2026, 7, 20),
        acquisition_failure_reason="direct blocked (WAF challenge signature detected)",
    )
    save_state(path, state)
    loaded = load_state(path)
    assert loaded == state
    assert "etag_confirms" in yaml.safe_load(path.read_text(encoding="utf-8"))


# --- OperationalState: backoff отказов конвертации (spec acquire-convert-seam-hardening §8, В11) ---


def test_legacy_state_without_convert_backoff_fields_stays_valid() -> None:
    """Обратная совместимость: та же дисциплина, что и у контура времени (§7) —
    новое поле без дефолта уронило бы КАЖДЫЙ существующий .state.yaml корпуса."""
    state = OperationalState.model_validate({"sha256": "a" * 64})
    assert state.convert_failed is None
    assert state.convert_failure_reason is None
    assert state.convert_failed_converter is None


def test_convert_backoff_fields_roundtrip_through_state_file(tmp_path: Path) -> None:
    path = tmp_path / ".state.yaml"
    state = OperationalState(
        convert_failed=dt.date(2026, 7, 27),
        convert_failure_reason="ocrmypdf завершился с кодом 1: timeout",
        convert_failed_converter="pdf@6",
    )
    save_state(path, state)
    assert load_state(path) == state


def test_state_rejects_unknown_field() -> None:
    """``extra="forbid"`` в силе и после расширения — опечатка в имени нового поля
    падает громко, а не оседает write-only мусором в сайдкаре."""
    with pytest.raises(ValidationError):
        OperationalState.model_validate({"etag_confirmations": 3})


# --- RejectionKind + probe-поля кандидата (§5) ---


def test_candidate_rejection_kind_defaults_to_none() -> None:
    """None == легаси == irrelevant-семантика: кандидаты, отклонённые до спека, не
    становятся молча «недобываемыми» (иначе весь старый отказный слой попал бы в
    популяцию (c) recheck и начал бы дёргать сеть)."""
    cand = CandidateRecord(connector_id="manual", retrieved_at=dt.date(2026, 7, 25), raw_hash="h" * 64)
    assert cand.rejected_kind is None
    assert cand.probe_checked is None
    assert cand.probe_finding is None


def test_candidate_rejection_kind_parses_enum() -> None:
    cand = CandidateRecord.model_validate(
        {
            "connector_id": "manual",
            "retrieved_at": "2026-07-25",
            "raw_hash": "h" * 64,
            "rejected_reason": "WAF blocks every rung",
            "rejected_kind": "unacquirable",
            "probe_checked": "2026-07-25",
            "probe_finding": "blocked: WAF challenge signature detected",
        }
    )
    assert cand.rejected_kind is RejectionKind.unacquirable
    assert cand.probe_checked == dt.date(2026, 7, 25)


def test_candidate_rejection_kind_rejects_unknown_value() -> None:
    """Закрытое множество: код ветвится по значению, свободная строка сломала бы
    маршрутизацию worksheet/recheck молча."""
    with pytest.raises(ValidationError):
        CandidateRecord.model_validate(
            {
                "connector_id": "manual",
                "retrieved_at": "2026-07-25",
                "raw_hash": "h" * 64,
                "rejected_kind": "maybe_later",
            }
        )


# --- superseded_ids: одно определение на recheck (§1) и graph-v2 (§2) ---


def test_superseded_ids_reads_forward_edge() -> None:
    successor = _record("me-law-2026", [Relation(type=RelationType.supersedes, target="me-law-2025")])
    assert superseded_ids([successor]) == {"me-law-2025"}


def test_superseded_ids_reads_reverse_edge() -> None:
    """``superseded_by`` у предшественника — легальная форма того же факта; recheck и
    graph-v2 обязаны видеть её одинаково, иначе документ, оформленный «обратным»
    ребром, вечно крутился бы в ротации как живой."""
    predecessor = _record(
        "me-law-2025", [Relation(type=RelationType.superseded_by, target="me-law-2026")]
    )
    assert superseded_ids([predecessor]) == {"me-law-2025"}


def test_superseded_ids_ignores_other_relation_types() -> None:
    rec = _record("sg-doc-2026", [Relation(type=RelationType.cites, target="eu-ai-act-2024")])
    assert superseded_ids([rec]) == set()


def test_superseded_ids_unions_both_directions() -> None:
    a = _record("a-doc-2026", [Relation(type=RelationType.supersedes, target="a-doc-2025")])
    b = _record("b-doc-2024", [Relation(type=RelationType.superseded_by, target="b-doc-2026")])
    assert superseded_ids([a, b]) == {"a-doc-2025", "b-doc-2024"}


def test_superseded_ids_empty_corpus() -> None:
    assert superseded_ids([]) == set()
