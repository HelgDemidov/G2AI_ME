"""Тесты цикла РЕДАКЦИЙ: дискриминатор `supersedes` в идентичности кандидата
(spec discovery-candidates-sharding §5).

Сквозной сценарий, который до этого спека был физически невозможен: новая редакция
закона живёт на ТОМ ЖЕ URL, что предшественник — dedup поглощал бы её как дубль, а
реконсиляция `pending_candidates` прятала бы её из worksheet.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from tests.support import valid_record

from core import schema
from discovery import manual, store
from discovery.dedup import dedup, normalize_url

_URL = "https://gov.example.org/law.pdf"


def _candidate(**overrides: object) -> schema.CandidateRecord:
    data: dict[str, object] = {
        "connector_id": "manual",
        "retrieved_at": "2026-07-25",
        "raw_hash": "a" * 64,
        "title": "Registration Law",
        "issuer": "Ministry",
        "language": "en",
        "source_url": _URL,
        "normalized_url": normalize_url(_URL),
    }
    data.update(overrides)
    return schema.CandidateRecord.model_validate(data)


def _predecessor(root: Path, *, relations: list[dict[str, str]] | None = None) -> schema.SourceRecord:
    """Записать в реестр запись-предшественника (id = valid_record's sg-...)."""
    data = valid_record()
    data["source_url"] = _URL
    if relations is not None:
        data["relations"] = relations
    rec = schema.SourceRecord.model_validate(data)
    schema.save_record(rec, root)
    return rec


# --- raw_hash: дискриминатор в канонической строке ------------------------------------


def test_raw_hash_differs_for_edition_with_identical_bibliography() -> None:
    """Одинаковая дата переиздания — реальный кейс: без дискриминатора обе записи получили
    бы ОДИН raw_hash, и `_resolve_candidate` падал бы «неоднозначен» для обеих."""
    date = dt.date(2026, 1, 1)
    base = manual.raw_hash_for_manual(_URL, "Registration Law", date)
    edition = manual.raw_hash_for_manual(_URL, "Registration Law", date, "sg-imda-mgf-agentic-2026")

    assert base != edition


def test_raw_hash_unchanged_when_no_supersedes() -> None:
    """Хэши всех существующих кандидатов не сдвигаются ни на бит (явный None == опущенный)."""
    date = dt.date(2026, 1, 1)
    assert manual.raw_hash_for_manual(_URL, "T", date) == manual.raw_hash_for_manual(_URL, "T", date, None)


# --- dedup: дискриминатор во всех трёх стратегиях -------------------------------------


def test_edition_not_absorbed_by_predecessor_candidate_same_url() -> None:
    predecessor_cand = _candidate(raw_hash="p" * 64)
    edition = _candidate(raw_hash="e" * 64, supersedes="sg-imda-mgf-agentic-2026")

    outcome = dedup([edition], existing=[predecessor_cand])

    assert outcome.fresh == [edition]
    assert outcome.absorbed == 0


def test_edition_not_absorbed_by_match_key_strategy() -> None:
    """Стратегия 2 (issuer+title+date) с ОДИНАКОВОЙ датой переиздания — тот угол, ради
    которого дискриминатор введён единообразно во все три ключа, а не только в URL."""
    same_date = dt.date(2026, 1, 1)
    predecessor_cand = _candidate(raw_hash="p" * 64, normalized_url=None, doc_date=same_date)
    edition = _candidate(
        raw_hash="e" * 64, normalized_url=None, doc_date=same_date, supersedes="sg-imda-mgf-agentic-2026"
    )

    outcome = dedup([edition], existing=[predecessor_cand])

    assert outcome.fresh == [edition]
    assert outcome.absorbed == 0


def test_edition_not_absorbed_by_content_hash_strategy() -> None:
    predecessor_cand = _candidate(raw_hash="p" * 64, normalized_url=None, title=None, content_hash="c1")
    edition = _candidate(
        raw_hash="e" * 64, normalized_url=None, title=None, content_hash="c1",
        supersedes="sg-imda-mgf-agentic-2026",
    )

    outcome = dedup([edition], existing=[predecessor_cand])

    assert outcome.fresh == [edition]
    assert outcome.absorbed == 0


def test_same_edition_twice_is_still_deduplicated() -> None:
    """Дискриминатор не ослабляет dedup: две записи об ОДНОЙ редакции — по-прежнему дубль."""
    first = _candidate(raw_hash="e" * 64, supersedes="sg-imda-mgf-agentic-2026")
    second = _candidate(
        connector_id="search:wb", raw_hash="f" * 64, supersedes="sg-imda-mgf-agentic-2026"
    )

    outcome = dedup([second], existing=[first])

    assert outcome.fresh == []
    assert outcome.absorbed == 1


def test_rejected_edition_is_not_resurrected() -> None:
    """Невоскрешение отклонённых в силе без оговорок — и для редакций тоже."""
    rejected = _candidate(
        raw_hash="e" * 64, supersedes="sg-imda-mgf-agentic-2026", rejected_reason="ложная тревога"
    )
    again = _candidate(
        connector_id="search:wb", raw_hash="f" * 64, supersedes="sg-imda-mgf-agentic-2026"
    )

    outcome = dedup([again], existing=[rejected])

    assert outcome.fresh == []
    assert outcome.absorbed == 1


# --- inject: валидация предшественника ------------------------------------------------


def test_inject_edition_creates_candidate_despite_same_url(tmp_path: Path) -> None:
    predecessor = _predecessor(tmp_path)
    manual.inject(
        url=_URL, title="Registration Law", issuer="Ministry", language="en", root=tmp_path,
    )

    cand, is_new = manual.inject(
        url=_URL, title="Registration Law", issuer="Ministry", language="en",
        supersedes=predecessor.id, root=tmp_path,
    )

    assert is_new
    assert cand.supersedes == predecessor.id
    assert len(store.load(tmp_path)) == 2  # предшественник-кандидат и редакция — обе записи


def test_inject_same_edition_twice_is_noop(tmp_path: Path) -> None:
    predecessor = _predecessor(tmp_path)
    manual.inject(
        url=_URL, title="Registration Law", issuer="Ministry", language="en",
        supersedes=predecessor.id, root=tmp_path,
    )

    _, is_new = manual.inject(
        url=_URL, title="Registration Law", issuer="Ministry", language="en",
        supersedes=predecessor.id, root=tmp_path,
    )

    assert is_new is False
    assert len(store.load(tmp_path)) == 1


def test_inject_rejects_unknown_predecessor(tmp_path: Path) -> None:
    """Опечатка в doc-id падает НА INJECT, а не при промоушене."""
    _predecessor(tmp_path)

    with pytest.raises(ValueError, match="нет в реестре корпуса"):
        manual.inject(
            url=_URL, title="Registration Law", issuer="Ministry", language="en",
            supersedes="me-no-such-document-2026", root=tmp_path,
        )

    assert store.load(tmp_path) == []


def test_inject_without_supersedes_does_not_read_registry(tmp_path: Path, monkeypatch: object) -> None:
    """Реестр читается ТОЛЬКО когда флаг задан (обычный inject не платит за обход дерева)."""
    def explode(_root: Path) -> list[schema.SourceRecord]:
        raise AssertionError("load_records не должен вызываться без --supersedes")

    monkeypatch.setattr(schema, "load_records", explode)  # type: ignore[attr-defined]
    _, is_new = manual.inject(
        url=_URL, title="Registration Law", issuer="Ministry", language="en", root=tmp_path,
    )

    assert is_new


# --- promote: авто-ребро supersedes ---------------------------------------------------


def _promote(cand: schema.CandidateRecord, **kwargs: object) -> schema.SourceRecord:
    return schema.promote_candidate(
        cand,
        id="me-registration-law-2027",
        entity_id="me",
        track=schema.Track.target_entity,
        issuer_type=schema.IssuerType.government,
        geo_scope=schema.GeoScope.national,
        doc_type="legislation",
        authority="binding_law",
        relevance=schema.Relevance(
            target_fit=schema.TargetFit.primary,
            axis="digital_sovereignty",
            assessed_stage=schema.AssessedStage.triage,
            rationale="new edition of a corpus record",
            assessed_date=dt.date(2026, 7, 25),
        ),
        **kwargs,  # type: ignore[arg-type]
    )


def test_promote_materializes_supersedes_edge() -> None:
    rec = _promote(_candidate(supersedes="sg-imda-mgf-agentic-2026"))

    assert [(r.type.value, r.target) for r in rec.relations] == [
        ("supersedes", "sg-imda-mgf-agentic-2026")
    ]


def test_promote_merges_auto_edge_with_curated_relations() -> None:
    rec = _promote(
        _candidate(supersedes="sg-imda-mgf-agentic-2026"),
        relations=[schema.Relation(type=schema.RelationType.implements, target="eu-ai-act-2024")],
    )

    assert [(r.type.value, r.target) for r in rec.relations] == [
        ("implements", "eu-ai-act-2024"),
        ("supersedes", "sg-imda-mgf-agentic-2026"),
    ]


def test_promote_does_not_duplicate_manually_written_edge() -> None:
    """Куратор вписал то же ребро руками в decisions.yaml — двойного не появляется."""
    rec = _promote(
        _candidate(supersedes="sg-imda-mgf-agentic-2026"),
        relations=[
            schema.Relation(type=schema.RelationType.supersedes, target="sg-imda-mgf-agentic-2026")
        ],
    )

    assert [(r.type.value, r.target) for r in rec.relations] == [
        ("supersedes", "sg-imda-mgf-agentic-2026")
    ]


def test_promote_without_supersedes_keeps_relations_untouched() -> None:
    rec = _promote(_candidate())
    assert rec.relations == []


# --- pending/worksheet: реконсиляция по паре (url, supersedes) ------------------------


def test_edition_stays_pending_while_only_predecessor_is_registered(tmp_path: Path) -> None:
    """Ядро фикса: совпадение URL с ПРЕДШЕСТВЕННИКОМ не гасит редакцию — иначе она
    никогда не попала бы в worksheet и штатный триаж редакций был бы невозможен."""
    predecessor = _predecessor(tmp_path)
    edition = _candidate(supersedes=predecessor.id)

    assert manual.pending_candidates([edition], [predecessor]) == [edition]


def test_edition_gasnet_after_its_own_promotion(tmp_path: Path) -> None:
    """Редакция перестаёт быть ждущей, когда промоутнута ОНА САМА (её ребро в реестре)."""
    predecessor = _predecessor(tmp_path)
    edition = _candidate(supersedes=predecessor.id)
    promoted_edition = _promote(edition)

    pending = manual.pending_candidates([edition], [predecessor, promoted_edition])

    assert pending == []


def test_plain_rediscovery_of_registered_url_is_not_pending(tmp_path: Path) -> None:
    """Обычное пере-обнаружение того же URL (без supersedes) по-прежнему гасится —
    регрессия исходного поведения реконсиляции."""
    predecessor = _predecessor(tmp_path)
    plain = _candidate()

    assert manual.pending_candidates([plain], [predecessor]) == []


def test_edition_of_other_predecessor_stays_pending(tmp_path: Path) -> None:
    """Промоушен редакции A не гасит кандидата-редакцию другого предшественника."""
    predecessor = _predecessor(tmp_path)
    promoted_edition = _promote(_candidate(supersedes=predecessor.id))
    other = _candidate(raw_hash="o" * 64, supersedes="eu-ai-act-2024")

    pending = manual.pending_candidates([other], [predecessor, promoted_edition])

    assert pending == [other]


def test_worksheet_shows_supersedes_column(tmp_path: Path) -> None:
    """Куратор обязан видеть, что перед ним редакция, а не дубль (иначе reject по ошибке)."""
    text = manual.render_worksheet([_candidate(supersedes="sg-imda-mgf-agentic-2026")])

    assert "| supersedes |" in text
    assert "sg-imda-mgf-agentic-2026" in text
    assert "НОВАЯ РЕДАКЦИЯ" in text  # шапка объясняет семантику колонки
