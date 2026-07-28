"""Property-тесты индексного dedup против независимого наивного оракула
(spec discovery-candidates-sharding §4; hypothesis — общепроектный стандарт для такого класса).

Прод-реализация схлопнула линейные сканы в dict-индексы. Эквивалентность КАНОНИЧЕСКОЙ
семантике (строгий приоритет стратегий url -> key -> идентичность в источнике над единым
пулом) доказывается сравнением с оракулом, живущим В ЭТОМ файле: наивные линейные сканы,
написанные независимо от прод-кода. Обёртка над самими индексами оракулом быть не может —
сравнение выродилось бы в тавтологию.
"""
from __future__ import annotations

import datetime as dt

from hypothesis import given, settings
from hypothesis import strategies as st

from core import schema
from discovery.dedup import dedup, normalize_url, normalized_title

# --- независимый наивный оракул -------------------------------------------------------


def _oracle_key(cand: schema.CandidateRecord) -> tuple[str, str, str] | None:
    if not (cand.title and cand.issuer):
        return None
    return (cand.issuer, normalized_title(cand.title), str(cand.doc_date))


def _oracle_url(cand: schema.CandidateRecord) -> str | None:
    if cand.url_provenance is schema.UrlProvenance.suspect:
        return None
    return normalize_url(cand.source_url) if cand.source_url else None


def _oracle_keyless(cand: schema.CandidateRecord) -> bool:
    return _oracle_url(cand) is None and _oracle_key(cand) is None


def _oracle_source(cand: schema.CandidateRecord) -> tuple[str, str, str | None]:
    return (cand.connector_id, cand.native_id or cand.raw_hash, cand.supersedes)


def _oracle_find(
    cand: schema.CandidateRecord, pool: list[schema.CandidateRecord]
) -> schema.CandidateRecord | None:
    """Линейные сканы по ЕДИНОМУ пулу в порядке приоритета стратегий."""
    url = _oracle_url(cand)
    if url:
        for other in pool:
            if _oracle_url(other) == url:
                return other
    key = _oracle_key(cand)
    if key is not None:
        for other in pool:
            if _oracle_key(other) == key:
                return other
    # стратегия 3 — только между бесключевыми, не как фолбэк промаха первых двух
    if _oracle_keyless(cand):
        ident = _oracle_source(cand)
        for other in pool:
            if _oracle_keyless(other) and _oracle_source(other) == ident:
                return other
    return None


def _oracle_dedup(
    new: list[schema.CandidateRecord], existing: list[schema.CandidateRecord]
) -> tuple[list[schema.CandidateRecord], int, list[tuple[schema.CandidateRecord, schema.CandidateRecord]]]:
    pool = list(existing)
    fresh: list[schema.CandidateRecord] = []
    absorptions: list[tuple[schema.CandidateRecord, schema.CandidateRecord]] = []
    for cand in new:
        match = _oracle_find(cand, pool)
        if match is not None:
            merged = list(getattr(match, "merged_connector_ids", None) or [])
            if cand.connector_id != match.connector_id and cand.connector_id not in merged:
                merged.append(cand.connector_id)
                match.merged_connector_ids = merged  # type: ignore[attr-defined]
            absorptions.append((cand, match))
            continue
        fresh.append(cand)
        pool.append(cand)
    return fresh, len(absorptions), absorptions


# --- генерация пулов с частыми коллизиями --------------------------------------------

# Крошечные алфавиты — намеренно: цель не «реалистичные данные», а густые коллизии по
# каждой из трёх стратегий и по нескольким сразу (кросс-стратегийный угол §4).
_URLS = st.sampled_from([None, "https://a.gov/1", "https://A.gov/1/", "https://a.gov/2"])
_URL_PROVENANCE = st.sampled_from(
    [None, schema.UrlProvenance.stated, schema.UrlProvenance.suspect]
)
_TITLES = st.sampled_from([None, "Doc One", "doc  one!", "Doc Two"])
_ISSUERS = st.sampled_from([None, "MinDigital", "MinFin"])
_DATES = st.sampled_from([None, dt.date(2026, 1, 1), dt.date(2026, 2, 2)])
_CONNECTORS = st.sampled_from(["manual", "agora", "search:wb"])
_NATIVE_IDS = st.sampled_from([None, "n1", "n2"])
_REJECTED = st.sampled_from([None, "вне обеих осей"])


@st.composite
def _candidate(draw: st.DrawFn, raw_hash: str) -> schema.CandidateRecord:
    return schema.CandidateRecord.model_validate(
        {
            "connector_id": draw(_CONNECTORS),
            "retrieved_at": dt.date(2026, 7, 25),
            "raw_hash": raw_hash,
            "title": draw(_TITLES),
            "issuer": draw(_ISSUERS),
            "doc_date": draw(_DATES),
            "source_url": draw(_URLS),
            "url_provenance": draw(_URL_PROVENANCE),
            "native_id": draw(_NATIVE_IDS),
            "rejected_reason": draw(_REJECTED),
        }
    )


@st.composite
def _pools(draw: st.DrawFn) -> tuple[list[schema.CandidateRecord], list[schema.CandidateRecord]]:
    n_existing = draw(st.integers(min_value=0, max_value=6))
    n_new = draw(st.integers(min_value=0, max_value=6))
    existing = [draw(_candidate(raw_hash=f"e{i}")) for i in range(n_existing)]
    new = [draw(_candidate(raw_hash=f"n{i}")) for i in range(n_new)]
    return existing, new


def _clone(records: list[schema.CandidateRecord]) -> list[schema.CandidateRecord]:
    """Глубокая копия: обе реализации мутируют existing на месте (merged_connector_ids)."""
    return [r.model_copy(deep=True) for r in records]


def _provenance(records: list[schema.CandidateRecord]) -> list[list[str] | None]:
    return [getattr(r, "merged_connector_ids", None) for r in records]


@given(pools=_pools())
@settings(max_examples=300)
def test_index_dedup_matches_naive_oracle(
    pools: tuple[list[schema.CandidateRecord], list[schema.CandidateRecord]],
) -> None:
    """Полное совпадение наблюдаемого поведения: какие кандидаты свежие, сколько
    поглощено, КОМУ достался провенанс поглощённого (цель merge — часть семантики)
    И какие именно пары (дубль, поглотитель) несёт ``DedupOutcome.absorptions``
    (spec discovery-acquire-seam-hardening §5, Г4 — не только счётчик)."""
    existing, new = pools

    prod_existing, prod_new = _clone(existing), _clone(new)
    oracle_existing, oracle_new = _clone(existing), _clone(new)

    outcome = dedup(prod_new, prod_existing)
    oracle_fresh, oracle_absorbed, oracle_absorptions = _oracle_dedup(oracle_new, oracle_existing)

    assert [c.raw_hash for c in outcome.fresh] == [c.raw_hash for c in oracle_fresh]
    assert outcome.absorbed == oracle_absorbed
    assert _provenance(prod_existing) == _provenance(oracle_existing)
    assert _provenance(prod_new) == _provenance(oracle_new)
    assert [(dup.raw_hash, absorber.raw_hash) for dup, absorber in outcome.absorptions] == [
        (dup.raw_hash, absorber.raw_hash) for dup, absorber in oracle_absorptions
    ]


@given(pools=_pools())
@settings(max_examples=200)
def test_dedup_invariants_hold(
    pools: tuple[list[schema.CandidateRecord], list[schema.CandidateRecord]],
) -> None:
    """Инварианты, не зависящие от оракула: баланс «поглощено + свежих == входных»,
    свежие — подмножество входных без повторов, отклонённые не воскресают."""
    existing, new = pools
    prod_existing, prod_new = _clone(existing), _clone(new)

    outcome = dedup(prod_new, prod_existing)

    assert outcome.absorbed + len(outcome.fresh) == len(prod_new)
    fresh_hashes = [c.raw_hash for c in outcome.fresh]
    assert len(fresh_hashes) == len(set(fresh_hashes))
    assert set(fresh_hashes) <= {c.raw_hash for c in prod_new}
    # ни одна existing-запись не потеряла/сменила rejected_reason
    assert [c.rejected_reason for c in prod_existing] == [c.rejected_reason for c in existing]


@given(pools=_pools())
@settings(max_examples=200)
def test_dedup_is_idempotent_against_persisted_pool(
    pools: tuple[list[schema.CandidateRecord], list[schema.CandidateRecord]],
) -> None:
    """Реконсиляционный инвариант discovery: повторный прогон по уже персистнутому
    результату поглощает ВСЁ, без исключений.

    Раньше исключением был кандидат без обоих ключей (ни достоверного URL, ни пары
    issuer+title): идентичности у него не было, и он честно приходил «свежим» каждый
    прогон. Идентичность записи в источнике (стратегия 3) закрывает этот класс —
    §23 бэклога.
    """
    existing, new = pools
    prod_existing, prod_new = _clone(existing), _clone(new)

    outcome = dedup(prod_new, prod_existing)
    fresh = outcome.fresh
    persisted = prod_existing + fresh

    again = dedup(_clone(fresh), persisted)

    assert again.fresh == []
    assert again.absorbed == len(fresh)
