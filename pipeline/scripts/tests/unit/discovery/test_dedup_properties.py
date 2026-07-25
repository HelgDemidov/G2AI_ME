"""Property-тесты индексного dedup против независимого наивного оракула
(spec discovery-candidates-sharding §4; hypothesis — общепроектный стандарт для такого класса).

Прод-реализация схлопнула три линейных скана в три dict-индекса. Эквивалентность
КАНОНИЧЕСКОЙ семантике (строгий приоритет стратегий url -> key -> hash над единым пулом)
доказывается сравнением с оракулом, живущим В ЭТОМ файле: наивные линейные сканы,
написанные независимо от прод-кода. Обёртка над самими индексами оракулом быть не может —
сравнение выродилось бы в тавтологию.
"""
from __future__ import annotations

import datetime as dt

from hypothesis import given, settings
from hypothesis import strategies as st

from core import schema
from discovery.dedup import dedup, normalized_title

# --- независимый наивный оракул -------------------------------------------------------


def _oracle_key(cand: schema.CandidateRecord) -> tuple[str, str, str] | None:
    if not (cand.title and cand.issuer):
        return None
    return (cand.issuer, normalized_title(cand.title), str(cand.doc_date))


def _oracle_find(
    cand: schema.CandidateRecord, pool: list[schema.CandidateRecord]
) -> schema.CandidateRecord | None:
    """Три линейных скана по ЕДИНОМУ пулу в порядке приоритета стратегий."""
    if cand.normalized_url:
        for other in pool:
            if other.normalized_url == cand.normalized_url:
                return other
    key = _oracle_key(cand)
    if key is not None:
        for other in pool:
            if _oracle_key(other) == key:
                return other
    if cand.content_hash:
        for other in pool:
            if other.content_hash == cand.content_hash:
                return other
    return None


def _oracle_dedup(
    new: list[schema.CandidateRecord], existing: list[schema.CandidateRecord]
) -> tuple[list[schema.CandidateRecord], int]:
    pool = list(existing)
    fresh: list[schema.CandidateRecord] = []
    absorbed = 0
    for cand in new:
        match = _oracle_find(cand, pool)
        if match is not None:
            merged = list(getattr(match, "merged_connector_ids", None) or [])
            if cand.connector_id != match.connector_id and cand.connector_id not in merged:
                merged.append(cand.connector_id)
                match.merged_connector_ids = merged  # type: ignore[attr-defined]
            absorbed += 1
            continue
        fresh.append(cand)
        pool.append(cand)
    return fresh, absorbed


# --- генерация пулов с частыми коллизиями --------------------------------------------

# Крошечные алфавиты — намеренно: цель не «реалистичные данные», а густые коллизии по
# каждой из трёх стратегий и по нескольким сразу (кросс-стратегийный угол §4).
_URLS = st.sampled_from([None, "https://a.gov/1", "https://a.gov/2"])
_TITLES = st.sampled_from([None, "Doc One", "doc  one!", "Doc Two"])
_ISSUERS = st.sampled_from([None, "MinDigital", "MinFin"])
_DATES = st.sampled_from([None, dt.date(2026, 1, 1), dt.date(2026, 2, 2)])
_HASHES = st.sampled_from([None, "c1", "c2"])
_CONNECTORS = st.sampled_from(["manual", "agora", "search:wb"])
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
            "normalized_url": draw(_URLS),
            "content_hash": draw(_HASHES),
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


def _has_dedup_key(cand: schema.CandidateRecord) -> bool:
    """Несёт ли кандидат хоть ОДИН из трёх ключей сравнения.

    Кандидат без всех трёх (ни URL, ни пары issuer+title, ни content_hash) не имеет
    идентичности для dedup и потому не дедуплицируется вовсе — свойство ИСХОДНОГО
    дизайна трёх стратегий (прежний линейный ``_find_match`` возвращал None так же),
    а не следствие индексации. Пинится тестом в ``test_dedup.py``.
    """
    return bool(cand.normalized_url or (cand.title and cand.issuer) or cand.content_hash)


@given(pools=_pools())
@settings(max_examples=300)
def test_index_dedup_matches_naive_oracle(
    pools: tuple[list[schema.CandidateRecord], list[schema.CandidateRecord]],
) -> None:
    """Полное совпадение наблюдаемого поведения: какие кандидаты свежие, сколько
    поглощено И КОМУ достался провенанс поглощённого (цель merge — часть семантики)."""
    existing, new = pools

    prod_existing, prod_new = _clone(existing), _clone(new)
    oracle_existing, oracle_new = _clone(existing), _clone(new)

    prod_fresh, prod_absorbed = dedup(prod_new, prod_existing)
    oracle_fresh, oracle_absorbed = _oracle_dedup(oracle_new, oracle_existing)

    assert [c.raw_hash for c in prod_fresh] == [c.raw_hash for c in oracle_fresh]
    assert prod_absorbed == oracle_absorbed
    assert _provenance(prod_existing) == _provenance(oracle_existing)
    assert _provenance(prod_new) == _provenance(oracle_new)


@given(pools=_pools())
@settings(max_examples=200)
def test_dedup_invariants_hold(
    pools: tuple[list[schema.CandidateRecord], list[schema.CandidateRecord]],
) -> None:
    """Инварианты, не зависящие от оракула: баланс «поглощено + свежих == входных»,
    свежие — подмножество входных без повторов, отклонённые не воскресают."""
    existing, new = pools
    prod_existing, prod_new = _clone(existing), _clone(new)

    fresh, absorbed = dedup(prod_new, prod_existing)

    assert absorbed + len(fresh) == len(prod_new)
    fresh_hashes = [c.raw_hash for c in fresh]
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
    результату поглощает всё, что несёт хоть один ключ сравнения.

    Единственное исключение — кандидат вообще БЕЗ ключей (см. ``_has_dedup_key``): у него
    нет идентичности, поэтому он честно приходит «свежим» снова. Тест пинит и это (точным
    равенством, не фильтрацией входа) — свойство видимо, а не замаскировано.
    """
    existing, new = pools
    prod_existing, prod_new = _clone(existing), _clone(new)

    fresh, _ = dedup(prod_new, prod_existing)
    persisted = prod_existing + fresh

    again_fresh, again_absorbed = dedup(_clone(fresh), persisted)

    keyless = [c.raw_hash for c in fresh if not _has_dedup_key(c)]
    assert [c.raw_hash for c in again_fresh] == keyless
    assert again_absorbed == len(fresh) - len(keyless)
