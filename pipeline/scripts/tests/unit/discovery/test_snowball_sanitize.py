"""Тесты discovery/connectors/snowball.py — санитизация URL, отсев самоссылок/уже-в-корпусе,
группировка аннотаций по uri (spec discovery-snowball §2.4, коммит 2). Property-based часть
(``group_by_uri``) — Hypothesis, общепроектный стандарт для геометрического/порядкового кода
(test-coverage-hardening): порядок ВХОДНОГО списка аннотаций не должен влиять на порядок
СГРУППИРОВАННОГО anchor-текста — только геометрия (top, x0)."""
from __future__ import annotations

import random
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core import schema
from discovery.connectors.snowball import (
    group_by_uri,
    is_self_or_corpus_link,
    sanitize_url,
)
from tests.support import valid_record

# --- sanitize_url ---


@pytest.mark.parametrize(
    "raw",
    [
        "https://ai.gov.eg/strategy.pdf",
        "http://example.org/doc",
        "https://example.org/doc).",  # хвостовая пунктуация — срезается, не отсеивает
        "https://example.org/doc»",
    ],
)
def test_sanitize_url_accepts_valid(raw: str) -> None:
    assert sanitize_url(raw) is not None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "http://a",  # живой пример GAIRI: нет точки в хосте И короче порога
        "mailto:research@oxfordinsights.com",  # не-http(s) схема
        "javascript:void(0)",
        "ftp://example.org/file",
        "https://noTLD",  # нет точки в хосте
        "http://x.co",  # < _MIN_URL_LENGTH (12 символов)
    ],
)
def test_sanitize_url_rejects_garbage(raw: str | None) -> None:
    assert sanitize_url(raw) is None


def test_sanitize_url_strips_trailing_punctuation_not_the_whole_url() -> None:
    assert sanitize_url("https://example.org/page).") == "https://example.org/page"
    assert sanitize_url("https://example.org/page»") == "https://example.org/page"


def test_sanitize_url_normalizes_to_nfc() -> None:
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "https://example.org/café")
    result = sanitize_url(decomposed)
    assert result == unicodedata.normalize("NFC", decomposed)


# --- is_self_or_corpus_link ---


def _record_with_url(url: str) -> schema.SourceRecord:
    data = valid_record()
    data["source_url"] = url
    return schema.SourceRecord.model_validate(data)


def test_self_link_is_filtered() -> None:
    assert is_self_or_corpus_link(
        "https://example.org/self", source_url="https://example.org/self", records=[]
    )


def test_corpus_link_is_filtered() -> None:
    other = _record_with_url("https://example.org/other-doc")
    assert is_self_or_corpus_link(
        "https://example.org/other-doc", source_url="https://example.org/self", records=[other]
    )


def test_unrelated_link_is_not_filtered() -> None:
    other = _record_with_url("https://example.org/other-doc")
    assert not is_self_or_corpus_link(
        "https://example.org/genuinely-new", source_url="https://example.org/self", records=[other]
    )


# --- group_by_uri: концретные сценарии ---


def test_group_by_uri_splits_by_uri_and_sorts_by_reading_order() -> None:
    annots = [
        {"uri": "https://a.org", "top": 100.0, "x0": 10.0},
        {"uri": "https://b.org", "top": 50.0, "x0": 10.0},
        {"uri": "https://a.org", "top": 50.0, "x0": 10.0},
    ]
    groups = group_by_uri(annots)
    assert set(groups) == {"https://a.org", "https://b.org"}
    assert [a["top"] for a in groups["https://a.org"]] == [50.0, 100.0]


def test_group_by_uri_ignores_annotations_without_uri() -> None:
    annots: list[dict[str, Any]] = [{"uri": None, "top": 0.0, "x0": 0.0}, {"top": 0.0, "x0": 0.0}]
    assert group_by_uri(annots) == {}


# --- group_by_uri: property — порядок входа не влияет на порядок группы ---


@st.composite
def _annot_group(draw: Any, min_size: int = 2, max_size: int = 8) -> list[dict[str, Any]]:
    """Синтетические аннотации ОДНОГО uri со случайными РАЗЛИЧНЫМИ (top, x0) — реальный
    инвариант: ``top``/``x0`` неотрицательны, как у настоящих pdfplumber-объектов.

    Различность пар — намеренное сужение домена: две аннотации на буквально идентичных
    координатах — вырожденный случай, где «порядок чтения» неопределён по построению
    (любой алгоритм, не только наш, развязывает такую тройную ничью произвольно чем-то
    ВНЕ геометрии) — не то, что проверяет этот инвариант."""
    pairs = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=600.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=min_size,
            max_size=max_size,
            unique=True,
        )
    )
    return [
        {"uri": "https://example.org/fixed", "top": top, "x0": x0, "_id": i}
        for i, (top, x0) in enumerate(pairs)
    ]


@given(annots=_annot_group())
@settings(max_examples=100)
def test_group_by_uri_order_invariant_to_input_shuffle(annots: list[dict[str, Any]]) -> None:
    shuffled = list(annots)
    random.Random(0).shuffle(shuffled)

    ordered_a = group_by_uri(annots)["https://example.org/fixed"]
    ordered_b = group_by_uri(shuffled)["https://example.org/fixed"]

    assert [a["_id"] for a in ordered_a] == [a["_id"] for a in ordered_b]
    # монотонность: результат отсортирован по (top, x0) вне зависимости от входа
    keys = [(a["top"], a["x0"]) for a in ordered_a]
    assert keys == sorted(keys)
