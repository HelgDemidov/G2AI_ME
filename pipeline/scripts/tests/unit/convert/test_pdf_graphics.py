"""Тесты pdf_graphics: элементы, дедуп, фильтр, кластеризация, guards. Вся
геометрия — синтетика, без единого реального PDF (spec convert-graphics §1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from convert.pdf_graphics import (
    Element,
    cluster_elements,
    dedupe_elements,
    filter_elements,
    region_guards_ok,
    region_words,
    union_bbox,
)

PAGE_W, PAGE_H = 600.0, 800.0
PAGE_AREA = PAGE_W * PAGE_H


@dataclass
class _Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


def _rect(x0: float, top: float, x1: float, bottom: float) -> Element:
    return Element("rect", x0, top, x1, bottom)


# --- dedupe_elements ---


def test_dedupe_collapses_stroke_fill_pair_within_tolerance() -> None:
    a = _rect(10.0, 10.0, 100.0, 60.0)
    b = _rect(10.2, 10.1, 100.1, 59.9)  # тот же бокс, stroke+fill вариация < 0.5pt
    assert dedupe_elements([a, b]) == [a]


def test_dedupe_keeps_distinct_boxes() -> None:
    a = _rect(10.0, 10.0, 100.0, 60.0)
    b = _rect(200.0, 10.0, 300.0, 60.0)  # далеко — не дубль
    assert dedupe_elements([a, b]) == [a, b]


def test_dedupe_does_not_collapse_across_kinds() -> None:
    rect = _rect(10.0, 10.0, 100.0, 60.0)
    image = Element("image", 10.0, 10.0, 100.0, 60.0)  # тот же bbox, другой kind
    assert dedupe_elements([rect, image]) == [rect, image]


# --- filter_elements ---


def test_filter_excludes_thin_ruler() -> None:
    ruler = _rect(10.0, 10.0, 400.0, 12.0)  # высота 2pt < ELEMENT_MIN_SIDE_PT (4pt)
    assert filter_elements([ruler], PAGE_AREA) == []


def test_filter_excludes_page_background() -> None:
    bg = _rect(0.0, 0.0, PAGE_W, PAGE_H)  # 100% площади > ELEMENT_MAX_PAGE_FRACTION (85%)
    assert filter_elements([bg], PAGE_AREA) == []


def test_filter_keeps_normal_box() -> None:
    box = _rect(10.0, 10.0, 110.0, 60.0)  # 100x50pt, обычный бокс фигуры
    assert filter_elements([box], PAGE_AREA) == [box]


# --- cluster_elements ---


def test_cluster_splits_two_distant_groups() -> None:
    near_a = _rect(10.0, 10.0, 50.0, 50.0)
    near_b = _rect(55.0, 10.0, 95.0, 50.0)  # 5pt зазор < CLUSTER_GAP_PT (8pt) — связан с near_a
    far = _rect(500.0, 500.0, 540.0, 540.0)  # далеко — отдельный кластер
    clusters = cluster_elements([near_a, near_b, far])
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_gap_threshold_determines_connectivity() -> None:
    a = _rect(0.0, 0.0, 10.0, 10.0)
    b = _rect(30.0, 0.0, 40.0, 10.0)  # зазор 20pt между bbox
    connected = cluster_elements([a, b], gap=15.0)  # 15*2=30 >= 20pt зазора -> связаны
    disconnected = cluster_elements([a, b], gap=5.0)  # 5*2=10 < 20pt -> раздельны
    assert len(connected) == 1
    assert len(disconnected) == 2


def test_cluster_single_element_is_its_own_cluster() -> None:
    a = _rect(0.0, 0.0, 10.0, 10.0)
    assert cluster_elements([a]) == [[a]]


# --- region_words / union_bbox ---


def test_union_bbox_spans_all_elements() -> None:
    a = _rect(10.0, 20.0, 30.0, 40.0)
    b = _rect(50.0, 5.0, 70.0, 25.0)
    assert union_bbox([a, b]) == (10.0, 5.0, 70.0, 40.0)


def test_region_words_selects_by_center() -> None:
    inside = _Word("inside", x0=15.0, x1=25.0, top=22.0, bottom=30.0)  # центр (20,26) внутри
    outside = _Word("outside", x0=200.0, x1=220.0, top=200.0, bottom=210.0)
    bbox = (10.0, 20.0, 30.0, 40.0)
    assert region_words([inside, outside], bbox) == [inside]


# --- region_guards_ok: три guard'а (§1, чартер §7.1) ---


def test_guards_reject_too_few_elements() -> None:
    """< REGION_MIN_ELEMENTS (3) — callout/плашка, не фигура."""
    elements = [_rect(0.0, 0.0, 50.0, 20.0), _rect(0.0, 25.0, 50.0, 45.0)]  # 2 элемента
    words: list[Any] = []
    assert region_guards_ok(elements, words, PAGE_AREA) is False


def test_guards_reject_near_full_page_region() -> None:
    """Регион на ~всю страницу — рамка/подложка, не фигура."""
    elements = [
        _rect(0.0, 0.0, PAGE_W, PAGE_H * 0.4),
        _rect(0.0, PAGE_H * 0.4, PAGE_W, PAGE_H * 0.7),
        _rect(0.0, PAGE_H * 0.7, PAGE_W, PAGE_H * 0.98),
    ]
    assert region_guards_ok(elements, [], PAGE_AREA) is False


def test_guards_reject_dense_prose_callout() -> None:
    """Плотный текст (> REGION_MAX_CHARS_PER_ELEMENT/элемент) — боксированная
    проза (callout/кейс-стади), не инфографика."""
    elements = [_rect(0.0, 0.0, 300.0, 100.0), _rect(0.0, 110.0, 300.0, 210.0), _rect(0.0, 220.0, 300.0, 320.0)]
    long_text = "x" * 800  # 800 симв. / 3 элемента = 267 > 250
    words = [_Word(long_text, x0=10.0, x1=290.0, top=50.0, bottom=60.0)]
    assert region_guards_ok(elements, words, PAGE_AREA) is False


def test_guards_accept_plausible_swot_region() -> None:
    """Правдоподобная SWOT-подобная фигура (4 бокса, короткие подписи) проходит
    все три guard'а."""
    elements = [
        _rect(0.0, 0.0, 100.0, 100.0), _rect(110.0, 0.0, 210.0, 100.0),
        _rect(0.0, 110.0, 100.0, 210.0), _rect(110.0, 110.0, 210.0, 210.0),
    ]
    words = [_Word("Strength", x0=10.0, x1=60.0, top=10.0, bottom=20.0)]
    assert region_guards_ok(elements, words, PAGE_AREA) is True
