"""Property-based тесты геометрии pdf_to_markdown (test-coverage-hardening, Hypothesis —
принят как общепроектный стандарт с этого спека, не разово для этого файла).

Существующий test_pdf_to_markdown.py тестирует конкретные вручную собранные сценарии
(каждый — регресс живого аудита); эти тесты проверяют ИНВАРИАНТЫ на случайных раскладках
Word — прямое лекарство от документированного в CLAUDE.md риска «пороги откалиброваны на
одном документе» (§ «Известное ограничение»)."""
from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from convert.pdf_to_markdown import (
    HEADING_MIN_RATIO,
    TINY_MARKER_RATIO,
    Word,
    compute_doc_stats,
    detect_columns,
    group_into_lines,
)


@st.composite
def _word_lists(draw: Any, min_size: int = 0, max_size: int = 25) -> list[Word]:
    """Синтетические Word со структурными инвариантами реальных: x1>x0, bottom>top,
    size>0 — теми же, что даёт load_words() на настоящем pdfplumber."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    out: list[Word] = []
    for i in range(n):
        x0 = draw(st.floats(min_value=0.0, max_value=550.0, allow_nan=False, allow_infinity=False))
        width = draw(st.floats(min_value=1.0, max_value=60.0, allow_nan=False, allow_infinity=False))
        top = draw(st.floats(min_value=0.0, max_value=750.0, allow_nan=False, allow_infinity=False))
        height = draw(st.floats(min_value=1.0, max_value=30.0, allow_nan=False, allow_infinity=False))
        size = draw(st.floats(min_value=1.0, max_value=30.0, allow_nan=False, allow_infinity=False))
        out.append(Word(f"w{i}", x0=x0, x1=x0 + width, top=top, bottom=top + height, size=size))
    return out


# --- group_into_lines: партиционирование — ни одно слово не теряется и не дублируется ---


@given(words=_word_lists())
@settings(max_examples=100)
def test_group_into_lines_partitions_all_words(words: list[Word]) -> None:
    lines = group_into_lines(words)
    flat = [w for line in lines for w in line]
    assert len(flat) == len(words)
    assert {id(w) for w in flat} == {id(w) for w in words}  # ровно те же объекты


@given(words=_word_lists(min_size=1))
@settings(max_examples=50)
def test_group_into_lines_never_empty_for_nonempty_input(words: list[Word]) -> None:
    assert group_into_lines(words) != []


@given(words=_word_lists())
@settings(max_examples=100)
def test_group_into_lines_sorted_by_top_non_decreasing(words: list[Word]) -> None:
    lines = group_into_lines(words)
    line_tops = [min(w.top for w in line) for line in lines]
    assert line_tops == sorted(line_tops)


# --- detect_columns: партиционирование ширины страницы, максимум 2 колонки ---


_PAGE_WIDTH = st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


@given(words=_word_lists(), page_width=_PAGE_WIDTH)
@settings(max_examples=100)
def test_detect_columns_never_crashes_and_tiles_page_width(words: list[Word], page_width: float) -> None:
    columns = detect_columns(words, page_width)
    assert len(columns) in (1, 2)  # известное ограничение: максимум один разрыв
    assert columns[0][0] == 0
    assert columns[-1][1] == page_width
    for (_a0, a1), (b0, _b1) in zip(columns, columns[1:]):  # noqa: B905 — 1-колоночный случай даёт 0 итераций
        assert a1 == b0  # колонки стыкуются без зазора/нахлёста


@given(page_width=_PAGE_WIDTH)
@settings(max_examples=20)
def test_detect_columns_empty_words_always_single_full_width_column(page_width: float) -> None:
    assert detect_columns([], page_width) == [(0, page_width)]


@given(words=_word_lists(), page_width=_PAGE_WIDTH)
@settings(max_examples=100)
def test_detect_columns_gap_never_slices_through_a_word(words: list[Word], page_width: float) -> None:
    """Найденный разрыв — по построению зона БЕЗ покрытия слов (mark() отмечает каждое
    слово целиком) — граница между колонками не может проходить сквозь середину слова."""
    columns = detect_columns(words, page_width)
    if len(columns) == 2:
        mid = columns[0][1]
        for w in words:
            assert not (w.x0 < mid < w.x1)


# --- compute_doc_stats: не падает на произвольном вводе, инварианты сводки ---


@given(
    words=_word_lists(max_size=15),
    page_height=st.floats(min_value=100.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_compute_doc_stats_never_crashes_and_respects_heading_ratio(
    words: list[Word], page_height: float
) -> None:
    stats = compute_doc_stats([(words, page_height)])
    assert stats.body_size > 0
    assert all(h >= stats.body_size * HEADING_MIN_RATIO - 1e-9 for h in stats.heading_sizes)
    assert stats.heading_sizes == sorted(stats.heading_sizes, reverse=True)
    assert stats.tiny_marker_max == pytest.approx(stats.body_size * TINY_MARKER_RATIO)
