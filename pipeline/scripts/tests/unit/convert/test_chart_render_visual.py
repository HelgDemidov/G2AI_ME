"""Рендер-валидация mermaid-вывода chart_render.py настоящим mermaid.js
(spec chart-data-extraction, постревью-фикс 2026-07-22): наши собственные
эвристики (санитизация лейблов, verify+fallback-to-table) и синтакс-
валидаторы (`mermaid-parser-bundle`/`mermaid.parse()`, проверены отдельно,
не вошли в постоянные зависимости) ловят ГРАММАТИЧЕСКИЕ нарушения, но не
СЕМАНТИЧЕСКИЕ/визуальные — живой пример, найденный на реальном рендере:
``pie title "T"`` синтаксически валиден (кавычки внутри плоской строки —
не ошибка грамматики), но mermaid.js рендерит кавычки БУКВАЛЬНО (заголовок
диаграммы pie title — plain string, не quote-delimited, в отличие от
data-лейблов). Только фактический рендер это ловит.

``mermaidx`` (dev-зависимость, requirements-dev.txt) — Python-native,
без Node/npm/браузера: гоняет настоящий mermaid.js через embedded QuickJS.
Не runtime-зависимость пайплайна (решение пользователя 2026-07-22) —
только тестовая; `_convert_xlsx`/`_convert_docx` продолжают эмитить
mermaid БЕЗ рендер-проверки на каждой конвертации."""
from __future__ import annotations

import re

import mermaidx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from convert.chart_data import ChartData, ChartSeries
from convert.chart_render import render_chart

_QUOTE_RE = re.compile(r'"')


def _mermaid_fence(rendered: str) -> str:
    assert "```mermaid" in rendered
    return rendered.split("```mermaid", 1)[1].split("```", 1)[0].strip()


def _render_ok(code: str) -> str:
    """Реальный рендер через mermaidx -> SVG-строка. Кидает, если mermaid.js
    не принял диаграмму — тест-функции проверяют это через отсутствие
    исключения (`pytest.raises` не используется, крах теста = сигнал)."""
    return mermaidx.render(code).svg()  # type: ignore[no-any-return]


def test_pie_with_title_renders_without_literal_quotes_in_title() -> None:
    """Регрессия (найдена на реальном рендере govtech-фикстуры, 2026-07-22):
    заголовок pie — плоская строка, не quote-delimited. Живой формат дефекта:
    mermaid.js рендерит заголовок в SVG как
    ``<text class="pieTitleText">&quot;Institutional Responsibility&quot;</text>``
    — кавычки HTML-энкодятся как ``&quot;``, не остаются буквальным ``"``
    (проверено ручной инспекцией SVG на СЛОМАННОЙ версии перед фиксом —
    наивная проверка на буквальный ``"`` эту форму пропустила бы)."""
    data = ChartData(
        chart_type="pie", title="Institutional Responsibility", value_axis_title=None,
        value_format=None, stacked=False, categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="pie"),),
    )
    rendered = render_chart(data)
    assert rendered is not None
    svg = _render_ok(_mermaid_fence(rendered))
    title_match = re.search(r'class="pieTitleText"[^>]*>(.*?)</text>', svg)
    assert title_match is not None, "pieTitleText не найден в SVG"
    title_text = title_match.group(1)
    assert not title_text.startswith("&quot;")
    assert not title_text.endswith("&quot;")
    assert title_text == "Institutional Responsibility"


def test_doughnut_with_many_categories_renders() -> None:
    data = ChartData(
        chart_type="doughnut", title="Institutional Responsibility for GovTech",
        value_axis_title=None, value_format="General", stacked=False,
        categories=("Unknown", "Autonomous Entity", "Ministry of ICT", "Other"),
        series=(ChartSeries(name="Economies", values=(15.0, 14.0, 93.0, 16.0), kind="doughnut"),),
    )
    rendered = render_chart(data)
    assert rendered is not None
    _render_ok(_mermaid_fence(rendered))


def test_xychart_bar_line_combo_renders() -> None:
    data = ChartData(
        chart_type="combo", title=None, value_axis_title="Average GTMI score",
        value_format="#,##0.000", stacked=False, categories=("A", "B", "C", "D"),
        series=(
            ChartSeries(name="Avg GTMI", values=(0.86, 0.61, 0.37, 0.14), kind="column"),
            ChartSeries(name="Reg Avg", values=(0.59, 0.59, 0.59, 0.59), kind="line"),
        ),
    )
    rendered = render_chart(data)
    assert rendered is not None
    _render_ok(_mermaid_fence(rendered))


def test_radar_multi_series_renders() -> None:
    data = ChartData(
        chart_type="radar", title="GovTech Maturity Index Components", value_axis_title=None,
        value_format=None, stacked=False, categories=("CGSI", "PSDI", "DCEI", "GTEI"),
        series=(
            ChartSeries(name="Regional Avg", values=(0.49, 0.51, 0.25, 0.46), kind="radar"),
            ChartSeries(name="Global Avg", values=(0.62, 0.66, 0.47, 0.61), kind="radar"),
        ),
    )
    rendered = render_chart(data)
    assert rendered is not None
    _render_ok(_mermaid_fence(rendered))


def test_pie_title_with_special_characters_still_renders() -> None:
    """Санитизация (``_sanitize_label``) вырезает ``[]{}()`` и заменяет
    двойные кавычки на одинарные ДО попадания в mermaid — проверяем, что
    после нашей очистки настоящий mermaid.js всё ещё принимает результат."""
    data = ChartData(
        chart_type="pie", title='Value (A) [1] "quoted"', value_axis_title=None,
        value_format=None, stacked=False, categories=("X", "Y"),
        series=(ChartSeries(name="S", values=(3.0, 4.0), kind="pie"),),
    )
    rendered = render_chart(data)
    assert rendered is not None
    _render_ok(_mermaid_fence(rendered))


# --- Hypothesis: узкая, но настоящая рендер-проверка (не только наши эвристики) ---
# max_examples мал (в отличие от чисто-эвристического property-теста в
# test_chart_render.py, 200 примеров) — каждый вызов mermaidx.render() стоит
# реальное время (десятки мс на прогретом движке), полный масштаб избыточен
# для этой конкретной проверки (ловит класс дефектов, не конкретные значения).

_LABEL_ALPHABET = "AaBbCc 0123456789"  # без []{}()"' — те уже покрыты test_pie_title_with_special_characters_still_renders
_labels = st.text(alphabet=_LABEL_ALPHABET, min_size=1, max_size=12)
_chart_types = st.sampled_from(["column", "bar", "line", "area", "pie", "doughnut", "radar"])


@st.composite
def _renderable_chart_data(draw: st.DrawFn) -> ChartData:
    chart_type = draw(_chart_types)
    n_cats = draw(st.integers(min_value=2, max_value=5))
    categories = tuple(draw(_labels) for _ in range(n_cats))
    values = tuple(draw(st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False)) for _ in range(n_cats))
    n_series = 1 if chart_type in ("pie", "doughnut") else draw(st.integers(min_value=1, max_value=3))
    series = tuple(
        ChartSeries(name=draw(_labels), values=values, kind=chart_type)
        for _ in range(n_series)
    )
    return ChartData(
        chart_type=chart_type,
        title=draw(st.one_of(st.none(), _labels)),
        value_axis_title=draw(st.one_of(st.none(), _labels)),
        value_format=None,
        stacked=False,
        categories=categories,
        series=series,
    )


@given(data=_renderable_chart_data())
@settings(max_examples=30, deadline=None)
def test_render_chart_mermaid_output_accepted_by_real_mermaid_js(data: ChartData) -> None:
    rendered = render_chart(data)
    if rendered is None or "```mermaid" not in rendered:
        return
    try:
        _render_ok(_mermaid_fence(rendered))
    except Exception as exc:  # noqa: BLE001 — любой отказ реального рендера — находка теста
        pytest.fail(f"mermaidx отказался рендерить сгенерированный mermaid: {exc}\n\n{rendered}")
