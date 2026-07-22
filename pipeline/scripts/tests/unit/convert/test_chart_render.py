"""Тесты chart_render.py (spec chart-data-extraction §3): ``render_chart``
(``ChartData`` -> GFM-таблица + mermaid). Опирается на конструирование
``ChartData``/``ChartSeries`` напрямую (чистые dataclasses, без XML) —
XML-к-``ChartData`` уже покрыт ``test_chart_data.py``."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from convert.chart_data import ChartData, ChartSeries
from convert.chart_render import render_chart


def _data(
    *,
    chart_type: str = "column",
    title: str | None = None,
    value_axis_title: str | None = None,
    value_format: str | None = None,
    stacked: bool = False,
    categories: tuple[str, ...] = (),
    series: tuple[ChartSeries, ...] = (),
) -> ChartData:
    return ChartData(
        chart_type=chart_type, title=title, value_axis_title=value_axis_title,
        value_format=value_format, stacked=stacked, categories=categories, series=series,
    )


def test_returns_none_for_empty_series() -> None:
    assert render_chart(_data(categories=("A",), series=())) is None


def test_returns_none_when_all_series_values_are_none() -> None:
    data = _data(categories=("A", "B"), series=(ChartSeries(name="S", values=(None, None), kind="column"),))
    assert render_chart(data) is None


def test_table_always_present() -> None:
    data = _data(categories=("A", "B"), series=(ChartSeries(name="S", values=(1.0, 2.0), kind="column"),))
    out = render_chart(data)
    assert out is not None
    assert "| Category | S |" in out
    assert "| A | 1 |" in out
    assert "| B | 2 |" in out


def test_none_value_renders_as_empty_table_cell() -> None:
    data = _data(categories=("A", "B"), series=(ChartSeries(name="S", values=(1.0, None), kind="column"),))
    out = render_chart(data)
    assert out is not None
    assert "| B |  |" in out


def test_order_is_caption_then_mermaid_then_table() -> None:
    data = _data(
        chart_type="pie", title="T", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="pie"),),
    )
    out = render_chart(data)
    assert out is not None
    assert out.index("T") < out.index("```mermaid") < out.index("| Category |")


def test_series_without_name_gets_synthetic_header() -> None:
    data = _data(categories=("A",), series=(ChartSeries(name=None, values=(1.0,), kind="column"),))
    out = render_chart(data)
    assert out is not None
    assert "| Category | Series 1 |" in out


def test_worked_example_chartdata_to_markdown_verbatim() -> None:
    """Golden-target, половина 2 (спек §Тестовое покрытие — ChartData ->
    markdown): та же ``ChartData``, что
    ``test_chart_data.py::test_worked_example_bar_chart_xml_to_chartdata_verbatim``
    получает из чарт-XML (собрана здесь напрямую, без XML — см. докстроку
    того теста про причину раздельности) -> точный конечный markdown,
    дословно."""
    data = _data(
        chart_type="column",
        title="Regional Comparison",
        value_axis_title="Score",
        value_format="0.0%",
        categories=("Montenegro", "Estonia"),
        series=(
            ChartSeries(name="2024", values=(0.42, 0.87), kind="column"),
            ChartSeries(name="2025", values=(0.55, 0.91), kind="column"),
        ),
    )
    assert render_chart(data) == (
        'Regional Comparison — Score\n\n'
        '```mermaid\n'
        'xychart-beta\n'
        'x-axis ["Montenegro", "Estonia"]\n'
        'y-axis "Score" 0 --> 0.91\n'
        'bar [0.42, 0.87]\n'
        'bar [0.55, 0.91]\n'
        '```\n\n'
        '| Category | 2024 | 2025 |\n'
        '| --- | --- | --- |\n'
        '| Montenegro | 42.0% | 55.0% |\n'
        '| Estonia | 87.0% | 91.0% |'
    )


def test_no_categories_synthesizes_index_row_labels() -> None:
    data = _data(categories=(), series=(ChartSeries(name="S", values=(1.0, 2.0, 3.0), kind="column"),))
    out = render_chart(data)
    assert out is not None
    assert "| 1 | 1 |" in out
    assert "| 3 | 3 |" in out


# --- mermaid маппинг ---


def _pie_data(n_series: int = 1, values: tuple[float, ...] = (1.0, 2.0)) -> ChartData:
    series = tuple(ChartSeries(name=f"S{i}", values=values, kind="pie") for i in range(n_series))
    return _data(chart_type="pie", title="Pie", categories=("A", "B"), series=series)


def test_pie_type_gets_pie_mermaid() -> None:
    out = render_chart(_pie_data())
    assert out is not None
    assert "```mermaid\npie" in out


def test_doughnut_type_maps_to_pie_mermaid() -> None:
    data = _data(
        chart_type="doughnut", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="doughnut"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid\npie" in out


def test_pie_with_more_than_one_series_falls_back_to_table_only() -> None:
    out = render_chart(_pie_data(n_series=2))
    assert out is not None
    assert "```mermaid" not in out


def test_pie_with_negative_value_falls_back_to_table_only() -> None:
    data = _data(
        chart_type="pie", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, -2.0), kind="pie"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid" not in out


def test_column_type_gets_xychart_beta() -> None:
    data = _data(
        chart_type="column", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="column"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid\nxychart-beta" in out
    assert "bar [1, 2]" in out


def test_line_type_gets_xychart_beta_with_line_series() -> None:
    data = _data(
        chart_type="line", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="line"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "line [1, 2]" in out


def test_area_type_gets_xychart_beta() -> None:
    data = _data(
        chart_type="area", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="area"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid\nxychart-beta" in out


def test_bar_line_combo_gets_overlay_with_both_kinds() -> None:
    data = _data(
        chart_type="combo", categories=("A", "B"),
        series=(
            ChartSeries(name="Bars", values=(1.0, 2.0), kind="column"),
            ChartSeries(name="Line", values=(3.0, 4.0), kind="line"),
        ),
    )
    out = render_chart(data)
    assert out is not None
    assert "bar [1, 2]" in out
    assert "line [3, 4]" in out


def test_radar_type_gets_radar_beta() -> None:
    data = _data(
        chart_type="radar", categories=("Axis1", "Axis2"),
        series=(ChartSeries(name="Series A", values=(0.5, 0.7), kind="radar"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid\nradar-beta" in out
    assert 'curve series_a["Series A"]{0.5, 0.7}' in out


def test_scatter_type_gets_table_only_no_mermaid_construct() -> None:
    data = _data(
        chart_type="scatter", categories=("1984", "1985"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="scatter"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid" not in out
    assert "| Category | S |" in out  # таблица всё равно есть — лосслесс


def test_stacked_bar_falls_back_to_table_only() -> None:
    data = _data(
        chart_type="column", stacked=True, categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="column"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid" not in out


def test_unrecognized_other_type_gets_table_only() -> None:
    data = _data(
        chart_type="other", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="other"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid" not in out


def test_mismatched_series_length_falls_back_to_table_only() -> None:
    """verify+fallback-to-table (спек §3): форма серии рискованна (длина не
    совпадает с категориями) -> mermaid снимается целиком, таблица честно
    показывает то, что есть."""
    data = _data(
        chart_type="column", categories=("A", "B", "C"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="column"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid" not in out


def test_sparse_none_in_series_falls_back_to_table_only_for_xychart() -> None:
    data = _data(
        chart_type="column", categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, None), kind="column"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "```mermaid" not in out


# --- Санитизация лейблов ---


def test_labels_with_brackets_and_quotes_are_sanitized_in_mermaid() -> None:
    data = _data(
        chart_type="pie", categories=('Value (A) [1] {x}', 'Contains "quotes"'),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="pie"),),
    )
    out = render_chart(data)
    assert out is not None
    fence = out.split("```mermaid", 1)[1].split("```", 1)[0]
    assert "[" not in fence.replace("x-axis [", "").replace("bar [", "").replace("line [", "")
    assert "(" not in fence and ")" not in fence and "{" not in fence
    assert "Value A 1 x" in fence
    assert "Contains 'quotes'" in fence


def test_empty_label_after_sanitization_gets_placeholder() -> None:
    data = _data(
        chart_type="pie", categories=("[]", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind="pie"),),
    )
    out = render_chart(data)
    assert out is not None
    assert '"?"' in out


# --- value_format / округление-фолбэк ---


def test_percent_format_scales_and_appends_percent_sign() -> None:
    data = _data(
        value_format="0.0%", categories=("A",),
        series=(ChartSeries(name="S", values=(0.589,), kind="column"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "| A | 58.9% |" in out


def test_decimal_format_uses_digit_count_from_code() -> None:
    data = _data(
        value_format="#,##0.000", categories=("A",),
        series=(ChartSeries(name="S", values=(0.58909698401216537,), kind="column"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "| A | 0.589 |" in out


def test_no_format_falls_back_to_rounding_not_raw_float() -> None:
    data = _data(
        categories=("A",), series=(ChartSeries(name="S", values=(0.58909698401216537,), kind="column"),),
    )
    out = render_chart(data)
    assert out is not None
    assert "0.58909698401216537" not in out
    assert "| A | 0.5891 |" in out


# --- Hypothesis: no-crash property (спек: mermaid для маппящихся типов
# синтаксически корректен) ---

_LABEL_ALPHABET = "AaBbCc []{}()\"'0123456789"
_labels = st.text(alphabet=_LABEL_ALPHABET, min_size=0, max_size=10)
_chart_types = st.sampled_from(
    ["column", "bar", "line", "area", "combo", "pie", "doughnut", "radar", "scatter", "other"]
)
_kinds = st.sampled_from(["column", "bar", "line", "area", "pie", "doughnut", "radar", "scatter", "other"])
_values = st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))


@st.composite
def _chart_data(draw: st.DrawFn) -> ChartData:
    chart_type = draw(_chart_types)
    categories = tuple(draw(_labels) for _ in range(draw(st.integers(min_value=0, max_value=5))))
    series = tuple(
        ChartSeries(
            name=draw(st.one_of(st.none(), _labels)),
            values=tuple(draw(st.lists(_values, min_size=0, max_size=5))),
            kind=draw(_kinds),
        )
        for _ in range(draw(st.integers(min_value=0, max_value=3)))
    )
    return ChartData(
        chart_type=chart_type,
        title=draw(st.one_of(st.none(), _labels)),
        value_axis_title=draw(st.one_of(st.none(), _labels)),
        value_format=draw(st.sampled_from([None, "0.0%", "#,##0.000", "General", "0"])),
        stacked=draw(st.booleans()),
        categories=categories,
        series=series,
    )


@given(data=_chart_data())
@settings(max_examples=200)
def test_render_chart_never_crashes_and_output_is_well_formed(data: ChartData) -> None:
    result = render_chart(data)
    if result is None:
        return
    assert "| Category |" in result
    if "```mermaid" in result:
        assert result.count("```mermaid") == 1
        assert result.count("```") == 2
        fence = result.split("```mermaid", 1)[1].split("```", 1)[0]
        for line in fence.splitlines():
            assert line.count('"') % 2 == 0
