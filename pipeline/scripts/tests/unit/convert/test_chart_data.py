"""Тесты chart_data.py (spec chart-data-extraction §3): ``parse_chart`` на
container-agnostic chart-XML (идентичная DrawingML-схема xlsx/docx). Чистый
XML in-memory, собранный ВРУЧНУЮ — ``openpyxl.chart`` для фикстур не годится:
эмпирически подтверждено (2026-07-22), что ``chart.add_data()`` пишет чарт
БЕЗ ``c:numCache``/``c:strCache`` вовсе (только ``<c:f>``-ссылку), а v1
парсера читает ИСКЛЮЧИТЕЛЬНО кэш. Формы XML ниже сверены с реальными чартами govtech-фикстуры
(strLit/xVal-yVal/серия без собственного ``c:cat`` — все три живые находки
этой сессии; фикстура сама с тех пор дважды переименована/пересобрана,
см. ``tests/fixtures/local/README.md``)."""
from __future__ import annotations

from lxml import etree

from convert.chart_data import ChartData, ChartSeries, parse_chart

_C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _pts(values: list[str]) -> str:
    return "".join(f'<c:pt idx="{i}"><c:v>{v}</c:v></c:pt>' for i, v in enumerate(values) if v is not None)


def _sparse_pts(indexed: dict[int, str]) -> str:
    return "".join(f'<c:pt idx="{i}"><c:v>{v}</c:v></c:pt>' for i, v in indexed.items())


def _str_cat(values: list[str], count: int | None = None) -> str:
    n = count if count is not None else len(values)
    return (
        f'<c:cat><c:strRef><c:f>Sheet1!$A$2:$A${n + 1}</c:f>'
        f'<c:strCache><c:ptCount val="{n}"/>{_pts(values)}</c:strCache></c:strRef></c:cat>'
    )


def _str_lit_cat(values: list[str]) -> str:
    return f'<c:cat><c:strLit><c:ptCount val="{len(values)}"/>{_pts(values)}</c:strLit></c:cat>'


def _num_val(values: list[str], fmt: str | None = None, count: int | None = None) -> str:
    n = count if count is not None else len(values)
    fmt_xml = f"<c:formatCode>{fmt}</c:formatCode>" if fmt else ""
    return (
        f'<c:val><c:numRef><c:f>Sheet1!$B$2:$B${n + 1}</c:f>'
        f'<c:numCache>{fmt_xml}<c:ptCount val="{n}"/>{_pts(values)}</c:numCache></c:numRef></c:val>'
    )


def _sparse_num_val(indexed: dict[int, str], count: int) -> str:
    return (
        f'<c:val><c:numRef><c:f>Sheet1!$B$2:$B${count + 1}</c:f>'
        f'<c:numCache><c:ptCount val="{count}"/>{_sparse_pts(indexed)}</c:numCache></c:numRef></c:val>'
    )


def _num_lit_val(values: list[str]) -> str:
    return f'<c:val><c:numLit><c:ptCount val="{len(values)}"/>{_pts(values)}</c:numLit></c:val>'


def _xval(values: list[str]) -> str:
    return (
        f'<c:xVal><c:numRef><c:f>Sheet1!$A$2:$A${len(values) + 1}</c:f>'
        f'<c:numCache><c:ptCount val="{len(values)}"/>{_pts(values)}</c:numCache></c:numRef></c:xVal>'
    )


def _yval(values: list[str]) -> str:
    return (
        f'<c:yVal><c:numRef><c:f>Sheet1!$B$2:$B${len(values) + 1}</c:f>'
        f'<c:numCache><c:ptCount val="{len(values)}"/>{_pts(values)}</c:numCache></c:numRef></c:yVal>'
    )


def _tx(name: str | None) -> str:
    if name is None:
        return ""
    return (
        '<c:tx><c:strRef><c:f>Sheet1!$B$1</c:f><c:strCache><c:ptCount val="1"/>'
        f'<c:pt idx="0"><c:v>{name}</c:v></c:pt></c:strCache></c:strRef></c:tx>'
    )


def _ser(idx: int, name: str | None, body: str) -> str:
    return f'<c:ser><c:idx val="{idx}"/><c:order val="{idx}"/>{_tx(name)}{body}</c:ser>'


def _bar_chart(sers: str, *, bar_dir: str = "col", grouping: str = "clustered") -> str:
    return f'<c:barChart><c:barDir val="{bar_dir}"/><c:grouping val="{grouping}"/>{sers}</c:barChart>'


def _plot_area(*chart_type_xmls: str, val_ax_title: str | None = None) -> str:
    val_ax = (
        f'<c:valAx><c:axId val="1"/><c:title><c:tx><c:rich><a:p><a:r>'
        f'<a:t>{val_ax_title}</a:t></a:r></a:p></c:rich></c:tx></c:title></c:valAx>'
        if val_ax_title else '<c:valAx><c:axId val="1"/></c:valAx>'
    )
    return f'<c:plotArea>{"".join(chart_type_xmls)}{val_ax}</c:plotArea>'


def _chart_root(plot_area_xml: str, *, title: str | None = None) -> etree._Element:
    title_xml = (
        f'<c:title><c:tx><c:rich><a:p><a:r><a:t>{title}</a:t></a:r></a:p></c:rich></c:tx></c:title>'
        if title else '<c:autoTitleDeleted val="1"/>'
    )
    xml = (
        f'<c:chartSpace xmlns:c="{_C}" xmlns:a="{_A}">'
        f"<c:chart>{title_xml}{plot_area_xml}</c:chart></c:chartSpace>"
    )
    return etree.fromstring(xml.encode())


def test_bar_chart_column_type_categories_and_series() -> None:
    sers = _ser(0, "Series A", _str_cat(["A", "B", "C"]) + _num_val(["1", "2", "3"]))
    root = _chart_root(_plot_area(_bar_chart(sers)), title="My Chart")
    data = parse_chart(root)
    assert data.chart_type == "column"
    assert data.title == "My Chart"
    assert data.categories == ("A", "B", "C")
    assert len(data.series) == 1
    assert data.series[0].name == "Series A"
    assert data.series[0].values == (1.0, 2.0, 3.0)
    assert data.series[0].kind == "column"


def test_bar_chart_bar_direction_gives_bar_type() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["1"]))
    root = _chart_root(_plot_area(_bar_chart(sers, bar_dir="bar")))
    assert parse_chart(root).chart_type == "bar"


def test_line_chart_type() -> None:
    sers = _ser(0, "S", _str_cat(["A", "B"]) + _num_val(["1", "2"]))
    root = _chart_root(_plot_area(f"<c:lineChart>{sers}</c:lineChart>"))
    data = parse_chart(root)
    assert data.chart_type == "line"
    assert data.series[0].kind == "line"


def test_pie_chart_type() -> None:
    sers = _ser(0, "S", _str_cat(["A", "B"]) + _num_val(["1", "2"]))
    root = _chart_root(_plot_area(f"<c:pieChart>{sers}</c:pieChart>"))
    assert parse_chart(root).chart_type == "pie"


def test_doughnut_chart_type() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["1"]))
    root = _chart_root(_plot_area(f"<c:doughnutChart>{sers}</c:doughnutChart>"))
    assert parse_chart(root).chart_type == "doughnut"


def test_radar_chart_type() -> None:
    sers = _ser(0, "S", _str_cat(["A", "B"]) + _num_val(["1", "2"]))
    root = _chart_root(_plot_area(f"<c:radarChart>{sers}</c:radarChart>"))
    data = parse_chart(root)
    assert data.chart_type == "radar"
    assert data.series[0].kind == "radar"


def test_area_chart_type() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["1"]))
    root = _chart_root(_plot_area(f"<c:areaChart>{sers}</c:areaChart>"))
    assert parse_chart(root).chart_type == "area"


def test_scatter_chart_uses_xval_yval_instead_of_cat_val() -> None:
    """Живой факт govtech-фикстуры: scatterChart несёт данные через
    ``c:xVal``/``c:yVal``, structurally отдельная пара от ``c:cat``/``c:val`` —
    без явного фолбэка чарт молча терял бы ВСЮ таблицу, не только mermaid."""
    sers = _ser(0, "Trend", _xval(["1984", "1985", "1986"]) + _yval(["10", "20", "30"]))
    root = _chart_root(_plot_area(f"<c:scatterChart>{sers}</c:scatterChart>"))
    data = parse_chart(root)
    assert data.chart_type == "scatter"
    assert data.categories == ("1984", "1985", "1986")
    assert data.series[0].values == (10.0, 20.0, 30.0)
    assert data.series[0].kind == "scatter"


def test_multiple_plot_area_types_give_combo_and_per_series_kind() -> None:
    bar = _bar_chart(_ser(0, "Bars", _str_cat(["A", "B"]) + _num_val(["1", "2"])))
    line = f'<c:lineChart>{_ser(1, "Line", _str_cat(["A", "B"]) + _num_val(["3", "4"]))}</c:lineChart>'
    root = _chart_root(_plot_area(bar, line))
    data = parse_chart(root)
    assert data.chart_type == "combo"
    kinds = {s.name: s.kind for s in data.series}
    assert kinds == {"Bars": "column", "Line": "line"}


def test_unrecognized_plot_area_element_gives_other() -> None:
    root = _chart_root(_plot_area("<c:bubbleChart><c:ser/></c:bubbleChart>"))
    assert parse_chart(root).chart_type == "other"


def test_no_plot_area_gives_other() -> None:
    root = _chart_root("")
    assert parse_chart(root).chart_type == "other"


def test_sparse_pt_idx_gives_none_or_empty_string_gaps() -> None:
    cat_xml = _str_cat(["A"], count=3)  # ptCount=3, но только idx=0 материализован ниже
    val_xml = _sparse_num_val({0: "1", 2: "3"}, count=3)
    root = _chart_root(_plot_area(_bar_chart(_ser(0, "S", cat_xml + val_xml))))
    data = parse_chart(root)
    assert data.categories == ("A", "", "")
    assert data.series[0].values == (1.0, None, 3.0)


def test_empty_numcache_gives_empty_series() -> None:
    sers = _ser(0, "S", _str_cat(["A", "B"]) + '<c:val><c:numRef><c:f>X</c:f><c:numCache/></c:numRef></c:val>')
    root = _chart_root(_plot_area(_bar_chart(sers)))
    data = parse_chart(root)
    assert data.series[0].values == ()


def test_missing_val_element_gives_empty_series() -> None:
    sers = _ser(0, "S", _str_cat(["A"]))
    root = _chart_root(_plot_area(_bar_chart(sers)))
    assert parse_chart(root).series[0].values == ()


def test_multi_series_chart() -> None:
    s1 = _ser(0, "First", _str_cat(["A", "B"]) + _num_val(["1", "2"]))
    s2 = _ser(1, "Second", _num_val(["3", "4"]))  # без своего <c:cat> — общий с первой
    root = _chart_root(_plot_area(_bar_chart(s1 + s2)))
    data = parse_chart(root)
    assert len(data.series) == 2
    assert [s.name for s in data.series] == ["First", "Second"]
    assert data.categories == ("A", "B")


def test_chart_without_title_gives_none() -> None:
    root = _chart_root(_plot_area(_bar_chart(_ser(0, "S", _str_cat(["A"]) + _num_val(["1"])))))
    assert parse_chart(root).title is None


def test_series_without_own_cat_falls_back_to_first_series_that_has_one() -> None:
    """Живой факт govtech-фикстуры chart1.xml: серия, идущая в документе
    ПЕРВОЙ, не несёт <c:cat> вовсе (Excel пишет категории лишь на одной
    серии, не обязательно первой) — категории должны найтись у ВТОРОЙ."""
    s1 = _ser(0, "NoCat", _num_val(["1", "2"]))
    s2 = _ser(1, "HasCat", _str_cat(["X", "Y"]) + _num_val(["3", "4"]))
    root = _chart_root(_plot_area(_bar_chart(s1 + s2)))
    data = parse_chart(root)
    assert data.categories == ("X", "Y")


def test_str_lit_categories_without_cell_reference() -> None:
    """Живой факт govtech chart6.xml: категории литеральные (c:strLit), без
    ссылки <c:f> на ячейку вовсе — structurally отдельная ветка от strRef."""
    sers = _ser(0, "S", _str_lit_cat(["A", "B", "C"]) + _num_val(["1", "2", "3"]))
    root = _chart_root(_plot_area(_bar_chart(sers)))
    assert parse_chart(root).categories == ("A", "B", "C")


def test_num_lit_values_without_cell_reference() -> None:
    sers = _ser(0, "S", _str_cat(["A", "B"]) + _num_lit_val(["10", "20"]))
    root = _chart_root(_plot_area(_bar_chart(sers)))
    assert parse_chart(root).series[0].values == (10.0, 20.0)


def test_value_axis_title_captured() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["1"]))
    root = _chart_root(_plot_area(_bar_chart(sers), val_ax_title="Average GTMI score"))
    assert parse_chart(root).value_axis_title == "Average GTMI score"


def test_value_format_captured_from_numcache() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["0.589"], fmt="#,##0.000"))
    root = _chart_root(_plot_area(_bar_chart(sers)))
    assert parse_chart(root).value_format == "#,##0.000"


def test_stacked_grouping_detected() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["1"]))
    root = _chart_root(_plot_area(_bar_chart(sers, grouping="stacked")))
    assert parse_chart(root).stacked is True


def test_percent_stacked_grouping_detected() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["1"]))
    root = _chart_root(_plot_area(_bar_chart(sers, grouping="percentStacked")))
    assert parse_chart(root).stacked is True


def test_clustered_grouping_is_not_stacked() -> None:
    sers = _ser(0, "S", _str_cat(["A"]) + _num_val(["1"]))
    root = _chart_root(_plot_area(_bar_chart(sers, grouping="clustered")))
    assert parse_chart(root).stacked is False


def test_worked_example_bar_chart_xml_to_chartdata_verbatim() -> None:
    """Golden-target, половина 1 (спек §Тестовое покрытие — XML -> ChartData):
    чарт-XML (2 категории х 2 серии, percent-формат, подпись оси) -> точный
    ``ChartData``, дословно. Половина 2 (ChartData -> точный markdown) —
    ``test_chart_render.py::test_worked_example_chartdata_to_markdown_verbatim``,
    та же ``ChartData`` собрана там напрямую (без XML) — раздельно, чтобы
    ``chart_data.py`` и ``chart_render.py`` оставались коммитируемы и
    тестируемы независимо друг от друга."""
    s1 = _ser(
        0, "2024",
        _str_cat(["Montenegro", "Estonia"]) + _num_val(["0.42", "0.87"], fmt="0.0%"),
    )
    s2 = _ser(1, "2025", _num_val(["0.55", "0.91"], fmt="0.0%"))
    root = _chart_root(
        _plot_area(_bar_chart(s1 + s2), val_ax_title="Score"), title="Regional Comparison"
    )
    assert parse_chart(root) == ChartData(
        chart_type="column",
        title="Regional Comparison",
        value_axis_title="Score",
        value_format="0.0%",
        stacked=False,
        categories=("Montenegro", "Estonia"),
        series=(
            ChartSeries(name="2024", values=(0.42, 0.87), kind="column"),
            ChartSeries(name="2025", values=(0.55, 0.91), kind="column"),
        ),
    )


def test_chart_data_and_series_are_frozen_dataclasses() -> None:
    series = ChartSeries(name="S", values=(1.0,), kind="column")
    data = ChartData(
        chart_type="column", title=None, value_axis_title=None, value_format=None,
        stacked=False, categories=("A",), series=(series,),
    )
    assert data.series[0].name == "S"
