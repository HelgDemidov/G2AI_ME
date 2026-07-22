"""Container-agnostic извлечение данных из нативного OOXML ``c:chart`` (spec
chart-data-extraction §3): чистый парсинг УЖЕ распарсенного chart-парта
(``xl/charts/chartN.xml`` ИЛИ ``word/charts/chartN.xml`` — идентичная
DrawingML-схема, разница только в контейнере-резолвере снаружи, который эта
функция не касается — см. ``xlsx_charts``/``docx_groups`` для навигации к
парту). Источник данных — ТОЛЬКО ``c:numCache``/``c:strCache`` (v1): данные,
которые авторское приложение УЖЕ закэшировало в самом chart-XML, без резолва
``<c:f>``-формул против книги. Диапазон-резолвер — вне скоупа v1 (см. Design
rationale спека): numCache присутствует практически во всех реальных чартах.

``parse_chart`` не бросает исключений на структурно неполном/непривычном
чарте — отсутствующий узел на любом шаге даёт пустой/``None`` результат
(вызывающая сторона решает про caption-фолбэк, см. ``chart_render.render_chart``
-> ``None``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxml import etree

_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"

# Локальное имя chart-type-элемента -> базовый chart_type (кроме barChart —
# требует разбора c:barDir, см. _bar_subtype). scatterChart несёт данные через
# c:xVal/c:yVal, а не c:cat/c:val (см. _cat_element/_val_element) — структурно
# другая пара тегов той же роли, обнаружено сверкой с реальной фикстурой
# govtech (5 scatter-чартов, 9 серий на общий year-xVal): без фолбэка на
# xVal/yVal эти чарты тихо теряли бы ВСЮ таблицу (не только mermaid).
_SIMPLE_CHART_TAGS = {
    "lineChart": "line",
    "pieChart": "pie",
    "doughnutChart": "doughnut",
    "radarChart": "radar",
    "scatterChart": "scatter",
    "areaChart": "area",
}
_ALL_CHART_TAGS = frozenset(_SIMPLE_CHART_TAGS) | {"barChart"}
_STACKED_GROUPINGS = frozenset({"stacked", "percentStacked"})


def _q(local: str) -> str:
    return f"{{{_NS}}}{local}"


@dataclass(frozen=True)
class ChartSeries:
    name: str | None
    values: tuple[float | None, ...]
    # Локальный chart_type ЭЛЕМЕНТА, несущего этот <c:ser> (не chart_type
    # чарта в целом) — нужен исключительно для bar+line combo: ChartData.
    # chart_type="combo" не говорит, какие серии рисовать как бары, какие как
    # линию (chart_render._mermaid_xychart).
    kind: str


@dataclass(frozen=True)
class ChartData:
    chart_type: str
    title: str | None
    value_axis_title: str | None
    value_format: str | None
    # c:grouping val="stacked"/"percentStacked" на любом bar/line/area-элементе
    # (реально встречается в govtech-фикстуре — 5+8 из 55 чартов): mermaid
    # xychart-beta стек НЕ поддерживает (issue #7392, только overlay) —
    # chart_render должен откатиться к таблице-только, не сымитировать стек
    # оверлеем (тихо исказило бы смысл — «сумма частей» превратилась бы в
    # «несколько наложенных рядов»).
    stacked: bool
    categories: tuple[str, ...]
    series: tuple[ChartSeries, ...]


def _text_of(el: Any) -> str | None:
    """Плоский текст rich-текстового узла (``c:title``/``c:valAx><c:title``):
    ``a:t``-раны через ``itertext()``, склеенные пробелом. В отличие от
    ``docx_groups._filter_caption_texts``/``xlsx_charts._chart_title`` (которые
    отсеивают числовой geometry-мусор координат из group/anchor XML) — здесь
    он не нужен: ``c:title`` несёт ТОЛЬКО текстовые раны, без geometry-тегов
    вроде ``wp:posOffset``/``a:ext``, что итерируются как текст в тех структурах."""
    if el is None:
        return None
    text = " ".join(t.strip() for t in el.itertext() if t.strip())
    return text or None


def _pt_count(cache_el: Any) -> int:
    pt_count_el = cache_el.find(_q("ptCount"))
    if pt_count_el is not None and pt_count_el.get("val") is not None:
        return int(pt_count_el.get("val"))
    idxs = [int(pt.get("idx", 0)) for pt in cache_el.findall(_q("pt"))]
    return (max(idxs) + 1) if idxs else 0


def _pt_value(pt_el: Any) -> str | None:
    v_el = pt_el.find(_q("v"))
    return v_el.text if v_el is not None else None


def _materialize_str(cache_el: Any) -> tuple[str, ...]:
    n = _pt_count(cache_el)
    values: list[str] = [""] * n
    for pt in cache_el.findall(_q("pt")):
        idx = int(pt.get("idx", 0))
        if 0 <= idx < n:
            v = _pt_value(pt)
            values[idx] = v if v is not None else ""
    return tuple(values)


def _materialize_num(cache_el: Any) -> tuple[float | None, ...]:
    n = _pt_count(cache_el)
    values: list[float | None] = [None] * n
    for pt in cache_el.findall(_q("pt")):
        idx = int(pt.get("idx", 0))
        if 0 <= idx < n:
            v = _pt_value(pt)
            values[idx] = float(v) if v is not None else None
    return tuple(values)


def _format_plain_num(v: float | None) -> str:
    if v is None:
        return ""
    return str(int(v)) if v == int(v) else str(v)


def _cat_element(ser_el: Any) -> Any:
    """``c:cat`` (bar/line/pie/radar/doughnut/area) или ``c:xVal`` (scatter,
    структурно та же роль — см. докстроку модуля/``_SIMPLE_CHART_TAGS``)."""
    cat = ser_el.find(_q("cat"))
    return cat if cat is not None else ser_el.find(_q("xVal"))


def _val_element(ser_el: Any) -> Any:
    """``c:val`` (bar/line/pie/radar/doughnut/area) или ``c:yVal`` (scatter)."""
    val = ser_el.find(_q("val"))
    return val if val is not None else ser_el.find(_q("yVal"))


def _categories(cat_el: Any) -> tuple[str, ...]:
    """``c:cat``/``c:xVal`` — 4 варианта источника (``CT_AxDataSource``):
    ``strRef``>``strCache`` (обычные текстовые категории по ссылке — типовой
    случай), ``numRef``>``numCache`` (числовые категории по ссылке — года
    scatter-серии), ``strLit``/``numLit`` (ЛИТЕРАЛЬНЫЕ данные без ссылки на
    ячейку вовсе — живой факт govtech-фикстуры, chart6.xml: категории заданы
    прямо в чарте, не через ``<c:f>``; структура ``ptCount``/``pt`` идентична
    ``*Cache``, поэтому те же ``_materialize_*`` подходят без изменений)."""
    if cat_el is None:
        return ()
    str_ref = cat_el.find(_q("strRef"))
    if str_ref is not None:
        cache = str_ref.find(_q("strCache"))
        return _materialize_str(cache) if cache is not None else ()
    str_lit = cat_el.find(_q("strLit"))
    if str_lit is not None:
        return _materialize_str(str_lit)
    num_ref = cat_el.find(_q("numRef"))
    if num_ref is not None:
        cache = num_ref.find(_q("numCache"))
        if cache is None:
            return ()
        return tuple(_format_plain_num(v) for v in _materialize_num(cache))
    num_lit = cat_el.find(_q("numLit"))
    if num_lit is not None:
        return tuple(_format_plain_num(v) for v in _materialize_num(num_lit))
    return ()


def _series_name(ser_el: Any) -> str | None:
    tx = ser_el.find(_q("tx"))
    if tx is None:
        return None
    str_ref = tx.find(_q("strRef"))
    if str_ref is not None:
        cache = str_ref.find(_q("strCache"))
        if cache is None:
            return None
        vals = _materialize_str(cache)
        return vals[0] if vals and vals[0] else None
    v_el = tx.find(_q("v"))  # литеральное имя серии (редко, но валидно)
    return v_el.text if v_el is not None else None


def _val_cache(val_el: Any) -> Any:
    """``c:val``/``c:yVal`` — ``numRef``>``numCache`` (типовая ссылка на
    ячейки) ИЛИ ``numLit`` (литеральные данные без ссылки — см. докстроку
    ``_categories``, тот же класс варианта в ``CT_NumDataSource``)."""
    num_ref = val_el.find(_q("numRef"))
    if num_ref is not None:
        return num_ref.find(_q("numCache"))
    return val_el.find(_q("numLit"))


def _series_values(ser_el: Any) -> tuple[float | None, ...]:
    val_el = _val_element(ser_el)
    if val_el is None:
        return ()
    cache = _val_cache(val_el)
    return _materialize_num(cache) if cache is not None else ()


def _series_format_code(ser_el: Any) -> str | None:
    val_el = _val_element(ser_el)
    if val_el is None:
        return None
    cache = _val_cache(val_el)
    if cache is None:
        return None
    fmt_el = cache.find(_q("formatCode"))
    return fmt_el.text if fmt_el is not None else None


def _bar_subtype(bar_chart_el: Any) -> str:
    bar_dir = bar_chart_el.find(_q("barDir"))
    val = bar_dir.get("val") if bar_dir is not None else None
    return "bar" if val == "bar" else "column"


def _tag_chart_type(tag: str, el: Any) -> str:
    if tag == "barChart":
        return _bar_subtype(el)
    return _SIMPLE_CHART_TAGS.get(tag, "other")


def _present_chart_elements(plot_area: Any) -> list[tuple[str, Any]]:
    out = []
    for child in plot_area:
        tag = etree.QName(child).localname
        if tag in _ALL_CHART_TAGS:
            out.append((tag, child))
    return out


def _chart_type(plot_area: Any) -> str:
    present = _present_chart_elements(plot_area)
    if not present:
        return "other"
    if len(present) > 1:
        return "combo"
    tag, el = present[0]
    return _tag_chart_type(tag, el)


def _is_stacked(plot_area: Any) -> bool:
    for tag in ("barChart", "lineChart", "areaChart"):
        for el in plot_area.findall(_q(tag)):
            grouping = el.find(_q("grouping"))
            if grouping is not None and grouping.get("val") in _STACKED_GROUPINGS:
                return True
    return False


def _series_kind(ser_el: Any) -> str:
    parent = ser_el.getparent()
    if parent is None:
        return "other"
    tag = etree.QName(parent).localname
    if tag not in _ALL_CHART_TAGS:
        return "other"
    return _tag_chart_type(tag, parent)


def _value_axis_title(chart_el: Any) -> str | None:
    """Первый ``c:valAx`` с непустым ``c:title`` (combo-чарты иногда несут
    несколько valAx — берём первый содержательный, не первый по документу)."""
    for val_ax in chart_el.findall(f".//{_q('valAx')}"):
        title = _text_of(val_ax.find(_q("title")))
        if title:
            return title
    return None


_EMPTY = ChartData(
    chart_type="other", title=None, value_axis_title=None, value_format=None,
    stacked=False, categories=(), series=(),
)


def parse_chart(chart_root: Any) -> ChartData:
    chart_el = chart_root.find(_q("chart"))
    if chart_el is None:
        return _EMPTY
    title = _text_of(chart_el.find(_q("title")))
    plot_area = chart_el.find(_q("plotArea"))
    if plot_area is None:
        return ChartData(
            chart_type="other", title=title, value_axis_title=None, value_format=None,
            stacked=False, categories=(), series=(),
        )

    chart_type = _chart_type(plot_area)
    stacked = _is_stacked(plot_area)
    value_axis_title = _value_axis_title(chart_el)

    ser_elements = plot_area.findall(f".//{_q('ser')}")
    # Категории живут на ОДНОЙ серии (обычно общей для всех — Excel не дублирует
    # c:cat/c:xVal на каждой), но НЕ обязательно на первой в порядке документа
    # (живой факт govtech-фикстуры, chart1.xml: серия, идущая в документе
    # первой, вовсе не несёт <c:cat> — категории у ВТОРОЙ). Берём первую серию,
    # у которой категории реально нашлись, а не слепо ser_elements[0].
    categories: tuple[str, ...] = ()
    for ser_el in ser_elements:
        found = _categories(_cat_element(ser_el))
        if found:
            categories = found
            break

    series: list[ChartSeries] = []
    value_format: str | None = None
    for ser_el in ser_elements:
        series.append(
            ChartSeries(name=_series_name(ser_el), values=_series_values(ser_el), kind=_series_kind(ser_el))
        )
        if value_format is None:
            fmt = _series_format_code(ser_el)
            if fmt is not None:
                value_format = fmt

    return ChartData(
        chart_type=chart_type,
        title=title,
        value_axis_title=value_axis_title,
        value_format=value_format,
        stacked=stacked,
        categories=categories,
        series=tuple(series),
    )
