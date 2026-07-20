"""Тесты xlsx_charts.py (spec convert-xlsx §3): детект встроенных чартов,
captions из c:title, id12 по XML-структуре чарта. Ни сети, ни LibreOffice —
чистый XML in-memory (openpyxl.chart строит реальный chart-парт)."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

from lxml import etree
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

from convert.xlsx_charts import (
    _anchor_col_row,
    _blank_foreign_cells,
    _is_compact,
    _ownership_ranges,
    _pad_range_axes,
    _q,
    _range_contains,
    extract_chart_workbook,
    extract_charts,
)


def _active(wb: Any) -> Any:
    """``wb.active`` типизирован как ``Worksheet | Chartsheet | None`` в
    стабах — свежий ``Workbook()`` всегда несёт активный лист-Worksheet."""
    ws = wb.active
    assert ws is not None
    return ws


def _workbook_with_chart(
    tmp_path: Path,
    *,
    title: str | None = "Chart Title",
    anchor: str = "D2",
    sheet_name: str = "Data",
    file_name: str = "raw.xlsx",
) -> Path:
    wb = Workbook()
    ws = _active(wb)
    ws.title = sheet_name
    ws.append(["Cat", "Val"])
    ws.append(["A", 1])
    ws.append(["B", 2])
    chart = BarChart()
    if title is not None:
        chart.title = title
    data = Reference(ws, min_col=2, min_row=1, max_row=3)
    cats = Reference(ws, min_col=1, min_row=2, max_row=3)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, anchor)
    raw = tmp_path / file_name
    wb.save(raw)
    return raw


def test_extract_charts_empty_workbook_returns_empty_list(tmp_path: Path) -> None:
    wb = Workbook()
    raw = tmp_path / "raw.xlsx"
    wb.save(raw)
    assert extract_charts(raw) == []


def test_extract_charts_detects_single_chart_with_title_and_anchor(tmp_path: Path) -> None:
    raw = _workbook_with_chart(tmp_path, title="Costs of LTE and 5G", anchor="D2", sheet_name="Data")
    charts = extract_charts(raw)
    assert len(charts) == 1
    chart = charts[0]
    assert chart.sheet == "Data"
    assert chart.anchor_cell == "D2"
    assert chart.captions == ("Costs of LTE and 5G",)


def test_extract_charts_no_title_gives_empty_captions(tmp_path: Path) -> None:
    raw = _workbook_with_chart(tmp_path, title=None)
    charts = extract_charts(raw)
    assert len(charts) == 1
    assert charts[0].captions == ()


def test_extract_charts_id12_stable_across_repeated_calls(tmp_path: Path) -> None:
    raw = _workbook_with_chart(tmp_path)
    id1 = extract_charts(raw)[0].id12
    id2 = extract_charts(raw)[0].id12
    assert id1 == id2
    assert len(id1) == 12


def test_extract_charts_different_titles_get_distinct_ids(tmp_path: Path) -> None:
    """Билдер рассчитан на один чарт на документ (симметрично docx-тестам) —
    сверяем на двух отдельных однокграфиковых документах, что id12 зависит от
    содержимого чарта (заголовок входит в chart-парт)."""
    raw_a = _workbook_with_chart(tmp_path, title="Title A", file_name="a.xlsx")
    raw_b = _workbook_with_chart(tmp_path, title="Title B", file_name="b.xlsx")
    id_a = extract_charts(raw_a)[0].id12
    id_b = extract_charts(raw_b)[0].id12
    assert id_a != id_b


def test_extract_charts_anchor_cell_reflects_position(tmp_path: Path) -> None:
    raw = _workbook_with_chart(tmp_path, anchor="G10")
    charts = extract_charts(raw)
    assert charts[0].anchor_cell == "G10"


def test_extract_charts_sheet_with_no_drawing_returns_no_charts(tmp_path: Path) -> None:
    wb = Workbook()
    ws = _active(wb)
    ws.append(["plain", "data"])
    raw = tmp_path / "raw.xlsx"
    wb.save(raw)
    assert extract_charts(raw) == []


# --- extract_chart_workbook (живой checkpoint WBG GovTech Dataset нашёл и
# исправил здесь реальный дефект: soffice игнорирует sheet_state при
# headless-конвертации и рендерит ВСЕ листы книги — страница 1 систематически
# была первым листом КНИГИ, не листом-хозяином чарта; листы, не референсированные
# формулами серий чарта, теперь физически удаляются, а хозяин переставляется
# на позицию 0. Формулы в окне печати конвертируются в статичные закэшированные
# значения — иначе formula-ячейки, ссылающиеся на УЖЕ удалённые листы, дают
# #ИМЯ?/пусто и глушат даже numCache/strCache самого чарта) ---


def _workbook_with_chart_and_extra_sheet(tmp_path: Path) -> tuple[Path, str]:
    """Лист-хозяин НЕ первый по порядку создания (сперва Other, потом Data
    с чартом) — проверяет и удаление лишнего листа, и переустановку хозяина
    на позицию 0 (страница 1 PDF = хозяин, независимо от исходного порядка)."""
    wb = Workbook()
    other = _active(wb)
    other.title = "Other"
    other.append(["irrelevant"])
    ws = wb.create_sheet("Data")
    ws.append(["Cat", "Val"])
    ws.append(["A", 1])
    ws.append(["B", 2])
    chart = BarChart()
    chart.title = "Chart Title"
    chart.add_data(Reference(ws, min_col=2, min_row=1, max_row=3), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=3))
    ws.add_chart(chart, "D2")
    raw = tmp_path / "raw.xlsx"
    wb.save(raw)
    return raw, extract_charts(raw)[0].id12


def _mini_sheet_names(mini: bytes) -> list[str]:
    _NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(io.BytesIO(mini)) as z:
        root = etree.fromstring(z.read("xl/workbook.xml"))
    sheets_el = root.find(f"{{{_NS_MAIN}}}sheets")
    assert sheets_el is not None
    return [el.get("name") for el in sheets_el]


def test_extract_chart_workbook_unknown_id_returns_none(tmp_path: Path) -> None:
    raw, _cid = _workbook_with_chart_and_extra_sheet(tmp_path)
    assert extract_chart_workbook(raw, "0" * 12) is None


def test_extract_chart_workbook_removes_unreferenced_sheet(tmp_path: Path) -> None:
    raw, cid = _workbook_with_chart_and_extra_sheet(tmp_path)
    mini = extract_chart_workbook(raw, cid)
    assert mini is not None
    assert _mini_sheet_names(mini) == ["Data"]  # "Other" не референсирован чартом -> удалён


def test_extract_chart_workbook_reorders_host_first(tmp_path: Path) -> None:
    raw, cid = _workbook_with_chart_and_extra_sheet(tmp_path)
    mini = extract_chart_workbook(raw, cid)
    assert mini is not None
    assert _mini_sheet_names(mini)[0] == "Data"


def _workbook_with_chart_and_formula_in_window(tmp_path: Path) -> tuple[Path, str]:
    """Формульная ячейка (``B5``) внутри окна печати якоря ``D2`` — openpyxl
    формулы не считает, поэтому кэш (``<v>``) патчится вручную поверх
    сохранённого файла (тот же приём, что ``test_converters._xlsx_with_cached_formula``)."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Data"
    ws.append(["Cat", "Val"])
    ws.append(["A", 1])
    ws.append(["B", 2])
    ws["B5"] = "=1+1"
    chart = BarChart()
    chart.title = "Chart Title"
    chart.add_data(Reference(ws, min_col=2, min_row=1, max_row=3), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=3))
    ws.add_chart(chart, "D2")
    raw = tmp_path / "raw.xlsx"
    wb.save(raw)

    orig = raw.read_bytes()
    with zipfile.ZipFile(io.BytesIO(orig)) as z:
        names = z.namelist()
        sheet_part = next(n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        sheet_xml = z.read(sheet_part).decode("utf-8")
    patched = sheet_xml.replace("<v></v>", "<v>4</v>")
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(orig)) as z, zipfile.ZipFile(buf, "w") as zo:
        for n in names:
            zo.writestr(n, patched if n == sheet_part else z.read(n))
    raw.write_bytes(buf.getvalue())
    return raw, extract_charts(raw)[0].id12


def test_extract_chart_workbook_strips_formula_keeps_cached_value(tmp_path: Path) -> None:
    """Формула в окне печати -> статичное значение: без этого ячейка,
    ссылающаяся (гипотетически) на удалённый лист, дала бы #ИМЯ?/пусто при
    пересчёте на открытии и заглушила бы даже закэшированные значения
    самого чарта (живой дефект checkpoint'а, см. докстроку extract_chart_workbook)."""
    raw, cid = _workbook_with_chart_and_formula_in_window(tmp_path)
    mini = extract_chart_workbook(raw, cid)
    assert mini is not None
    with zipfile.ZipFile(io.BytesIO(mini)) as z:
        sheet_part = next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        sheet_xml = z.read(sheet_part).decode("utf-8")
    assert "<f>1+1</f>" not in sheet_xml
    assert "<v>4</v>" in sheet_xml


# --- взаимное владение/исключение соседних чартов (spec §3, живой checkpoint
# WBG GovTech — раунд харденинга: соседние чарты на одном листе делят одну
# сетку ячеек, окно печати цели неизбежно задевает их край, если явно не
# разметить «занято другим»; ниже — pure-function тесты для самой хитрой
# логики (data_keep ВСЕГДА побеждает exclude, bbox_keep — НЕТ) и один
# end-to-end тест на реальной мини-книге с двумя чартами ---


def _anchor_xml(from_col0: int, from_row0: int, *, to_col0: int | None = None, to_row0: int | None = None) -> Any:
    tag = "twoCellAnchor" if to_col0 is not None else "oneCellAnchor"
    anchor = etree.Element(_q("xdr", tag))
    frm = etree.SubElement(anchor, _q("xdr", "from"))
    etree.SubElement(frm, _q("xdr", "col")).text = str(from_col0)
    etree.SubElement(frm, _q("xdr", "row")).text = str(from_row0)
    if to_col0 is not None:
        to = etree.SubElement(anchor, _q("xdr", "to"))
        etree.SubElement(to, _q("xdr", "col")).text = str(to_col0)
        etree.SubElement(to, _q("xdr", "row")).text = str(to_row0)
    return anchor


def _chart_root_with_refs(sheet: str, refs: list[str]) -> Any:
    """Минимальное дерево с <c:f> — ``_chart_refs``/``_ownership_ranges`` читают
    ТОЛЬКО ``.//c:f`` где-либо внутри, остальная схема чарта не нужна."""
    root = etree.Element(_q("c", "chartSpace"))
    for ref in refs:
        etree.SubElement(root, _q("c", "f")).text = f"{sheet}!{ref}"
    return root


def test_ownership_ranges_own_refs_kept_neighbor_refs_excluded() -> None:
    target_anchor = _anchor_xml(1, 1, to_col0=4, to_row0=4)  # 0-indexed -> B2:E5
    target_root = _chart_root_with_refs("Data", ["$B$1:$B$3"])
    neighbor_anchor = _anchor_xml(8, 1, to_col0=11, to_row0=4)  # I2:L5
    neighbor_root = _chart_root_with_refs("Data", ["$I$1:$I$3"])
    siblings = [(target_anchor, target_root), (neighbor_anchor, neighbor_root)]

    anchor_col, anchor_row = _anchor_col_row(target_anchor)
    data_keep, bbox_keep, exclude = _ownership_ranges(
        target_root, target_anchor, anchor_col, anchor_row, "Data", siblings
    )

    assert any(_range_contains(r, 2, 2) for r in data_keep)  # своя $B$1:$B$3 (col=2)
    assert not any(_range_contains(r, 2, 2) for r in exclude)
    assert any(_range_contains(r, 9, 2) for r in exclude)  # чужая $I$1:$I$3 (col=9)
    assert not any(_range_contains(r, 9, 2) for r in data_keep)
    assert not any(_range_contains(r, 9, 2) for r in bbox_keep)


def _sheet_xml_with_cells(refs: list[str]) -> Any:
    """Минимальный ``<worksheet><sheetData>`` с пустыми ячейками по заданным
    ссылкам — ``_blank_foreign_cells`` смотрит только на ``r``-атрибуты."""
    rows: dict[int, list[str]] = {}
    for ref in refs:
        m = re.match(r"^([A-Z]+)(\d+)$", ref)
        assert m is not None
        rows.setdefault(int(m.group(2)), []).append(ref)
    root = etree.Element(_q("main", "worksheet"))
    sheet_data = etree.SubElement(root, _q("main", "sheetData"))
    for row_num in sorted(rows):
        row_el = etree.SubElement(sheet_data, _q("main", "row"))
        row_el.set("r", str(row_num))
        for ref in rows[row_num]:
            etree.SubElement(row_el, _q("main", "c")).set("r", ref)
    return root


def _surviving_refs(sheet_root: Any) -> set[str]:
    sheet_data = sheet_root.find(_q("main", "sheetData"))
    return {c.get("r") for row_el in sheet_data for c in row_el}


def test_blank_foreign_cells_data_keep_wins_over_overlapping_exclude() -> None:
    """CE12-класс живого дефекта: ячейка попадает И в data_keep (заслуженное
    владение цели), И в exclude (явное владение соседа) одновременно —
    data_keep обязан победить (bbox_keep цели в этот вызов НЕ передаётся,
    ровно по этой причине, см. докстроку ``_ownership_ranges``)."""
    data_keep = [(12, 1, 16, 5)]  # своя область цели, включает (14, 2)
    exclude = [(8, 1, 37, 28)]  # чужой bbox-запас, ТОЖЕ включает (14, 2)
    print_window = (1, 1, 40, 30)
    sheet_root = _sheet_xml_with_cells(["N2", "P10", "A1"])  # N2=(14,2) P10=(16,10) A1=(1,1)
    _blank_foreign_cells(sheet_root, print_window, data_keep, exclude)
    remaining = _surviving_refs(sheet_root)
    assert "N2" in remaining  # в data_keep — выживает несмотря на пересечение с exclude
    assert "P10" not in remaining  # только в exclude — чужое, вычищено
    assert "A1" in remaining  # не заявлена никем — default-keep, не default-blank


def test_is_compact_within_and_beyond_threshold() -> None:
    assert _is_compact((1, 1, 5, 10), 25) is True
    assert _is_compact((1, 1, 1, 197), 25) is False  # узкая, но очень длинная сторона
    assert _is_compact((1, 1, 30, 3), 25) is False


def test_pad_range_axes_asymmetric() -> None:
    assert _pad_range_axes((5, 5, 10, 10), 1, 2) == (4, 3, 11, 12)


def test_pad_range_axes_clamps_at_one() -> None:
    assert _pad_range_axes((1, 1, 3, 3), 2, 5) == (1, 1, 5, 8)


def _workbook_with_two_charts_same_sheet(tmp_path: Path) -> tuple[Path, str]:
    """Два чарта на одном листе, анкеры близко (A1/J1) — ``oneCellAnchor`` без
    ``xdr:to`` (openpyxl-дефолт, проверено живьём) даёт предсказуемый
    ``_ONE_CELL_FALLBACK=25``-клеточный bbox, поэтому их запасы гарантированно
    пересекаются без нужды патчить XML вручную. Целевые данные (столбец N)
    намеренно лежат ВНУТРИ bbox-запаса соседа — воспроизводит живой
    CE12-прецедент на реальной мини-книге (не только на синтетическом XML)."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Data"
    ws["A1"], ws["N1"] = "CatA", "ValA"
    ws["A2"], ws["N2"] = "X", 1
    ws["A3"], ws["N3"] = "Y", 2
    ws["J1"], ws["K1"] = "CatB", "ValB"
    ws["J2"], ws["K2"] = "P", 10
    ws["J3"], ws["K3"] = "Q", 20
    ws["P10"] = "unclaimed foreign label"  # никем не референсировано напрямую

    chart_a = BarChart()
    chart_a.title = "Chart A"
    chart_a.add_data(Reference(ws, min_col=14, min_row=1, max_row=3), titles_from_data=True)
    chart_a.set_categories(Reference(ws, min_col=1, min_row=2, max_row=3))
    ws.add_chart(chart_a, "A1")

    chart_b = BarChart()
    chart_b.title = "Chart B"
    chart_b.add_data(Reference(ws, min_col=11, min_row=1, max_row=3), titles_from_data=True)
    chart_b.set_categories(Reference(ws, min_col=10, min_row=2, max_row=3))
    ws.add_chart(chart_b, "J1")

    raw = tmp_path / "raw.xlsx"
    wb.save(raw)
    charts = extract_charts(raw)
    target_id = next(c.id12 for c in charts if c.captions == ("Chart A",))
    return raw, target_id


def test_extract_chart_workbook_keeps_own_data_blanks_unclaimed_neighbor_cell(tmp_path: Path) -> None:
    raw, target_id = _workbook_with_two_charts_same_sheet(tmp_path)
    mini = extract_chart_workbook(raw, target_id)
    assert mini is not None
    with zipfile.ZipFile(io.BytesIO(mini)) as z:
        sheet_part = next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        sheet_xml = z.read(sheet_part).decode("utf-8")
    assert 'r="N2"' in sheet_xml  # своя данные цели — выживают несмотря на попадание в bbox-запас соседа
    assert 'r="P10"' not in sheet_xml  # ничья ячейка внутри чужого bbox-запаса — вычищена
