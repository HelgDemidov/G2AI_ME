"""Тесты xlsx_charts.py (spec convert-xlsx §3): детект встроенных чартов,
captions из c:title, id12 по XML-структуре чарта. Ни сети, ни LibreOffice —
чистый XML in-memory (openpyxl.chart строит реальный chart-парт)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

from convert.xlsx_charts import extract_charts


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
