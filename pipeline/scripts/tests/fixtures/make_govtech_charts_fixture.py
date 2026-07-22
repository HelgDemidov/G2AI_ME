"""Регенератор `tests/fixtures/local/govtech-2025-charts.xlsx` из полного
World Bank GovTech Maturity Index Dataset (см. `tests/fixtures/local/README.md`
за источником/провенансом).

Хирургическая обрезка листов: сохраняет ВСЕ 55 встроенных чартов
byte-identical (``xl/charts/*``, ``xl/drawings/*`` и drawing-relationship
в `_rels` листа не трогаются вовсе) — `chart_data.py::parse_chart` читает
ТОЛЬКО закэшированные `<c:numCache>`/`<c:strCache>` внутри самого
chart-парта, не живые ячейки листа, поэтому обрезка данных листа не может
повлиять на извлечение/рендер чартов. Убирает единственный источник тяжести
исходной книги: `sheetData` за пределами `MAX_ROW`x`MAX_COL` и раздутые
per-cell `<hyperlinks>` на внешние источники (13192 штук на одном листе,
2.6MB `.rels`) — `_rels`-файл листа пересобирается по фактически
оставшимся `r:id`-ссылкам, поэтому осиротевших relationship-id не остаётся.

Запуск (исходник — заново скачанный `WBG_GovTech_Dataset_Dec2025.xlsx`,
см. README за URL; сам он не хранится в репозитории):
    .venv/bin/python pipeline/scripts/tests/fixtures/make_govtech_charts_fixture.py \\
        <путь-к-полной-книге.xlsx> pipeline/scripts/tests/fixtures/local/govtech-2025-charts.xlsx
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from lxml import etree

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_MAX_ROW = 30
_MAX_COL = 20
_SHEET_RE = re.compile(r"xl/worksheets/sheet\d+\.xml$")
_SHEET_RELS_RE = re.compile(r"xl/worksheets/_rels/(sheet\d+)\.xml\.rels$")


def _col_to_idx(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref)
    assert match is not None
    col = 0
    for ch in match.group(1):
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col


def _ref_in_bounds(ref: str) -> bool:
    for part in ref.split(":"):
        match = re.match(r"([A-Z]+)(\d+)", part)
        if match is None or int(match.group(2)) > _MAX_ROW or _col_to_idx(part) > _MAX_COL:
            return False
    return True


def trim_sheet_xml(xml_bytes: bytes) -> bytes:
    root = etree.fromstring(xml_bytes)
    sheet_data = root.find(f"{{{_MAIN_NS}}}sheetData")
    if sheet_data is not None:
        for row in list(sheet_data.findall(f"{{{_MAIN_NS}}}row")):
            if int(row.get("r", "0")) > _MAX_ROW:
                sheet_data.remove(row)
                continue
            for cell in list(row.findall(f"{{{_MAIN_NS}}}c")):
                ref = cell.get("r", "")
                if ref and _col_to_idx(ref) > _MAX_COL:
                    row.remove(cell)
    for group_tag, item_tag in (("mergeCells", "mergeCell"), ("hyperlinks", "hyperlink")):
        group = root.find(f"{{{_MAIN_NS}}}{group_tag}")
        if group is None:
            continue
        kept = 0
        for item in list(group.findall(f"{{{_MAIN_NS}}}{item_tag}")):
            ref = item.get("ref", "")
            if ref and _ref_in_bounds(ref):
                kept += 1
            else:
                group.remove(item)
        if group_tag == "mergeCells":
            group.set("count", str(kept))
        if kept == 0:
            parent = group.getparent()
            assert parent is not None
            parent.remove(group)
    dimension = root.find(f"{{{_MAIN_NS}}}dimension")
    if dimension is not None:
        dimension.set("ref", f"A1:T{_MAX_ROW}")
    result: bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    return result


def trim_rels(rels_bytes: bytes, kept_ids: set[str]) -> bytes:
    root = etree.fromstring(rels_bytes)
    for rel in list(root.findall(f"{{{_RELS_NS}}}Relationship")):
        if rel.get("Id") not in kept_ids:
            root.remove(rel)
    result: bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    return result


def referenced_rids(xml_bytes: bytes) -> set[str]:
    root = etree.fromstring(xml_bytes)
    return {value for element in root.iter() for key, value in element.attrib.items() if key == f"{{{_R_NS}}}id"}


def build_fixture(src: Path, dst: Path) -> None:
    trimmed_sheets: dict[str, bytes] = {}
    with zipfile.ZipFile(src) as zin:
        for name in zin.namelist():
            if _SHEET_RE.match(name):
                trimmed_sheets[name] = trim_sheet_xml(zin.read(name))
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                name = item.filename
                if name in trimmed_sheets:
                    zout.writestr(item, trimmed_sheets[name])
                    continue
                rels_match = _SHEET_RELS_RE.match(name)
                if rels_match is not None:
                    sheet_name = f"xl/worksheets/{rels_match.group(1)}.xml"
                    kept_ids = referenced_rids(trimmed_sheets[sheet_name])
                    zout.writestr(item, trim_rels(zin.read(name), kept_ids))
                    continue
                zout.writestr(item, zin.read(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="полная книга WBG_GovTech_Dataset_*.xlsx")
    parser.add_argument("dest", type=Path, help="путь для govtech-2025-charts.xlsx")
    args = parser.parse_args()
    build_fixture(args.source, args.dest)


if __name__ == "__main__":
    main()
