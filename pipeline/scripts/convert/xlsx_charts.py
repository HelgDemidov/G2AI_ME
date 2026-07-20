"""Встроенные чарты xlsx (spec convert-xlsx §3): архитектурный аналог
``docx_groups.py``, адаптированный под безпотоковую (cell-anchored) модель —
у xlsx-чарта нет «места в абзаце», есть лист + якорная ячейка, поэтому вместо
сентинел-в-потоке маркер декларативно позиционируется сразу после таблицы
своего листа (``converters._convert_xlsx``). Отсюда и другое имя функции
детекта: в отличие от docx-аналога (``extract_and_strip_groups`` — переписывает
``word/document.xml``, вырезая группу), здесь ``raw`` НЕ модифицируется вовсе —
называть это «strip» было бы неточно.

Детект: ``xl/worksheets/sheetK.xml`` несёт ``<drawing r:id="rIdX"/>`` ->
``xl/worksheets/_rels/sheetK.xml.rels`` -> ``xl/drawings/drawingM.xml`` ->
``<xdr:oneCellAnchor>``/``<xdr:twoCellAnchor>`` с ``<xdr:from>`` (якорная
ячейка: 0-indexed col/row) и ``graphicFrame``, несущим ``c:chart`` (ссылка на
``xl/charts/chartN.xml`` через ``xl/drawings/_rels/drawingM.xml.rels``).
Имя листа -> путь его XML-парта — через ``xl/workbook.xml`` (имя -> r:id) +
``xl/_rels/workbook.xml.rels`` (r:id -> Target); три разных источника
относительных путей (workbook/лист/drawing) требуют resolve относительно
директории КАЖДОГО конкретного source-парта (не хардкод «xl/», как у docx,
где все rels живут в одной ``word/``, см. ``_resolve_target``).

``id12`` — sha256 XML-СТРУКТУРЫ чарта (``etree.tostring`` парсенного
``xl/charts/chartN.xml``), НЕ байтов рендера: тот же принцип, что у
docx-групп (единственное отклонение, на котором дважды спотыкались тесты
docx — здесь зафиксировано в докстроке заранее). ``captions`` — из
``c:title`` чарт-парта: идентичная DrawingML-схема ``c:tx/c:rich/a:p/a:r/a:t``,
что у нативных docx-чартов (§2-ter convert-docx) — код адаптируется под
другие XML-пути, не копируется 1:1.
"""
from __future__ import annotations

import hashlib
import io
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree
from openpyxl.utils import column_index_from_string, get_column_letter

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}
_NUMERIC_JUNK_RE = re.compile(r"^-?\d+$")  # posOffset/ext координаты в itertext(), см. docx_groups
_CELL_RANGE_RE = re.compile(r"^\$?(?P<c1>[A-Z]+)\$?(?P<r1>\d+)(?::\$?(?P<c2>[A-Z]+)\$?(?P<r2>\d+))?$")
_CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _q(prefix: str, local: str) -> str:
    return f"{{{_NS[prefix]}}}{local}"


@dataclass(frozen=True)
class XlsxChart:
    id12: str
    sheet: str
    anchor_cell: str
    captions: tuple[str, ...]


def _filter_caption_texts(texts: Any) -> tuple[str, ...]:
    """Тот же фильтр, что ``docx_groups._filter_caption_texts`` — отсеивает
    числовой геометрический мусор и дедуплицирует по порядку появления."""
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        s = t.strip()
        if not s or _NUMERIC_JUNK_RE.match(s) or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


def _rel_targets(z: zipfile.ZipFile, part: str) -> dict[str, str]:
    """rId -> сырой Target (ещё не resolve-нутый) для ``part``, из соседнего .rels."""
    rels_name = f"{posixpath.dirname(part)}/_rels/{posixpath.basename(part)}.rels"
    if rels_name not in z.namelist():
        return {}
    root = etree.fromstring(z.read(rels_name))
    return {rel.get("Id"): rel.get("Target") for rel in root if rel.get("Id")}


def _resolve_target(source_part: str, target: str) -> str:
    """OPC-резолв Target относительно СВОЕГО source-парта: абсолютный
    (ведущий ``/``) -> package-root, иначе -> относительно директории
    source_part (не хардкод одной директории — в отличие от docx, где все
    rels живут в ``word/``, здесь по цепочке участвуют ``xl/``,
    ``xl/worksheets/``, ``xl/drawings/``)."""
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))


def _sheet_parts(z: zipfile.ZipFile) -> dict[str, str]:
    """Имя листа (как в ``wb.sheetnames``) -> путь XML-парта листа в zip."""
    names = set(z.namelist())
    if "xl/workbook.xml" not in names:
        return {}
    rel_targets = _rel_targets(z, "xl/workbook.xml")
    root = etree.fromstring(z.read("xl/workbook.xml"))
    out: dict[str, str] = {}
    for sheet_el in root.findall(f".//{_q('main', 'sheet')}"):
        name, rid = sheet_el.get("name"), sheet_el.get(_q("r", "id"))
        if name is None or rid is None or rid not in rel_targets:
            continue
        part = _resolve_target("xl/workbook.xml", rel_targets[rid])
        if part in names:
            out[name] = part
    return out


def _chart_anchors(drawing_root: Any) -> list[tuple[Any, Any]]:
    """(anchor, chart_ref) для каждого ``xdr:oneCellAnchor``/``xdr:twoCellAnchor``,
    несущего ``c:chart`` где-то внутри (обычно в ``graphicFrame``)."""
    pairs: list[tuple[Any, Any]] = []
    for tag in ("oneCellAnchor", "twoCellAnchor"):
        for anchor in drawing_root.findall(_q("xdr", tag)):
            chart_ref = anchor.find(f".//{_q('c', 'chart')}")
            if chart_ref is not None:
                pairs.append((anchor, chart_ref))
    return pairs


def _anchor_col_row(anchor: Any) -> tuple[int, int]:
    """``xdr:from`` (0-indexed col/row) -> (col, row) 1-indexed, как
    ``openpyxl.utils.column_index_from_string``/``ws.cell`` ожидают."""
    frm = anchor.find(_q("xdr", "from"))
    return int(frm.findtext(_q("xdr", "col"))) + 1, int(frm.findtext(_q("xdr", "row"))) + 1


def _anchor_cell(anchor: Any) -> str:
    """``xdr:from`` (0-indexed col/row) -> Excel-ссылка на ячейку (``D2``)."""
    col, row = _anchor_col_row(anchor)
    return f"{get_column_letter(col)}{row}"


def _chart_title(chart_root: Any) -> tuple[str, ...]:
    title = chart_root.find(f".//{_q('c', 'title')}")
    if title is None:
        return ()
    return _filter_caption_texts(title.itertext())


def extract_charts(raw: Path) -> list[XlsxChart]:
    """Все встроенные чарты workbook, лист за листом. ``raw`` НЕ изменяется
    (см. докстроку модуля). Малформед/недостижимая ссылка на любом шаге
    цепочки (лист без drawing, drawing без rels, чарт-парт отсутствует) —
    честно пропускается (terminal safety net — конвертация не падает на
    повреждённом OOXML, симметрично ``_classify_docx``)."""
    with zipfile.ZipFile(raw) as z:
        names = set(z.namelist())
        charts: list[XlsxChart] = []
        for sheet_name, sheet_part in _sheet_parts(z).items():
            if sheet_part not in names:
                continue
            sheet_rels = _rel_targets(z, sheet_part)
            drawing_ref = etree.fromstring(z.read(sheet_part)).find(_q("main", "drawing"))
            if drawing_ref is None:
                continue
            drid = drawing_ref.get(_q("r", "id"))
            if drid is None or drid not in sheet_rels:
                continue
            drawing_part = _resolve_target(sheet_part, sheet_rels[drid])
            if drawing_part not in names:
                continue
            drawing_rels = _rel_targets(z, drawing_part)
            drawing_root = etree.fromstring(z.read(drawing_part))
            for anchor, chart_ref in _chart_anchors(drawing_root):
                crid = chart_ref.get(_q("r", "id"))
                if crid is None or crid not in drawing_rels:
                    continue
                chart_part = _resolve_target(drawing_part, drawing_rels[crid])
                if chart_part not in names:
                    continue
                chart_root = etree.fromstring(z.read(chart_part))
                charts.append(
                    XlsxChart(
                        id12=hashlib.sha256(etree.tostring(chart_root)).hexdigest()[:12],
                        sheet=sheet_name,
                        anchor_cell=_anchor_cell(anchor),
                        captions=_chart_title(chart_root),
                    )
                )
        return charts


def _parse_ref_range(ref_after_bang: str) -> tuple[int, int, int, int] | None:
    """``$BQ$27:$BQ$34`` -> ``(col_lo, row_lo, col_hi, row_hi)`` (1-indexed,
    inclusive). Одиночная ячейка (без ``:``) -> вырожденный диапазон
    ``col_lo==col_hi``. ``None`` — не распарсилось (не наш формат ссылки)."""
    m = _CELL_RANGE_RE.match(ref_after_bang.strip())
    if not m:
        return None
    c1, r1 = column_index_from_string(m.group("c1")), int(m.group("r1"))
    if m.group("c2"):
        c2, r2 = column_index_from_string(m.group("c2")), int(m.group("r2"))
    else:
        c2, r2 = c1, r1
    return (min(c1, c2), min(r1, r2), max(c1, c2), max(r1, r2))


def _chart_refs(chart_root: Any) -> list[tuple[str, str]]:
    """(sheet_name, cell_range_text) для каждой ``<c:f>`` формулы серии чарта
    (раскавычивание листа с пробелами: ``'My Sheet'!$A$1`` -> ``My Sheet``,
    ``''``->``'`` — стандартное Excel-экранирование апострофа)."""
    out: list[tuple[str, str]] = []
    for f in chart_root.findall(f".//{_q('c', 'f')}"):
        text = f.text or ""
        if "!" not in text:
            continue
        sheet, _, cell_range = text.partition("!")
        sheet = sheet.strip()
        if sheet.startswith("'") and sheet.endswith("'"):
            sheet = sheet[1:-1].replace("''", "'")
        if sheet:
            out.append((sheet, cell_range))
    return out


def _chart_referenced_sheets(chart_root: Any) -> set[str]:
    """Имена листов, на ячейки которых ссылаются серии чарта — нужны, чтобы
    сохранить эти листы в мини-книге целиком (данные должны остаться
    разрешимыми), даже если чарт физически привязан к другому листу."""
    return {sheet for sheet, _ in _chart_refs(chart_root)}


_WINDOW_PAD_BACK = 5    # запас ДО якоря/диапазона данных (заголовки/подписи левее-выше)
_WINDOW_PAD_FORWARD = 30  # запас ПОСЛЕ — визуальный бокс чарта обычно шире его точечных данных


def _host_window(chart_root: Any, host_sheet: str, anchor_col: int, anchor_row: int) -> tuple[int, int, int, int]:
    """Диапазон ячеек листа-хозяина, который обязан пережить обрезку:
    объединение якорной позиции (1-indexed) чарта и ВСЕХ диапазонов данных,
    на которые ссылаются его серии НА ЭТОМ ЖЕ листе (``<c:f>`` часто указывает
    на ячейки в стороне от визуального якоря — таблица-источник данных
    доughnut/bar чарта, например, не обязана лежать прямо под ним), плюс
    фиксированный запас. Живой checkpoint (WBG GovTech, доughnut на
    Stats!BO37, данные в Stats!$BQ$27:$BR$34) — без запаса по данным окно
    держало бы только визуальный якорь и чарт рендерился бы ПУСТЫМ."""
    col_lo, row_lo, col_hi, row_hi = anchor_col, anchor_row, anchor_col, anchor_row
    for sheet, cell_range in _chart_refs(chart_root):
        if sheet != host_sheet:
            continue
        parsed = _parse_ref_range(cell_range)
        if parsed is None:
            continue
        c_lo, r_lo, c_hi, r_hi = parsed
        col_lo, row_lo = min(col_lo, c_lo), min(row_lo, r_lo)
        col_hi, row_hi = max(col_hi, c_hi), max(row_hi, r_hi)
    return (
        max(1, col_lo - _WINDOW_PAD_BACK),
        max(1, row_lo - _WINDOW_PAD_BACK),
        col_hi + _WINDOW_PAD_FORWARD,
        row_hi + _WINDOW_PAD_FORWARD,
    )


_OWN_RANGE_PAD = 2         # запас вокруг СВОИХ данных (c:f-диапазонов цели)
_OWN_BBOX_COL_PAD = 1      # запас вокруг bbox цели — ПО СТОЛБЦАМ (см. ниже)
_OWN_BBOX_ROW_PAD = 2      # запас вокруг bbox цели — ПО СТРОКАМ (см. ниже)
# Асимметрия проверена экспериментом 2026-07-20: симметричный pad=1 закрывал
# зазор до случайного соседа СПРАВА от bbox radar-чарта (справочная таблица
# индикаторов вплотную к его правому краю), но давал регресс на 3 из 5
# пилотных чартов — И ВСЕ ТРИ регресса были обрезкой СВЕРХУ (внешний
# заголовок над таблицей/чартом у doughnut/bar; заголовок САМОГО ЧАРТА на
# combo, верхняя кромка текста касалась края кропа). Разные оси, разная
# нужная величина запаса: справа — где типично соседи впритык — можно
# ужимать; сверху — где типично живёт заголовок цели/секции — сжимать
# рискованно. Раздельные константы вместо одной дают то же закрытие зазора
# без единого регресса (см. живую проверку в extract_chart_workbook).
_NEIGHBOR_RANGE_PAD = 2    # соседа НЕ ужимаем (это чужая территория — щедрость
# здесь не вредит цели, зато ловит «итоговую» строку/колонку соседа сразу под
# его данными, не входящую в его c:f-ссылки явно — живой checkpoint: Total-
# строка соседнего bar-чарта "Regional distribution" на том же листе)
_ONE_CELL_FALLBACK = 25    # запас якоря без xdr:to (oneCellAnchor, только EMU-размер, не ячейки)


def _chart_own_bbox(anchor: Any, anchor_col: int, anchor_row: int) -> tuple[int, int, int, int]:
    """Точный визуальный bbox чарта (``xdr:from``->``xdr:to``, 1-indexed), БЕЗ
    запаса. ``xdr:twoCellAnchor`` несёт точную конечную ячейку; у
    ``xdr:oneCellAnchor`` есть только EMU-размер без привязки к ячейкам —
    честный фиксированный запас вместо ложно-точной EMU-конвертации."""
    to = anchor.find(_q("xdr", "to"))
    if to is not None:
        return anchor_col, anchor_row, int(to.findtext(_q("xdr", "col"))) + 1, int(to.findtext(_q("xdr", "row"))) + 1
    return anchor_col, anchor_row, anchor_col + _ONE_CELL_FALLBACK, anchor_row + _ONE_CELL_FALLBACK


def _pad_range(rng: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return _pad_range_axes(rng, pad, pad)


def _pad_range_axes(rng: tuple[int, int, int, int], col_pad: int, row_pad: int) -> tuple[int, int, int, int]:
    c1, r1, c2, r2 = rng
    return (max(1, c1 - col_pad), max(1, r1 - row_pad), c2 + col_pad, r2 + row_pad)


def _range_contains(rng: tuple[int, int, int, int], col: int, row: int) -> bool:
    c1, r1, c2, r2 = rng
    return c1 <= col <= c2 and r1 <= row <= r2


def _bounding_union(ranges: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    cols_lo, rows_lo, cols_hi, rows_hi = zip(*ranges, strict=True)
    return (min(cols_lo), min(rows_lo), max(cols_hi), max(rows_hi))


_MAX_TABLE_SPAN = 25  # своя опорная табличка (5-10 строк) — да, показываем;
# полноценный построчный датасет (живой checkpoint: scatter-чарт со 197
# строками, по экономике на строку) — нет: печатная область растягивается
# под его размер, fitToPage сжимает страницу, и сам чарт — то, что должно
# быть главным объектом кропа — превращается в мелкий фрагмент в углу.
# Порог — по ЛЮБОЙ стороне диапазона (сторона таблицы бывает узкой, но очень
# длинной, как раз кейс scatter: 1 столбец x 197 строк).


def _is_compact(rng: tuple[int, int, int, int], max_span: int) -> bool:
    c1, r1, c2, r2 = rng
    return (c2 - c1 + 1) <= max_span and (r2 - r1 + 1) <= max_span


def _ownership_ranges(
    chart_root: Any,
    keep_anchor: Any,
    anchor_col: int,
    anchor_row: int,
    host_sheet: str,
    siblings: list[tuple[Any, Any]],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Взаимный учёт якорей ВСЕХ чартов листа-хозяина (по образцу того, как
    docx-группы физически исключают друг друга при вырезке — здесь объекты
    делят один и тот же лист ячеек, поэтому «исключение» не вырезка блока, а
    разметка «эта ячейка занята другим объектом»). Возвращает
    ``(data_keep, bbox_keep, exclude_ranges)`` — ТРИ, не два, списка: bbox и
    данные цели РАЗВЕДЕНЫ намеренно (живой дефект: bbox чарта — это ЗАПАС «на
    всякий случай» вокруг его визуального прямоугольника, не заслуженное
    владение конкретной ячейкой; если позволить bbox-запасу побеждать чужое
    ЯВНОЕ исключение, соседняя таблица данных снова просочится, стоит ей
    попасть в зону запаса — ровно так и произошло на живом checkpoint'е,
    CE12 оказалась и в exclude ЧУЖОГО чарта, и в bbox-запасе цели, и «keep
    побеждает» пропускало её):

    - ``data_keep`` — диапазоны данных СОБСТВЕННЫХ серий цели на этом листе
      (``_chart_refs``), с запасом. ВСЕГДА побеждает — заслуженное владение.
    - ``bbox_keep`` — визуальный bbox цели, с запасом. Нужен ТОЛЬКО чтобы
      печатная область включала сам рисунок чарта; уступает чужому exclude.
    - ``exclude_ranges`` — bbox + собственные c:f-диапазоны КАЖДОГО ДРУГОГО
      чарта того же drawing-парта, с ТЕМ ЖЕ запасом, что у своих (иначе
      «итоговая» строка соседа сразу под его данными, не входящая в его
      c:f явно, осталась бы ничейной и просочилась бы — живой дефект)."""
    data_keep: list[tuple[int, int, int, int]] = []
    for sheet, cell_range in _chart_refs(chart_root):
        if sheet != host_sheet:
            continue
        parsed = _parse_ref_range(cell_range)
        if parsed is not None:
            data_keep.append(_pad_range(parsed, _OWN_RANGE_PAD))
    bbox_keep = [
        _pad_range_axes(
            _chart_own_bbox(keep_anchor, anchor_col, anchor_row), _OWN_BBOX_COL_PAD, _OWN_BBOX_ROW_PAD
        )
    ]

    exclude: list[tuple[int, int, int, int]] = []
    for anchor, sib_root in siblings:
        if anchor is keep_anchor:
            continue
        sib_col, sib_row = _anchor_col_row(anchor)
        exclude.append(_pad_range(_chart_own_bbox(anchor, sib_col, sib_row), _NEIGHBOR_RANGE_PAD))
        for sheet, cell_range in _chart_refs(sib_root):
            if sheet != host_sheet:
                continue
            parsed = _parse_ref_range(cell_range)
            if parsed is not None:
                exclude.append(_pad_range(parsed, _NEIGHBOR_RANGE_PAD))
    return data_keep, bbox_keep, exclude


def _blank_foreign_cells(
    sheet_root: Any,
    print_window: tuple[int, int, int, int],
    data_keep: list[tuple[int, int, int, int]],
    exclude_ranges: list[tuple[int, int, int, int]],
) -> None:
    """In-place: внутри ``print_window`` убрать ячейки, попавшие в ЧЕЙ-ТО
    ЧУЖОЙ (``exclude_ranges``) диапазон и НЕ попавшие ни в один диапазон
    СОБСТВЕННЫХ ДАННЫХ цели (``data_keep`` — заслуженное владение, ВСЕГДА
    побеждает; ``bbox_keep`` цели сюда намеренно НЕ передаётся — bbox-запас
    не должен перебивать чужое явное исключение, см. ``_ownership_ranges``).
    Ячейка, не заявленная НИКЕМ (не в data_keep и не в exclude — например,
    самостоятельный заголовок-подпись раздела) остаётся как есть: default-
    keep, не default-blank — вычищаем только ДОКАЗАННО чужое, не гадаем."""
    pw_c1, pw_r1, pw_c2, pw_r2 = print_window
    sheet_data = sheet_root.find(_q("main", "sheetData"))
    for row_el in list(sheet_data):
        r = int(row_el.get("r"))
        if not (pw_r1 <= r <= pw_r2):
            continue
        for c_el in list(row_el):
            m = _CELL_REF_RE.match(c_el.get("r", ""))
            if m is None:
                continue
            col = column_index_from_string(m.group(1))
            if not (pw_c1 <= col <= pw_c2):
                continue
            if any(_range_contains(kr, col, r) for kr in data_keep):
                continue
            if any(_range_contains(er, col, r) for er in exclude_ranges):
                row_el.remove(c_el)


def _trim_and_devalue_sheet(sheet_root: Any, window: tuple[int, int, int, int]) -> None:
    """In-place: строки/ячейки листа ВНЕ ``window`` — удалить; ячейки С
    формулой ВНУТРИ ``window`` — конвертировать в статичное значение (``<f>``
    убрать, оставить закэшированный ``<v>``/``t``). Обе меры нужны вместе:
    обрезка не даёт soffice распечатать книгу многостраничным гигантом
    (живой checkpoint: 214x96 лист -> 33 страницы книги, из них 10 даже
    после обрезки ДО этого шага), а девалидация формул убирает зависимость от
    УДАЛЁННЫХ листов (formula-ячейки вида ``=GTMI_Data!...`` иначе дают
    #ИМЯ?/пустой результат при пересчёте на открытии, ПОДАВЛЯЯ даже
    закэшированные значения САМОГО чарта — см. Design rationale спека)."""
    col_lo, row_lo, col_hi, row_hi = window
    sheet_data = sheet_root.find(_q("main", "sheetData"))
    for row_el in list(sheet_data):
        r = int(row_el.get("r"))
        if not (row_lo <= r <= row_hi):
            sheet_data.remove(row_el)
            continue
        for c_el in list(row_el):
            m = _CELL_REF_RE.match(c_el.get("r", ""))
            if m is None or not (col_lo <= column_index_from_string(m.group(1)) <= col_hi):
                row_el.remove(c_el)
                continue
            f_el = c_el.find(_q("main", "f"))
            if f_el is not None:
                c_el.remove(f_el)


def _set_single_page_print_area(sheet_root: Any, wb_root: Any, sheet_name: str, window: tuple[int, int, int, int]) -> None:
    """In-place: ``pageSetUpPr fitToPage`` + ``pageSetup fitToWidth/Height=1``
    на листе (масштабирует печать до ОДНОЙ страницы независимо от размера
    ``window``) + ``_xlnm.Print_Area`` definedName, ограничивающий печать
    ИМЕННО этим диапазоном (иначе LibreOffice печатает ВЕСЬ used-range листа,
    включая давно пустые — но формально «использованные» — колонки/строки
    перед ``window``, живой checkpoint: без Print_Area те же данные давали
    10 страниц вместо 1). ``localSheetId="0"`` — лист-хозяин уже переставлен
    на позицию 0 к моменту вызова (см. ``extract_chart_workbook``)."""
    sheetpr = sheet_root.find(_q("main", "sheetPr"))
    if sheetpr is None:
        sheetpr = etree.Element(_q("main", "sheetPr"))
        sheet_root.insert(0, sheetpr)
    etree.SubElement(sheetpr, _q("main", "pageSetUpPr")).set("fitToPage", "1")

    existing_ps = sheet_root.find(_q("main", "pageSetup"))
    if existing_ps is not None:
        sheet_root.remove(existing_ps)
    page_setup = etree.SubElement(sheet_root, _q("main", "pageSetup"))
    page_setup.set("fitToWidth", "1")
    page_setup.set("fitToHeight", "1")
    page_setup.set("orientation", "landscape")

    col_lo, row_lo, col_hi, row_hi = window
    defined_names = etree.SubElement(wb_root, _q("main", "definedNames"))
    print_area = etree.SubElement(defined_names, _q("main", "definedName"))
    print_area.set("name", "_xlnm.Print_Area")
    print_area.set("localSheetId", "0")
    print_area.text = (
        f"{sheet_name}!${get_column_letter(col_lo)}${row_lo}:${get_column_letter(col_hi)}${row_hi}"
    )


def extract_chart_workbook(raw: Path, id12: str) -> bytes | None:
    """Пересобрать мини-книгу для изолированного рендера чарта с данным
    ``id12`` (см. ``figures_vlm._render_xlsx_chart``).

    Живой checkpoint (WBG GovTech Dataset, 9 листов, 214x96 лист-хозяин, 55
    чартов, реальный документ) опроверг ТРИ исходных предположения:

    1. «soffice --convert-to pdf рендерит только видимые листы» — ложно:
       LibreOffice ИГНОРИРУЕТ ``sheet_state`` при headless-конвертации и
       рендерит ВСЕ листы в порядке документа (9 листов -> 33 страницы), а
       рендер брал ``pages[0]`` — систематически захватывая содержимое
       ПЕРВОГО листа КНИГИ, не листа-хозяина чарта. Исправлено: листы, не
       являющиеся ни хозяином, ни референсированные формулами серий чарта
       (``_chart_referenced_sheets``), физически УДАЛЯЮТСЯ (не прячутся —
       прятать бесполезно), лист-хозяин переставляется на позицию 0.
    2. «печать всего листа-хозяина уместится на одной странице» — тоже
       ложно на реальном 214x96 листе: страница 1 после фикса (1) оказывалась
       ДАЛЕКО от визуальной позиции чарта (used-range листа огромен).
       Исправлено: ``_host_window`` + ``_trim_and_devalue_sheet`` обрезают
       лист-хозяин до диапазона (якорь ∪ диапазоны данных серий чарта на
       этом листе) + запас, ``_set_single_page_print_area`` гарантирует ровно
       одну печатную страницу. Обрезка формул до статичных закэшированных
       значений (не просто обрезка строк/столбцов) ОБЯЗАТЕЛЬНА: без нeё
       formula-ячейки листа-хозяина, ссылающиеся на УЖЕ УДАЛЁННЫЕ листы,
       дают #ИМЯ?/пусто при пересчёте на открытии, и это перекрывает даже
       закэшированные ``c:numCache``/``c:strCache`` значения самого чарта.
    3. «печать области (1) заведомо содержит ТОЛЬКО целевой чарт» — тоже
       ложно на плотном дашборд-листе: соседние чарты/их таблицы-источники
       лежат в паре строк/столбцов от цели, и прямоугольная печатная область
       неизбежно цепляет их край. Исправлено: ``_ownership_ranges`` —
       взаимный учёт якорей ВСЕХ чартов drawing-парта (по духу похоже на то,
       как docx-группы физически исключают друг друга при вырезке — здесь
       объекты делят один лист ячеек, поэтому «исключение» это разметка
       «занято другим», не вырезка блока). Печатная область — ограничивающий
       прямоугольник СОБСТВЕННЫХ диапазонов цели (bbox + c:f-данные), а не
       фиксированная догадка. ``data_keep`` (собственные данные) и
       ``bbox_keep`` (визуальный запас) РАЗВЕДЕНЫ по силе: данные — заслуженное
       владение, побеждают ВСЕГДА; bbox — просто запас «на всякий случай», не
       должен перебивать ЧУЖОЕ явное владение (живой дефект: ячейка чужой
       таблицы, ошибочно попавшая в bbox-запас цели, проходила как «своя» —
       запас не равен владению). ``_blank_foreign_cells`` вычищает внутри
       печатной области только ДОКАЗАННО чужое (заявленное диапазоном
       соседа — с ТЕМ ЖЕ запасом, что у своих, иначе «итоговая» строка
       соседа сразу под его данными осталась бы ничейной), оставляя
       незаявленный контекст (например, самостоятельный заголовок раздела)
       как есть.
    4. «своя таблица-источник ВСЕГДА стоит подтягивать в кроп» — тоже не
       универсально: маленькая опорная табличка (5-10 строк) — полезная
       страховка от неточного прочтения чарта VLM-моделью, но чарт вроде
       scatter/line, чьи серии ссылаются на диапазон в сотни строк (по точке
       на строку), при том же подходе растягивает печатную область под весь
       датасет — ``fitToPage`` сжимает лист в один кадр, и сам чарт (то, что
       ДОЛЖНО быть главным объектом) съёживается до нечитаемого фрагмента.
       ``_MAX_TABLE_SPAN`` — диапазон крупнее порога по любой стороне не
       тянет печатную область под себя (``compact_data_keep`` в отличие от
       полного ``data_keep``), но остаётся живым в ``data_window`` (резолв
       формул) и в приоритете ``_blank_foreign_cells`` (не может быть по
       ошибке зачищен как чужой).

    ``definedNames`` оригинала (кроме нового Print_Area) и ``xl/calcChain.xml``
    отброшены целиком — не нужны для одноразового рендера, их
    ``localSheetId``-индексы всё равно рассинхронизировались бы после удаления
    листов. ``[Content_Types].xml`` подчищен от Override-записей удалённых
    частей (иначе пакет декларирует несуществующие парты).

    Известные ограничения (не устранены этим фиксом, документированы честно):
    (а) если референсированный чартом НЕ-хозяйский лист сам несёт формулы,
    ссылающиеся на ТРЕТЬИ (удалённые) листы — та же проблема из (2)
    воспроизведётся на нём одним уровнем глубже; полный транзитивный анализ
    зависимостей не реализован (не встретилось в живом checkpoint — все
    проверенные чарты ссылались только на свой же лист-хозяин); (б) контент,
    НЕ заявленный НИ одним чартом (например, независимый текстовый заголовок
    раздела дашборда ИЛИ справочная таблица, не питающая ни одну формулу
    серий) из кропа цели не убирается — эвристика (3) вычищает только
    доказанно чужое, не гадает по умолчанию (сознательный компромисс —
    убирать «неопознанное» рискует стереть легитимные заголовки/подписи
    рядом с целью, которые тоже static, см. §Design rationale спека), поэтому
    такой «ничейный» контент иногда остаётся виден рядом с целью.

    drawing-парт листа-хозяина по-прежнему обрезается до ОДНОГО целевого
    чарта (иначе кроп заденет соседний чарт/фигуру того же drawing-парта).
    None — чарт с таким id12 не найден при пере-детекции (raw изменился?)."""
    with zipfile.ZipFile(raw) as z:
        names = z.namelist()
        name_set = set(names)
        target: tuple[str, str, Any, Any] | None = None  # (host_sheet, drawing_part, keep_anchor, chart_root)
        drawing_cache: dict[str, Any] = {}
        sheet_parts = _sheet_parts(z)
        for sheet_name, sheet_part in sheet_parts.items():
            if sheet_part not in name_set:
                continue
            sheet_rels = _rel_targets(z, sheet_part)
            drawing_ref = etree.fromstring(z.read(sheet_part)).find(_q("main", "drawing"))
            if drawing_ref is None:
                continue
            drid = drawing_ref.get(_q("r", "id"))
            if drid is None or drid not in sheet_rels:
                continue
            drawing_part = _resolve_target(sheet_part, sheet_rels[drid])
            if drawing_part not in name_set:
                continue
            drawing_rels = _rel_targets(z, drawing_part)
            drawing_root = drawing_cache.setdefault(drawing_part, etree.fromstring(z.read(drawing_part)))
            for anchor, chart_ref in _chart_anchors(drawing_root):
                crid = chart_ref.get(_q("r", "id"))
                if crid is None or crid not in drawing_rels:
                    continue
                chart_part = _resolve_target(drawing_part, drawing_rels[crid])
                if chart_part not in name_set:
                    continue
                chart_root = etree.fromstring(z.read(chart_part))
                if hashlib.sha256(etree.tostring(chart_root)).hexdigest()[:12] == id12:
                    target = (sheet_name, drawing_part, anchor, chart_root)
                    break
            if target is not None:
                break
        if target is None:
            return None
        host_sheet, drawing_part, keep_anchor, chart_root = target
        drawing_root = drawing_cache[drawing_part]
        keep_sheets = {host_sheet} | _chart_referenced_sheets(chart_root)
        anchor_col, anchor_row = _anchor_col_row(keep_anchor)

        # Взаимный учёт якорей ВСЕХ чартов drawing-парта (siblings) — ДО обрезки
        # drawing до одного целевого анкера (см. _ownership_ranges).
        drawing_rels = _rel_targets(z, drawing_part)
        siblings: list[tuple[Any, Any]] = []
        for anchor, chart_ref in _chart_anchors(drawing_root):
            crid = chart_ref.get(_q("r", "id"))
            if crid is None or crid not in drawing_rels:
                continue
            sib_chart_part = _resolve_target(drawing_part, drawing_rels[crid])
            if sib_chart_part not in name_set:
                continue
            siblings.append((anchor, etree.fromstring(z.read(sib_chart_part))))
        data_keep, bbox_keep, exclude_ranges = _ownership_ranges(
            chart_root, keep_anchor, anchor_col, anchor_row, host_sheet, siblings
        )
        # Гигантский диапазон (полноценный датасет, не опорная табличка) не
        # тянет печатную область под себя (см. _MAX_TABLE_SPAN) — но остаётся
        # в data_keep для приоритета в _blank_foreign_cells и в data_window
        # для резолва формул (обрезки данных не происходит вовсе).
        compact_data_keep = [r for r in data_keep if _is_compact(r, _MAX_TABLE_SPAN)]
        print_window = _bounding_union(compact_data_keep + bbox_keep)

        for tag in ("oneCellAnchor", "twoCellAnchor"):
            for anchor in list(drawing_root.findall(_q("xdr", tag))):
                if anchor is not keep_anchor:
                    drawing_root.remove(anchor)
        new_drawing_xml = etree.tostring(drawing_root, xml_declaration=True, encoding="UTF-8", standalone=True)

        wb_root = etree.fromstring(z.read("xl/workbook.xml"))
        sheets_el = wb_root.find(f".//{_q('main', 'sheets')}")
        removed_parts: set[str] = set()
        for sheet_el in list(sheets_el):
            name = sheet_el.get("name")
            if name in keep_sheets:
                if name == host_sheet:
                    sheet_el.set("state", "visible")  # покрывает случай «чарт жил на скрытом листе»
                continue
            sheets_el.remove(sheet_el)
            part = sheet_parts.get(name)
            if part is not None:
                removed_parts.add(part)
        host_el = next(el for el in sheets_el if el.get("name") == host_sheet)
        sheets_el.remove(host_el)
        sheets_el.insert(0, host_el)  # страница 1 PDF = хозяин, независимо от порядка в оригинале

        defined_names_el = wb_root.find(f".//{_q('main', 'definedNames')}")
        if defined_names_el is not None:
            wb_root.remove(defined_names_el)  # localSheetId рассинхронизировался бы после удаления листов

        host_sheet_part = sheet_parts[host_sheet]
        host_root = etree.fromstring(z.read(host_sheet_part))
        # data_window (генерозный, для резолва формул) ГАРАНТИРОВАННО объемлет
        # print_window (bounding union data_keep+bbox_keep, те же «свои»
        # диапазоны с меньшим запасом, 30/5 vs 2) — обрезка не разрежет то,
        # что должно выжить.
        data_window = _host_window(chart_root, host_sheet, anchor_col, anchor_row)
        _trim_and_devalue_sheet(host_root, data_window)
        _blank_foreign_cells(host_root, print_window, data_keep, exclude_ranges)
        _set_single_page_print_area(host_root, wb_root, host_sheet, print_window)
        new_host_sheet_xml = etree.tostring(host_root, xml_declaration=True, encoding="UTF-8", standalone=True)

        new_wb_xml = etree.tostring(wb_root, xml_declaration=True, encoding="UTF-8", standalone=True)

        ct_root = etree.fromstring(z.read("[Content_Types].xml"))
        removed_part_names = {f"/{p}" for p in removed_parts}
        for override in list(ct_root):
            if etree.QName(override).localname == "Override" and override.get("PartName") in removed_part_names:
                ct_root.remove(override)
        new_ct_xml = etree.tostring(ct_root, xml_declaration=True, encoding="UTF-8", standalone=True)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zo:
            for n in names:
                if n in removed_parts or n == "xl/calcChain.xml":
                    continue
                if n == drawing_part:
                    zo.writestr(n, new_drawing_xml)
                elif n == "xl/workbook.xml":
                    zo.writestr(n, new_wb_xml)
                elif n == host_sheet_part:
                    zo.writestr(n, new_host_sheet_xml)
                elif n == "[Content_Types].xml":
                    zo.writestr(n, new_ct_xml)
                else:
                    zo.writestr(n, z.read(n))
        return buf.getvalue()


def render_chart_marker(chart: XlsxChart) -> str:
    caption_line = "; ".join(chart.captions) if chart.captions else "(нет текста)"
    return (
        f"> [Figure, xlsx chart {chart.id12} on {chart.sheet}!{chart.anchor_cell} — "
        f"chart content not analyzed]\n"
        f"> captions: {caption_line}"
    )
