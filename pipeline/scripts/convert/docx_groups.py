"""Composite-группы docx (spec convert-docx §2-ter): Word рисует сложную
инфографику как ГРУППУ фигур (``mc:AlternateContent``/``mc:Choice``/``wpg:wgp``
+ VML-``mc:Fallback``) — mammoth обходит такую группу поэлементно, распадая ОДНУ
диаграмму на россыпь растровых фрагментов + бессвязные строки текста (живой
кейс, см. спек §2-ter.1: 3/3 настоящих инфографик тестовой вырезки распадались
именно так). Этот модуль детектирует такие группы ДО передачи документа
mammoth, вырезает их целиком (заменяя на текстовый сентинел, который mammoth
пронесёт как обычный текст на своём месте), собирает id вложенных media
(чтобы фолбэк-проход ``converters._docx_image_markers`` их не задублировал)
и текстовые подписи группы (zero-loss на случай недоступности VLM).

Детект: top-level блок body содержит ``mc:AlternateContent``, чей ``mc:Choice``
несёт ``wpg:wgp`` (современный DrawingML-группа фигур) — сигнал специфичный и
надёжный (прототип 2026-07-20: 3/3 реальных диаграмм, 0 ложных срабатываний).

Вторая категория (kind="chart", §2-ter ultimate-тест 2026-07-20): голый
``w:drawing`` с ``c:chart``-анкером ВНЕ ``mc:AlternateContent`` — классический
нативный Word-чарт (данные в embeddings/*.xlsx). У него нет Fallback-картинки
вовсе: mammoth его просто НЕ видит — класс «тихая потеря» (живой кейс: bar-chart
CAPEX/OPEX в реальном отчёте). Обрабатывается тем же конвейером
сентинел→маркер→soffice-рендер: мини-docx наследует ВЕСЬ zip оригинала (см.
``extract_group_docx``), поэтому chart-парты/rels/xlsx доезжают автоматически.
chartEx-диаграммы нового поколения (sunburst и т.п.) сюда СОЗНАТЕЛЬНО не входят:
Word по построению кладёт им в ``mc:Fallback`` готовую PNG-отрисовку — её
забирает штатный mammoth-путь инлайн-картинок (позиция + VLM бесплатно).
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

from convert import chart_render
from convert.chart_data import ChartData, parse_chart

_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

SENTINEL_PREFIX = "DOCXGROUPSENTINEL"
# \**...\** поглощает возможное bold-обрамление markdownify: сентинел наследует
# rPr того run'а, чьё содержимое заменил (живой кейс — блок с bold в фикстуре).
_SENTINEL_SCAN_RE = re.compile(r"\**" + SENTINEL_PREFIX + r"(?P<id>[0-9a-f]{12})\**")
_NUMERIC_JUNK_RE = re.compile(r"^-?\d+$")  # posOffset/extent координаты в itertext()


def _q(prefix: str, local: str) -> str:
    return f"{{{_NS[prefix]}}}{local}"


@dataclass(frozen=True)
class DocxGroup:
    id12: str
    media_ids: frozenset[str]
    captions: tuple[str, ...]
    kind: str = "group"  # "group" (wpg-группа фигур) | "chart" (нативный c:chart)
    # kind="chart" ТОЛЬКО (spec chart-data-extraction §4.2): распарсенные данные
    # чарта для data-driven резолюции (inject_group_markers). kind="group" —
    # всегда None, group-путь остаётся на VLM без изменений.
    chart_data: ChartData | None = None


def _rel_targets(z: zipfile.ZipFile, part: str) -> dict[str, str]:
    """rId -> Target для ``part`` (напр. ``word/document.xml``), из соседнего .rels."""
    rels_name = f"{posixpath.dirname(part)}/_rels/{posixpath.basename(part)}.rels"
    if rels_name not in z.namelist():
        return {}
    root = etree.fromstring(z.read(rels_name))
    return {rel.get("Id"): rel.get("Target") for rel in root if rel.get("Id")}


def _group_media_ids(
    ac: Any, rel_targets: dict[str, str], z: zipfile.ZipFile, names: set[str]
) -> frozenset[str]:
    ids: set[str] = set()
    for el in ac.iter():
        for attr in ("embed", "id", "link"):
            rid = el.get(_q("r", attr))
            if rid is None or rid not in rel_targets:
                continue
            media = posixpath.normpath(posixpath.join("word", rel_targets[rid]))
            if media.startswith("word/media/") and media in names:
                ids.add(hashlib.sha256(z.read(media)).hexdigest()[:12])
    return frozenset(ids)


def _filter_caption_texts(texts: Any) -> tuple[str, ...]:
    """Общий фильтр текстов под captions: ``itertext()`` тянет и числовой мусор
    координат (``wp:posOffset``/``a:ext`` несут значение как текстовое
    содержимое элемента, не атрибут) — отсеиваем строки, целиком состоящие из
    цифр (настоящие подписи содержат буквы); дедуп по порядку появления
    (proofErr иногда дробит слово на несколько run — не склеиваем, честно
    передаём как есть)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        s = t.strip()
        if not s or _NUMERIC_JUNK_RE.match(s) or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


def _group_captions(ac: Any) -> tuple[str, ...]:
    """Текст группы (captions под маркером, zero-loss без VLM)."""
    return _filter_caption_texts(ac.itertext())


def _chart_root(
    drawing: Any, rel_targets: dict[str, str], z: zipfile.ZipFile, names: set[str]
) -> Any | None:
    """Распарсенный chart-парт (``word/charts/chartN.xml``) референсированный
    ``w:drawing``-анкером — общий резолв для captions (``_chart_captions``) И
    данных (``chart_data.parse_chart``, spec chart-data-extraction §4.2). None —
    анкер/rel/парт недостижимы (малформед OOXML, честно пропускается)."""
    chart_ref = drawing.find(f".//{_q('c', 'chart')}")
    rid = chart_ref.get(_q("r", "id")) if chart_ref is not None else None
    if rid is None or rid not in rel_targets:
        return None
    part = posixpath.normpath(posixpath.join("word", rel_targets[rid]))
    if part not in names:
        return None
    return etree.fromstring(z.read(part))


def _chart_captions(chart_root: Any | None) -> tuple[str, ...]:
    """Заголовок чарта из его chart-парта (``c:title``): сам ``w:drawing``-анкер
    текста не несёт (данные и заголовок живут в ``word/charts/chartN.xml``).
    Берём ТОЛЬКО title, не весь парт — иначе captions затопило бы подписями
    осей/категорий/значений."""
    if chart_root is None:
        return ()
    title = chart_root.find(f".//{_q('c', 'title')}")
    if title is None:
        return ()
    return _filter_caption_texts(title.itertext())


def _iter_group_acs(body: Any) -> list[tuple[Any, Any]]:
    """Топ-level (block, ac) пары для всех composite-групп ``body`` — общая
    точка детекта для ``extract_and_strip_groups``/``extract_group_docx``
    (единственный критерий: ``mc:Choice`` несёт ``wpg:wgp``, см. докстроку
    модуля)."""
    pairs: list[tuple[Any, Any]] = []
    for block in list(body):
        for ac in block.findall(f".//{_q('mc', 'AlternateContent')}"):
            choice = ac.find(_q("mc", "Choice"))
            if choice is None or choice.find(f".//{_q('wpg', 'wgp')}") is None:
                continue
            pairs.append((block, ac))
    return pairs


def _iter_chart_drawings(body: Any) -> list[tuple[Any, Any]]:
    """Топ-level (block, drawing) пары для голых chart-анкеров: ``w:drawing``
    с ``c:chart`` ВНЕ ``mc:AlternateContent``. Вложенные в AC чарты (chartEx:
    Choice несёт cx-диаграмму, Fallback — готовую PNG-картинку от Word)
    сознательно пропускаются — их забирает mammoth-путь инлайн-картинок
    (см. докстроку модуля)."""
    pairs: list[tuple[Any, Any]] = []
    for block in list(body):
        for drawing in block.findall(f".//{_q('w', 'drawing')}"):
            if drawing.find(f".//{_q('c', 'chart')}") is None:
                continue
            anc = drawing.getparent()
            inside_ac = False
            while anc is not None and anc is not block:
                if anc.tag == _q("mc", "AlternateContent"):
                    inside_ac = True
                    break
                anc = anc.getparent()
            if not inside_ac:
                pairs.append((block, drawing))
    return pairs


def _iter_objects(body: Any) -> list[tuple[Any, Any, str]]:
    """Все вырезаемые объекты body: (block, element, kind) — единая точка
    для ``extract_and_strip_groups``/``extract_group_docx``."""
    objects = [(b, el, "group") for b, el in _iter_group_acs(body)]
    objects += [(b, el, "chart") for b, el in _iter_chart_drawings(body)]
    return objects


def extract_and_strip_groups(raw: Path) -> tuple[bytes, list[DocxGroup]]:
    """Вернуть (переписанный zip docx, найденные группы). Ноль групп -> байты
    БАЙТ-В-БАЙТ идентичны ``raw.read_bytes()`` (документ без composite-групп —
    большинство docx — платит только за один проход детекта, ноль риска
    случайно исказить содержимое)."""
    orig = raw.read_bytes()
    with zipfile.ZipFile(io.BytesIO(orig)) as z:
        names = set(z.namelist())
        if "word/document.xml" not in names:
            return orig, []
        rel_targets = _rel_targets(z, "word/document.xml")
        tree = etree.fromstring(z.read("word/document.xml"))
        body = tree.find(_q("w", "body"))
        groups: list[DocxGroup] = []
        for _block, el, kind in _iter_objects(body):
            media_ids = _group_media_ids(el, rel_targets, z, names)
            parsed_chart_data: ChartData | None = None
            if kind == "group":
                captions = _group_captions(el)
            else:
                chart_root = _chart_root(el, rel_targets, z, names)
                captions = _chart_captions(chart_root)
                if chart_root is not None:
                    parsed_chart_data = parse_chart(chart_root)
            id12 = hashlib.sha256(etree.tostring(el)).hexdigest()[:12]
            groups.append(
                DocxGroup(
                    id12=id12, media_ids=media_ids, captions=captions, kind=kind,
                    chart_data=parsed_chart_data,
                )
            )

            run = el.getparent()
            sentinel = etree.Element(_q("w", "t"))
            sentinel.set(_XML_SPACE, "preserve")
            sentinel.text = f"{SENTINEL_PREFIX}{id12}"
            run.replace(el, sentinel)
        if not groups:
            return orig, []
        new_doc_xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zo:
            for n in z.namelist():
                zo.writestr(n, new_doc_xml if n == "word/document.xml" else z.read(n))
        return buf.getvalue(), groups


def extract_group_docx(raw: Path, id12: str) -> bytes | None:
    """Пересобрать мини-docx, содержащий ТОЛЬКО блок с данной группой (+
    ``sectPr`` оригинала для геометрии страницы) — под изолированный рендер
    через soffice (``figures_vlm._render_docx_group``). Прототип 2026-07-20:
    все 3 диаграммы тестовой вырезки отрендерились этим способом ЦЕЛИКОМ.
    None — группа с таким id12 не найдена при пере-детекции (raw изменился?)."""
    with zipfile.ZipFile(raw) as z:
        names = z.namelist()
        tree = etree.fromstring(z.read("word/document.xml"))
        body = tree.find(_q("w", "body"))
        blocks = list(body)
        target = next(
            (block for block, el, _kind in _iter_objects(body) if hashlib.sha256(etree.tostring(el)).hexdigest()[:12] == id12),
            None,
        )
        if target is None:
            return None
        sect = blocks[-1] if etree.QName(blocks[-1]).localname == "sectPr" else None
        for block in blocks:
            body.remove(block)
        body.append(target)
        if sect is not None and sect is not target:
            body.append(sect)
        new_doc_xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zo:
            for n in names:
                zo.writestr(n, new_doc_xml if n == "word/document.xml" else z.read(n))
        return buf.getvalue()


def _render_group_marker(id12: str, captions: tuple[str, ...], kind: str = "group") -> str:
    caption_line = "; ".join(captions) if captions else "(нет текста)"
    noun = "composite" if kind == "group" else "chart"
    return (
        f"\n\n> [Figure, docx {kind} {id12} — {noun} content not analyzed]\n"
        f"> captions: {caption_line}\n\n"
    )


def inject_group_markers(text: str, groups: list[DocxGroup]) -> str:
    """Заменить текстовые сентинелы (пережившие mammoth+markdownify на месте
    вырезанной группы, см. ``extract_and_strip_groups``) на итоговый блок.

    kind="group" — БЕЗ ИЗМЕНЕНИЙ, честный VLM-маркер (spec chart-data-extraction
    §4.2/§2: group-путь остаётся на VLM). kind="chart" — data-driven (spec §4.2):
    ``chart_render.render_chart(group.chart_data)``, если извлечение непустое;
    пустое извлечение (нет numCache и т.п.) -> ТОТ ЖЕ честный маркер, что и
    раньше (caption-фолбэк, zero-loss без VLM). Позиция сохраняется точно —
    сентинел заменяется IN-PLACE (spec §4.4: docx-провенанс = сама позиция в
    потоке, отдельная строка не нужна, в отличие от xlsx)."""
    if not groups:
        return text
    by_id = {g.id12: g for g in groups}

    def _replace(m: re.Match[str]) -> str:
        group = by_id.get(m.group("id"))
        if group is None:  # практически невозможно (id12 — sha256), но не падаем
            return m.group(0)
        if group.kind == "chart" and group.chart_data is not None:
            rendered = chart_render.render_chart(group.chart_data)
            if rendered is not None:
                return f"\n\n{rendered}\n\n"
        return _render_group_marker(group.id12, group.captions, group.kind)

    return _SENTINEL_SCAN_RE.sub(_replace, text)


def all_media_ids(groups: list[DocxGroup]) -> frozenset[str]:
    """Объединение media_ids всех групп — «поглощённые» id для фолбэк-прохода
    (``converters._docx_image_markers(raw, placed=...)``): куски группы не
    должны всплыть повторно ни инлайн, ни в ``## Figures (position unknown)``."""
    if not groups:
        return frozenset()
    return frozenset().union(*(g.media_ids for g in groups))
