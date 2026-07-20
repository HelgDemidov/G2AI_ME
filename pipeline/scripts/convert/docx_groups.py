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

_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
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


def _group_captions(ac: Any) -> tuple[str, ...]:
    """Текст группы (captions под маркером, zero-loss без VLM): ``itertext()``
    тянет и числовой мусор координат (``wp:posOffset``/``a:ext`` несут значение
    как текстовое содержимое элемента, не атрибут) — отсеиваем строки, целиком
    состоящие из цифр (настоящие подписи содержат буквы); дедуп по порядку
    появления (proofErr иногда дробит слово на несколько run — не склеиваем,
    честно передаём как есть)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in ac.itertext():
        s = t.strip()
        if not s or _NUMERIC_JUNK_RE.match(s) or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


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
        for block in list(body):
            for ac in block.findall(f".//{_q('mc', 'AlternateContent')}"):
                choice = ac.find(_q("mc", "Choice"))
                if choice is None or choice.find(f".//{_q('wpg', 'wgp')}") is None:
                    continue
                media_ids = _group_media_ids(ac, rel_targets, z, names)
                captions = _group_captions(ac)
                id12 = hashlib.sha256(etree.tostring(ac)).hexdigest()[:12]
                groups.append(DocxGroup(id12=id12, media_ids=media_ids, captions=captions))

                run = ac.getparent()
                sentinel = etree.Element(_q("w", "t"))
                sentinel.set(_XML_SPACE, "preserve")
                sentinel.text = f"{SENTINEL_PREFIX}{id12}"
                run.replace(ac, sentinel)
        if not groups:
            return orig, []
        new_doc_xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zo:
            for n in z.namelist():
                zo.writestr(n, new_doc_xml if n == "word/document.xml" else z.read(n))
        return buf.getvalue(), groups


def _render_group_marker(id12: str, captions: tuple[str, ...]) -> str:
    caption_line = "; ".join(captions) if captions else "(нет текста)"
    return (
        f"\n\n> [Figure, docx group {id12} — composite content not analyzed]\n"
        f"> captions: {caption_line}\n\n"
    )


def inject_group_markers(text: str, groups: list[DocxGroup]) -> str:
    """Заменить текстовые сентинелы (пережившие mammoth+markdownify на месте
    вырезанной группы, см. ``extract_and_strip_groups``) на честный
    маркер-блок с сохранёнными подписями."""
    if not groups:
        return text
    by_id = {g.id12: g for g in groups}

    def _replace(m: re.Match[str]) -> str:
        group = by_id.get(m.group("id"))
        if group is None:  # практически невозможно (id12 — sha256), но не падаем
            return m.group(0)
        return _render_group_marker(group.id12, group.captions)

    return _SENTINEL_SCAN_RE.sub(_replace, text)


def all_media_ids(groups: list[DocxGroup]) -> frozenset[str]:
    """Объединение media_ids всех групп — «поглощённые» id для фолбэк-прохода
    (``converters._docx_image_markers(raw, placed=...)``): куски группы не
    должны всплыть повторно ни инлайн, ни в ``## Figures (position unknown)``."""
    if not groups:
        return frozenset()
    return frozenset().union(*(g.media_ids for g in groups))
