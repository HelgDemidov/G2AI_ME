"""Векторная детекция инфографики: элементы, дедуп, кластеризация, guards.

Заменяет word-gap-эвристику (детекцию «диаграммы» по разрывам между словами —
известное слабое место, ложно срабатывавшее на TOC/SWOT, см. CLAUDE.md) на
геометрию: rects/curves/images страницы группируются в кластеры, и только
кластеры, прошедшие три guard'а против съедания прозы, становятся кандидатами
на реконструкцию (спек convert-graphics §1).

Вся логика этого модуля — чистые функции над примитивами (bbox/Element),
тестируемые синтетикой без единого реального PDF: главный урок предыдущего
ревью — геометрия конвертера была не покрыта тестами вообще.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

BBox = tuple[float, float, float, float]  # (x0, top, x1, bottom)

# --- дедуп / фильтр / кластеризация ---
DEDUPE_ROUND_PT = 0.5          # stroke+fill дубли одного бокса (PDF рисует их 2 объектами)
ELEMENT_MIN_SIDE_PT = 4.0      # тоньше — линейки/подчёркивания, не элементы фигур
ELEMENT_MAX_PAGE_FRACTION = 0.85  # крупнее — фон страницы
CLUSTER_GAP_PT = 8.0           # расширение bbox при кластеризации (связность)

# --- guards региона (чартер §7.1: сомнение => проза) ---
REGION_MIN_ELEMENTS = 3            # < 3 элементов — декор/плашка, не фигура (callout-guard)
REGION_MAX_PAGE_FRACTION = 0.9     # регион на ~всю страницу — рамка/подложка, не фигура
REGION_MAX_CHARS_PER_ELEMENT = 250  # больше текста на элемент — боксированная проза (callout)


class WordLike(Protocol):
    """Структурная совместимость с ``pdf_to_markdown.Word`` — без импорта (нет
    циклической зависимости convert/pdf_graphics.py <-> convert/pdf_to_markdown.py)."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass(frozen=True)
class Element:
    """Векторный/растровый объект страницы (rect/curve/image), участвующий в
    детекции регионов. ``content_hash`` заполнен только для kind="image" —
    используется растровой политикой (§1.4), к геометрии региона отношения не имеет."""

    kind: str  # "rect" | "curve" | "image"
    x0: float
    top: float
    x1: float
    bottom: float
    content_hash: str | None = None


def _bbox(e: Element) -> BBox:
    return (e.x0, e.top, e.x1, e.bottom)


def _area(bbox: BBox) -> float:
    x0, top, x1, bottom = bbox
    return max(x1 - x0, 0.0) * max(bottom - top, 0.0)


def _center_in_bbox(x0: float, top: float, x1: float, bottom: float, bbox: BBox) -> bool:
    cx, cy = (x0 + x1) / 2, (top + bottom) / 2
    bx0, btop, bx1, bbottom = bbox
    return bx0 <= cx <= bx1 and btop <= cy <= bbottom


def _element_center_in_any_bbox(e: Element, bboxes: list[BBox]) -> bool:
    return any(_center_in_bbox(e.x0, e.top, e.x1, e.bottom, b) for b in bboxes)


def _word_center_in_bbox(w: Any, bbox: BBox) -> bool:
    return _center_in_bbox(w.x0, w.top, w.x1, w.bottom, bbox)


def _image_content_hash(img: Any) -> str | None:
    """sha256 сырых байт потока изображения (дедуп декора §1.4). Битый/зашифрованный
    поток -> None (дальше трактуется как заведомо уникальный — не декор).

    pdfplumber отдаёт объект ``pdfminer.pdftypes.PDFStream``; байты — только через
    ``.get_data()`` (проверено эмпирически 2026-07-16: атрибут ``.rawdata`` до
    вызова ``get_data()`` равен None, а не сырым байтам — использовать его
    напрямую было бы тихой порчей дедупа)."""
    try:
        return hashlib.sha256(img["stream"].get_data()).hexdigest()
    except Exception:  # noqa: BLE001 — любой отказ декодирования = «не наш кейс», не крах конвертации
        return None


def collect_elements(page: Any) -> list[Element]:
    """pdfplumber-адаптер: rects+curves+images страницы -> Element. ``lines`` не
    берём — на обоих документах корпуса их 0, а как класс это линейки/подчёркивания
    (уже отсекаются ``filter_elements`` у rects/curves, будь они там)."""
    out: list[Element] = []
    for r in page.rects:
        out.append(Element("rect", r["x0"], r["top"], r["x1"], r["bottom"]))
    for c in page.curves:
        out.append(Element("curve", c["x0"], c["top"], c["x1"], c["bottom"]))
    for img in page.images:
        out.append(
            Element(
                "image", img["x0"], img["top"], img["x1"], img["bottom"],
                content_hash=_image_content_hash(img),
            )
        )
    return out


def dedupe_elements(elements: list[Element]) -> list[Element]:
    """Схлопнуть дубли одного бокса: bbox округляется до ``DEDUPE_ROUND_PT``,
    первый встреченный побеждает (kind тоже участвует в ключе — rect и image с
    одинаковым bbox не должны схлопываться друг в друга)."""
    seen: set[tuple[str, float, float, float, float]] = set()
    out: list[Element] = []
    for e in elements:
        key = (
            e.kind,
            round(e.x0 / DEDUPE_ROUND_PT), round(e.top / DEDUPE_ROUND_PT),
            round(e.x1 / DEDUPE_ROUND_PT), round(e.bottom / DEDUPE_ROUND_PT),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def filter_elements(elements: list[Element], page_area: float) -> list[Element]:
    """Отсечь линейки/подчёркивания (тоньше ``ELEMENT_MIN_SIDE_PT`` по меньшей
    стороне) и фон страницы (крупнее ``ELEMENT_MAX_PAGE_FRACTION`` площади)."""
    out = []
    for e in elements:
        w, h = e.x1 - e.x0, e.bottom - e.top
        if min(w, h) < ELEMENT_MIN_SIDE_PT:
            continue
        if _area(_bbox(e)) > ELEMENT_MAX_PAGE_FRACTION * page_area:
            continue
        out.append(e)
    return out


def _expand(bbox: BBox, gap: float) -> BBox:
    x0, top, x1, bottom = bbox
    return (x0 - gap, top - gap, x1 + gap, bottom + gap)


def _intersects(a: BBox, b: BBox) -> bool:
    ax0, atop, ax1, abottom = a
    bx0, btop, bx1, bbottom = b
    return ax0 < bx1 and bx0 < ax1 and atop < bbottom and btop < abottom


def cluster_elements(elements: list[Element], gap: float = CLUSTER_GAP_PT) -> list[list[Element]]:
    """Связные компоненты по пересечению bbox, расширенных на ``gap`` в каждую
    сторону. Граф на ~200 элементах страницы — O(n^2) пар приемлем (доли мс)."""
    n = len(elements)
    expanded = [_expand(_bbox(e), gap) for e in elements]
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _intersects(expanded[i], expanded[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = [False] * n
    clusters: list[list[Element]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack, component = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        clusters.append([elements[k] for k in component])
    return clusters


def union_bbox(elements: list[Element]) -> BBox:
    return (
        min(e.x0 for e in elements), min(e.top for e in elements),
        max(e.x1 for e in elements), max(e.bottom for e in elements),
    )


def region_words(words: list[Any], bbox: BBox) -> list[Any]:
    """Слова, чей центр попадает внутрь bbox региона (изымаются из прозы вызывающей
    стороной — этот модуль только вычисляет принадлежность, ничего не мутирует)."""
    return [w for w in words if _word_center_in_bbox(w, bbox)]


def region_guards_ok(elements: list[Element], words: list[Any], page_area: float) -> bool:
    """Три guard'а против съедания прозы (чартер convert-graphics §7.1: ложно-
    положительный регион — регресс хуже статус-кво). ВСЕ три обязаны пройти,
    иначе кластер распускается — его слова остаются в прозе (вызывающая сторона
    просто не изымает их)."""
    if len(elements) < REGION_MIN_ELEMENTS:
        return False
    bbox = union_bbox(elements)
    if _area(bbox) > REGION_MAX_PAGE_FRACTION * page_area:
        return False
    if not elements:  # защита от деления на 0 (сюда не дойдёт при guard'е выше, но честно)
        return False
    total_chars = sum(len(w.text) for w in words)
    if total_chars / len(elements) > REGION_MAX_CHARS_PER_ELEMENT:
        return False
    return True
