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
REGION_MAX_CHARS_PER_ELEMENT = 100  # больше текста на элемент — боксированная проза (callout)
# Калибровано аудитом на реальном корпусе 2026-07-16 (spec convert-graphics §4,
# открытый вопрос): стартовое значение 250 пропускало явный кейс-стади-callout
# Сингапура p.37 (17 элементов, 138 симв./элемент — целые абзацы прозы под
# маркером «Figure»). Легитимные фигуры на обоих документах — 2.9..74.6
# симв./элемент (макс. — Сингапур p.10, decision-матрица с длинными подписями);
# погран. кейс из спека (Эстония p.23, 42 rect/3168 симв.) — 75.4, тоже легитимен.
# 100 — с запасом выше обоих легитимных максимумов и с запасом ниже 138.3.


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
    """Guard'ы против съедания прозы (чартер convert-graphics §7.1: ложно-
    положительный регион — регресс хуже статус-кво). ВСЕ обязаны пройти, иначе
    кластер распускается — его слова остаются в прозе (вызывающая сторона
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
    if total_chars == 0:
        # Ни одного слова в bbox региона (типично — мелкая декоративная иконка,
        # нарисованная россыпью curve-сегментов, напр. Сингапур p.26: 4 таких
        # кластера с пустыми Labels). Дёшево распустить: слова региона и так
        # пусты, распускание НИЧЕГО не возвращает в прозу (нечего возвращать) —
        # но маркер с пустым Labels добавлял бы чистый шум (найдено аудитом
        # 2026-07-16, spec §4). Не отменяет opaque-маркер для фигур С текстом.
        return False
    return True


# --- классификация (грид/sequence), region-hash, Region, detect_regions ---

GRID_SNAP_PT = 3.0          # допуск защёлкивания краёв rect'ов в колонки/строки
GRID_MAX_COLS = 6
GRID_MAX_ROWS = 12
GRID_MIN_OCCUPANCY = 0.9    # доля заполненных ячеек грида
SEQ_MIN_BOXES = 3
SEQ_MAX_BOXES = 8
SEQ_AXIS_OVERLAP = 0.5      # мин. доля перекрытия по поперечной оси (от меньшей стороны)
_LINE_TOP_TOLERANCE = 2.5   # тот же допуск, что pdf_to_markdown.LINE_TOP_TOLERANCE


@dataclass
class Region:
    """Классифицированный регион: reconstructed (grid/sequence) несёт
    cells/items, opaque — только слова (Labels маркера рендерятся из них).
    ``id`` — стабильный короткий hash (region_id) для машинной адресации
    (VLM-шов чартера §4.3; вычисляется для ЛЮБОГО региона, не только opaque —
    полезно для аудита/отладки, дёшево)."""

    bbox: BBox
    elements: list[Element]
    words: list[Any]
    kind: str  # "grid" | "sequence" | "opaque"
    id: str
    cells: list[list[str]] | None = None
    items: list[str] | None = None


def _line_texts(words: list[Any]) -> list[str]:
    """Группирует слова региона в строки читательского порядка (top, затем x0).
    Общий примитив для текста ячейки грида / подписи sequence-элемента / Labels
    opaque-маркера — самодостаточен (без импорта pdf_to_markdown.group_into_lines,
    никакой циклической зависимости, design rationale WordLike-Protocol)."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.top, w.x0))
    lines: list[list[Any]] = []
    for w in ordered:
        if lines and abs(w.top - lines[-1][-1].top) <= _LINE_TOP_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [" ".join(x.text for x in sorted(line, key=lambda w: w.x0)) for line in lines]


def _join_reading_order(words: list[Any]) -> str:
    return " ".join(_line_texts(words))


def _cluster_1d(values: list[float], tol: float) -> list[float]:
    """1D-кластеризация с допуском: соседние (после сортировки) значения ближе
    ``tol`` схлопываются в группу; представитель группы — среднее."""
    if not values:
        return []
    vals = sorted(values)
    groups: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _nearest_index(value: float, representatives: list[float]) -> int:
    return min(range(len(representatives)), key=lambda i: abs(representatives[i] - value))


def try_grid(elements: list[Element], words: list[Any]) -> list[list[str]] | None:
    """Грид-предикат (§1.1): rect'ы защёлкиваются в колонки/строки по x0/top
    (допуск ``GRID_SNAP_PT``), 2..GRID_MAX_COLS x 2..GRID_MAX_ROWS, ни один rect
    не делит ячейку с другим, заполненность >= GRID_MIN_OCCUPANCY. Любое
    нарушение -> None — фабрикация «почти грида» запрещена."""
    rects = [e for e in elements if e.kind == "rect"]
    if not rects:
        return None
    cols = _cluster_1d([r.x0 for r in rects], GRID_SNAP_PT)
    rows = _cluster_1d([r.top for r in rects], GRID_SNAP_PT)
    if not (2 <= len(cols) <= GRID_MAX_COLS and 2 <= len(rows) <= GRID_MAX_ROWS):
        return None

    cell_map: dict[tuple[int, int], Element] = {}
    for r in rects:
        key = (_nearest_index(r.top, rows), _nearest_index(r.x0, cols))
        if key in cell_map:
            return None  # два rect в одной ячейке — не чистый грид
        cell_map[key] = r

    if len(rects) < GRID_MIN_OCCUPANCY * len(rows) * len(cols):
        return None

    cells: list[list[str]] = []
    for row_idx in range(len(rows)):
        row_cells: list[str] = []
        for col_idx in range(len(cols)):
            rect = cell_map.get((row_idx, col_idx))
            row_cells.append("" if rect is None else _join_reading_order(region_words(words, _bbox(rect))))
        cells.append(row_cells)
    return cells


def _vertical_overlap(a: Element, b: Element) -> float:
    return max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))


def _horizontal_overlap(a: Element, b: Element) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _bbox_overlap(a: Element, b: Element) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.top < b.bottom and b.top < a.bottom


def _axis_chain_ok(rects: list[Element], overlap: Any, size: Any) -> bool:
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            min_size = min(size(rects[i]), size(rects[j]))
            if min_size <= 0 or overlap(rects[i], rects[j]) < SEQ_AXIS_OVERLAP * min_size:
                return False
    return True


def try_sequence(elements: list[Element], words: list[Any]) -> list[str] | None:
    """Sequence-предикат (§1.2): SEQ_MIN_BOXES..SEQ_MAX_BOXES непересекающихся
    rect'ов на одной оси (все пары перекрываются по поперечной оси на
    >= SEQ_AXIS_OVERLAP от меньшей стороны) -> упорядоченные подписи. Направление
    потока/стрелки НЕ угадывается (чартер §4.2) — только один линейный порядок
    (по x0 для горизонтальной цепи, по top для вертикальной)."""
    rects = [e for e in elements if e.kind == "rect"]
    if not (SEQ_MIN_BOXES <= len(rects) <= SEQ_MAX_BOXES):
        return None
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _bbox_overlap(rects[i], rects[j]):
                return None

    if _axis_chain_ok(rects, _vertical_overlap, lambda r: r.bottom - r.top):
        ordered = sorted(rects, key=lambda r: r.x0)
    elif _axis_chain_ok(rects, _horizontal_overlap, lambda r: r.x1 - r.x0):
        ordered = sorted(rects, key=lambda r: r.top)
    else:
        return None

    return [_join_reading_order(region_words(words, _bbox(r))) for r in ordered]


def region_hash(page: int, bbox: BBox, texts: list[str]) -> str:
    """sha256(page|округлённый_bbox|отсортированные_тексты) — контракт VLM-шва
    (чартер §4.3): детерминирован, НЕ зависит от рендера пикселей (рендер
    нестабилен между версиями pdfium). Округление до 1pt — стабильность против
    float-шума; сортировка текстов — независимость от порядка извлечения."""
    payload = (
        f"{page}|{round(bbox[0])},{round(bbox[1])},{round(bbox[2])},{round(bbox[3])}|"
        + "\x1f".join(sorted(t.strip() for t in texts if t.strip()))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def region_id(page: int, bbox: BBox, texts: list[str]) -> str:
    return region_hash(page, bbox, texts)[:12]


def detect_regions(
    page: int,
    elements: list[Element],
    words: list[Any],
    page_width: float,
    page_height: float,
    table_bboxes: list[BBox],
) -> tuple[list[Region], list[Any]]:
    """Главный конвейер §1: intake (искл. images — только через раздельную
    растровую политику §1.4/classify_images, НЕ кластеризацию; искл. уже
    забранные таблицами) -> dedupe -> filter -> cluster -> guards ->
    классификация (грид/sequence/opaque). ``page`` — 1-based номер страницы
    (входит в region_hash). Регионы, не прошедшие guard'ы, распускаются — их
    слова просто не изымаются из ``words``.

    Images НЕ участвуют в кластеризации (калибровка по реальному аудиту
    2026-07-16, spec convert-graphics §4): в некоторых PDF текстовые строки
    (напр. TOC с dot-leader'ами) растеризованы как МНОЖЕСТВО мелких image-
    "срезов" одной высоты — по геометрии неотличимых от кластера иконок
    реального флоучарта, но не несущих собственной графической ценности
    (реальный текст рядом — в word-слое, растеризация лишь визуальная).
    Настоящие флоучарты (напр. Сингапур p.6: 27 rect + 52 curve) остаются
    детектируемыми и без images в кластере — векторная форма их и так несёт."""
    page_area = page_width * page_height
    candidates = [e for e in elements if e.kind != "image"]
    candidates = [e for e in candidates if not _element_center_in_any_bbox(e, table_bboxes)]
    candidates = dedupe_elements(candidates)
    candidates = filter_elements(candidates, page_area)

    regions: list[Region] = []
    consumed: set[int] = set()
    for cluster in cluster_elements(candidates):
        bbox = union_bbox(cluster)
        r_words = region_words(words, bbox)
        if not region_guards_ok(cluster, r_words, page_area):
            continue
        rid = region_id(page, bbox, _line_texts(r_words))
        cells = try_grid(cluster, r_words)
        if cells is not None:
            regions.append(Region(bbox, cluster, r_words, "grid", rid, cells=cells))
        else:
            items = try_sequence(cluster, r_words)
            if items is not None:
                regions.append(Region(bbox, cluster, r_words, "sequence", rid, items=items))
            else:
                regions.append(Region(bbox, cluster, r_words, "opaque", rid))
        consumed.update(id(w) for w in r_words)

    remaining = [w for w in words if id(w) not in consumed]
    return regions, remaining


def render_region_block(region: Region, page: int) -> str:
    """Точный маркер-грамматика §2. Грид/sequence несут общий заголовок
    «Reconstructed infographic»; opaque — честный «structure not reconstructed»
    с сохранёнными подписями (порядок чтения не гарантирован)."""
    if region.kind == "grid":
        assert region.cells is not None
        return _render_grid(region.cells, page, region.id)
    if region.kind == "sequence":
        assert region.items is not None
        return _render_sequence(region.items, page, region.id)
    return _render_opaque(region.words, page, region.id)


def _render_grid(cells: list[list[str]], page: int, rid: str) -> str:
    if not cells:
        return f"> [Reconstructed infographic, p. {page}, region {rid}]"
    header, *body = cells
    lines = [
        f"> [Reconstructed infographic, p. {page}, region {rid}]",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_sequence(items: list[str], page: int, rid: str) -> str:
    lines = [f"> [Reconstructed infographic, p. {page}, region {rid}]", ""]
    lines.extend(f"{i}. {text}" for i, text in enumerate(items, start=1))
    return "\n".join(lines)


def _render_opaque(words: list[Any], page: int, rid: str) -> str:
    labels = "; ".join(_line_texts(words))
    return (
        f"> [Figure, p. {page}, region {rid} — structure not reconstructed]\n"
        f"> Labels (reading order not guaranteed): {labels}"
    )


# --- растровая политика (§1.4, чартер §4.1): детектировать, НЕ распознавать ---

RASTER_DECOR_MIN_REPEATS = 3     # изображение чаще — декор/логотип (частота ПО ДОКУМЕНТУ)
RASTER_MIN_PAGE_FRACTION = 0.08  # мельче — иконка, молча


def document_hash_counts(all_images: list[Element]) -> dict[str, int]:
    """Частота ``content_hash`` по ВСЕМУ документу, не странице — декор/логотип
    типично повторяется по разу НА КАЖДОЙ странице (header/footer), а не
    многократно на одной; частота меньше документа не поймала бы это (§1.4)."""
    counts: dict[str, int] = {}
    for img in all_images:
        if img.content_hash is not None:
            counts[img.content_hash] = counts.get(img.content_hash, 0) + 1
    return counts


def classify_images(
    page_images: list[Element],
    consumed_bboxes: list[BBox],
    hash_counts: dict[str, int],
    page_area: float,
) -> list[Element]:
    """Изображения ОДНОЙ страницы, заслуживающие отдельного маркера «raster
    content not analyzed»: не поглощены уже детектированным регионом
    (``consumed_bboxes`` — bbox'ы Region из detect_regions той же страницы), не
    повторяются >= RASTER_DECOR_MIN_REPEATS раз по документу (декор — молча),
    площадь на странице >= RASTER_MIN_PAGE_FRACTION (иначе — иконка, молча).
    ``hash_counts`` — предпосчитан ``document_hash_counts`` по ВСЕМУ документу."""
    loose = [img for img in page_images if not _element_center_in_any_bbox(img, consumed_bboxes)]
    marked: list[Element] = []
    for img in loose:
        if img.content_hash is not None and hash_counts.get(img.content_hash, 0) >= RASTER_DECOR_MIN_REPEATS:
            continue
        if _area(_bbox(img)) < RASTER_MIN_PAGE_FRACTION * page_area:
            continue
        marked.append(img)
    return marked


def image_id(image: Element, page: int) -> str:
    """12-hex id маркера растра (spec convert-cloud-tier §4): из ``content_hash``
    (уже посчитан для дедупа §1.4, дешёво) — та же величина, по которой
    ``figures_vlm`` §5 матчит маркер с изображением при пере-детекции (stream-
    хэш стабилен между прогонами, рендер пикселей — нет). ``content_hash=None``
    (сбой декодирования потока, редкость — см. ``_image_content_hash``) не
    должен оставлять маркер без адреса: детерминированный fallback по
    (page, bbox) — тот же принцип, что ``region_hash`` для текстовых регионов."""
    if image.content_hash is not None:
        return image.content_hash[:12]
    payload = f"{page}|{round(image.x0)},{round(image.top)},{round(image.x1)},{round(image.bottom)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def render_raster_marker(page: int, image: Element) -> str:
    return f"> [Image, p. {page}, image {image_id(image, page)} — raster content not analyzed]"
