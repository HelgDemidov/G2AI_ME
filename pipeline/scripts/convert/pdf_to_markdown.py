#!/usr/bin/env python3
"""PDF -> Markdown через pdfplumber: восстановление порядка чтения в многоколоночной
вёрстке (проекционный gap-анализ), иерархия заголовков по кластерам font-size (не по
regex на нумерации — работает и для документов без "1.1.1"-нумерации), колонтитулы/
номера страниц отсеиваются по частоте ЦЕЛЫХ строк (не отдельных слов - иначе частые
короткие слова колонтитула стираются и из основного текста), крошечные (обычно <65%
от размера тела) надстрочные номера сносок вычищаются как шум перед анализом разрывов.
Таблицы - через pdfplumber.extract_tables(). Инфографика (SWOT-матрицы, боксовые
последовательности, флоучарты) — векторная детекция pdf_graphics.py (spec
convert-graphics): реконструкция в семантический Markdown там, где геометрия
однозначна (грид/последовательность), иначе честный маркер с сохранёнными подписями —
вместо прежней word-gap-эвристики (снята §3 п.4: ложно срабатывала на оглавлении/SWOT,
см. CLAUDE.md). Все блоки (таблицы/регионы/растр-маркеры) вставляются ПОЗИЦИОННО в
поток колонки по вертикальной позиции — таблицы больше не приклеиваются в конец страницы.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.table import Table

from convert import pdf_graphics

MIN_GAP_PT = 14.0            # мин. ширина "пустого" промежутка, чтобы считать его границей колонок
COLUMN_ZONE = (0.30, 0.70)   # разрыв должен начинаться в этой доле ширины страницы (не путать с полями)
MIN_GAP_HEIGHT_FRAC = 0.45   # разрыв должен покрывать не менее этой доли высоты контентной зоны
HEADER_FOOTER_BAND_FRAC = 0.09
BOILERPLATE_MIN_PAGE_FRACTION = 0.25
HEADING_MIN_RATIO = 1.15      # во сколько раз крупнее тела, чтобы считаться заголовком
TINY_MARKER_RATIO = 0.65      # порог для надстрочных номеров сносок (сильно мельче и текста сносок, и тела)
LINE_TOP_TOLERANCE = 2.5
PARA_GAP_RATIO = 1.6
DOT_LEADER_RE = re.compile(r"\.{3,}")
BOLD_HEADING_MAX_CHARS = 80   # A1: bold-фолбэк заголовков — макс. длина строки-кандидата
_BOLD_RE = re.compile(r"bold|black|heavy", re.I)  # покрывает Arial-BoldMT, subset-префиксы ABCDEF+...-Bold
_LIST_MARKER_RE = re.compile(r"^[-–•*]\s")


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    fontname: str = ""


@dataclass
class DocStats:
    body_size: float
    heading_sizes: list[float]
    tiny_marker_max: float
    boilerplate_norms: set[str]


def normalize_line(text: str) -> str:
    s = text.strip()
    if "|" in s:
        s = s.split("|", 1)[0].strip()
    return re.sub(r"\d+", "#", s)


def _word_fontname(word_chars: list[dict[str, Any]]) -> str:
    """Fontname слова, ТОЛЬКО если оно целиком (все символы) bold — иначе "".

    ``page.extract_words(extra_attrs=["fontname"])`` группирует символы в слова
    через ``itertools.groupby`` по КАЖДОМУ extra_attr — смена fontname внутри
    слова БЕЗ пробела (bold-термин впритык к пунктуации: "Controls:", типичный
    паттерн списков термин-двоеточие) резала бы слово надвое и вставляла
    ложный пробел при ``" ".join(w.text for w in line)`` ВЕЗДЕ в документе, не
    только в заголовках (регресс, найденный живым аудитом sg/IMDA). Поэтому
    границы слов определяются БЕЗ fontname (``extra_attrs=["size"]``, как до
    A1) — bold смотрим отдельно, по сырым символам через ``return_chars=True``.
    """
    if not word_chars:
        return ""
    fontnames = [c.get("fontname", "") or "" for c in word_chars]
    if all(_BOLD_RE.search(f) for f in fontnames):
        return fontnames[0]
    return ""


def load_words(page: Page) -> list[Word]:
    raw = page.extract_words(extra_attrs=["size"], return_chars=True)
    out = []
    for w in raw:
        text = DOT_LEADER_RE.sub("", w["text"]).strip()  # артефакт оглавления, в т.ч. приклеенный к слову
        if not text:
            continue
        out.append(Word(
            text, w["x0"], w["x1"], w["top"], w["bottom"], round(w["size"], 1), _word_fontname(w["chars"]),
        ))
    return out


MIN_TABLE_NONEMPTY_CELLS = 5  # отсекает случайные 2x2-обломки диаграмм, но не настоящие
                              # таблицы с объединёнными ячейками (там тоже низкая ДОЛЯ
                              # заполненности - решает именно абсолютное число, не доля)


def get_real_tables(page: Page) -> list[Table]:
    """pdfplumber нередко дробит одну настоящую многоколоночную таблицу (или
    фрагменты диаграммы с частичными разделительными линиями) на кучу шумных
    "таблиц" с почти пустыми ячейками - это не таблицы, а обычный текст/подписи;
    оставляем только объекты с >=2 колонками, >=1 строкой данных и достаточной
    заполненностью содержимым."""
    real = []
    for t in page.find_tables():
        rows = t.extract()
        if not rows or len(rows[0]) < 2 or len(rows) < 2:
            continue
        nonempty = sum(1 for row in rows for cell in row if cell and cell.strip())
        if nonempty < MIN_TABLE_NONEMPTY_CELLS:
            continue
        real.append(t)
    return real


def word_in_any_bbox(word: Word, bboxes: list[tuple[float, float, float, float]]) -> bool:
    cx, cy = (word.x0 + word.x1) / 2, (word.top + word.bottom) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for (x0, top, x1, bottom) in bboxes)


def group_into_lines(words: list[Word]) -> list[list[Word]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.top, w.x0))
    lines: list[list[Word]] = []
    current = [ordered[0]]
    for w in ordered[1:]:
        if abs(w.top - current[-1].top) <= LINE_TOP_TOLERANCE:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x.x0))
            current = [w]
    lines.append(sorted(current, key=lambda x: x.x0))
    return lines


def compute_doc_stats(pages: list[tuple[list[Word], float]]) -> DocStats:
    """``pages`` — список (слова_страницы, высота_страницы): полосы колонтитулов
    считаются по высоте КАЖДОЙ страницы, а не единой глобальной — документы со
    смешанной ориентацией (портретное тело + альбомные приложения-таблицы,
    типично для гос-стратегий) иначе получали бы неверные полосы на альбомных
    страницах. ``body_size``/``heading_sizes``/``boilerplate_norms`` остаются
    документ-глобальными — шрифты и колонтитулы общие для всего документа.
    """
    size_char_counts: Counter[float] = Counter()
    boilerplate_line_counts: Counter[str] = Counter()
    n_pages = len(pages)

    for words, page_height in pages:
        top_band = page_height * HEADER_FOOTER_BAND_FRAC
        bottom_band = page_height * (1 - HEADER_FOOTER_BAND_FRAC)
        for w in words:
            size_char_counts[w.size] += len(w.text)
        lines = group_into_lines(words)
        seen: set[str] = set()
        for line in lines:
            top = min(w.top for w in line)
            bottom = max(w.bottom for w in line)
            if top <= top_band or bottom >= bottom_band:
                norm = normalize_line(" ".join(w.text for w in line))
                if norm and norm not in seen:
                    boilerplate_line_counts[norm] += 1
                    seen.add(norm)

    boilerplate_norms = {
        norm for norm, cnt in boilerplate_line_counts.items()
        if cnt / n_pages >= BOILERPLATE_MIN_PAGE_FRACTION
    }
    body_size = size_char_counts.most_common(1)[0][0] if size_char_counts else 11.0
    heading_sizes = sorted(
        {s for s in size_char_counts if s >= body_size * HEADING_MIN_RATIO},
        reverse=True,
    )
    tiny_marker_max = body_size * TINY_MARKER_RATIO
    return DocStats(body_size, heading_sizes, tiny_marker_max, boilerplate_norms)


def strip_boilerplate_and_page_numbers(words: list[Word], page_height: float, stats: DocStats) -> list[Word]:
    top_band = page_height * HEADER_FOOTER_BAND_FRAC
    bottom_band = page_height * (1 - HEADER_FOOTER_BAND_FRAC)
    lines = group_into_lines(words)
    kept: list[Word] = []
    for line in lines:
        top = min(w.top for w in line)
        bottom = max(w.bottom for w in line)
        text = " ".join(w.text for w in line)
        in_band = top <= top_band or bottom >= bottom_band
        if in_band:
            if text.strip().isdigit():
                continue  # голый номер страницы
            if normalize_line(text) in stats.boilerplate_norms:
                continue  # повторяющийся колонтитул
        kept.extend(line)
    return kept


def detect_columns(words: list[Word], page_width: float) -> list[tuple[float, float]]:
    if not words:
        return [(0, page_width)]
    content_top = min(w.top for w in words)
    content_bottom = max(w.bottom for w in words)
    content_height = max(content_bottom - content_top, 1.0)

    resolution = 2.0
    n_bins = int(page_width / resolution) + 1
    covered = [False] * n_bins

    def mark(x0: float, x1: float) -> None:
        lo = max(0, int(x0 / resolution))
        hi = min(n_bins - 1, int(x1 / resolution))
        for i in range(lo, hi + 1):
            covered[i] = True

    for w in words:
        mark(w.x0, w.x1)

    gaps: list[tuple[float, float]] = []
    i = 0
    while i < n_bins:
        if not covered[i]:
            j = i
            while j < n_bins and not covered[j]:
                j += 1
            gap_x0, gap_x1 = i * resolution, j * resolution
            if gap_x1 - gap_x0 >= MIN_GAP_PT:
                zone_lo, zone_hi = COLUMN_ZONE[0] * page_width, COLUMN_ZONE[1] * page_width
                if zone_lo <= (gap_x0 + gap_x1) / 2 <= zone_hi:
                    ys_left = [w.top for w in words if w.x1 <= gap_x0]
                    ys_right = [w.top for w in words if w.x0 >= gap_x1]
                    if ys_left and ys_right:
                        span = max(max(ys_left), max(ys_right)) - min(min(ys_left), min(ys_right))
                        if span >= content_height * MIN_GAP_HEIGHT_FRAC:
                            gaps.append((gap_x0, gap_x1))
            i = j
        else:
            i += 1

    if not gaps:
        return [(0, page_width)]
    gap = max(gaps, key=lambda g: g[1] - g[0])
    mid = (gap[0] + gap[1]) / 2
    return [(0, mid), (mid, page_width)]


def heading_level(size: float, stats: DocStats) -> int | None:
    for idx, hsize in enumerate(stats.heading_sizes):
        if abs(size - hsize) < 0.3:
            return idx + 1
    return None


def _is_bold(fontname: str) -> bool:
    return bool(_BOLD_RE.search(fontname))


def _median_line_gap(lines: list[list[Word]]) -> float:
    gaps = [lines[i][0].top - lines[i - 1][0].bottom for i in range(1, len(lines))]
    return sorted(gaps)[len(gaps) // 2] if gaps else 0.0


def _bold_run(lines: list[list[Word]], start: int, med_gap: float) -> list[list[Word]] | None:
    """Максимальный прогон СМЕЖНЫХ строк (тот же визуальный абзац — разрыв не
    шире ``med_gap * PARA_GAP_RATIO``), где КАЖДАЯ строка целиком bold.

    Ловит регресс, найденный живым аудитом (sg, IMDA Agentic AI): bold как
    стилистическое ВЫДЕЛЕНИЕ (lead-in первых строк абзаца, перенос заголовка
    на вторую строку колонки) — не то же самое, что bold как признак
    ОТДЕЛЬНОГО заголовка. Прогон обрывается на первой НЕ-bold строке того же
    абзаца; различаем ДВА случая такого обрыва по первому символу этой
    НЕ-bold строки: строчная буква -> грамматическое продолжение ТОГО ЖЕ
    предложения (лид-ин длинного абзаца, sg: "These greater capabilities...
    new" + "risks.") -> прогон дисквалифицируется ЦЕЛИКОМ; что угодно другое
    (буллет/цифра/заглавная — новый пункт списка/таблицы, ee: "Tugevused
    Nõrkused" + "• Digitaalselt...") -> прогон изолирован как есть, не
    дисквалифицируется (короткий bold-ярлык физически «приклеен» к телу без
    вертикального разрыва, но грамматически НЕ продолжает его).
    """
    if not all(_is_bold(w.fontname) for w in lines[start]):
        return None
    run = [lines[start]]
    i = start + 1
    while i < len(lines):
        gap = lines[i][0].top - lines[i - 1][0].bottom
        if med_gap > 0 and gap > med_gap * PARA_GAP_RATIO:
            break  # новый абзац — дальше уже не наш прогон
        if all(_is_bold(w.fontname) for w in lines[i]):
            run.append(lines[i])
            i += 1
            continue
        continuation = " ".join(w.text for w in lines[i]).lstrip()
        if continuation[:1].islower():
            return None  # bold обрывается серединой предложения -> не заголовок целиком
        break  # bold обрывается новым пунктом/абзацем -> прогон изолирован как есть
    return run


def _bold_heading_level(run: list[list[Word]], stats: DocStats) -> int | None:
    """A1-фолбэк: срабатывает ТОЛЬКО когда размерная кластеризация молчит
    (``heading_level`` вернул None) — документ с одинаковым кеглем заголовка и
    тела (CLAUDE.md, известное ограничение) иначе пропускался бы целиком.
    Уровень — всегда НИЖЕ всех размерных уровней документа (bold — более слабый
    сигнал, чем явно больший кегль); ``run`` уже подтверждён целиком-bold и
    визуально изолированным (``_bold_run``) — здесь только guard'ы кегля/длины/
    пунктуации по объединённому тексту; при любом непрохождении прогон
    остаётся прозой (чартер §2.5).
    """
    dominant_size = Counter(w.size for line in run for w in line).most_common(1)[0][0]
    if dominant_size < stats.body_size:
        return None
    text = " ".join(" ".join(w.text for w in line) for line in run)
    if len(text) > BOLD_HEADING_MAX_CHARS:
        return None
    if text.endswith((".", ";", ":")) or _LIST_MARKER_RE.match(text):
        return None
    return min(len(stats.heading_sizes) + 1, 6)


def render_lines_as_paragraphs(lines: list[list[Word]]) -> str:
    if not lines:
        return ""
    med_gap = _median_line_gap(lines)
    out: list[str] = []
    buf: list[str] = [" ".join(w.text for w in lines[0])]
    for i in range(1, len(lines)):
        gap = lines[i][0].top - lines[i - 1][0].bottom
        text = " ".join(w.text for w in lines[i])
        if med_gap > 0 and gap > med_gap * PARA_GAP_RATIO:
            out.append(" ".join(buf))
            buf = [text]
        else:
            buf.append(text)
    out.append(" ".join(buf))
    return "\n\n".join(out)


def _render_column_with_blocks(
    lines: list[list[Word]], blocks: list[tuple[pdf_graphics.BBox, str]], stats: DocStats
) -> str:
    """Мёржит строки прозы колонки с уже отрендеренными блоками (таблицы/
    регионы/растр-маркеры) той же колонки по вертикальной позиции (``top``) —
    позиционная вставка (spec convert-graphics §3 п.3), заменяет прежнее
    приклеивание блоков в конец. Заголовки/абзацы внутри непрерывных прозных
    прогонов между блоками — прежняя логика render_lines_with_diagram_detection
    минус диаграммная ветка (снята §3 п.4: word-gap-эвристика ложно
    срабатывала на TOC/SWOT, её работу забрала векторная детекция pdf_graphics)."""
    stream: list[tuple[float, str, Any]] = [
        (min(w.top for w in line), "line", line) for line in lines
    ] + [(bbox[1], "block", text) for bbox, text in blocks]
    stream.sort(key=lambda item: item[0])

    rendered: list[str] = []
    prose_buf: list[list[Word]] = []

    def flush_prose() -> None:
        nonlocal prose_buf
        if not prose_buf:
            return
        segments: list[str] = []
        para_buf: list[list[Word]] = []
        med_gap = _median_line_gap(prose_buf)  # локальный масштаб «тот же абзац» для bold-прогонов
        i = 0
        n = len(prose_buf)
        while i < n:
            line = prose_buf[i]
            dominant_size = Counter(w.size for w in line).most_common(1)[0][0]
            level = heading_level(dominant_size, stats)
            if level is not None:
                if para_buf:
                    segments.append(render_lines_as_paragraphs(para_buf))
                    para_buf = []
                segments.append(f"{'#' * min(level, 6)} {' '.join(w.text for w in line)}")
                i += 1
                continue

            run = _bold_run(prose_buf, i, med_gap)
            if run is not None:
                level = _bold_heading_level(run, stats)
                if level is not None:
                    if para_buf:
                        segments.append(render_lines_as_paragraphs(para_buf))
                        para_buf = []
                    text = " ".join(" ".join(w.text for w in ln) for ln in run)
                    segments.append(f"{'#' * min(level, 6)} {text}")
                    i += len(run)
                    continue
                # прогон целиком-bold и изолирован, но не прошёл guard (длина/
                # пунктуация) — весь прогон СРАЗУ уходит в прозу; иначе «хвост»
                # переоценивался бы САМ ПО СЕБЕ со следующей итерации и мог бы
                # ложно спромоутиться в одиночный заголовок (регресс, найденный
                # живым аудитом sg: «...third-party» + «ecosystem boundaries»
                # длиннее 80 симв. вместе, но «ecosystem boundaries» одна короче).
                para_buf.extend(run)
                i += len(run)
                continue

            para_buf.append(line)
            i += 1
        if para_buf:
            segments.append(render_lines_as_paragraphs(para_buf))
        rendered.append("\n\n".join(s for s in segments if s.strip()))
        prose_buf = []

    for _, kind, payload in stream:
        if kind == "line":
            prose_buf.append(payload)
        else:
            flush_prose()
            rendered.append(payload)
    flush_prose()
    return "\n\n".join(r for r in rendered if r.strip())


def render_page(
    words: list[Word],
    page_width: float,
    stats: DocStats,
    blocks: list[tuple[pdf_graphics.BBox, str]],
) -> str:
    """``words`` — уже очищенные и лишённые слов детектированных регионов
    (вызывающая сторона: см. ``convert``). ``blocks`` — отрендеренный текст
    таблиц/регионов/растр-маркеров этой страницы с их bbox, для позиционной
    вставки; назначаются колонке по центру-x bbox."""
    columns = detect_columns(words, page_width)
    column_texts: list[str] = []
    for x0, x1 in columns:
        col_words = [w for w in words if x0 <= (w.x0 + w.x1) / 2 < x1]
        lines = group_into_lines(col_words)
        col_blocks = [(bbox, text) for bbox, text in blocks if x0 <= (bbox[0] + bbox[2]) / 2 < x1]
        column_texts.append(_render_column_with_blocks(lines, col_blocks, stats))

    return "\n\n".join(t for t in column_texts if t.strip())


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    """A3: колонки, пустые во ВСЕХ строках (включая шапку), удаляются — обломки
    объединённых ячеек pdfplumber часто оставляют такие колонки-призраки.
    Вырожденный случай (все колонки пусты) уже отсечён guard'ом непустоты в
    ``render_tables`` (``any(any(cell for cell in row) for row in raw_rows)``)."""
    if not rows:
        return rows
    n_cols = len(rows[0])
    keep = [any(row[i].strip() for row in rows) for i in range(n_cols)]
    if all(keep):
        return rows
    return [[cell for cell, k in zip(row, keep, strict=True) if k] for row in rows]


def render_tables(real_tables: list[Table]) -> list[tuple[pdf_graphics.BBox, str]]:
    """(bbox, markdown) на таблицу — bbox нужен для позиционной вставки в
    поток колонки (render_page), не только для рендера."""
    out: list[tuple[pdf_graphics.BBox, str]] = []
    for table in real_tables:
        raw_rows = table.extract()
        if not any(any(cell for cell in row) for row in raw_rows):
            continue
        clean_rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in raw_rows]
        clean_rows = _drop_empty_columns(clean_rows)
        header, *body = clean_rows
        md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in body:
            md.append("| " + " | ".join(row) + " |")
        out.append((table.bbox, "\n".join(md)))
    return out


_TABLE_LINE_RE = re.compile(r"^\|.*\|$")


def _is_table_block(block: str) -> bool:
    lines = [line for line in block.split("\n") if line.strip()]
    return bool(lines) and all(_TABLE_LINE_RE.match(line) for line in lines)


def merge_split_tables(md: str) -> str:
    """A2: сливает многостраничные таблицы, раздробленные на соседние блоки с
    повторяющейся шапкой (наблюдённый паттерн Эстонии) — markdown-пост-проход
    поверх УЖЕ готового текста (страницы соединены ``"\\n\\n"``, колонтитулы уже
    сняты), не bbox-уровень: правка изолирована от самой хрупкой части
    конвертера — позиционной вставки. Продолжения БЕЗ повторной шапки
    сознательно НЕ сливаются (precision-first: нет надёжного предиката отличить
    продолжение от соседней независимой таблицы, повтор шапки — наблюдённый
    паттерн, не гипотеза)."""
    blocks = md.split("\n\n")
    merged: list[str] = []
    for block in blocks:
        if merged and _is_table_block(block) and _is_table_block(merged[-1]):
            prev_lines = merged[-1].split("\n")
            cur_lines = block.split("\n")
            if len(prev_lines) >= 2 and len(cur_lines) >= 2:
                prev_header, cur_header = prev_lines[0].strip(), cur_lines[0].strip()
                same_cols = prev_header.count("|") == cur_header.count("|")
                if same_cols and prev_header == cur_header:
                    merged[-1] = "\n".join([*prev_lines, *cur_lines[2:]])  # без шапки+разделителя 2-го блока
                    continue
        merged.append(block)
    return "\n\n".join(merged)


def convert(pdf_path: str, out_path: str) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            raise RuntimeError(f"{pdf_path}: PDF без страниц")
        pages_words = [load_words(p) for p in pdf.pages]
        stats = compute_doc_stats(list(zip(pages_words, (p.height for p in pdf.pages))))

        print(f"body_size={stats.body_size}, heading_sizes={stats.heading_sizes}, "
              f"tiny_marker_max={stats.tiny_marker_max:.1f}", file=sys.stderr)
        print(f"колонтитулов/повторяющихся строк обнаружено: {len(stats.boilerplate_norms)}", file=sys.stderr)

        # graphics-pass (spec convert-graphics §1/§1.4): элементы + document-wide
        # частота content_hash растра собираются ДО постраничного рендера —
        # декор/логотип определяется частотой ПО ВСЕМУ документу, не странице.
        pages_elements = [pdf_graphics.collect_elements(p) for p in pdf.pages]
        all_images = [e for elements in pages_elements for e in elements if e.kind == "image"]
        hash_counts = pdf_graphics.document_hash_counts(all_images)

        n_grid = n_sequence = n_opaque = n_raster = 0
        out_parts: list[str] = []
        for page_num, (page, words, elements) in enumerate(
            zip(pdf.pages, pages_words, pages_elements), start=1
        ):
            real_tables = get_real_tables(page)
            table_bboxes = [t.bbox for t in real_tables]

            clean_words = strip_boilerplate_and_page_numbers(words, page.height, stats)
            clean_words = [w for w in clean_words if not word_in_any_bbox(w, table_bboxes)]
            main_words = [w for w in clean_words if w.size > stats.tiny_marker_max]

            # регионы: слова изъяты из прозы ДО detect_columns (заодно чинит
            # ложную 2-колоночность от широкой фигуры в центре, §3 п.2).
            regions, remaining_words = pdf_graphics.detect_regions(
                page_num, elements, main_words, page.width, page.height, table_bboxes
            )
            for r in regions:
                if r.kind == "grid":
                    n_grid += 1
                elif r.kind == "sequence":
                    n_sequence += 1
                else:
                    n_opaque += 1

            loose_images = [e for e in elements if e.kind == "image"]
            raster_targets = pdf_graphics.classify_images(
                loose_images, [r.bbox for r in regions], hash_counts, page.width * page.height
            )
            n_raster += len(raster_targets)

            blocks: list[tuple[pdf_graphics.BBox, str]] = [
                *render_tables(real_tables),
                *((r.bbox, pdf_graphics.render_region_block(r, page_num)) for r in regions),
                *(
                    ((img.x0, img.top, img.x1, img.bottom), pdf_graphics.render_raster_marker(page_num))
                    for img in raster_targets
                ),
            ]
            # ширина/высота — СВОЕЙ страницы (не первой): корректно для смешанной
            # ориентации (портретное тело + альбомные приложения-таблицы).
            out_parts.append(render_page(remaining_words, page.width, stats, blocks))

        print(
            f"инфографика: {n_grid} грид -> таблица, {n_sequence} sequence -> список, "
            f"{n_opaque} нереконструировано (маркер); растр: {n_raster} маркеров",
            file=sys.stderr,
        )

        full_text = "\n\n".join(p for p in out_parts if p.strip())
        full_text = merge_split_tables(full_text)  # A2: склейка многостраничных таблиц
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
