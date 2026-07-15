#!/usr/bin/env python3
"""PDF -> Markdown через pdfplumber: восстановление порядка чтения в многоколоночной
вёрстке (проекционный gap-анализ), иерархия заголовков по кластерам font-size (не по
regex на нумерации — работает и для документов без "1.1.1"-нумерации), колонтитулы/
номера страниц отсеиваются по частоте ЦЕЛЫХ строк (не отдельных слов - иначе частые
короткие слова колонтитула стираются и из основного текста), крошечные (обычно <65%
от размера тела) надстрочные номера сносок вычищаются как шум перед анализом разрывов
(иначе создают ложный "признак диаграммы"), реальные диаграммы/инфографика определяются
по разбегу локальных горизонтальных промежутков между словами и помечаются как "требует
ручной проверки" вместо тихой порчи текста. Таблицы - через pdfplumber.extract_tables().
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.table import Table

MIN_GAP_PT = 14.0            # мин. ширина "пустого" промежутка, чтобы считать его границей колонок
COLUMN_ZONE = (0.30, 0.70)   # разрыв должен начинаться в этой доле ширины страницы (не путать с полями)
MIN_GAP_HEIGHT_FRAC = 0.45   # разрыв должен покрывать не менее этой доли высоты контентной зоны
HEADER_FOOTER_BAND_FRAC = 0.09
BOILERPLATE_MIN_PAGE_FRACTION = 0.25
HEADING_MIN_RATIO = 1.15      # во сколько раз крупнее тела, чтобы считаться заголовком
TINY_MARKER_RATIO = 0.65      # порог для надстрочных номеров сносок (сильно мельче и текста сносок, и тела)
LINE_TOP_TOLERANCE = 2.5
PARA_GAP_RATIO = 1.6
WORD_GAP_THRESHOLD = 25.0     # разрыв между соседними словами в "строке", похожий на диаграмму/схему
DIAGRAM_MIN_RUN = 3           # столько подряд "подозрительных" строк, чтобы считать блок диаграммой
DOT_LEADER_RE = re.compile(r"\.{3,}")


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float


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


def load_words(page: Page) -> list[Word]:
    raw = page.extract_words(extra_attrs=["size"])
    out = []
    for w in raw:
        text = DOT_LEADER_RE.sub("", w["text"]).strip()  # артефакт оглавления, в т.ч. приклеенный к слову
        if not text:
            continue
        out.append(Word(text, w["x0"], w["x1"], w["top"], w["bottom"], round(w["size"], 1)))
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


def compute_doc_stats(pages_words: list[list[Word]], page_height: float) -> DocStats:
    size_char_counts: Counter[float] = Counter()
    boilerplate_line_counts: Counter[str] = Counter()
    n_pages = len(pages_words)
    top_band = page_height * HEADER_FOOTER_BAND_FRAC
    bottom_band = page_height * (1 - HEADER_FOOTER_BAND_FRAC)

    for words in pages_words:
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


def line_is_diagram_like(line: list[Word]) -> bool:
    if len(line) <= 1:
        return True  # одинокое слово/номер - типично для узлов схемы
    gaps = [line[i + 1].x0 - line[i].x1 for i in range(len(line) - 1)]
    return max(gaps) > WORD_GAP_THRESHOLD


def render_lines_as_paragraphs(lines: list[list[Word]]) -> str:
    if not lines:
        return ""
    gaps = [lines[i][0].top - lines[i - 1][0].bottom for i in range(1, len(lines))]
    med_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
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


def render_lines_with_diagram_detection(lines: list[list[Word]], stats: DocStats) -> str:
    """Разбивает строки колонки на обычные абзацы/заголовки и на подозрительные
    "диаграммные" прогоны (>= DIAGRAM_MIN_RUN строк подряд с широким разрывом/одним словом)."""
    rendered: list[str] = []
    i = 0
    n = len(lines)
    prose_buf: list[list[Word]] = []

    def flush_prose() -> None:
        nonlocal prose_buf
        if not prose_buf:
            return
        segments: list[str] = []
        para_buf: list[list[Word]] = []
        for line in prose_buf:
            dominant_size = Counter(w.size for w in line).most_common(1)[0][0]
            level = heading_level(dominant_size, stats)
            if level is not None:
                if para_buf:
                    segments.append(render_lines_as_paragraphs(para_buf))
                    para_buf = []
                segments.append(f"{'#' * min(level, 6)} {' '.join(w.text for w in line)}")
            else:
                para_buf.append(line)
        if para_buf:
            segments.append(render_lines_as_paragraphs(para_buf))
        rendered.append("\n\n".join(s for s in segments if s.strip()))
        prose_buf = []

    while i < n:
        if line_is_diagram_like(lines[i]):
            j = i
            while j < n and line_is_diagram_like(lines[j]):
                j += 1
            run = lines[i:j]
            if len(run) >= DIAGRAM_MIN_RUN:
                flush_prose()
                raw = "; ".join(" ".join(w.text for w in ln) for ln in run)
                rendered.append(
                    "> **[Требует ручной проверки — вероятная диаграмма/инфографика]**\n"
                    f"> Извлечённые фрагменты текста (порядок не гарантирован): {raw}"
                )
            else:
                prose_buf.extend(run)  # слишком короткий прогон - не диаграмма, обычный текст
            i = j
        else:
            prose_buf.append(lines[i])
            i += 1
    flush_prose()
    return "\n\n".join(r for r in rendered if r.strip())


def render_page(
    words: list[Word],
    page_width: float,
    page_height: float,
    stats: DocStats,
    table_bboxes: list[tuple[float, float, float, float]],
) -> str:
    clean_words = strip_boilerplate_and_page_numbers(words, page_height, stats)
    clean_words = [w for w in clean_words if not word_in_any_bbox(w, table_bboxes)]
    main_words = [w for w in clean_words if w.size > stats.tiny_marker_max]

    columns = detect_columns(main_words, page_width)
    column_texts: list[str] = []
    for x0, x1 in columns:
        col_words = [w for w in main_words if x0 <= (w.x0 + w.x1) / 2 < x1]
        lines = group_into_lines(col_words)
        column_texts.append(render_lines_with_diagram_detection(lines, stats))

    return "\n\n".join(t for t in column_texts if t.strip())


def render_tables(real_tables: list[Table]) -> list[str]:
    tables_md = []
    for table in real_tables:
        raw_rows = table.extract()
        if not any(any(cell for cell in row) for row in raw_rows):
            continue
        clean_rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in raw_rows]
        header, *body = clean_rows
        md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in body:
            md.append("| " + " | ".join(row) + " |")
        tables_md.append("\n".join(md))
    return tables_md


def convert(pdf_path: str, out_path: str) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        pages_words = [load_words(p) for p in pdf.pages]
        page_height = pdf.pages[0].height
        page_width = pdf.pages[0].width
        stats = compute_doc_stats(pages_words, page_height)

        print(f"body_size={stats.body_size}, heading_sizes={stats.heading_sizes}, "
              f"tiny_marker_max={stats.tiny_marker_max:.1f}", file=sys.stderr)
        print(f"колонтитулов/повторяющихся строк обнаружено: {len(stats.boilerplate_norms)}", file=sys.stderr)

        out_parts: list[str] = []
        for page, words in zip(pdf.pages, pages_words):
            real_tables = get_real_tables(page)
            table_bboxes = [t.bbox for t in real_tables]
            text = render_page(words, page_width, page_height, stats, table_bboxes)
            tables = render_tables(real_tables)
            if tables:
                text += "\n\n" + "\n\n".join(tables)
            out_parts.append(text)

        full_text = "\n\n".join(p for p in out_parts if p.strip())
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
