"""Тесты pdf_to_markdown: per-page высота/ширина (смешанная ориентация), guard на
пустой PDF, позиционная вставка блоков графика-пасса, снятие word-gap-эвристики."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from convert import pdf_graphics
from convert.pdf_to_markdown import (
    DocStats,
    Word,
    _drop_empty_columns,
    _is_bold,
    _render_column_with_blocks,
    compute_doc_stats,
    convert,
    merge_split_tables,
)


def test_compute_doc_stats_uses_own_height_per_page_for_boilerplate_band() -> None:
    """Полоса колонтитула считается по высоте КАЖДОЙ страницы: заголовочная строка
    у top=60 попадает в полосу большой (800pt) страницы, но не маленькой (200pt) —
    если бы band считался по единой (напр. первой) высоте, эта строка никогда не
    попала бы ни в одну полосу и не была бы распознана как повторяющийся колонтитул."""
    header = Word("HEADERTEXT", x0=10.0, x1=100.0, top=60.0, bottom=72.0, size=10.0)
    small_empty: list[Word] = []
    big_with_header = [header]

    pages = [
        (small_empty, 200.0),   # band = 200*0.09 = 18 — top=60 сюда бы не попал
        (big_with_header, 800.0),  # band = 800*0.09 = 72 — top=60 попадает
        (small_empty, 200.0),
        (big_with_header, 800.0),
    ]
    stats = compute_doc_stats(pages)
    assert "HEADERTEXT" in stats.boilerplate_norms


def test_compute_doc_stats_empty_pages_no_boilerplate() -> None:
    stats = compute_doc_stats([([], 300.0), ([], 300.0)])
    assert stats.boilerplate_norms == set()
    assert stats.body_size == 11.0  # дефолт при пустом size_char_counts


class _FakePage:
    def __init__(self, height: float) -> None:
        self.height = height


class _FakeEmptyPdf:
    def __init__(self) -> None:
        self.pages: list[_FakePage] = []

    def __enter__(self) -> "_FakeEmptyPdf":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def test_convert_raises_on_pdf_without_pages(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr("convert.pdf_to_markdown.pdfplumber.open", lambda path: _FakeEmptyPdf())
    with pytest.raises(RuntimeError, match="без страниц"):
        convert(str(tmp_path / "in.pdf"), str(tmp_path / "out.md"))


# --- _render_column_with_blocks: позиционная вставка (spec convert-graphics §3 п.3) ---

_FLAT_STATS = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())


def test_block_inserted_between_two_paragraphs_by_vertical_position() -> None:
    """Блок с top МЕЖДУ двумя абзацами оказывается МЕЖДУ ними в выводе, а не в
    конце (прежнее поведение — все таблицы/маркеры приклеивались в конец страницы)."""
    before = [Word("Before", x0=0.0, x1=50.0, top=10.0, bottom=20.0, size=10.0)]
    after = [Word("After", x0=0.0, x1=50.0, top=100.0, bottom=110.0, size=10.0)]
    block_bbox = (0.0, 50.0, 100.0, 60.0)  # top=50 — между 10 и 100
    out = _render_column_with_blocks([before, after], [(block_bbox, "> [block marker]")], _FLAT_STATS)
    assert out == "Before\n\n> [block marker]\n\nAfter"


def test_block_at_page_start_appears_before_all_prose() -> None:
    only = [Word("Only paragraph", x0=0.0, x1=80.0, top=50.0, bottom=60.0, size=10.0)]
    block_bbox = (0.0, 0.0, 100.0, 10.0)  # top=0 — раньше единственного абзаца
    out = _render_column_with_blocks([only], [(block_bbox, "> [early marker]")], _FLAT_STATS)
    assert out == "> [early marker]\n\nOnly paragraph"


# --- инвариант «ничего не потеряно»: слова региона — в блоке, не в прозе ---


def test_region_words_excluded_from_prose_present_in_block_exactly_once() -> None:
    grid_elements = [
        pdf_graphics.Element("rect", 0.0, 0.0, 100.0, 100.0),
        pdf_graphics.Element("rect", 110.0, 0.0, 210.0, 100.0),
        pdf_graphics.Element("rect", 0.0, 110.0, 100.0, 210.0),
        pdf_graphics.Element("rect", 110.0, 110.0, 210.0, 210.0),
    ]
    cell_words = [
        Word("Strength", x0=10.0, x1=60.0, top=10.0, bottom=20.0, size=10.0),
        Word("Weakness", x0=120.0, x1=170.0, top=10.0, bottom=20.0, size=10.0),
        Word("Opportunity", x0=10.0, x1=60.0, top=120.0, bottom=130.0, size=10.0),
        Word("Threat", x0=120.0, x1=170.0, top=120.0, bottom=130.0, size=10.0),
    ]
    outside_word = Word("Unrelated prose sentence", x0=0.0, x1=100.0, top=300.0, bottom=310.0, size=10.0)

    regions, remaining = pdf_graphics.detect_regions(
        page=1, elements=grid_elements, words=[*cell_words, outside_word],
        page_width=600.0, page_height=800.0, table_bboxes=[],
    )
    assert len(regions) == 1  # SWOT-грид распознан как один регион
    assert remaining == [outside_word]  # только внешнее слово осталось в прозе

    blocks = [(regions[0].bbox, pdf_graphics.render_region_block(regions[0], page=1))]
    prose_lines = [[w] for w in remaining]
    rendered = _render_column_with_blocks(prose_lines, blocks, _FLAT_STATS)

    for w in cell_words:
        assert w.text not in rendered.split(blocks[0][1])[0]  # не в прозной части (до блока)
        assert w.text in blocks[0][1]  # ровно в блоке
    assert outside_word.text in rendered


# --- снятие word-gap-эвристики (§3 п.4): россыпь коротких строк -> проза, без маркеров ---


def test_scattered_short_lines_render_as_prose_without_diagram_markers() -> None:
    """Строки-одиночки с широкими промежутками между словами — раньше триггерили
    word-gap-эвристику («вероятная диаграмма»). Без единого векторного элемента
    поблизости это теперь просто проза: слова на месте, разметки-маркера нет."""
    lines = [
        [Word("Alpha", x0=0.0, x1=30.0, top=10.0, bottom=20.0, size=10.0)],
        [Word("Beta", x0=200.0, x1=230.0, top=40.0, bottom=50.0, size=10.0)],
        [Word("Gamma", x0=400.0, x1=430.0, top=70.0, bottom=80.0, size=10.0)],
    ]
    out = _render_column_with_blocks(lines, [], _FLAT_STATS)
    assert "> [" not in out
    assert "Alpha" in out and "Beta" in out and "Gamma" in out


# --- A1: bold-фолбэк заголовков (документ с одинаковым кеглем заголовка/тела) ---


@pytest.mark.parametrize(
    ("fontname", "expected"),
    [
        ("Arial-BoldMT", True),
        ("DejaVuSans-Bold", True),
        ("Helvetica-Black", True),
        ("Roboto-Heavy", True),
        ("ABCDEF+Arial-Bold", True),  # subset-префикс
        ("Arial-Regular", False),
        ("Times-Italic", False),
        ("", False),
    ],
)
def test_is_bold(fontname: str, expected: bool) -> None:
    assert _is_bold(fontname) is expected


def _bold_line(text: str, size: float = 10.0) -> list[Word]:
    return [Word(w, x0=0.0, x1=10.0, top=0.0, bottom=10.0, size=size, fontname="Arial-Bold") for w in text.split()]


def test_bold_fallback_promotes_body_size_bold_line_below_all_sizes() -> None:
    """Документ с реальными размерными заголовками: bold-строка кегля тела не
    конкурирует с ними — уходит на уровень СРАЗУ ПОД последним размерным."""
    stats = DocStats(body_size=10.0, heading_sizes=[16.0, 13.0], tiny_marker_max=6.5, boilerplate_norms=set())
    line = _bold_line("Bold Heading")
    out = _render_column_with_blocks([line], [], stats)
    assert out == "### Bold Heading"  # len(heading_sizes)+1 = 3


def test_bold_fallback_gives_h1_when_document_has_no_size_headings() -> None:
    """Кейс, ради которого фикс: документ без единого размерного заголовка
    (одинаковый кегль заголовка/тела) — bold-строка не пропадает, а даёт #."""
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    line = _bold_line("Bold Heading")
    out = _render_column_with_blocks([line], [], stats)
    assert out == "# Bold Heading"


def test_bold_fallback_rejects_trailing_period() -> None:
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    line = _bold_line("Bold sentence.")
    out = _render_column_with_blocks([line], [], stats)
    assert "#" not in out
    assert "Bold sentence." in out


def test_bold_fallback_rejects_line_over_max_chars() -> None:
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    long_text = "Word " * 20  # far over BOLD_HEADING_MAX_CHARS=80
    line = _bold_line(long_text.strip())
    out = _render_column_with_blocks([line], [], stats)
    assert not out.startswith("#")


def test_bold_fallback_rejects_size_below_body() -> None:
    """Bold-сноски/подписи мельче тела не промоутятся в заголовки."""
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    line = _bold_line("Tiny bold caption", size=8.0)
    out = _render_column_with_blocks([line], [], stats)
    assert "#" not in out


def test_bold_fallback_rejects_list_marker() -> None:
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    line = [
        Word("-", x0=0.0, x1=5.0, top=0.0, bottom=10.0, size=10.0, fontname="Arial-Bold"),
        Word("item", x0=6.0, x1=20.0, top=0.0, bottom=10.0, size=10.0, fontname="Arial-Bold"),
    ]
    out = _render_column_with_blocks([line], [], stats)
    assert "#" not in out
    assert "- item" in out


# --- A3: _drop_empty_columns ---


def test_drop_empty_columns_removes_column_empty_in_all_rows() -> None:
    rows = [["Name", "", "Value"], ["a", "", "1"], ["b", "", "2"]]
    assert _drop_empty_columns(rows) == [["Name", "Value"], ["a", "1"], ["b", "2"]]


def test_drop_empty_columns_keeps_column_empty_only_in_data() -> None:
    """Пусто только в данных (шапка непуста) — колонка остаётся: это может быть
    легитимно разреженная колонка (не обломок)."""
    rows = [["Name", "Note"], ["a", ""], ["b", ""]]
    assert _drop_empty_columns(rows) == rows


def test_drop_empty_columns_noop_when_all_nonempty() -> None:
    rows = [["Name", "Value"], ["a", "1"]]
    assert _drop_empty_columns(rows) == rows


# --- A2: merge_split_tables ---


def test_merge_split_tables_identical_header_across_blank_line() -> None:
    md = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
        "\n\n"
        "| A | B |\n| --- | --- |\n| 3 | 4 |"
    )
    out = merge_split_tables(md)
    assert out == "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    assert out.count("| A | B |") == 1  # шапка не повторена


def test_merge_split_tables_different_headers_not_merged() -> None:
    md = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
        "\n\n"
        "| X | Y |\n| --- | --- |\n| 3 | 4 |"
    )
    assert merge_split_tables(md) == md


def test_merge_split_tables_different_column_count_not_merged() -> None:
    md = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
        "\n\n"
        "| A | B | C |\n| --- | --- | --- |\n| 3 | 4 | 5 |"
    )
    assert merge_split_tables(md) == md


def test_merge_split_tables_prose_between_tables_not_merged() -> None:
    md = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
        "\n\nSome prose paragraph in between.\n\n"
        "| A | B |\n| --- | --- |\n| 3 | 4 |"
    )
    assert merge_split_tables(md) == md


def test_merge_split_tables_idempotent() -> None:
    md = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
        "\n\n"
        "| A | B |\n| --- | --- |\n| 3 | 4 |"
    )
    once = merge_split_tables(md)
    assert merge_split_tables(once) == once
