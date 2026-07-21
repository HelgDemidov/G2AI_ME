"""Тесты pdf_to_markdown: per-page высота/ширина (смешанная ориентация), guard на
пустой PDF, позиционная вставка блоков графика-пасса, снятие word-gap-эвристики."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pdfplumber
import pytest

from convert import pdf_graphics
from convert.pdf_to_markdown import (
    DocStats,
    Word,
    _bold_run,
    _drop_empty_columns,
    _is_bold,
    _median_line_gap,
    _render_column_with_blocks,
    _word_fontname,
    compute_doc_stats,
    compute_page_graphics,
    convert,
    detect_columns,
    get_real_tables,
    heading_level,
    load_words,
    merge_split_tables,
    normalize_line,
    render_lines_as_paragraphs,
    render_page,
    render_tables,
    strip_boilerplate_and_page_numbers,
    word_in_any_bbox,
)
from tests.support import build_pdf


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


# --- _word_fontname: bold ТОЛЬКО если ВСЕ символы слова bold ---


def test_word_fontname_all_bold_returns_representative_name() -> None:
    chars = [{"fontname": "Arial-Bold"}, {"fontname": "Arial-Bold"}]
    assert _word_fontname(chars) == "Arial-Bold"


def test_word_fontname_mixed_bold_and_regular_returns_empty() -> None:
    chars = [{"fontname": "Arial-Bold"}, {"fontname": "Arial"}]
    assert _word_fontname(chars) == ""


def test_word_fontname_empty_chars_returns_empty() -> None:
    assert _word_fontname([]) == ""


# --- load_words: регресс живого аудита (sg, IMDA Agentic AI) — границы слов
# НЕ должны зависеть от fontname, иначе bold-термин впритык к пунктуации
# ("Controls:") режется надвое и при рендере получает ложный пробел ---


class _FakeWordsPage:
    def __init__(self, words: list[dict[str, Any]]) -> None:
        self._words = words
        self.calls: list[dict[str, Any]] = []

    def extract_words(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self._words


def test_load_words_requests_word_boundaries_without_fontname() -> None:
    page = _FakeWordsPage([
        {
            "text": "Controls:", "x0": 0.0, "x1": 40.0, "top": 0.0, "bottom": 10.0, "size": 10.0,
            "chars": [{"fontname": "Arial-Bold"}] * 8 + [{"fontname": "Arial"}],  # ':' не bold
        },
    ])
    words = load_words(page)  # type: ignore[arg-type]
    assert page.calls == [{"extra_attrs": ["size"], "return_chars": True}]  # без "fontname"
    assert [w.text for w in words] == ["Controls:"]  # не разрезано на два токена
    assert words[0].fontname == ""  # смешанный bold/non-bold -> не bold целиком


def test_load_words_marks_fully_bold_word() -> None:
    page = _FakeWordsPage([
        {
            "text": "Heading", "x0": 0.0, "x1": 40.0, "top": 0.0, "bottom": 10.0, "size": 10.0,
            "chars": [{"fontname": "Arial-Bold"}] * 7,
        },
    ])
    words = load_words(page)  # type: ignore[arg-type]
    assert words[0].fontname == "Arial-Bold"


# --- _bold_run / _median_line_gap: изоляция bold-прогона (регресс sg + целевой кейс ee) ---


def _line(text: str, top: float, *, bold: bool = False, size: float = 10.0) -> list[Word]:
    fontname = "Arial-Bold" if bold else "Arial"
    return [
        Word(w, x0=0.0, x1=10.0, top=top, bottom=top + 10.0, size=size, fontname=fontname)
        for w in text.split()
    ]


def test_median_line_gap_empty_or_single_line_is_zero() -> None:
    assert _median_line_gap([]) == 0.0
    assert _median_line_gap([_line("only", top=0.0)]) == 0.0


def test_bold_run_single_isolated_line() -> None:
    lines = [_line("Heading", top=0.0, bold=True)]
    assert _bold_run(lines, 0, med_gap=10.0) == [lines[0]]


def test_bold_run_not_bold_returns_none() -> None:
    lines = [_line("Plain text", top=0.0, bold=False)]
    assert _bold_run(lines, 0, med_gap=10.0) is None


def test_bold_run_extends_across_tight_gap_when_both_bold() -> None:
    """Перенос заголовка на вторую строку колонки (sg: "3. Implement technical
    controls and" + "processes") — обе bold, тесный разрыв -> ОДИН прогон."""
    lines = [_line("First line", top=0.0, bold=True), _line("second part", top=10.0, bold=True)]
    assert _bold_run(lines, 0, med_gap=10.0) == lines


def test_bold_run_stops_without_disqualifying_on_big_gap() -> None:
    lines = [_line("Heading", top=0.0, bold=True), _line("unrelated later text", top=100.0, bold=False)]
    assert _bold_run(lines, 0, med_gap=10.0) == [lines[0]]


def test_bold_run_disqualified_by_lowercase_continuation_same_paragraph() -> None:
    """Регресс sg: bold lead-in ("...bring forth new") впритык (тот же абзац)
    к НЕ-bold продолжению, начинающемуся со строчной буквы ("risks are
    severe") -> грамматическое продолжение ОДНОГО предложения, весь прогон
    дисквалифицируется (остаётся прозой)."""
    lines = [_line("bring forth new", top=0.0, bold=True), _line("risks are severe", top=10.0, bold=False)]
    assert _bold_run(lines, 0, med_gap=10.0) is None


def test_bold_run_not_disqualified_by_bullet_continuation_same_paragraph() -> None:
    """Целевой кейс ee: короткий bold-ярлык ("Tugevused Nõrkused") впритык к
    НЕ-bold буллет-списку (новый пункт, НЕ грамматическое продолжение) ->
    прогон остаётся изолированным как есть."""
    bullet_line = [
        Word("•", x0=0.0, x1=5.0, top=10.0, bottom=20.0, size=10.0, fontname="Arial"),
        *_line("Digitally innovative", top=10.0, bold=False),
    ]
    lines = [_line("Tugevused Nõrkused", top=0.0, bold=True), bullet_line]
    assert _bold_run(lines, 0, med_gap=10.0) == [lines[0]]


def test_flush_prose_flushes_paragraph_before_bold_fallback_heading() -> None:
    """para_buf уже накопил прозу к моменту bold-фолбэк заголовка — флашится ПЕРЕД ним
    (зеркало test_flush_prose_promotes_heading_by_size_after_prose_paragraph для
    размерного пути)."""
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    prose = [Word("Intro", x0=0.0, x1=30.0, top=0.0, bottom=10.0, size=10.0, fontname="Arial")]
    heading = [Word("Bold Heading", x0=0.0, x1=80.0, top=100.0, bottom=110.0, size=10.0, fontname="Arial-Bold")]
    out = _render_column_with_blocks([prose, heading], [], stats)
    assert out == "Intro\n\n# Bold Heading"


def test_flush_prose_promotes_isolated_bold_label_but_not_lead_in_sentence() -> None:
    """Сквозной регресс-тест через _render_column_with_blocks: короткий
    изолированный bold-ярлык становится заголовком, а bold lead-in длинного
    предложения — остаётся ЦЕЛЬНЫМ прозным абзацем (не рвётся на фрагмент)."""
    label = _line("Tugevused Nõrkused", top=0.0, bold=True)
    bullet = [
        Word("•", x0=0.0, x1=5.0, top=10.0, bottom=20.0, size=10.0, fontname="Arial"),
        *_line("Digitally innovative ecosystem", top=10.0, bold=False),
    ]
    lead_in = _line("bring forth new", top=200.0, bold=True)
    continuation = _line("risks are severe indeed", top=210.0, bold=False)
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())

    out = _render_column_with_blocks([label, bullet, lead_in, continuation], [], stats)

    assert "# Tugevused Nõrkused" in out
    assert "bring forth new risks are severe indeed" in out  # цельный абзац, не разорван
    assert "# bring forth new" not in out


# --- normalize_line: строка-таблицы с "|" (test-coverage-hardening) ---


def test_normalize_line_strips_content_after_pipe() -> None:
    assert normalize_line("Page 3 | extra junk") == "Page #"


# --- load_words: слово, ставшее пустым после срезки DOT_LEADER_RE ---


def test_load_words_skips_word_empty_after_dot_leader_strip() -> None:
    page = _FakeWordsPage([
        {"text": "...", "x0": 0.0, "x1": 10.0, "top": 0.0, "bottom": 10.0, "size": 10.0, "chars": []},
        {"text": "Real", "x0": 20.0, "x1": 40.0, "top": 0.0, "bottom": 10.0, "size": 10.0, "chars": []},
    ])
    words = load_words(page)  # type: ignore[arg-type]
    assert [w.text for w in words] == ["Real"]


# --- heading_level / flush_prose: размерная (не bold-фолбэк) промоция заголовка — ни
# один существующий тест не давал документу реальные heading_sizes И строку без bold ---


def test_heading_level_no_match_returns_none() -> None:
    stats = DocStats(body_size=10.0, heading_sizes=[16.0], tiny_marker_max=6.5, boilerplate_norms=set())
    assert heading_level(10.0, stats) is None


def test_flush_prose_promotes_heading_by_matching_font_size() -> None:
    stats = DocStats(body_size=10.0, heading_sizes=[16.0], tiny_marker_max=6.5, boilerplate_norms=set())
    line = [Word("Big Heading", x0=0.0, x1=80.0, top=0.0, bottom=16.0, size=16.0, fontname="Arial")]
    out = _render_column_with_blocks([line], [], stats)
    assert out == "# Big Heading"


def test_flush_prose_promotes_heading_by_size_after_prose_paragraph() -> None:
    """para_buf уже накопил прозу к моменту размерного заголовка — флашится ПЕРЕД ним."""
    stats = DocStats(body_size=10.0, heading_sizes=[16.0], tiny_marker_max=6.5, boilerplate_norms=set())
    prose = [Word("Intro", x0=0.0, x1=30.0, top=0.0, bottom=10.0, size=10.0, fontname="Arial")]
    heading = [Word("Big Heading", x0=0.0, x1=80.0, top=20.0, bottom=36.0, size=16.0, fontname="Arial")]
    out = _render_column_with_blocks([prose, heading], [], stats)
    assert out == "Intro\n\n# Big Heading"


# --- render_lines_as_paragraphs: пустой ввод + разбивка на абзацы по широкому разрыву ---


def test_render_lines_as_paragraphs_empty_returns_empty_string() -> None:
    assert render_lines_as_paragraphs([]) == ""


def test_render_lines_as_paragraphs_splits_on_wide_gap() -> None:
    """3 тесных разрыва (2pt) задают med_gap=2 -> порог 3.2pt; 4-й разрыв (116pt) намного
    шире порога -> новый абзац."""
    lines = [
        [Word("A", x0=0.0, x1=10.0, top=0.0, bottom=10.0, size=10.0)],
        [Word("B", x0=0.0, x1=10.0, top=12.0, bottom=22.0, size=10.0)],
        [Word("C", x0=0.0, x1=10.0, top=24.0, bottom=34.0, size=10.0)],
        [Word("D", x0=0.0, x1=10.0, top=150.0, bottom=160.0, size=10.0)],
    ]
    assert render_lines_as_paragraphs(lines) == "A B C\n\nD"


# --- _drop_empty_columns: пустой ввод ---


def test_drop_empty_columns_empty_rows_returns_as_is() -> None:
    assert _drop_empty_columns([]) == []


# --- word_in_any_bbox (test-coverage-hardening: чистая функция, была без единого теста) ---


def test_word_in_any_bbox_true_when_center_inside() -> None:
    w = Word("x", x0=10.0, x1=20.0, top=10.0, bottom=20.0, size=10.0)
    assert word_in_any_bbox(w, [(0.0, 0.0, 30.0, 30.0)]) is True


def test_word_in_any_bbox_false_when_outside_all_bboxes() -> None:
    w = Word("x", x0=100.0, x1=110.0, top=100.0, bottom=110.0, size=10.0)
    assert word_in_any_bbox(w, [(0.0, 0.0, 30.0, 30.0)]) is False


def test_word_in_any_bbox_empty_bboxes_returns_false() -> None:
    w = Word("x", x0=10.0, x1=20.0, top=10.0, bottom=20.0, size=10.0)
    assert word_in_any_bbox(w, []) is False


# --- strip_boilerplate_and_page_numbers (чистая функция, была без единого теста) ---


def test_strip_boilerplate_drops_bare_page_number_in_footer_band() -> None:
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    page_number = Word("7", x0=300.0, x1=310.0, top=780.0, bottom=790.0, size=10.0)  # нижняя полоса
    body = Word("Body", x0=50.0, x1=90.0, top=400.0, bottom=410.0, size=10.0)
    out = strip_boilerplate_and_page_numbers([page_number, body], page_height=800.0, stats=stats)
    assert [w.text for w in out] == ["Body"]


def test_strip_boilerplate_drops_repeated_header_line() -> None:
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms={"HEADER"})
    header = Word("HEADER", x0=50.0, x1=100.0, top=10.0, bottom=20.0, size=10.0)  # верхняя полоса
    body = Word("Body", x0=50.0, x1=90.0, top=400.0, bottom=410.0, size=10.0)
    out = strip_boilerplate_and_page_numbers([header, body], page_height=800.0, stats=stats)
    assert [w.text for w in out] == ["Body"]


def test_strip_boilerplate_keeps_in_band_text_not_matching_boilerplate() -> None:
    """В полосе колонтитула, но не голый номер и не известный повтор — легитимный
    контент (напр. заголовок раздела у самого верха страницы), не вычищается."""
    stats = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())
    heading = Word("Introduction", x0=50.0, x1=150.0, top=10.0, bottom=20.0, size=14.0)
    out = strip_boilerplate_and_page_numbers([heading], page_height=800.0, stats=stats)
    assert [w.text for w in out] == ["Introduction"]


# --- detect_columns (чистая функция, флагманский геометрический детектор — был без единого теста) ---


def test_detect_columns_no_gap_returns_full_width() -> None:
    words = [Word("word", x0=10.0, x1=590.0, top=10.0, bottom=20.0, size=10.0)]
    assert detect_columns(words, page_width=600.0) == [(0, 600.0)]


def test_detect_columns_empty_words_returns_full_width() -> None:
    assert detect_columns([], page_width=600.0) == [(0, 600.0)]


def test_detect_columns_splits_on_wide_central_gap() -> None:
    """Левая колонка x∈[50,250], правая [350,550] — разрыв [250,350]=100pt ≥ MIN_GAP_PT,
    центр (300) внутри COLUMN_ZONE (180-420 при ширине 600), обе стороны — 5 строк
    (span ≥ MIN_GAP_HEIGHT_FRAC контентной высоты)."""
    left = [
        Word(f"L{i}", x0=50.0, x1=250.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    right = [
        Word(f"R{i}", x0=350.0, x1=550.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    columns = detect_columns([*left, *right], page_width=600.0)
    assert len(columns) == 2
    assert 250.0 <= columns[0][1] <= 350.0  # граница где-то внутри разрыва


def test_detect_columns_ignores_gap_outside_column_zone() -> None:
    """Разрыв [100,150] геометрически валиден (шире MIN_GAP_PT, вертикально протяжённый),
    но его центр (125) вне COLUMN_ZONE (180-420) — не путать с полями страницы."""
    left = [
        Word("L", x0=0.0, x1=100.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    right = [
        Word("R", x0=150.0, x1=600.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    assert detect_columns([*left, *right], page_width=600.0) == [(0, 600.0)]


def test_detect_columns_ignores_short_vertical_gap() -> None:
    """Разрыв в зоне, но обе стороны — одна строка на одной высоте: геометрический span
    почти нулевой (не покрывает MIN_GAP_HEIGHT_FRAC контентной высоты) — вероятно короткая
    надпись/строка таблицы, не двухколоночная вёрстка."""
    left = Word("L", x0=50.0, x1=250.0, top=10.0, bottom=20.0, size=10.0)
    right = Word("R", x0=350.0, x1=550.0, top=10.0, bottom=20.0, size=10.0)
    assert detect_columns([left, right], page_width=600.0) == [(0, 600.0)]


# --- render_page (чистая функция — оркестрирует detect_columns/group_into_lines/
# _render_column_with_blocks; была без единого теста) ---


_RP_STATS = DocStats(body_size=10.0, heading_sizes=[], tiny_marker_max=6.5, boilerplate_norms=set())


def test_render_page_single_column_renders_prose() -> None:
    words = [Word("Hello", x0=10.0, x1=60.0, top=10.0, bottom=20.0, size=10.0)]
    assert render_page(words, page_width=600.0, stats=_RP_STATS, blocks=[]) == "Hello"


def test_render_page_two_columns_orders_left_before_right() -> None:
    left = [
        Word("LeftWord", x0=50.0, x1=200.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    right = [
        Word("RightWord", x0=350.0, x1=550.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    out = render_page([*left, *right], page_width=600.0, stats=_RP_STATS, blocks=[])
    assert out.index("LeftWord") < out.index("RightWord")


def test_render_page_places_block_in_matching_column() -> None:
    left = [
        Word("LeftWord", x0=50.0, x1=200.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    right = [
        Word("RightWord", x0=350.0, x1=550.0, top=float(i * 20), bottom=float(i * 20 + 10), size=10.0)
        for i in range(5)
    ]
    right_block_bbox = (350.0, 5.0, 550.0, 15.0)  # центр-x внутри правой колонки (>275)
    out = render_page(
        [*left, *right], page_width=600.0, stats=_RP_STATS, blocks=[(right_block_bbox, "> [block]")]
    )
    assert "> [block]" in out
    assert out.index("LeftWord") < out.index("> [block]")


# --- render_tables (нужен только shape объекта Table — .extract()/.bbox — лёгкий локальный
# фейк, не reportlab: сама pdfplumber-геометрия здесь не участвует, только рендер) ---


class _FakeTable:
    def __init__(self, rows: list[list[str | None]], bbox: tuple[float, float, float, float]) -> None:
        self._rows = rows
        self.bbox = bbox

    def extract(self) -> list[list[str | None]]:
        return self._rows


def test_render_tables_produces_markdown_with_bbox() -> None:
    table = _FakeTable([["A", "B"], ["1", "2"]], bbox=(0.0, 0.0, 100.0, 40.0))
    out = render_tables([table])  # type: ignore[list-item]
    assert len(out) == 1
    bbox, md = out[0]
    assert bbox == (0.0, 0.0, 100.0, 40.0)
    assert md == "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_render_tables_skips_fully_empty_table() -> None:
    table = _FakeTable([["", ""], ["", ""]], bbox=(0.0, 0.0, 100.0, 40.0))
    assert render_tables([table]) == []  # type: ignore[list-item]


def test_render_tables_normalizes_newlines_in_cells() -> None:
    table = _FakeTable([["Head\n1", "Head 2"], ["a\nb", "c"]], bbox=(0.0, 0.0, 100.0, 40.0))
    out = render_tables([table])  # type: ignore[list-item]
    assert "Head 1" in out[0][1]


# --- get_real_tables / compute_page_graphics: реальные объекты pdfplumber (test-coverage-
# hardening §3.B.1) — синтетические PDF через reportlab (tests.support.build_pdf), не мок:
# find_tables()/extract_words() — геометрия самой pdfplumber, тестировать мок было бы
# тестированием мока, не нашего кода. ---


def test_get_real_tables_finds_real_grid_table(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(
        build_pdf(table=([["A", "B"], ["1", "2"], ["3", "4"]], 50.0, 50.0, 80.0, 20.0))
    )
    with pdfplumber.open(pdf_path) as pdf:
        tables = get_real_tables(pdf.pages[0])
    assert len(tables) == 1
    rows = tables[0].extract()
    assert rows[0] == ["A", "B"]
    assert rows[1] == ["1", "2"]
    assert rows[2] == ["3", "4"]


def test_get_real_tables_filters_sparse_fragment(tmp_path: Path) -> None:
    """< MIN_TABLE_NONEMPTY_CELLS непустых ячеек — не считается настоящей таблицей
    (реальный обломок диаграммы pdfplumber иногда детектирует как «таблицу»)."""
    pdf_path = tmp_path / "sparse.pdf"
    pdf_path.write_bytes(build_pdf(table=([["", ""], ["", "x"]], 50.0, 50.0, 80.0, 20.0)))
    with pdfplumber.open(pdf_path) as pdf:
        assert get_real_tables(pdf.pages[0]) == []


def test_compute_page_graphics_extracts_text_and_table_from_real_pdf(tmp_path: Path) -> None:
    """Заголовок/тело позиционированы НИЖЕ полосы колонтитула (top 9% страницы, ~71pt при
    792pt высоте) — иначе на однострочичном (1-страничном) документе ЛЮБАЯ строка в этой
    полосе классифицировалась бы как boilerplate (частота 1/1 страниц ≥ 25%-порога) —
    ожидаемое поведение эвристики, не относящееся к цели этого теста."""
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(
        build_pdf(
            lines=[("Document Title", 50.0, 120.0, 16.0), ("Some body prose here.", 50.0, 160.0, 10.0)],
            table=([["Col A", "Col B"], ["1", "2"], ["3", "4"]], 50.0, 220.0, 80.0, 20.0),
        )
    )
    doc = compute_page_graphics(str(pdf_path))
    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert len(page.real_tables) == 1
    assert doc.stats.body_size > 0
    all_text = " ".join(w.text for w in page.remaining_words)
    assert "Document" in all_text
    assert "prose" in all_text
    assert not any("Col" in w.text for w in page.remaining_words)  # содержимое таблицы не в прозе
