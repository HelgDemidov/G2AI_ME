"""Тесты convert/lint.py: C1 авто-QA (spec convert-hardening) — чистые функции,
без сети/модели/pdfplumber."""
from __future__ import annotations

from convert.lint import lint_conversion


def test_no_headings_flagged() -> None:
    defects = lint_conversion("Just plain prose, no markdown headings anywhere.", raw_text_chars=None, fmt="pdf")
    assert "no-headings" in defects


def test_headings_present_no_defect() -> None:
    defects = lint_conversion("# Title\n\nSome body text.", raw_text_chars=None, fmt="pdf")
    assert "no-headings" not in defects


def test_text_loss_flagged_when_ratio_below_threshold() -> None:
    md = "# Title\n\nShort."
    defects = lint_conversion(md, raw_text_chars=1000, fmt="pdf")  # md text tiny vs raw
    assert any(d.startswith("text-loss") for d in defects)


def test_text_loss_not_flagged_when_ratio_above_threshold() -> None:
    md = "# Title\n\n" + ("word " * 50)
    defects = lint_conversion(md, raw_text_chars=10, fmt="pdf")  # md text >> raw
    assert not any(d.startswith("text-loss") for d in defects)


def test_text_loss_skipped_when_raw_text_chars_none() -> None:
    """html-путь: raw_text_chars=None -> ratio неинформативен, проверка не выполняется
    вообще, даже если md почти пуст."""
    defects = lint_conversion("# T\n\nx", raw_text_chars=None, fmt="html")
    assert not any(d.startswith("text-loss") for d in defects)


def test_ragged_table_flagged() -> None:
    md = "# Title\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| only-one-cell |"
    defects = lint_conversion(md, raw_text_chars=None, fmt="pdf")
    assert any(d.startswith("table-ragged") for d in defects)


def test_clean_table_not_flagged() -> None:
    md = "# Title\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    defects = lint_conversion(md, raw_text_chars=None, fmt="pdf")
    assert not any(d.startswith("table-ragged") for d in defects)


def test_clean_document_no_defects() -> None:
    md = "# Title\n\nEnough body prose to comfortably clear the text-loss ratio threshold.\n\n" \
         "| A | B |\n| --- | --- |\n| 1 | 2 |"
    defects = lint_conversion(md, raw_text_chars=10, fmt="pdf")
    assert defects == []


def test_frontmatter_stripped_before_heading_check() -> None:
    """Frontmatter сам начинается с '---', не '#' — не должен создавать ложный
    сигнал «заголовки есть», если тело документа без единого #."""
    md = "---\nid: test-doc\n---\n\nJust prose, no headings in the body."
    defects = lint_conversion(md, raw_text_chars=None, fmt="pdf")
    assert "no-headings" in defects
