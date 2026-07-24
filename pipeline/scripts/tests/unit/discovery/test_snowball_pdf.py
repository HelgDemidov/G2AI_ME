"""Тесты discovery/connectors/snowball.py — экстрактор PDF-аннотаций (spec discovery-snowball
§2.1/§2.4, коммит 2). Реальные объекты ``pdfplumber``/reportlab (``tests.support.build_pdf``,
методология test-coverage-hardening) — пере-реализация геометрии pdfplumber моком тестировала
бы мок, не наш код.
"""
from __future__ import annotations

from pathlib import Path

from discovery.connectors.snowball import RawLink, extract_pdf_annotation_links
from tests.support import build_pdf


def test_single_line_link_extracted_with_anchor_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "single.pdf"
    pdf_path.write_bytes(
        build_pdf(
            lines=[("AI Action Plan.", 50.0, 60.0, 12.0)],
            links=[("https://www.whitehouse.gov/ai-plan", 50.0, 55.0, 200.0, 72.0)],
        )
    )
    links = extract_pdf_annotation_links(pdf_path)
    assert links == [
        RawLink(url="https://www.whitehouse.gov/ai-plan", anchor="AI Action Plan.", page_number=1)
    ]


def test_pdf_without_annotations_yields_zero_links(tmp_path: Path) -> None:
    pdf_path = tmp_path / "plain.pdf"
    pdf_path.write_bytes(build_pdf(lines=[("no links here", 50.0, 60.0, 12.0)]))
    assert extract_pdf_annotation_links(pdf_path) == []


def test_wrapped_link_two_annotations_same_uri_merge_in_reading_order(tmp_path: Path) -> None:
    """Живой пример GAIRI: обёрнутая ссылка = ДВЕ отдельные аннотации одного uri, каждая
    на своей строке — anchor должен склеиться в порядке чтения (top, x0), не наоборот.

    Строки разнесены на 50pt (реальный line-height многократно меньше) — намеренный запас,
    чтобы rect-боксы двух аннотаций гарантированно не задевали друг друга по вертикали
    (glyph bbox чуть шире номинального font-size — ~2.5pt над/под y_from_top, см. живой
    промер ``pdfplumber page.chars``); тесная разметка иначе даёт ложный overlap и дубли."""
    uri = "https://www.canada.ca/en/government/system/digital-government"
    pdf_path = tmp_path / "wrapped.pdf"
    pdf_path.write_bytes(
        build_pdf(
            lines=[
                ("Canada's AI Strategy for the", 50.0, 60.0, 12.0),
                ("Federal Public Service", 50.0, 110.0, 12.0),
            ],
            links=[
                (uri, 50.0, 55.0, 230.0, 80.0),  # первая строка
                (uri, 50.0, 105.0, 200.0, 130.0),  # вторая строка
            ],
        )
    )
    links = extract_pdf_annotation_links(pdf_path)
    assert len(links) == 1
    assert links[0].url == uri
    assert links[0].anchor == "Canada's AI Strategy for the Federal Public Service"


def test_garbage_uri_like_http_a_is_dropped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "garbage.pdf"
    pdf_path.write_bytes(
        build_pdf(
            lines=[("a", 50.0, 60.0, 12.0)],
            links=[("http://a", 50.0, 55.0, 60.0, 72.0)],
        )
    )
    assert extract_pdf_annotation_links(pdf_path) == []


def test_two_different_uris_on_same_page_both_extracted(tmp_path: Path) -> None:
    pdf_path = tmp_path / "two.pdf"
    pdf_path.write_bytes(
        build_pdf(
            lines=[("First link", 50.0, 60.0, 12.0), ("Second link", 50.0, 100.0, 12.0)],
            links=[
                ("https://example.org/first", 50.0, 55.0, 150.0, 72.0),
                ("https://example.org/second", 50.0, 95.0, 160.0, 112.0),
            ],
        )
    )
    links = extract_pdf_annotation_links(pdf_path)
    urls = {link.url for link in links}
    assert urls == {"https://example.org/first", "https://example.org/second"}


def test_annotation_with_no_overlapping_text_has_empty_anchor(tmp_path: Path) -> None:
    """Аннотация, под которой нет текста (пустая область) — anchor="", находка НЕ теряется
    (фолбэк на сегмент пути URL — работа маппинга, коммит 4, не экстрактора)."""
    pdf_path = tmp_path / "no_text.pdf"
    pdf_path.write_bytes(
        build_pdf(
            lines=[],
            links=[("https://example.org/empty-anchor", 500.0, 700.0, 580.0, 720.0)],
        )
    )
    links = extract_pdf_annotation_links(pdf_path)
    assert links == [RawLink(url="https://example.org/empty-anchor", anchor="", page_number=1)]
