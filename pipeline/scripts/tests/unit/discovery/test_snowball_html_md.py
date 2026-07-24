"""Тесты discovery/connectors/snowball.py — экстракторы href raw.html и напечатанных URL
doc.md (spec discovery-snowball §2.2/§2.3, коммит 3)."""
from __future__ import annotations

from pathlib import Path

from discovery.connectors.snowball import (
    RawLink,
    extract_html_href_links,
    extract_printed_urls,
)

# --- extract_html_href_links ---


def test_absolute_href_extracted_with_anchor_text(tmp_path: Path) -> None:
    html = '<html><body><p>See <a href="https://example.org/report.pdf">the report</a>.</p></body></html>'
    path = tmp_path / "raw.html"
    path.write_text(html, encoding="utf-8")
    links = extract_html_href_links(path, source_url="https://gov.example/page")
    assert links == [RawLink(url="https://example.org/report.pdf", anchor="the report")]


def test_relative_href_resolved_against_source_url(tmp_path: Path) -> None:
    html = '<a href="/docs/other.pdf">other doc</a>'
    path = tmp_path / "raw.html"
    path.write_text(html, encoding="utf-8")
    links = extract_html_href_links(path, source_url="https://gov.example/section/page.html")
    assert links[0].url == "https://gov.example/docs/other.pdf"


def test_fragment_only_href_is_dropped(tmp_path: Path) -> None:
    html = '<a href="#section-2">jump</a><a href="https://example.org/real">real</a>'
    path = tmp_path / "raw.html"
    path.write_text(html, encoding="utf-8")
    links = extract_html_href_links(path, source_url="https://gov.example/page")
    assert [link.url for link in links] == ["https://example.org/real"]


def test_mailto_and_javascript_hrefs_are_dropped(tmp_path: Path) -> None:
    html = (
        '<a href="mailto:info@gov.example">mail</a>'
        '<a href="javascript:void(0)">js</a>'
        '<a href="https://example.org/real2">real</a>'
    )
    path = tmp_path / "raw.html"
    path.write_text(html, encoding="utf-8")
    links = extract_html_href_links(path, source_url="https://gov.example/page")
    assert [link.url for link in links] == ["https://example.org/real2"]


def test_href_without_anchor_text_has_empty_anchor(tmp_path: Path) -> None:
    html = '<a href="https://example.org/icon-link"><img src="i.png"/></a>'
    path = tmp_path / "raw.html"
    path.write_text(html, encoding="utf-8")
    links = extract_html_href_links(path, source_url="https://gov.example/page")
    assert links == [RawLink(url="https://example.org/icon-link", anchor="")]


def test_html_without_any_links_yields_zero(tmp_path: Path) -> None:
    path = tmp_path / "raw.html"
    path.write_text("<p>no links here</p>", encoding="utf-8")
    assert extract_html_href_links(path, source_url="https://gov.example/page") == []


# --- extract_printed_urls ---


def test_url_in_prose_extracted_with_line_as_context(tmp_path: Path) -> None:
    doc_md = tmp_path / "doc.md"
    doc_md.write_text(
        "See https://ai.gov.eg/strategy.pdf for the full text.\n\nNo url on this line.\n",
        encoding="utf-8",
    )
    links = extract_printed_urls(doc_md)
    assert len(links) == 1
    assert links[0].url == "https://ai.gov.eg/strategy.pdf"
    assert "See" in links[0].anchor and "full text" in links[0].anchor
    assert links[0].ocr_text_url is False


def test_trailing_punctuation_stripped_from_printed_url(tmp_path: Path) -> None:
    doc_md = tmp_path / "doc.md"
    doc_md.write_text("Cf. (https://example.org/doc).\n", encoding="utf-8")
    links = extract_printed_urls(doc_md)
    assert links[0].url == "https://example.org/doc"


def test_doc_without_any_url_yields_zero(tmp_path: Path) -> None:
    doc_md = tmp_path / "doc.md"
    doc_md.write_text("# Title\n\nJust prose, no links.\n", encoding="utf-8")
    assert extract_printed_urls(doc_md) == []


def test_ocr_normalized_flag_propagates_to_every_finding(tmp_path: Path) -> None:
    doc_md = tmp_path / "doc.md"
    doc_md.write_text(
        "https://example.org/one\nsome text\nhttps://example.org/two\n", encoding="utf-8"
    )
    links = extract_printed_urls(doc_md, ocr_normalized=True)
    assert len(links) == 2
    assert all(link.ocr_text_url for link in links)


def test_multiple_urls_on_same_line_both_extracted(tmp_path: Path) -> None:
    doc_md = tmp_path / "doc.md"
    doc_md.write_text("https://example.org/a and https://example.org/b\n", encoding="utf-8")
    links = extract_printed_urls(doc_md)
    assert {link.url for link in links} == {"https://example.org/a", "https://example.org/b"}
