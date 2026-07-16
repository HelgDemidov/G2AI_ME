"""Тесты pdf_to_markdown: per-page высота/ширина (смешанная ориентация), guard на пустой PDF."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from convert.pdf_to_markdown import Word, compute_doc_stats, convert


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
