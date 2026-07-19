"""Тесты реестра HTML-препроцессоров (B1, spec convert-hardening): first-match
диспетчеризация, тест-реестр подхватывается без правок ядра (паттерн test-реестра
конвертеров)."""
from __future__ import annotations

from typing import Any

from convert import html_preprocess

_ELI_FIXTURE = (
    b"<html><body><div id=\"cpt_I\"><p class=\"oj-ti-section-1\">CHAPTER I</p>"
    b"<p class=\"oj-normal\">Body text.</p></div></body></html>"
)
_NON_ELI_FIXTURE = b"<html><body><p>Just a regular paragraph.</p></body></html>"


def test_eli_fixture_is_transformed() -> None:
    out = html_preprocess.apply(_ELI_FIXTURE)
    assert b"<h1>" in out


def test_non_eli_html_returned_byte_for_byte() -> None:
    assert html_preprocess.apply(_NON_ELI_FIXTURE) == _NON_ELI_FIXTURE


def test_first_matching_preprocessor_wins(monkeypatch: Any) -> None:
    """Fake-препроцессор, зарегистрированный тестом, подхватывается БЕЗ правок
    ядра — доказывает, что apply() формат-агностичен и работает через реестр,
    не через хардкод eli."""
    calls: list[str] = []

    def fake_matcher(html: bytes) -> bool:
        return b"FAKE-PORTAL" in html

    def fake_transform(html: bytes) -> bytes:
        calls.append("fake")
        return b"TRANSFORMED"

    monkeypatch.setattr(
        html_preprocess,
        "_PREPROCESSORS",
        [("fake", fake_matcher, fake_transform), *html_preprocess._PREPROCESSORS],
    )
    out = html_preprocess.apply(b"<html>FAKE-PORTAL content</html>")
    assert out == b"TRANSFORMED"
    assert calls == ["fake"]


def test_non_matching_fake_preprocessor_falls_through_to_eli(monkeypatch: Any) -> None:
    def fake_matcher(html: bytes) -> bool:
        return False

    def fake_transform(html: bytes) -> bytes:
        return b"SHOULD NOT BE CALLED"

    monkeypatch.setattr(
        html_preprocess,
        "_PREPROCESSORS",
        [("fake", fake_matcher, fake_transform), *html_preprocess._PREPROCESSORS],
    )
    out = html_preprocess.apply(_ELI_FIXTURE)
    assert b"<h1>" in out
