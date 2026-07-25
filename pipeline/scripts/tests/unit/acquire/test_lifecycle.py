"""Контур времени, сторона ACQUIRE — spec post-acquisition-lifecycle.

Захват серверных валидаторов (§2), проактивный снимок SavePageNow (§4) и чистый
классификатор recheck (§2). Сеть всюду замокана — тот же паттерн, что в
``test_acquisition.py``: реальных запросов ни один unit-тест не делает.
"""
from __future__ import annotations

from acquire import acquisition
from core import schema


def _headers(*extra: str, status: int = 200) -> str:
    lines = [f"HTTP/1.1 {status} OK", "Content-Type: application/pdf", *extra]
    return "\r\n".join(lines) + "\r\n"


# --- §2: валидаторы снимаются из уже разобранного заголовочного дампа ---


def test_classify_response_captures_etag_and_last_modified() -> None:
    """Ноль новых запросов: заголовки и без того парсятся ради WAF-детекции."""
    result = acquisition.classify_response(
        b"%PDF-1.7 body",
        _headers('ETag: "abc123"', "Last-Modified: Wed, 21 Oct 2026 07:28:00 GMT"),
    )
    assert result.outcome is acquisition.AcquisitionOutcome.ok
    assert result.etag == '"abc123"'
    assert result.last_modified == "Wed, 21 Oct 2026 07:28:00 GMT"


def test_classify_response_without_validators_gives_none() -> None:
    """Гос-порталы часто не отдают валидаторов вовсе — это штатное состояние, не отказ."""
    result = acquisition.classify_response(b"%PDF-1.7 body", _headers())
    assert result.outcome is acquisition.AcquisitionOutcome.ok
    assert result.etag is None and result.last_modified is None


def test_classify_response_captures_validators_on_blocked_too() -> None:
    """Валидаторы не участвуют в самой классификации: спорный ответ остаётся спорным,
    а поля заполняются единообразно (решает, писать ли их, вызывающая сторона)."""
    result = acquisition.classify_response(
        b"<html>Attention Required</html>", _headers('ETag: "waf"'), expected=schema.SourceFormat.html
    )
    assert result.outcome is acquisition.AcquisitionOutcome.blocked
    assert result.etag == '"waf"'


def test_classify_response_validators_from_last_redirect_hop() -> None:
    """``_headers_from_text`` держит только последний hop — валидатор редиректа (301)
    не должен подменить валидатор финального документа."""
    headers = (
        'HTTP/1.1 301 Moved\r\nETag: "redirect-hop"\r\n'
        'HTTP/1.1 200 OK\r\nETag: "final-doc"\r\n'
    )
    result = acquisition.classify_response(b"%PDF-1.7 body", headers)
    assert result.etag == '"final-doc"'


def test_validator_capture_rungs_are_publisher_owned_only() -> None:
    """Гейт §2 — не деталь реализации, а контракт: archive/browser/manual выдают
    заголовки, которые издателю не принадлежат, и попадание их в state сделало бы
    каждую будущую проверку сравнением мусора с мусором."""
    assert acquisition.VALIDATOR_CAPTURE_RUNGS == {
        schema.AcquisitionMethod.direct,
        schema.AcquisitionMethod.official_alt,
    }


def test_browser_rung_synthetic_headers_carry_no_validators() -> None:
    """Ступень browser синтезирует заголовки сама — валидаторов там нет by construction
    (проверяем на том же синтетическом блоке, который строит ``_try_browser_rung``)."""
    result = acquisition.classify_response(
        b"<html>" + b"x" * 2000 + b"</html>",
        "HTTP/1.1 200\r\nContent-Type: text/html; charset=utf-8\r\n",
        expected=schema.SourceFormat.html,
    )
    assert result.outcome is acquisition.AcquisitionOutcome.ok
    assert result.etag is None and result.last_modified is None
