"""Контур времени, сторона ACQUIRE — spec post-acquisition-lifecycle.

Захват серверных валидаторов (§2), проактивный снимок SavePageNow (§4) и чистый
классификатор recheck (§2). Сеть всюду замокана — тот же паттерн, что в
``test_acquisition.py``: реальных запросов ни один unit-тест не делает.
"""
from __future__ import annotations

import subprocess
from typing import Any

from acquire import acquisition
from core import schema

# Снято ДО autouse-заглушки герметичности (tests/unit/conftest.py подменяет
# ``acquisition.request_snapshot``, чтобы ни один тест добычи не ходил в сеть) —
# тестам САМОЙ функции нужна настоящая реализация, а не заглушка.
_REAL_REQUEST_SNAPSHOT = acquisition.request_snapshot


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


# --- AWS WAF: челлендж без тела (находка живой приёмки 2026-07-25) ---

_AWS_CHALLENGE_HEADERS = (
    "HTTP/1.1 202 Accepted\r\nServer: CloudFront\r\nContent-Length: 0\r\n"
    "x-amzn-waf-action: challenge\r\nContent-Type: text/html; charset=UTF-8\r\n"
)


def test_aws_waf_challenge_detected_despite_empty_body() -> None:
    """eur-lex.europa.eu переехал за CloudFront: HTTP 202, ПУСТОЕ тело, действие WAF
    объявлено заголовком. Маркеры тела при таком ответе сработать не могут в принципе,
    поэтому без отпечатка заголовка причина писалась бы как «unexpected content» —
    исход верный, диагноз ложный, а он оседает в .state.yaml для куратора."""
    result = acquisition.classify_response(b"", _AWS_CHALLENGE_HEADERS, expected=schema.SourceFormat.html)
    assert result.outcome is acquisition.AcquisitionOutcome.blocked
    assert result.reason == "WAF challenge signature detected"


def test_aws_waf_header_without_challenge_action_is_not_a_fingerprint() -> None:
    """Проверяется ЗНАЧЕНИЕ, а не наличие заголовка — тот же урок, что с `Server: BigIP`
    (ложноположителен на реальном контенте нескольких гос-сайтов)."""
    headers = (
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nx-amzn-waf-action: allow\r\n"
    )
    result = acquisition.classify_response(
        b"<html>" + b"x" * 2000 + b"</html>", headers, expected=schema.SourceFormat.html
    )
    assert result.outcome is acquisition.AcquisitionOutcome.ok


def test_aws_waf_challenge_detected_for_pdf_records_too() -> None:
    result = acquisition.classify_response(b"", _AWS_CHALLENGE_HEADERS)
    assert result.outcome is acquisition.AcquisitionOutcome.blocked
    assert result.reason == "WAF challenge signature detected"


def test_aws_waf_challenge_detected_by_format_agnostic_probe() -> None:
    """Популяция (c) recheck пользуется тем же знанием — иначе недобываемый кандидат
    за AWS WAF репортился бы «неожиданный ответ», а не «канал закрыт»."""
    result = acquisition.classify_probe(b"", _AWS_CHALLENGE_HEADERS)
    assert result.outcome is acquisition.AcquisitionOutcome.blocked
    assert result.reason == "WAF challenge signature detected"


# --- §4: SavePageNow — fire-and-forget, никогда не исключение ---


def _capture_curl(monkeypatch: Any, *, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_request_snapshot_hits_savepagenow_endpoint(monkeypatch: Any) -> None:
    calls = _capture_curl(monkeypatch)
    assert _REAL_REQUEST_SNAPSHOT("https://example.gov/strategy.pdf") is True
    assert calls[0][-1] == "https://web.archive.org/save/https://example.gov/strategy.pdf"
    assert "--max-time" in calls[0]  # снимок — страховка, а не часть критического пути


def test_request_snapshot_returns_false_on_curl_error(monkeypatch: Any) -> None:
    """Отказ Wayback не обязан ничего ронять: ретраев нет, гарантии нет — долговечность
    держит локальный raw, а не этот вызов."""
    _capture_curl(monkeypatch, returncode=28)
    assert _REAL_REQUEST_SNAPSHOT("https://example.gov/doc.pdf") is False


def test_request_snapshot_quotes_url_with_space(monkeypatch: Any) -> None:
    """spec acquire-convert-seam-hardening §7, В9-код: пробел/не-URL-символ в
    query-строке ломал бы итоговый SPN-запрос без квотинга — argv-элемент, не
    shell, поэтому эффект «отказ», не инъекция, но всё равно ломающий страховку."""
    calls = _capture_curl(monkeypatch)
    _REAL_REQUEST_SNAPSHOT("https://example.gov/doc?title=two words")
    assert calls[0][-1] == "https://web.archive.org/save/https://example.gov/doc?title=two%20words"


def test_request_snapshot_preserves_plain_url_unchanged(monkeypatch: Any) -> None:
    """Квотинг сохраняет структурные символы САМОГО url (:/?&=#%) — типовой URL без
    пробелов проходит байт-в-байт, регресс-guard на test_request_snapshot_hits_savepagenow_endpoint."""
    calls = _capture_curl(monkeypatch)
    _REAL_REQUEST_SNAPSHOT("https://example.gov/path?a=1&b=2#frag")
    assert calls[0][-1] == "https://web.archive.org/save/https://example.gov/path?a=1&b=2#frag"


def test_request_snapshot_survives_missing_curl(monkeypatch: Any) -> None:
    def boom(cmd: list[str], **kw: Any) -> None:
        raise OSError("curl не найден")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _REAL_REQUEST_SNAPSHOT("https://example.gov/doc.pdf") is False


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
