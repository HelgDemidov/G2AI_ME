"""Тесты лестницы добычи: детекция блока (без реальной сети — файлы/subprocess подделываются)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from acquisition import AcquisitionOutcome, classify_response, fetch_and_classify

REAL_PDF_BODY = b"%PDF-1.6\n" + b"x" * 4000  # больше MIN_EXPECTED_PDF_SIZE

CLOUDFLARE_BLOCK_HEADERS = (
    "HTTP/2 403 \n"
    "date: Wed, 15 Jul 2026 15:43:50 GMT\n"
    "content-type: text/html; charset=UTF-8\n"
    "cf-ray: a1b9e294cfd4e293-BEG\n"
    "server: cloudflare\n"
)
CLOUDFLARE_BLOCK_BODY = b"<html><body>Sorry, you have been blocked</body></html>"

CF_COOKIE_ONLY_HEADERS = (
    "HTTP/2 200 \n"
    "content-type: text/html; charset=UTF-8\n"
    "set-cookie: __cf_bm=0Cmh0PjncfxIOzhyqGyzrl7s; HttpOnly; SameSite=None; Secure\n"
)

DEAD_404_HEADERS = "HTTP/1.1 404 Not Found\ncontent-type: text/html\n"
DEAD_410_HEADERS = "HTTP/1.1 410 Gone\ncontent-type: text/html\n"

OK_HEADERS = "HTTP/1.1 200 OK\ncontent-type: application/pdf\n"

REDIRECT_THEN_BLOCK_HEADERS = (
    "HTTP/1.1 301 Moved Permanently\n"
    "location: https://ai.gov.ae/final.pdf\n"
    "\n"
    "HTTP/2 403 \n"
    "cf-ray: a1b9e294cfd4e293-BEG\n"
    "content-type: text/html\n"
)


def test_classify_valid_pdf_is_ok() -> None:
    result = classify_response(REAL_PDF_BODY, OK_HEADERS)
    assert result.outcome == AcquisitionOutcome.ok
    assert result.http_status == 200


def test_classify_cloudflare_block_via_cf_ray_header() -> None:
    result = classify_response(CLOUDFLARE_BLOCK_BODY, CLOUDFLARE_BLOCK_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked
    assert result.http_status == 403


def test_classify_cloudflare_block_via_cookie_only() -> None:
    # 200 + __cf_bm cookie + не-PDF тело -> тоже блок (сама Cloudflare-кука уже отпечаток,
    # даже без cf-ray в этом конкретном ответе).
    result = classify_response(b"<html>checking your browser</html>", CF_COOKIE_ONLY_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked


def test_classify_challenge_body_marker() -> None:
    body = b"<html><h1>Attention Required! | Cloudflare</h1></html>"
    result = classify_response(body, CLOUDFLARE_BLOCK_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked
    assert "WAF" in result.reason or "challenge" in result.reason


def test_classify_dead_404() -> None:
    result = classify_response(b"<html>not found</html>", DEAD_404_HEADERS)
    assert result.outcome == AcquisitionOutcome.dead
    assert result.http_status == 404


def test_classify_dead_410() -> None:
    result = classify_response(b"", DEAD_410_HEADERS)
    assert result.outcome == AcquisitionOutcome.dead
    assert result.http_status == 410


def test_classify_small_unexpected_body_is_blocked() -> None:
    # 200, не PDF, без явных challenge-маркеров, но подозрительно маленький -> блок, не "ok".
    result = classify_response(b"tiny", OK_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked


def test_classify_redirect_keeps_only_final_hop_status_and_headers() -> None:
    # Финальный статус (403) должен победить промежуточный 301; cf-ray виден только
    # в финальном блоке заголовков -> детектор не должен потеряться в редиректе.
    result = classify_response(CLOUDFLARE_BLOCK_BODY, REDIRECT_THEN_BLOCK_HEADERS)
    assert result.http_status == 403
    assert result.outcome == AcquisitionOutcome.blocked


def test_fetch_and_classify_omits_dash_f_and_captures_headers(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """curl вызывается БЕЗ -f (иначе тело/заголовки блока/dead-ответа теряются —
    см. решение №1 спека) и с -D для заголовков.
    """
    captured_cmd: list[str] = []

    def fake_run(cmd: list[str], check: bool) -> None:  # noqa: FBT001 — сигнатура subprocess.run
        captured_cmd.extend(cmd)
        d_index = cmd.index("-D")
        o_index = cmd.index("-o")
        Path(cmd[d_index + 1]).write_text(CLOUDFLARE_BLOCK_HEADERS, encoding="utf-8")
        Path(cmd[o_index + 1]).write_bytes(CLOUDFLARE_BLOCK_BODY)

    monkeypatch.setattr("acquisition.subprocess.run", fake_run)
    dest = tmp_path / "doc.pdf"
    result = fetch_and_classify("https://ai.gov.ae/doc.pdf", dest, user_agent="test-ua")

    assert "-f" not in captured_cmd
    assert "-D" in captured_cmd
    assert result.outcome == AcquisitionOutcome.blocked
    assert result.http_status == 403
