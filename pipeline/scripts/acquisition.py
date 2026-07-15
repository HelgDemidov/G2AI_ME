"""Acquisition ladder for ``source_url`` downloads: block detection, ladder
routing, manual watch-folder ingestion, and Wayback archive fallback.

See ``pipeline/setup/source-acquisition-ladder/spec.md``. This module holds
the WAF-aware acquisition logic; ``run_pipeline.py`` stays a thin orchestrator
that calls into it, same separation as ``build_graph.py``/``corpus_index.py``.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AcquisitionOutcome(str, Enum):
    """Result of classifying a single download attempt."""

    ok = "ok"
    blocked = "blocked"
    dead = "dead"


@dataclass
class ClassifiedResponse:
    outcome: AcquisitionOutcome
    http_status: int | None
    reason: str


class AcquisitionBlocked(RuntimeError):
    """A direct/official_alt attempt hit a WAF/challenge signature (see §3 of the spec)."""


class AcquisitionDead(RuntimeError):
    """A direct/official_alt attempt confirmed the URL no longer resolves (404/410)."""


# Challenge pages we've observed (ai.gov.ae/Cloudflare) are small HTML, well under this.
MIN_EXPECTED_PDF_SIZE = 2048
CHALLENGE_BODY_MARKERS = (b"Attention Required", b"cf-chl", b"Just a moment", b"cf_captcha", b"turnstile")
DEAD_STATUS_CODES = {404, 410}


def _status_from_headers_text(headers_text: str) -> int | None:
    """Last ``HTTP/...`` status line wins (curl -D with -L appends one block per redirect hop)."""
    status: int | None = None
    for line in headers_text.splitlines():
        if line.upper().startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                status = int(parts[1])
    return status


def _headers_from_text(headers_text: str) -> dict[str, str]:
    """Parse a curl -D dump, keeping only the final hop's header block."""
    headers: dict[str, str] = {}
    for line in headers_text.splitlines():
        if line.upper().startswith("HTTP/"):
            headers = {}  # new hop -> reset, keep only the last block
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return headers


def _has_cloudflare_fingerprint(headers: dict[str, str]) -> bool:
    if "cf-ray" in headers:
        return True
    return "__cf_bm" in headers.get("set-cookie", "")


def classify_response(
    body: bytes, headers_text: str, *, expect_pdf: bool = True
) -> ClassifiedResponse:
    """Pure classification: given a response body and raw ``curl -D`` header dump,
    decide whether the download is a real document, a WAF block, or a dead URL.
    """
    status = _status_from_headers_text(headers_text)
    headers = _headers_from_text(headers_text)

    if status in DEAD_STATUS_CODES:
        return ClassifiedResponse(AcquisitionOutcome.dead, status, f"HTTP {status}")

    if expect_pdf and body.startswith(b"%PDF"):
        return ClassifiedResponse(AcquisitionOutcome.ok, status, "valid PDF")

    if _has_cloudflare_fingerprint(headers) or any(m in body for m in CHALLENGE_BODY_MARKERS):
        return ClassifiedResponse(AcquisitionOutcome.blocked, status, "WAF challenge signature detected")

    if len(body) < MIN_EXPECTED_PDF_SIZE:
        return ClassifiedResponse(AcquisitionOutcome.blocked, status, "response too small to be the expected document")

    return ClassifiedResponse(AcquisitionOutcome.blocked, status, "unexpected content (not a valid PDF)")


def fetch_and_classify(
    url: str, dest: Path, *, user_agent: str, timeout: int = 30
) -> ClassifiedResponse:
    """Single download attempt (no ladder stepping — that's the caller's job).

    Deliberately omits ``-f``: a hard HTTP error (403/404) must still land its
    status/body so ``classify_response`` can tell a block apart from a dead URL —
    with ``-f`` curl discards the response before we ever see it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="acq-headers-", suffix=".txt", delete=False) as tmp:
        headers_path = Path(tmp.name)
    try:
        cmd = [
            "curl", "-sSL", "--retry", "3", "--retry-delay", "2",
            "--connect-timeout", str(timeout), "-A", user_agent,
            "-D", str(headers_path), "-o", str(dest), url,
        ]
        subprocess.run(cmd, check=True)
        headers_text = headers_path.read_text(encoding="utf-8", errors="replace")
        body = dest.read_bytes() if dest.exists() else b""
        return classify_response(body, headers_text)
    finally:
        headers_path.unlink(missing_ok=True)
