"""Acquisition ladder for ``source_url`` downloads: block detection, ladder
routing, manual watch-folder ingestion, and Wayback archive fallback.

See ``pipeline/setup/source-acquisition-ladder/spec.md``. This module holds
the WAF-aware acquisition logic; ``run_pipeline.py`` stays a thin orchestrator
that calls into it, same separation as ``build_graph.py``/``corpus_index.py``.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

import schema


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
    """The ladder's automatic rungs (direct/official_alt) are exhausted by a WAF
    block (see §3/§4 of the spec) — manual acquisition is needed.
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"manual acquisition needed for {url}: {reason}")


class AcquisitionDead(RuntimeError):
    """``source_url`` is confirmed gone (404/410) — archive fallback is needed,
    or manual if the record is ``sensitivity: confidential`` (§9, archive unavailable).
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"archive acquisition needed for {url}: {reason}")


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


# --- ladder routing (§2/§4/§9 of the spec) ---
_FIDELITY_BY_AUTOMATIC_RUNG = {
    schema.AcquisitionMethod.direct: schema.Fidelity.live,
    schema.AcquisitionMethod.official_alt: schema.Fidelity.rehost,
}


def next_rung(
    outcome: AcquisitionOutcome,
    rung: schema.AcquisitionMethod,
    *,
    has_official_alt: bool,
    sensitivity: schema.Sensitivity,
) -> schema.AcquisitionMethod | None:
    """Given the outcome of attempting ``rung``, decide the next ladder rung.

    Returns ``None`` if ``rung`` succeeded. See spec §2 (block -> manual is
    terminal for "alive but blocked"; dead -> archive is terminal for "gone"),
    §4 (official_alt is a block-bypass tried once, right after direct), and
    §9 (confidential records never reach archive, even when dead).
    """
    if outcome is AcquisitionOutcome.ok:
        return None
    if outcome is AcquisitionOutcome.dead:
        if sensitivity is schema.Sensitivity.confidential:
            return schema.AcquisitionMethod.manual
        return schema.AcquisitionMethod.archive
    # blocked
    if rung is schema.AcquisitionMethod.direct and has_official_alt:
        return schema.AcquisitionMethod.official_alt
    return schema.AcquisitionMethod.manual


@dataclass
class LadderResult:
    method: schema.AcquisitionMethod
    fidelity: schema.Fidelity
    classified: ClassifiedResponse


def run_ladder(rec: schema.SourceRecord, dest: Path, *, user_agent: str) -> LadderResult:
    """Automatic portion of the ladder: try ``direct``, then ``official_alt`` once
    if blocked and available. Always starts fresh from ``direct`` — the ladder
    deliberately does not cache "known blocked" across runs in this version
    (§5: WAF state can flip either way; curl's own ``--retry`` already avoids
    hammering a *transient* failure, and a block/dead classification is a
    single-shot signal, not something curl retries internally).

    Raises ``AcquisitionBlocked`` when manual acquisition is needed, or
    ``AcquisitionDead`` when archive fallback is needed. Does not itself
    perform manual/archive acquisition or persist anything to ``sources.yaml``
    — those are the caller's job (see run_pipeline.py and commits 4/5/6).
    """
    has_alt = bool(rec.official_alt_url)
    rung = schema.AcquisitionMethod.direct
    classified = fetch_and_classify(rec.source_url, dest, user_agent=user_agent)

    while True:
        nxt = next_rung(classified.outcome, rung, has_official_alt=has_alt, sensitivity=rec.sensitivity)
        if nxt is None:
            return LadderResult(rung, _FIDELITY_BY_AUTOMATIC_RUNG[rung], classified)
        if nxt is schema.AcquisitionMethod.official_alt:
            assert rec.official_alt_url is not None  # has_alt guarantees this
            rung = nxt
            classified = fetch_and_classify(rec.official_alt_url, dest, user_agent=user_agent)
            continue
        if nxt is schema.AcquisitionMethod.manual:
            if classified.outcome is AcquisitionOutcome.dead:
                raise AcquisitionBlocked(
                    rec.source_url,
                    f"{rung.value} confirmed dead ({classified.reason}) but sensitivity=confidential — archive unavailable",
                )
            raise AcquisitionBlocked(rec.source_url, f"{rung.value} blocked ({classified.reason})")
        raise AcquisitionDead(rec.source_url, f"{rung.value} confirmed dead ({classified.reason})")


# --- persisting acquisition state back to sources.yaml (round-trip-safe) ---
def persist_acquisition_state(
    sources_path: Path,
    record_id: str,
    *,
    acquisition_method: schema.AcquisitionMethod,
    fidelity: schema.Fidelity,
    checked: _dt.date,
    retrieved_snapshot_date: _dt.date | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Point-patch acquisition_method/acquisition_checked/fidelity(/retrieved_snapshot_date)
    for one record in ``sources.yaml``, leaving everything else in the file byte-identical.

    Uses ``ruamel.yaml`` round-trip mode, NOT plain PyYAML: a naive
    load-then-``yaml.safe_dump`` would silently drop the file's top comment
    header and reformat multi-line ``summary``/``notes`` fields (PyYAML's data
    model has no concept of comments or original scalar style). Atomic write
    (tmp -> rename), same pattern as the ``.md`` frontmatter sync.

    Returns ``{field: (old_value, new_value)}`` for fields that actually
    changed — empty dict if nothing needed updating (caller uses this to
    print a summary rather than writing silently; see spec §"Доработки
    оркестратора").
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    with sources_path.open("r", encoding="utf-8") as fh:
        data = yaml.load(fh)

    record = next((item for item in data if item.get("id") == record_id), None)
    if record is None:
        raise ValueError(f"запись '{record_id}' не найдена в {sources_path}")

    patch: dict[str, Any] = {
        "acquisition_method": acquisition_method.value,
        "acquisition_checked": checked,
        "fidelity": fidelity.value,
    }
    if retrieved_snapshot_date is not None:
        patch["retrieved_snapshot_date"] = retrieved_snapshot_date

    changed: dict[str, tuple[Any, Any]] = {}
    for key, new_value in patch.items():
        old_value = record.get(key)
        if old_value != new_value:
            changed[key] = (old_value, new_value)
            record[key] = new_value

    if not changed:
        return changed

    tmp = sources_path.with_name(sources_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    tmp.replace(sources_path)
    return changed
