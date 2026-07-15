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
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pdfplumber
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


class ManualAcquisitionTimeout(RuntimeError):
    """No valid, unambiguous candidate appeared in the watch window before the deadline."""


class ManualAcquisitionConflict(RuntimeError):
    """More than one valid candidate appeared in the watch window — not guessing which is right."""


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


# --- manual watch-folder ingestion (§6 of the spec) ---
DEFAULT_MANUAL_TIMEOUT_SECONDS = 300.0  # разумный запас на клик + докачку одного PDF
DEFAULT_POLL_INTERVAL_SECONDS = 1.0


def default_watch_dir() -> Path:
    """Системная папка загрузок пользователя, через ``xdg-user-dir DOWNLOAD``
    (не хардкодить строку — локаль-зависима, см. §6: на этой машине она кириллическая).
    """
    try:
        result = subprocess.run(
            ["xdg-user-dir", "DOWNLOAD"], capture_output=True, text=True, check=True
        )
        resolved = result.stdout.strip()
        if resolved:
            return Path(resolved)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return Path.home() / "Downloads"  # разумный fallback без xdg-user-dir


def _looks_like_candidate_pdf(path: Path) -> tuple[bool, str]:
    """Дешёвая магия сначала (не читаем весь файл ради больших не-PDF в папке
    загрузок), pdfplumber — только если магия совпала. Возвращает (валиден, причина).
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
    except OSError as exc:
        return False, f"не удалось прочитать: {exc}"
    if not head.startswith(b"%PDF"):
        return False, "не PDF (нет магии %PDF)"
    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
    except Exception as exc:  # noqa: BLE001 — битый/недописанный PDF, не наш кандидат (пока)
        return False, f"PDF повреждён или ещё не дописан: {exc}"
    if n_pages < 1:
        return False, "0 страниц"
    return True, f"{n_pages} стр."


def watch_and_ingest(
    dest: Path,
    *,
    watch_dir: Path,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_MANUAL_TIMEOUT_SECONDS,
) -> Path:
    """Следить за ``watch_dir`` до появления ОДНОГО валидного PDF-кандидата, перенести
    его в ``dest``. Первая же итерация — это и есть начальное сканирование уже
    лежащих файлов (резюмируемость после прерванного прогона — §6), отдельного
    "initial scan" шага не требуется по построению.

    НЕ кеширует отказы кандидатов между итерациями: недописанный браузером файл
    должен быть перепроверен на следующем цикле, а не забыт навсегда. Батч
    (>1 валидных кандидатов одновременно) не разрешается угадыванием — конфликт.

    ``now``/``sleep`` инжектируемы ради детерминированных тестов (без реального sleep).
    """
    deadline = now() + timeout
    while True:
        candidates = [
            path
            for path in sorted(watch_dir.iterdir())
            if path.is_file() and _looks_like_candidate_pdf(path)[0]
        ]
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            raise ManualAcquisitionConflict(
                f"{len(candidates)} валидных кандидата одновременно в {watch_dir}: {names} — разрешите вручную"
            )
        if len(candidates) == 1:
            dest.parent.mkdir(parents=True, exist_ok=True)
            candidates[0].replace(dest)
            return dest
        if now() >= deadline:
            raise ManualAcquisitionTimeout(f"не дождался файла в {watch_dir} за {timeout:.0f}с")
        sleep(poll_interval)


def acquire_manually(
    rec: schema.SourceRecord,
    dest: Path,
    *,
    watch_dir: Path | None = None,
    timeout: float = DEFAULT_MANUAL_TIMEOUT_SECONDS,
) -> LadderResult:
    """1-клик путь: открыть ``rec.source_url`` в браузере пользователя, дождаться
    файла в watch-папке, перенести в ``dest``. Сбой ``xdg-open`` не фатален —
    URL уже напечатан вызывающей стороной, пользователь может кликнуть сам.
    """
    subprocess.run(["xdg-open", rec.source_url], check=False)
    watch_and_ingest(dest, watch_dir=watch_dir or default_watch_dir(), timeout=timeout)
    return LadderResult(
        schema.AcquisitionMethod.manual,
        schema.Fidelity.manual,
        ClassifiedResponse(AcquisitionOutcome.ok, None, "manual acquisition via watch-folder"),
    )
