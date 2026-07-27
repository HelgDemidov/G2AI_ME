"""Acquisition ladder for ``source_url`` downloads: block detection, ladder
routing, manual watch-folder ingestion, and Wayback archive fallback.

See ``docs/pipeline/acquire/tech_specs/source-acquisition-ladder/spec.md``. This module holds
the WAF-aware acquisition logic; ``run_pipeline.py`` stays a thin orchestrator
that calls into it, same separation as ``build_graph.py``/``corpus_index.py``.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import quote

import pdfplumber

from core import browser_resolver, schema

logger = logging.getLogger("acquisition")


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
    # Серверные валидаторы ответа verbatim (spec post-acquisition-lifecycle §2).
    # Заполняет ``classify_response`` из уже разобранных заголовков — ноль новых
    # запросов: добыча и без того парсит ``curl -D``-дамп. Пути, не имеющие
    # настоящего HTTP-ответа (watch-folder, синтетические заголовки браузерной
    # ступени), оставляют None по построению.
    etag: str | None = None
    last_modified: str | None = None


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
MIN_EXPECTED_HTML_SIZE = 1024
DOCX_MAGIC = b"PK\x03\x04"  # docx = zip-контейнер (OOXML), сигнатура ZIP local-file-header
MIN_EXPECTED_DOCX_SIZE = 4096
XLSX_MAGIC = DOCX_MAGIC  # xlsx — тот же OOXML/zip-контейнер, та же сигнатура
MIN_EXPECTED_XLSX_SIZE = 4096  # тот же порядок величины: пустая .xlsx тоже несёт zip+styles+theme overhead
# F5/BigIP и Akamai-сигнатуры (спек aiforgood-standards §5, ОБЯЗАТЕЛЬНЫЙ фикс): размерный
# порог выше — совпадение, не гарантия (F5-заглушка mcit.gov.sa: HTTP 200, text/html,
# 42 936 Б — на два порядка крупнее MIN_EXPECTED_HTML_SIZE, проходила бы как валидный HTML
# без сигнатуры тела). НЕ детектировать по заголовку Server: BigIP — ложноположителен
# (mcit.gov.eg/tdra.gov.ae отдают тот же заголовок с реальным контентом).
CHALLENGE_BODY_MARKERS = (
    b"Attention Required", b"cf-chl", b"Just a moment", b"cf_captcha", b"turnstile",
    b"Request Rejected", b"Unauthorized Access", b"errors.edgesuite.net",
)
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


# AWS WAF (CloudFront) объявляет своё действие ОТДЕЛЬНЫМ заголовком и отдаёт челлендж
# ПУСТЫМ телом с кодом 202 — маркеры тела при таком ответе не могут сработать в
# принципе. Найдено живьём 2026-07-25 на eur-lex.europa.eu: `HTTP/1.1 202 Accepted`,
# `Content-Length: 0`, `x-amzn-waf-action: challenge`. Значение проверяется, а не сам
# факт наличия заголовка — тот же урок, что с `Server: BigIP` (ложноположителен на
# реальном контенте).
_AWS_WAF_ACTIONS = frozenset({"challenge", "captcha"})


def _has_waf_header_fingerprint(headers: dict[str, str]) -> bool:
    if "cf-ray" in headers:
        return True
    if "__cf_bm" in headers.get("set-cookie", ""):
        return True
    return headers.get("x-amzn-waf-action", "").strip().lower() in _AWS_WAF_ACTIONS


def classify_response(
    body: bytes, headers_text: str, expected: schema.SourceFormat = schema.SourceFormat.pdf
) -> ClassifiedResponse:
    """Pure classification: given a response body and raw ``curl -D`` header dump,
    decide whether the download is a real document, a WAF block, or a dead URL.

    ``expected`` dispatches to a format-specific ok-branch (§3.2 of the convert
    chartier) via ``_FORMAT_PROFILES`` (spec acquire-convert-seam-hardening §6, В6:
    a registry mirroring convert's ``_CONVERTERS``, not an if/elif chain — a new
    format is one dict entry, not edits scattered across the module); ``expected=pdf``
    is the default so every pre-existing caller keeps its exact behaviour unchanged.
    """
    status = _status_from_headers_text(headers_text)
    headers = _headers_from_text(headers_text)

    if status in DEAD_STATUS_CODES:
        result = ClassifiedResponse(AcquisitionOutcome.dead, status, f"HTTP {status}")
    else:
        result = _FORMAT_PROFILES[expected].classify(body, headers, status)

    # Валидаторы навешиваются ЗДЕСЬ, а не в каждой формат-ветке: они не зависят от
    # формата и не участвуют в самой классификации (спорный ответ остаётся спорным
    # независимо от того, прислал ли сервер ETag).
    result.etag = headers.get("etag")
    result.last_modified = headers.get("last-modified")
    return result


def _format_mismatch_note(headers: dict[str, str], expected: schema.SourceFormat) -> str:
    """Диагноз частого промаха лестницы (spec discovery-acquire-seam-hardening §8,
    Г7): сервер вернул HTML, когда ожидался другой формат — типичный признак
    рассинхрона курируемого ``source_format`` с реальным содержимым (документ на
    самом деле HTML-страница), а не WAF-блока (тот уже отловлен маркерами тела/
    заголовков раньше по коду). Зеркало уже существующего диагноза
    ``_classify_html`` для обратного случая («сервер отдал PDF при
    source_format=html»). Пустая строка — не наш случай, ничего не добавляем;
    только СТРОКА-ПРИЧИНА меняется, исход классификации (``blocked``) — тот же."""
    if "text/html" not in headers.get("content-type", ""):
        return ""
    return f" — сервер отдал HTML при ожидании {expected.value} (проверьте source_format: ошибка формата?)"


def _classify_pdf(body: bytes, headers: dict[str, str], status: int | None) -> ClassifiedResponse:
    if body.startswith(b"%PDF"):
        return ClassifiedResponse(AcquisitionOutcome.ok, status, "valid PDF")

    if _has_waf_header_fingerprint(headers) or any(m in body for m in CHALLENGE_BODY_MARKERS):
        return ClassifiedResponse(AcquisitionOutcome.blocked, status, "WAF challenge signature detected")

    if len(body) < MIN_EXPECTED_PDF_SIZE:
        return ClassifiedResponse(
            AcquisitionOutcome.blocked, status,
            "response too small to be the expected document"
            + _format_mismatch_note(headers, schema.SourceFormat.pdf),
        )

    return ClassifiedResponse(
        AcquisitionOutcome.blocked, status,
        "unexpected content (not a valid PDF)" + _format_mismatch_note(headers, schema.SourceFormat.pdf),
    )


def _classify_docx(body: bytes, headers: dict[str, str], status: int | None) -> ClassifiedResponse:
    """spec convert-docx §3. Zip-магия (``DOCX_MAGIC``) — необходимое, НЕ достаточное
    условие (любой zip пройдёт эту проверку) — терминальная страховка на
    неразличимость от честного docx здесь: mammoth/markdownify поднимут
    ``ConversionError`` при конвертации не-docx zip'а (см. ``convert/converters._convert_docx``)."""
    if _has_waf_header_fingerprint(headers) or any(m in body for m in CHALLENGE_BODY_MARKERS):
        return ClassifiedResponse(AcquisitionOutcome.blocked, status, "WAF challenge signature detected")

    if status == 200 and body.startswith(DOCX_MAGIC) and len(body) >= MIN_EXPECTED_DOCX_SIZE:
        return ClassifiedResponse(AcquisitionOutcome.ok, status, "valid DOCX (zip magic)")

    return ClassifiedResponse(
        AcquisitionOutcome.blocked, status,
        "unexpected content (not the expected DOCX document)"
        + _format_mismatch_note(headers, schema.SourceFormat.docx),
    )


def _classify_xlsx(body: bytes, headers: dict[str, str], status: int | None) -> ClassifiedResponse:
    """spec convert-xlsx §5. Дословное зеркало ``_classify_docx`` — тот же
    контейнер OOXML/zip, та же терминальная страховка (zip-магия — необходимое,
    НЕ достаточное условие; ``openpyxl`` поднимет ``ConversionError`` при
    конвертации не-xlsx zip'а, см. ``convert/converters._convert_xlsx``)."""
    if _has_waf_header_fingerprint(headers) or any(m in body for m in CHALLENGE_BODY_MARKERS):
        return ClassifiedResponse(AcquisitionOutcome.blocked, status, "WAF challenge signature detected")

    if status == 200 and body.startswith(XLSX_MAGIC) and len(body) >= MIN_EXPECTED_XLSX_SIZE:
        return ClassifiedResponse(AcquisitionOutcome.ok, status, "valid XLSX (zip magic)")

    return ClassifiedResponse(
        AcquisitionOutcome.blocked, status,
        "unexpected content (not the expected XLSX document)"
        + _format_mismatch_note(headers, schema.SourceFormat.xlsx),
    )


def _classify_html(body: bytes, headers: dict[str, str], status: int | None) -> ClassifiedResponse:
    # WAF-challenge check comes BEFORE the content-type check: a challenge page
    # is itself served as "200 text/html" — a content-type-only check would wave
    # it straight through as "ok".
    if _has_waf_header_fingerprint(headers) or any(m in body for m in CHALLENGE_BODY_MARKERS):
        return ClassifiedResponse(AcquisitionOutcome.blocked, status, "WAF challenge signature detected")

    if body.startswith(b"%PDF"):
        return ClassifiedResponse(
            AcquisitionOutcome.blocked, status,
            "server returned PDF but source_format=html (curator mismatch?)",
        )

    content_type = headers.get("content-type", "")
    if status == 200 and "text/html" in content_type and len(body) >= MIN_EXPECTED_HTML_SIZE:
        return ClassifiedResponse(AcquisitionOutcome.ok, status, "valid HTML")

    return ClassifiedResponse(
        AcquisitionOutcome.blocked, status, "unexpected content (not the expected HTML document)"
    )


@dataclass(frozen=True)
class FormatProfile:
    """Формат-пакет добычи: диспетчер классификации + CDX-mimetype + флаги двух
    формат-специфичных ступеней лестницы, ОДНА запись на формат (spec
    acquire-convert-seam-hardening §6, В6) — зеркало convert-слоя ``_CONVERTERS``
    (Strategy-реестр по расширению raw.*). До этого спека формат-ось acquire-стороны
    была асимметрична: if/elif в ``classify_response`` + четыре отдельные россыпи
    (``MIN_EXPECTED_*``/магии в самих ``_classify_*``, ``_CDX_MIMETYPE_BY_FORMAT``,
    PDF-only гейт watch-folder, HTML-only гейт browser-ступени в ``run_ladder``) —
    новый формат требовал ~6 правок в разных местах модуля.

    Behavior-preserving by design: ``classify`` указывает на СУЩЕСТВУЮЩИЕ
    ``_classify_pdf/html/docx/xlsx``, которые НЕ нормализованы под общую форму —
    у pdf магия проверяется ДО WAF-отпечатка, у html/docx/xlsx WAF-отпечаток ПЕРВЫЙ
    (реальное поведение, не стилистическая разница; общая форма либо сломала бы
    его, либо обросла бы флагами до нечитаемости). Реестр лишь диспетчеризует
    существующие функции по формату, байт-идентично прежнему if/elif.
    """

    classify: Callable[[bytes, dict[str, str], int | None], ClassifiedResponse]
    cdx_mimetype: str
    watch_folder: bool  # manual watch-folder способен принять формат (v1: только pdf)
    browser_rung: bool  # рендер-дамп — легитимный финальный артефакт (v1: только html)


_FORMAT_PROFILES: dict[schema.SourceFormat, FormatProfile] = {
    schema.SourceFormat.pdf: FormatProfile(
        classify=_classify_pdf, cdx_mimetype="application/pdf",
        watch_folder=True, browser_rung=False,
    ),
    schema.SourceFormat.html: FormatProfile(
        classify=_classify_html, cdx_mimetype="text/html",
        watch_folder=False, browser_rung=True,
    ),
    schema.SourceFormat.docx: FormatProfile(
        classify=_classify_docx,
        cdx_mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        watch_folder=False, browser_rung=False,
    ),
    schema.SourceFormat.xlsx: FormatProfile(
        classify=_classify_xlsx,
        cdx_mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        watch_folder=False, browser_rung=False,
    ),
}


def classify_probe(body: bytes, headers_text: str) -> ClassifiedResponse:
    """Формат-агностичная классификация probe (spec post-acquisition-lifecycle §2).

    Отвечает на вопрос «не закрыт ли канал», а НЕ «тот ли это документ»: у кандидата
    слоя discovery ``source_format`` вовсе нет — его объявляет куратор только при
    триаже. С наивным дефолтом ``expected=pdf`` любой оживший HTML-источник навсегда
    оставался бы «not a valid PDF» -> blocked, и «стало добываемо» было бы недостижимо
    для половины кандидатов. Формат подтвердят триаж и добыча.

    Порядок проверок повторяет форматные классификаторы: challenge ПЕРЕД статусом —
    заглушка WAF сама приходит как ``200``.

    ``status in (200, 206)`` (spec discovery-acquire-seam-hardening §11, Г12):
    ``206 Partial Content`` — штатный ответ сервера на Range-запрос
    (``fetch_raw(byte_cap=...)``, которым эта функция обслуживает популяцию (c)) —
    живой канал, не аномалия.
    """
    status = _status_from_headers_text(headers_text)
    headers = _headers_from_text(headers_text)

    if _has_waf_header_fingerprint(headers) or any(m in body for m in CHALLENGE_BODY_MARKERS):
        return ClassifiedResponse(AcquisitionOutcome.blocked, status, "WAF challenge signature detected")
    if status in DEAD_STATUS_CODES:
        return ClassifiedResponse(AcquisitionOutcome.dead, status, f"HTTP {status}")
    if status in (200, 206) and body:
        return ClassifiedResponse(AcquisitionOutcome.ok, status, f"HTTP {status}, тело {len(body)} Б")
    return ClassifiedResponse(
        AcquisitionOutcome.blocked, status, f"неожиданный ответ (HTTP {status}, тело {len(body)} Б)"
    )


# curl exit codes (stable across decades): 6 = couldn't resolve host, 7 = failed
# to connect. Both mean "this URL is unreachable" — the same terminal signal as
# a confirmed-dead HTTP status, so the ladder should treat them identically.
_CURL_UNREACHABLE_CODES = (6, 7)


@dataclass
class RawResponse:
    """Ответ ДО интерпретации: тело + заголовочный дамп, без классификации.

    Существует ради второго потребителя curl-обвязки — условного запроса
    recheck-контура (spec post-acquisition-lifecycle §2), которому нужно выбрать
    классификатор самому (форматный для записей корпуса, формат-агностичный для
    кандидатов) и увидеть ``304``, у которого тела нет вовсе.
    """

    status: int | None
    headers_text: str
    body: bytes
    unreachable_reason: str | None = None  # curl exit 6/7 — DNS/connect


def fetch_raw(
    url: str,
    dest: Path,
    *,
    user_agent: str,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 30,
    total_timeout: int = 300,
    byte_cap: int | None = None,
) -> RawResponse:
    """Одна попытка скачивания без интерпретации результата.

    Deliberately omits ``-f``: a hard HTTP error (403/404) must still land its
    status/body so the caller's classifier can tell a block apart from a dead URL —
    with ``-f`` curl discards the response before we ever see it.

    A network-level curl failure (exit 6/7 — DNS/connect unreachable) surfaces as
    ``unreachable_reason`` rather than an exception: this is the single most common
    shape of "the URL is gone" (a decommissioned government domain), and the ladder
    must route it to the archive rung instead of crashing. Offline-vs-dead-domain is
    not disambiguated here — see design rationale in the spec: the archive rung's own
    curl call fails the same way, so the worst case is one wasted archive attempt, not
    silent corruption. ``--max-time`` bounds the whole transfer (``--connect-timeout``
    alone doesn't cap a stalled transfer on a slow LTE link).

    ``byte_cap`` (spec discovery-acquire-seam-hardening §11, Г12) — best-effort
    ``curl -r 0-{cap-1}`` (Range-запрос): сервер вправе проигнорировать и отдать
    полное тело (поведение прежнее, никакого дополнительного парсинга ответа не
    требуется — статус ``206``/``200`` классификатор уже различает). Опущено
    (``None``) — прежнее поведение без потолка байтов.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="acq-headers-", suffix=".txt", delete=False) as tmp:
        headers_path = Path(tmp.name)
    try:
        cmd = [
            "curl", "-sSL", "--retry", "3", "--retry-delay", "2",
            "--connect-timeout", str(timeout), "--max-time", str(total_timeout),
            "-A", user_agent,
        ]
        if byte_cap is not None:
            cmd += ["-r", f"0-{byte_cap - 1}"]
        for name, value in (extra_headers or {}).items():
            cmd += ["-H", f"{name}: {value}"]
        cmd += ["-D", str(headers_path), "-o", str(dest), url]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode in _CURL_UNREACHABLE_CODES:
            return RawResponse(
                None, "", b"", f"curl exit {proc.returncode}: host unreachable (DNS/connect)"
            )
        if proc.returncode != 0:
            # 28 = timeout after --retry already exhausted transients; anything
            # else is an unexpected curl failure — not a classification, a bug/env issue.
            raise RuntimeError(f"curl failed (exit {proc.returncode}) for {url}")
        headers_text = headers_path.read_text(encoding="utf-8", errors="replace")
        body = dest.read_bytes() if dest.exists() else b""
        return RawResponse(_status_from_headers_text(headers_text), headers_text, body)
    finally:
        headers_path.unlink(missing_ok=True)


def fetch_and_classify(
    url: str,
    dest: Path,
    *,
    user_agent: str,
    timeout: int = 30,
    total_timeout: int = 300,
    expected: schema.SourceFormat = schema.SourceFormat.pdf,
) -> ClassifiedResponse:
    """Single download attempt (no ladder stepping — that's the caller's job):
    ``fetch_raw`` + форматная классификация."""
    raw = fetch_raw(url, dest, user_agent=user_agent, timeout=timeout, total_timeout=total_timeout)
    if raw.unreachable_reason is not None:
        return ClassifiedResponse(AcquisitionOutcome.dead, None, raw.unreachable_reason)
    return classify_response(raw.body, raw.headers_text, expected)


# Ступени, чьи валидаторы принадлежат ИЗДАТЕЛЮ и потому годятся для будущего
# условного запроса (spec post-acquisition-lifecycle §2). Гейт обязателен: archive
# отдаёт заголовки Wayback (recheck сравнивал бы ETag архива с ответом официального
# сайта -> мусорный drift на каждой ротации), browser синтезирует заголовки сам,
# manual вовсе не имеет HTTP-ответа.
VALIDATOR_CAPTURE_RUNGS = frozenset(
    {schema.AcquisitionMethod.direct, schema.AcquisitionMethod.official_alt}
)

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
        if not schema.external_disclosure_allowed(sensitivity):
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
    retrieved_snapshot_date: _dt.date | None = None  # заполняется только archive-путём


def _try_browser_rung(rec: schema.SourceRecord, dest: Path) -> ClassifiedResponse:
    """Одна попытка резолва через headless-браузер (``core/browser_resolver``) —
    ТОЛЬКО для ``expected=html`` (спек headless-browser-resolver: единственный
    формат, для которого рендер-дамп — легитимный финальный артефакт; PDF/DOCX/XLSX
    через браузер не тестировались, см. «Вне скоупа» спека). Инструментальный
    отказ (Node/lightpanda недоступны или упали) трактуется как ``blocked``, не
    пробрасывается исключением — следующая ступень (``manual``) должна сработать
    как обычно, будто резолвера вовсе нет.

    Синтезируем ``curl -D``-совместимый заголовочный блок и прогоняем контент
    через уже существующий ``classify_response`` — переиспользуем WAF-маркеры и
    размерный порог, а не дублируем их: движок-резолвер и без того не даёт
    настоящих HTTP-заголовков (браузер их не эмитит наружу), контент —
    единственный сигнал, который у нас есть.
    """
    try:
        result = browser_resolver.resolve(rec.source_url)
    except browser_resolver.BrowserResolverUnavailable as exc:
        return ClassifiedResponse(AcquisitionOutcome.blocked, None, f"browser resolver unavailable: {exc}")
    if not result.ok:
        return ClassifiedResponse(AcquisitionOutcome.blocked, None, f"browser resolver: {result.error}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = result.html.encode("utf-8")
    dest.write_bytes(body)
    synthetic_headers = "HTTP/1.1 200\r\nContent-Type: text/html; charset=utf-8\r\n"
    return classify_response(body, synthetic_headers, expected=schema.SourceFormat.html)


def run_ladder(rec: schema.SourceRecord, dest: Path, *, user_agent: str) -> LadderResult:
    """Automatic portion of the ladder: try ``direct``, then ``official_alt`` once
    if blocked and available. Always starts fresh from ``direct`` — the ladder
    deliberately does not cache "known blocked" across runs in this version
    (§5: WAF state can flip either way; curl's own ``--retry`` already avoids
    hammering a *transient* failure, and a block/dead classification is a
    single-shot signal, not something curl retries internally).

    Raises ``AcquisitionBlocked`` when manual acquisition is needed, or
    ``AcquisitionDead`` when archive fallback is needed. Does not itself
    perform manual/archive acquisition or persist any state (``.state.yaml``)
    — those are the caller's job (see run_pipeline.py and commits 4/5/6).
    """
    has_alt = bool(rec.official_alt_url)
    rung = schema.AcquisitionMethod.direct
    classified = fetch_and_classify(rec.source_url, dest, user_agent=user_agent, expected=rec.source_format)

    while True:
        nxt = next_rung(classified.outcome, rung, has_official_alt=has_alt, sensitivity=rec.sensitivity)
        if nxt is None:
            return LadderResult(rung, _FIDELITY_BY_AUTOMATIC_RUNG[rung], classified)
        if nxt is schema.AcquisitionMethod.official_alt:
            assert rec.official_alt_url is not None  # has_alt guarantees this
            rung = nxt
            classified = fetch_and_classify(
                rec.official_alt_url, dest, user_agent=user_agent, expected=rec.source_format
            )
            continue
        if nxt is schema.AcquisitionMethod.manual:
            if (
                classified.outcome is AcquisitionOutcome.blocked
                and _FORMAT_PROFILES[rec.source_format].browser_rung
                and browser_resolver.is_available()
            ):
                # Заблокированный HTML — единственный формат/исход, где ступень
                # manual (watch-folder) и без браузера уже была PDF-only тупиком
                # (см. guard ниже) — резолвер даёт РЕАЛЬНОЕ автоматическое
                # восстановление там, где раньше был безусловный терминальный отказ.
                browser_classified = _try_browser_rung(rec, dest)
                if browser_classified.outcome is AcquisitionOutcome.ok:
                    return LadderResult(schema.AcquisitionMethod.browser, schema.Fidelity.rendered, browser_classified)
                classified = browser_classified  # несём вперёд ПОСЛЕДНЮЮ причину (браузерную), не устаревшую curl-причину
            if not _FORMAT_PROFILES[rec.source_format].watch_folder:
                # manual watch-folder matching (title_matcher, PDF-magic sniff) is
                # PDF-only in v1 — a non-PDF record can't ride that path (§3 of
                # convert-html spec); adoption via --no-download still works.
                raise AcquisitionBlocked(
                    rec.source_url,
                    f"manual watch-folder поддерживает только PDF; сохраните страницу вручную в "
                    f"<doc_dir>/raw.{rec.source_format.value} и перезапустите с --no-download "
                    f"(последняя причина: {classified.reason})",
                )
            if classified.outcome is AcquisitionOutcome.dead:
                raise AcquisitionBlocked(
                    rec.source_url,
                    f"{rung.value} confirmed dead ({classified.reason}) but sensitivity=confidential — archive unavailable",
                )
            raise AcquisitionBlocked(rec.source_url, f"{rung.value} blocked ({classified.reason})")
        raise AcquisitionDead(rec.source_url, f"{rung.value} confirmed dead ({classified.reason})")


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


def _looks_like_candidate_pdf(path: Path) -> tuple[bool, str, str]:
    """Дешёвая магия сначала (не читаем весь файл ради больших не-PDF в папке
    загрузок), pdfplumber — только если магия совпала (один парс, не два — тут же
    извлекается текст 1-й страницы для матчера принадлежности, §5b спека
    ingest-hardening). Возвращает (валиден, причина, текст_первой_страницы).
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
    except OSError as exc:
        return False, f"не удалось прочитать: {exc}", ""
    if not head.startswith(b"%PDF"):
        return False, "не PDF (нет магии %PDF)", ""
    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
            first_page_text = (pdf.pages[0].extract_text() or "") if n_pages else ""
    except Exception as exc:  # noqa: BLE001 — битый/недописанный PDF, не наш кандидат (пока)
        return False, f"PDF повреждён или ещё не дописан: {exc}", ""
    if n_pages < 1:
        return False, "0 страниц", ""
    return True, f"{n_pages} стр.", first_page_text


TITLE_MATCH_MIN_TOKEN_FRACTION = 0.4  # доля содержательных слов title, которую нужно найти на 1-й странице


def title_matcher(rec: schema.SourceRecord) -> Callable[[str], tuple[bool, str]]:
    """Предикат принадлежности PDF документу ``rec`` по тексту 1-й страницы
    (текст передаётся вызывающей стороной — уже извлечён ``watch_and_ingest`` при
    валидации кандидата, повторного парсинга PDF не требуется).

    Грубый порог (40% содержательных слов title, юникод-«словами» >=4 буквы —
    работает и для диакритики õ/ä/č) — достаточен против целевого сценария
    (посторонний файл, лежавший в папке загрузок), при этом не ломается на
    небуквальных/переводных заголовках. Пустой набор токенов (крайне короткий
    title) — сверка пропускается: не блокировать добычу из-за отсутствия сигнала.
    """
    tokens = {t for t in re.findall(r"[^\W\d_]{4,}", rec.title.casefold())}

    def match(first_page_text: str) -> tuple[bool, str]:
        if not tokens:
            return True, "нет содержательных токенов в title — сверка пропущена"
        text = first_page_text.casefold()
        found = sum(1 for t in tokens if t in text)
        need = max(1, round(len(tokens) * TITLE_MATCH_MIN_TOKEN_FRACTION))
        return found >= need, f"{found}/{len(tokens)} токенов титула"

    return match


def watch_and_ingest(
    dest: Path,
    *,
    watch_dir: Path,
    matcher: Callable[[str], tuple[bool, str]] | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_MANUAL_TIMEOUT_SECONDS,
) -> Path:
    """Следить за ``watch_dir`` до появления ОДНОГО валидного PDF-кандидата, перенести
    его в ``dest``. Первая же итерация — это и есть начальное сканирование уже
    лежащих файлов (резюмируемость после прерванного прогона — §6), отдельного
    "initial scan" шага не требуется по построению.

    ``matcher`` (опционален) — дополнительный гейт принадлежности поверх «валиден
    как PDF» (обычно ``title_matcher(rec)``): кандидаты, не прошедшие его,
    логируются как отвергнутые и не участвуют в конфликте. ``None`` — прежнее
    поведение (первый валидный PDF принимается без сверки) — используется, когда
    пользователь явно указал выделенную ``--watch-dir``, где сверка избыточна.

    НЕ кеширует ОТКАЗЫ кандидатов между итерациями: недописанный браузером файл
    должен быть перепроверен на следующем цикле, а не забыт навсегда. Инспекция
    (магия+pdfplumber+matcher) мемоизируется по ``(path, size, mtime_ns)`` —
    неизменившийся посторонний файл не парсится pdfplumber заново на каждой
    итерации поллинга; изменившийся (size/mtime другие) честно перепроверяется.
    Батч (>1 ПРОШЕДШИХ кандидата одновременно) не разрешается угадыванием — конфликт.

    ``now``/``sleep`` инжектируемы ради детерминированных тестов (без реального sleep).
    """
    deadline = now() + timeout
    cache: dict[tuple[Path, int, int], tuple[bool, str, str]] = {}
    rejected: dict[str, str] = {}  # имя файла -> причина отказа (для таймаут-диагностики)

    def inspect(path: Path) -> tuple[bool, str, str]:
        try:
            st = path.stat()
        except OSError as exc:
            return False, f"не удалось прочитать: {exc}", ""
        key = (path, st.st_size, st.st_mtime_ns)
        if key not in cache:
            cache[key] = _looks_like_candidate_pdf(path)
        return cache[key]

    while True:
        candidates: list[Path] = []
        for path in sorted(watch_dir.iterdir()):
            if not path.is_file():
                continue
            valid, reason, text = inspect(path)
            if not valid:
                rejected[path.name] = reason
                continue
            if matcher is not None:
                belongs, match_reason = matcher(text)
                if not belongs:
                    rejected[path.name] = match_reason
                    continue
            candidates.append(path)
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
            detail = "; ".join(f"{name}: {reason}" for name, reason in rejected.items())
            suffix = f" (отвергнутые кандидаты — {detail})" if detail else ""
            raise ManualAcquisitionTimeout(f"не дождался файла в {watch_dir} за {timeout:.0f}с{suffix}")
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

    Сверка принадлежности по титулу (``title_matcher``) применяется ТОЛЬКО к
    дефолтной (системной) watch-папке. Явный ``--watch-dir`` — escape hatch:
    пользователь сам изолировал папку под этот прогон, дополнительная сверка
    избыточна и рискует отсеять скан без текстового слоя.
    """
    subprocess.run(["xdg-open", rec.source_url], check=False)
    matcher = title_matcher(rec) if watch_dir is None else None
    watch_and_ingest(
        dest, watch_dir=watch_dir or default_watch_dir(), matcher=matcher, timeout=timeout
    )
    return LadderResult(
        schema.AcquisitionMethod.manual,
        schema.Fidelity.manual,
        ClassifiedResponse(AcquisitionOutcome.ok, None, "manual acquisition via watch-folder"),
    )


# --- archive fallback via Wayback CDX (§8 of the spec — only for confirmed-dead URLs) ---
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"


class ArchiveUnavailable(RuntimeError):
    """No usable Wayback snapshot exists for source_url — the document is genuinely gone,
    not just from the official host. A real, reportable terminal outcome (§8: the last
    rung of the ladder), not a bug.
    """


@dataclass
class ArchiveSnapshot:
    timestamp: str  # YYYYMMDDHHMMSS (Wayback's format)
    snapshot_url: str  # .../web/<timestamp>id_/<original> — "id_" = raw bytes, no Wayback toolbar


def find_wayback_snapshot(
    original_url: str, *, mimetype: str = "application/pdf", timeout: int = 30
) -> ArchiveSnapshot | None:
    """CDX lookup for the freshest 200/``mimetype`` snapshot of ``original_url``.

    Returns ``None`` if none found. ``limit=-N`` asks the CDX server for the
    LAST N results directly (verified against the official CDX server README,
    2026-07-16: "Set limit=-N to return the last N results"). A plain
    ``limit=20`` (no sign) instead returns the FIRST 20 — CDX sorts ascending
    by timestamp, so for a URL with more than 20 matching snapshots that
    silently picked a stale one instead of the freshest (still not a fidelity
    guarantee either way, see spec §7 — the record's ``fidelity`` is set to
    ``archived_snapshot``, never ``live``, regardless).
    """
    query = (
        f"{WAYBACK_CDX_URL}?url={quote(original_url, safe='')}"
        f"&output=text&filter=statuscode:200&filter=mimetype:{mimetype}&limit=-5"
    )
    result = subprocess.run(
        ["curl", "-sS", "--connect-timeout", str(timeout), "--max-time", str(timeout), query],
        check=True, capture_output=True, text=True,
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return None
    # CDX text format: <urlkey> <timestamp> <original> <mimetype> <statuscode> <digest> <length>
    fields = lines[-1].split()
    if len(fields) < 2:
        return None  # неразбираемая строка — трактуем как «снимка нет», не IndexError
    timestamp = fields[1]
    return ArchiveSnapshot(timestamp, f"https://web.archive.org/web/{timestamp}id_/{original_url}")


# --- проактивный снимок: SavePageNow при добыче (§4 spec post-acquisition-lifecycle) ---
WAYBACK_SAVE_URL = "https://web.archive.org/save/"
SPN_TIMEOUT_SECONDS = 20  # жёсткий потолок: снимок — страховка, а не часть критического пути


def request_snapshot(url: str, *, timeout: int = SPN_TIMEOUT_SECONDS) -> bool:
    """Попросить Wayback сохранить ТЕКУЩУЮ редакцию ``url`` (SPN2, анонимно).

    Fire-and-forget: тело ответа не читается, ретраев нет, ЛЮБОЙ отказ — WARNING в
    лог и ``False``, никогда не исключение. Долговечность обеспечивает локальный raw
    (и будущий R2-бэкап), а это — страховка на случай смерти официального URL: когда
    archive-ступень лестницы понадобится, снимок нужной редакции уже будет существовать.
    Ровно практика, которой Wikipedia автоархивирует каждую внешнюю ссылку.

    Анонимного доступа достаточно на нашем темпе (единицы запросов за прогон);
    аутентифицированный SPN с более высокими лимитами — эскалация на случай, если
    анонимный начнёт отказывать.

    ``url`` квотится (spec acquire-convert-seam-hardening §7, В9-код) — та же
    дисциплина, что уже применена в ``find_wayback_snapshot`` для CDX-запроса:
    пробелы/иные не-URL-символы в query-строке ``url`` иначе ломали бы итоговый
    SPN-запрос (отказ, не инъекция — argv-элемент, не shell). ``safe`` сохраняет
    структурные символы САМОГО ``url`` (он и без внешнего квотинга уже валидный URL).
    """
    try:
        proc = subprocess.run(
            [
                "curl", "-sS", "-o", os.devnull,
                "--connect-timeout", str(timeout), "--max-time", str(timeout),
                f"{WAYBACK_SAVE_URL}{quote(url, safe=':/?&=#%')}",
            ],
            check=False,
        )
    except OSError as exc:  # curl отсутствует/не запускается — не повод ронять добычу
        logger.warning("SavePageNow недоступен для %s: %s", url, exc)
        return False
    if proc.returncode != 0:
        logger.warning("SavePageNow не ответил для %s (curl exit %d)", url, proc.returncode)
        return False
    return True


def fetch_from_archive(rec: schema.SourceRecord, dest: Path, *, user_agent: str, timeout: int = 30) -> LadderResult:
    """Last rung of the ladder: only ever reached for a confirmed-dead source_url
    (never for a block — see §2/§9; sensitivity=confidential never reaches here,
    ``next_rung`` routes those to manual instead).

    Defense-in-depth guard (spec acquire-convert-seam-hardening §5, В5): routing
    through ``next_rung`` already excludes confidential records, but this is the
    ONLY function in the module that actually discloses bytes to a third party
    (Wayback) — worth a head-guard against a future caller that reaches here by a
    path other than ``next_rung``, not just documentation of the current one.
    """
    if not schema.external_disclosure_allowed(rec.sensitivity):
        raise RuntimeError(f"{rec.id}: archive-ступень недоступна для sensitivity=confidential")
    # spec acquire-convert-seam-hardening §6, В6: реестр вместо .get(..., "application/pdf") —
    # новый член SourceFormat без зарегистрированного профиля даёт громкий KeyError
    # ("забыли добавить формат"), а не тихо-неверный CDX-mimetype-фильтр.
    mimetype = _FORMAT_PROFILES[rec.source_format].cdx_mimetype
    snapshot = find_wayback_snapshot(rec.source_url, mimetype=mimetype, timeout=timeout)
    if snapshot is None:
        raise ArchiveUnavailable(f"нет снимка Wayback для {rec.source_url}")
    classified = fetch_and_classify(
        snapshot.snapshot_url, dest, user_agent=user_agent, timeout=timeout, expected=rec.source_format
    )
    if classified.outcome is not AcquisitionOutcome.ok:
        raise ArchiveUnavailable(
            f"снимок {snapshot.snapshot_url} не является валидным {rec.source_format.value.upper()}: {classified.reason}"
        )
    snapshot_date = _dt.datetime.strptime(snapshot.timestamp[:8], "%Y%m%d").date()
    return LadderResult(
        schema.AcquisitionMethod.archive, schema.Fidelity.archived_snapshot, classified, snapshot_date
    )
