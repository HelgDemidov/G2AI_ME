"""discovery/dedup.py — кросс-коннекторный dedup кандидатов (spec discovery-core §3).

Ключи сравнения по убыванию надёжности (чартер §4.4): ``normalized_url`` -> ``(issuer,
normalized_title, doc_date)`` -> ``content_hash``. Без fuzzy-библиотек — детерминизм важнее
recall (остаточные дубли дочистит человек на worksheet, discovery-manual).
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

from core.schema import CandidateRecord

_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_url(url: str) -> str:
    """URL -> ключ сравнения: lower-host, без fragment, без trailing ``/``, http==https.

    Схема нормализуется в фиксированную ``https`` (значение не для перехода по ссылке,
    только для сравнения) — реальный ``source_url`` документа не трогается.
    """
    parts = urlsplit(url)
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit(("https", netloc, path, parts.query, ""))


def normalized_title(title: str) -> str:
    """Заголовок -> ключ сравнения: нижний регистр, только буквы/цифры (юникод-aware).

    Схлопывает и пробелы, и пунктуацию/дефисы разом — "AI Act" / "ai-act" / "AI  Act."
    дают один ключ. Диакритика (č/š/đ) сохраняется как буква, не отбрасывается.
    """
    return _NON_WORD_RE.sub("", title.lower())


def _match_key(cand: CandidateRecord) -> tuple[str, str, str] | None:
    if not (cand.title and cand.issuer):
        return None
    return (cand.issuer, normalized_title(cand.title), str(cand.doc_date))


def _find_match(cand: CandidateRecord, pool: list[CandidateRecord]) -> CandidateRecord | None:
    if cand.normalized_url:
        for other in pool:
            if other.normalized_url == cand.normalized_url:
                return other
    key = _match_key(cand)
    if key is not None:
        for other in pool:
            if _match_key(other) == key:
                return other
    if cand.content_hash:
        for other in pool:
            if other.content_hash == cand.content_hash:
                return other
    return None


def _merge_provenance(existing: CandidateRecord, dup: CandidateRecord) -> None:
    """Дописать provenance поглощённого дубля в existing — НИКОГДА не перезаписывая его поля."""
    merged: list[str] = list(getattr(existing, "merged_connector_ids", None) or [])
    if dup.connector_id != existing.connector_id and dup.connector_id not in merged:
        merged.append(dup.connector_id)
        existing.merged_connector_ids = merged  # type: ignore[attr-defined]  # extra="allow"


def dedup(
    new: list[CandidateRecord], existing: list[CandidateRecord]
) -> tuple[list[CandidateRecord], int]:
    """Разложить ``new`` на (свежие-после-dedup, счётчик поглощённых).

    Дубль внутри ``new`` -> первый выигрывает, второй сливается в него. Дубль против
    ``existing`` (включая отклонённых триажем — они персистят с ``rejected_reason`` и
    не должны воскресать как "свежие") -> ``new``-кандидат НЕ добавляется, его
    ``connector_id`` дописывается в ``merged_connector_ids`` существующего (объект
    ``existing`` мутируется на месте — вызывающая сторона персистит его вместе с ``fresh``).
    """
    fresh: list[CandidateRecord] = []
    absorbed = 0

    for cand in new:
        match = _find_match(cand, existing) or _find_match(cand, fresh)
        if match is not None:
            _merge_provenance(match, cand)
            absorbed += 1
            continue
        fresh.append(cand)

    return fresh, absorbed
