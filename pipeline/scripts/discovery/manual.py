"""discovery/manual.py — ручной инжект + worksheet/apply батч-триаж (spec discovery-manual).

Три операции, делящие store/dedup-обвязку discovery-core:
``inject`` (кандидат от куратора/directed-search), ``pending_candidates``/``render_worksheet``
(реконсиляционная таблица ждущих) и ``apply_decisions`` (batch promote/reject). CLI-обёртка —
``discover.py`` (``inject``/``worksheet``/``apply`` subcommands).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

from core import schema
from discovery import dedup, store


def raw_hash_for_manual(normalized_url: str, title: str, doc_date: dt.date | None) -> str:
    """sha256 канонической строки идентичности ручного/directed-кандидата.

    В отличие от коннекторных кандидатов, у ручного нет нативной записи-источника, откуда
    обычно берётся ``raw_hash`` — идентичность конструируется из уже нормализованного URL,
    заголовка и (опциональной) даты документа. Детерминирован: те же входы -> тот же хэш.
    """
    canonical = f"{normalized_url}|{title}|{doc_date.isoformat() if doc_date else ''}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def inject(
    *,
    url: str,
    title: str,
    issuer: str,
    language: str,
    jurisdiction: str | None = None,
    date: dt.date | None = None,
    summary: str | None = None,
    kind: schema.ConnectorKind = schema.ConnectorKind.manual,
    campaign: str | None = None,
    query: str | None = None,
    rights: schema.Rights | None = None,
    sensitivity: schema.Sensitivity | None = None,
    root: Path = schema.DEFAULT_SOURCES,
) -> tuple[schema.CandidateRecord, bool]:
    """Завести ручного/directed-search кандидата (spec discovery-manual §2).

    Не скачивает, не оценивает — только строит ``CandidateRecord``, прогоняет через
    кросс-коннекторный ``dedup`` против уже персистнутых кандидатов и сохраняет store.
    Повторный inject той же ссылки — no-op (dedup ловит совпадение, включая уже отклонённые
    триажем — они не должны воскресать как "свежие").

    Возвращает ``(candidate, is_new)``: при ``is_new=False`` — по возможности возвращается
    СУЩЕСТВУЮЩАЯ запись (по совпадению ``normalized_url``), чтобы вызывающая сторона могла
    сообщить куратору причину (уже есть / уже отклонён и почему), не только сам факт дубля.
    """
    if kind == schema.ConnectorKind.directed_search:
        if not campaign:
            raise ValueError("directed_search: --campaign обязателен")
        if not query:
            raise ValueError("directed_search: --query обязателен")
        connector_id = f"search:{campaign}"
        source_ref = query
    else:
        connector_id = "manual"
        source_ref = url

    normalized = dedup.normalize_url(url)
    cand = schema.CandidateRecord(
        connector_id=connector_id,
        connector_kind=kind,
        retrieved_at=dt.date.today(),
        source_ref=source_ref,
        raw_hash=raw_hash_for_manual(normalized, title, date),
        title=title,
        issuer=issuer,
        jurisdiction=jurisdiction,
        source_url=url,
        doc_date=date,
        language=language,
        rights=rights,
        sensitivity=sensitivity,
        native_summary=summary,
        matched_query=query,
        normalized_url=normalized,
    )

    candidates_path = root / "candidates.yaml"
    existing = store.load(candidates_path)
    fresh, absorbed = dedup.dedup([cand], existing)
    store.save(existing + fresh, candidates_path)

    if fresh:
        return cand, True
    matched = next((c for c in existing if c.normalized_url == normalized), None)
    assert absorbed  # dedup гарантирует: не fresh -> поглощён кем-то из existing
    return matched or cand, False
