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


def pending_candidates(
    candidates: list[schema.CandidateRecord], records: list[schema.SourceRecord]
) -> list[schema.CandidateRecord]:
    """«Ждущие» кандидаты — вычисляется реконсиляцией, не хранимым статусом (spec §3).

    Кандидат «ждущий», если у него нет ``rejected_reason`` И его URL (``normalized_url``,
    либо ``source_url`` нормализованный на лету) не совпадает ни с одним ``source_url``
    записи реестра. Кандидат без URL вовсе (совпадение только по ``content_hash``/тайтлу)
    реконсиляцией по URL отфильтровать нельзя — остаётся ждущим (безопасный дефолт: не
    прячем от куратора то, чего не можем уверенно сопоставить).
    """
    registered_urls = {dedup.normalize_url(r.source_url) for r in records}
    pending: list[schema.CandidateRecord] = []
    for cand in candidates:
        if cand.rejected_reason is not None:
            continue
        url = cand.normalized_url or (dedup.normalize_url(cand.source_url) if cand.source_url else None)
        if url is not None and url in registered_urls:
            continue
        pending.append(cand)
    return pending


_WORKSHEET_HEADER = """\
# Триаж-worksheet — ждущие кандидаты

Инструкция: для каждой строки — решение в decisions.yaml (`discover.py apply decisions.yaml`),
ключ — `raw_hash` (первые 12 символов ниже, либо полный хэш; должен быть уникальным префиксом
среди ждущих). Формат decisions.yaml — spec discovery-manual §4.

Заполняя `admit`:
- `relations` — если связь с другим документом реестра видна уже сейчас (`implements`/`cites`/…),
  указать сразу: второго прохода по документу не будет (pre-wave требование graph-v2).
- `source_format` — поддерживает `html`/`docx`/`xlsx` помимо `pdf` (дефолт); сверить с квотой
  форматов волны.
"""


def render_worksheet(pending: list[schema.CandidateRecord]) -> str:
    """Markdown-таблица ждущих кандидатов + шапка-инструкция (spec §3)."""
    lines = [_WORKSHEET_HEADER, ""]
    lines.append(
        "| raw_hash | title | issuer | jurisdiction | doc_date | connector_id "
        "| native_tags/matched_query | source_url |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for cand in pending:
        tags = ", ".join(cand.native_tags) if cand.native_tags else (cand.matched_query or "")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                cand.raw_hash[:12],
                cand.title or "",
                cand.issuer or "",
                cand.jurisdiction or "",
                cand.doc_date.isoformat() if cand.doc_date else "",
                cand.connector_id,
                tags,
                cand.source_url or "",
            )
        )
    return "\n".join(lines) + "\n"
