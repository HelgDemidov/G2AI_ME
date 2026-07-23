"""discovery/manual.py — ручной инжект + worksheet/apply батч-триаж (spec discovery-manual).

Три операции, делящие store/dedup-обвязку discovery-core:
``inject`` (кандидат от куратора/directed-search), ``pending_candidates``/``render_worksheet``
(реконсиляционная таблица ждущих) и ``apply_decisions`` (batch promote/reject). CLI-обёртка —
``discover.py`` (``inject``/``worksheet``/``apply`` subcommands).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    else:
        connector_id = "manual"

    normalized = dedup.normalize_url(url)
    # архетип канала отдельным полем не хранится — он выводится из грамматики
    # connector_id ("manual" | "search:<кампания>"), см. docstring CandidateRecord
    cand = schema.CandidateRecord(
        connector_id=connector_id,
        retrieved_at=dt.date.today(),
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
среди ждущих). Порядок ключей решения — конвенция читаемости (код к порядку безразличен):
`id` первым (кем кандидат станет), `action` последним (итог отработки). Весь СОДЕРЖАТЕЛЬНЫЙ
контент решений — АНГЛИЙСКИЙ (`rationale`/`summary`/`reason`; языковая унификация данных).

```yaml
- id: me-example-strategy-2026
  raw_hash: "abc123def456"
  entity_id: me
  issuer_type: government
  geo_scope: national
  doc_type: national_strategy
  source_format: pdf
  relevance:
    target_fit: primary
    axis: agentic_g2ai
    assessed_stage: triage
    rationale: "ONLY the relevance factors for this tier/axis — do NOT restate the summary"
    assessed_date: 2026-07-21
  topics: [ai-governance]
  summary: "2-3 sentences EN — what the document IS"
  relations:
    - {type: implements, target: eu-ai-act-2024}
  hidden_fields: [authority, track]  # выведет apply (карта дефолтов ниже) и впишет в meta.yaml
                                     # полными значениями; переопределить — добавить одноимённый
                                     # ключ (authority:/track:) прямо в это решение
  action: admit

- raw_hash: "fed456abc789"
  reason: "outside both axes: marketing overview"
  action: reject
```

Заполняя `admit`:
- `rationale` ≠ `summary`: summary описывает документ, rationale — ТОЛЬКО факторы релевантности
  (почему этот тир по этой оси); не дублировать одно в другом.
- Дефолты (сводка apply напечатает, во что развернулись; RAG/фасеты видят полные значения):
  `authority` — из doc_type: legislation→binding_law, regulation→regulation,
  report/academic_paper→report, guidance/framework/national_strategy→soft_law,
  technical_standard/technical_spec→voluntary_standard. Исключения (draft!) — задавать явно.
  `track` — jurisdiction=me→montenegro, think_tank/academia→research-papers, иначе intl-xperience.
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


_ADMIT_REQUIRED = (
    "id",
    "entity_id",
    "issuer_type",
    "geo_scope",
    "doc_type",
    "relevance",
)

# Дефолты admit-решения (ревью 2026-07-21): куратор в типовом решении НЕ пишет
# authority/track — apply выводит их и МАТЕРИАЛИЗУЕТ в meta.yaml полными значениями
# (слой знаний/фасеты ничего не теряют — «скрытость» существует только во входном
# файле решений). Явный ключ в решении всегда побеждает дефолт; исключения (draft!)
# куратор задаёт явно. Ключ `hidden_fields` в решении — чистая аннотация для человека
# (какие поля выведены дефолтом), apply его не читает.
#
# authority — из doc_type (жанр почти всегда определяет нормативную силу; полное
# покрытие текущего vocab_doc_types — при органическом росте словаря дополнять карту,
# иначе apply честно потребует явный authority):
_AUTHORITY_BY_DOC_TYPE = {
    "national_strategy": "soft_law",
    "framework": "soft_law",
    "guidance": "soft_law",
    "legislation": "binding_law",
    "regulation": "regulation",
    "technical_standard": "voluntary_standard",
    "technical_spec": "voluntary_standard",
    "report": "report",
    "academic_paper": "report",
}


def _default_track(jurisdiction: str | None, issuer_type: schema.IssuerType) -> schema.Track:
    """track — из jurisdiction/issuer_type. Приоритет: jurisdiction=me -> montenegro
    (прецедент me-undp-aila-2025: igo-доклад О Черногории живёт в треке montenegro),
    затем think_tank/academia -> research-papers (вторичная аналитика), иначе
    intl-xperience. На всех 6 документах текущего корпуса дефолт угадывает точно."""
    if jurisdiction == "me":
        return schema.Track.montenegro
    if issuer_type in (schema.IssuerType.think_tank, schema.IssuerType.academia):
        return schema.Track.research_papers
    return schema.Track.intl_xperience


@dataclass(frozen=True)
class ApplyOutcome:
    """Итог применения одного решения (spec §4) — per-решение, не рвёт остальной батч."""

    raw_hash: str
    action: str
    ok: bool
    detail: str


@dataclass
class ApplySummary:
    outcomes: list[ApplyOutcome] = field(default_factory=list)
    dry_run: bool = False

    @property
    def errors(self) -> list[ApplyOutcome]:
        return [o for o in self.outcomes if not o.ok]


def _resolve_candidate(
    raw_hash_prefix: str, candidates: list[schema.CandidateRecord]
) -> schema.CandidateRecord:
    """Найти кандидата по ``raw_hash`` (полный или уникальный префикс ``>=12`` символов)."""
    if len(raw_hash_prefix) < 12:
        raise ValueError(f"raw_hash слишком короткий префикс (нужно >=12 символов): {raw_hash_prefix!r}")
    matches = [c for c in candidates if c.raw_hash.startswith(raw_hash_prefix)]
    if not matches:
        raise ValueError(f"raw_hash не найден среди кандидатов: {raw_hash_prefix!r}")
    if len(matches) > 1:
        raise ValueError(f"raw_hash неоднозначен ({len(matches)} совпадений): {raw_hash_prefix!r}")
    return matches[0]


def _build_admit_record(
    cand: schema.CandidateRecord, decision: dict[str, Any]
) -> tuple[schema.SourceRecord, list[str]]:
    """Построить ``SourceRecord`` из ``admit``-решения (промоушен, ничего не пишет на диск).

    Возвращает ``(запись, применённые_дефолты)`` — второй элемент вида
    ``["authority=soft_law", "track=montenegro"]`` для эха в сводке apply (куратор
    видит, во что развернулись опущенные поля)."""
    missing = [k for k in _ADMIT_REQUIRED if k not in decision]
    if missing:
        raise ValueError(f"admit: отсутствуют обязательные поля: {', '.join(missing)}")

    defaulted: list[str] = []
    issuer_type = schema.IssuerType(decision["issuer_type"])

    authority = decision.get("authority")
    if authority is None:
        authority = _AUTHORITY_BY_DOC_TYPE.get(decision["doc_type"])
        if authority is None:
            raise ValueError(
                f"admit: authority не указан, а для doc_type '{decision['doc_type']}' "
                "нет дефолта — задайте authority явно"
            )
        defaulted.append(f"authority={authority}")

    if "track" in decision:
        track = schema.Track(decision["track"])
    else:
        track = _default_track(cand.jurisdiction, issuer_type)
        defaulted.append(f"track={track.value}")

    relations_raw = decision.get("relations")
    rec = schema.promote_candidate(
        cand,
        id=decision["id"],
        entity_id=decision["entity_id"],
        track=track,
        issuer_type=issuer_type,
        geo_scope=schema.GeoScope(decision["geo_scope"]),
        doc_type=decision["doc_type"],
        authority=authority,
        relevance=schema.Relevance.model_validate(decision["relevance"]),
        source_format=schema.SourceFormat(decision.get("source_format", "pdf")),
        topics=decision.get("topics"),
        g2ai_pattern=decision.get("g2ai_pattern"),
        summary=decision.get("summary"),
        relations=[schema.Relation.model_validate(r) for r in relations_raw] if relations_raw else None,
        language=decision.get("language"),
    )
    return rec, defaulted


def apply_decisions(
    decisions: list[dict[str, Any]],
    *,
    root: Path = schema.DEFAULT_SOURCES,
    dry_run: bool = False,
) -> ApplySummary:
    """Применить batch решений triage (spec §4): ``reject`` -> ``rejected_reason``, ``admit`` ->
    ``promote_candidate`` + ``save_record``.

    Ошибка одного решения (не найден raw_hash, неполный admit, конфликт meta.yaml) не рвёт
    батч — попадает в ``ApplySummary.errors``, остальные решения применяются. ``dry_run`` строит
    план (валидирует admit-решения через ``promote_candidate`` целиком, включая enum/pydantic
    ошибки) без записи store/meta.yaml.
    """
    candidates_path = root / "candidates.yaml"
    candidates = store.load(candidates_path)
    outcomes: list[ApplyOutcome] = []
    store_changed = False

    for decision in decisions:
        raw_hash_key = str(decision.get("raw_hash") or "")
        action = decision.get("action")

        if not raw_hash_key or action not in ("admit", "reject"):
            outcomes.append(
                ApplyOutcome(
                    raw_hash=raw_hash_key,
                    action=str(action),
                    ok=False,
                    detail="raw_hash обязателен, action должен быть 'admit' или 'reject'",
                )
            )
            continue

        try:
            cand = _resolve_candidate(raw_hash_key, candidates)
        except ValueError as exc:
            outcomes.append(ApplyOutcome(raw_hash=raw_hash_key, action=action, ok=False, detail=str(exc)))
            continue

        if action == "reject":
            if cand.rejected_reason is not None:
                outcomes.append(
                    ApplyOutcome(
                        raw_hash=cand.raw_hash,
                        action=action,
                        ok=True,
                        detail=f"уже был отклонён ранее (без изменений): {cand.rejected_reason}",
                    )
                )
                continue
            reason = decision.get("reason") or "отклонено триажем (без указанной причины)"
            if dry_run:
                outcomes.append(
                    ApplyOutcome(
                        raw_hash=cand.raw_hash, action=action, ok=True, detail=f"план: отклонить ({reason})"
                    )
                )
                continue
            cand.rejected_reason = reason
            store_changed = True
            outcomes.append(ApplyOutcome(raw_hash=cand.raw_hash, action=action, ok=True, detail="отклонён"))
            continue

        # action == "admit"
        try:
            rec, defaulted = _build_admit_record(cand, decision)
        except ValueError as exc:
            outcomes.append(ApplyOutcome(raw_hash=cand.raw_hash, action=action, ok=False, detail=str(exc)))
            continue

        # эхо дефолтов: куратор видит, во что развернулись опущенные поля
        suffix = f" (по дефолту: {', '.join(defaulted)})" if defaulted else ""

        if dry_run:
            outcomes.append(
                ApplyOutcome(
                    raw_hash=cand.raw_hash, action=action, ok=True,
                    detail=f"план: допустить как {rec.id}{suffix}",
                )
            )
            continue

        try:
            schema.save_record(rec, root)
        except ValueError as exc:
            outcomes.append(ApplyOutcome(raw_hash=cand.raw_hash, action=action, ok=False, detail=str(exc)))
            continue

        outcomes.append(
            ApplyOutcome(
                raw_hash=cand.raw_hash, action=action, ok=True, detail=f"допущен как {rec.id}{suffix}"
            )
        )

    if not dry_run and store_changed:
        store.save(candidates, candidates_path)

    return ApplySummary(outcomes=outcomes, dry_run=dry_run)
