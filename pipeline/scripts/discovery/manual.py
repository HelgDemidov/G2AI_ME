"""discovery/manual.py — ручной инжект + worksheet/apply батч-триаж (spec discovery-manual).

Три операции, делящие store/dedup-обвязку discovery-core:
``inject`` (кандидат от куратора/directed-search), ``pending_candidates``/``render_worksheet``
(реконсиляционная таблица ждущих) и ``apply_decisions`` (batch promote/reject). CLI-обёртка —
``discover.py`` (``inject``/``worksheet``/``apply`` subcommands).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core import schema
from core.env import REPO_ROOT
from discovery import dedup, store

TARGET_ENTITIES_CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "target_entities.yaml"


def raw_hash_for_manual(
    normalized_url: str, title: str, doc_date: dt.date | None, supersedes: str | None = None
) -> str:
    """sha256 канонической строки идентичности ручного/directed-кандидата.

    В отличие от коннекторных кандидатов, у ручного нет нативной записи-источника, откуда
    обычно берётся ``raw_hash`` — идентичность конструируется из уже нормализованного URL,
    заголовка и (опциональной) даты документа. Детерминирован: те же входы -> тот же хэш.

    ``supersedes`` (spec discovery-candidates-sharding §5) входит в строку ТОЛЬКО когда
    задан — хэши всех существующих кандидатов не меняются ни на бит. Без этого редакция с
    совпадающими (url, title, doc_date) — а одинаковая дата переиздания реальна — дала бы
    ``raw_hash``, БАЙТ-В-БАЙТ равный хэшу кандидата-предшественника (тот персистит в store
    навсегда), и ``_resolve_candidate`` падал бы «raw_hash неоднозначен» для ОБОИХ: ключ
    worksheet/apply сломан у обеих записей разом.
    """
    canonical = f"{normalized_url}|{title}|{doc_date.isoformat() if doc_date else ''}"
    if supersedes is not None:
        canonical = f"{canonical}|{supersedes}"
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
    supersedes: str | None = None,
    root: Path = schema.DEFAULT_SOURCES,
) -> tuple[schema.CandidateRecord, bool]:
    """Завести ручного/directed-search кандидата (spec discovery-manual §2).

    Не скачивает, не оценивает — только строит ``CandidateRecord``, прогоняет через
    кросс-коннекторный ``dedup`` против уже персистнутых кандидатов и сохраняет store.
    Повторный inject той же ссылки — no-op (dedup ловит совпадение, включая уже отклонённые
    триажем — они не должны воскресать как "свежие").

    ``supersedes`` (spec discovery-candidates-sharding §5) — doc-id записи корпуса, которую
    кандидат сознательно ЗАМЕНЯЕТ (новая редакция на том же URL). Единственная дверь
    остаётся единственной: редакция входит через тот же inject, но с ЯВНЫМ намерением
    куратора. Валидация — предшественник обязан существовать в реестре: опечатка падает
    здесь, а не при промоушене (реестр читается только когда флаг задан).

    Возвращает ``(candidate, is_new)``: при ``is_new=False`` — по возможности возвращается
    СУЩЕСТВУЮЩАЯ запись (по совпадению пары ``(normalized_url, supersedes)``), чтобы
    вызывающая сторона могла сообщить куратору причину (уже есть / уже отклонён и почему),
    не только сам факт дубля.
    """
    if kind == schema.ConnectorKind.directed_search:
        if not campaign:
            raise ValueError("directed_search: --campaign обязателен")
        if not query:
            raise ValueError("directed_search: --query обязателен")
        connector_id = f"search:{campaign}"
    else:
        connector_id = "manual"

    if supersedes is not None:
        known_ids = {rec.id for rec in schema.load_records(root)}
        if supersedes not in known_ids:
            raise ValueError(
                f"--supersedes {supersedes!r}: такого документа нет в реестре корпуса "
                "(редакция обязана ссылаться на существующего предшественника)"
            )

    normalized = dedup.normalize_url(url)
    # архетип канала отдельным полем не хранится — он выводится из грамматики
    # connector_id ("manual" | "search:<кампания>"), см. docstring CandidateRecord
    cand = schema.CandidateRecord(
        connector_id=connector_id,
        retrieved_at=dt.date.today(),
        raw_hash=raw_hash_for_manual(normalized, title, date, supersedes),
        title=title,
        # Заголовок назвал человек (--title обязателен) — по определению stated
        # (spec triage-intake-hardening §3).
        title_provenance=schema.TitleProvenance.stated,
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
        supersedes=supersedes,
        native_format_hint=dedup.format_hint_from_url(url),
    )

    existing = store.load(root)
    outcome = dedup.dedup([cand], existing)
    store.save(existing + outcome.fresh, root)

    if outcome.fresh:
        return cand, True
    # spec discovery-acquire-seam-hardening §5, Г4: поглотитель берётся из
    # absorptions (реальный, любой из трёх стратегий) — не поиском по URL-паре,
    # который находил бы None и при поглощении стратегией 2/3 давал куратору
    # ложное «уже есть» без причины отказа.
    assert outcome.absorptions  # dedup гарантирует: не fresh -> поглощён кем-то из existing
    _, absorber = outcome.absorptions[0]
    return absorber, False


def registered_pairs(records: list[schema.SourceRecord]) -> set[tuple[str, str | None]]:
    """Пары ``(нормализованный source_url, заменяемый doc-id | None)``, уже представленные
    в реестре — правая часть реконсиляции ``pending_candidates``/``unacquirable_candidates``,
    двух очередей слоя кандидатов (spec discovery-acquire-seam-hardening §4).

    Публично, хотя оба вызывающих — соседи по модулю: это словарь предметной области
    слоя (реконсиляция «что уже в реестре»), а не деталь одной функции. Кросс-слойного
    потребителя у него нет и быть не должно — ротацию recheck кормит уже реконсилированным
    списком оркестратор (ревью PR #54, см. ``acquire.recheck.due_candidates``): здесь
    нормализуются URL, а ``normalize_url`` — знание слоя DISCOVERY.

    Каждая запись регистрирует ``(url, None)`` — URL как таковой корпусом покрыт — И
    ``(url, target)`` для каждого своего ребра ``supersedes``: последнее означает «редакция,
    заменяющая target, УЖЕ промоутнута». Разделение нужно, чтобы обычное пере-обнаружение
    того же URL не всплывало ждущим, а конкретная редакция гасилась только собственным
    промоушеном.
    """
    pairs: set[tuple[str, str | None]] = set()
    for rec in records:
        url = dedup.normalize_url(rec.source_url)
        pairs.add((url, None))
        for rel in rec.relations:
            if rel.type is schema.RelationType.supersedes:
                pairs.add((url, rel.target))
    return pairs


def pending_candidates(
    candidates: list[schema.CandidateRecord], records: list[schema.SourceRecord]
) -> list[schema.CandidateRecord]:
    """«Ждущие» кандидаты — вычисляется реконсиляцией, не хранимым статусом (spec §3).

    Кандидат «ждущий», если у него нет ``rejected_reason`` И его пара
    ``(URL, supersedes)`` не представлена в реестре (см. ``registered_pairs``). Кандидат
    без URL вовсе (совпадение только по паре issuer+тайтл) реконсиляцией по URL
    отфильтровать нельзя — остаётся ждущим (безопасный дефолт: не прячем от куратора то,
    чего не можем уверенно сопоставить).

    **Почему пара, а не голый URL** (spec discovery-candidates-sharding §5): новая редакция
    живёт на ТОМ ЖЕ URL, что предшественник. При сверке по одному URL кандидат-редакция
    гасился бы НЕМЕДЛЕННО — совпадением с предшественником — и никогда не попадал бы в
    worksheet; штатный батч-триаж редакций был бы физически невозможен, а цикл разрешения
    дрейфа (спек post-acquisition-lifecycle §6) обрывался бы на первом же шаге. Редакция
    перестаёт быть ждущей, когда промоутнута ОНА САМА (её ребро ``supersedes`` в реестре).
    Куратор, переписавший авто-ребро при триаже, вернёт кандидата в worksheet — видимый
    сбой, не тихий.
    """
    registered = registered_pairs(records)
    pending: list[schema.CandidateRecord] = []
    for cand in candidates:
        if cand.rejected_reason is not None:
            continue
        url = cand.normalized_url or (dedup.normalize_url(cand.source_url) if cand.source_url else None)
        if url is not None and (url, cand.supersedes) in registered:
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
  admission:
    axis: agentic_g2ai
    rationale: "ONLY the relevance factors for this axis — do NOT restate the summary"
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

- raw_hash: "0123456789ab"
  reason: "wanted, but every ladder rung is WAF-blocked"
  reject_kind: unacquirable   # см. ниже: НЕ терминальный отказ, а очередь ожидания
  action: reject
```

Заполняя `admit`:
- `rationale` ≠ `summary`: summary описывает документ, rationale — ТОЛЬКО факторы релевантности
  (почему эта ось); не дублировать одно в другом.
- `title`/`issuer`/`source_url` — опциональные override'ы (той же формы, что `language` ниже):
  колонка `missing` таблицы ниже показывает, каких из четырёх условно-обязательных полей
  (`title`/`issuer`/`language`/`source_url`) у кандидата нет — задайте недостающее ключом
  того же имени в решении; отсутствие И у кандидата, И в решении — отказ с именем поля.
- `missing` показывает поле и при НЕПУСТОМ значении в таблице — когда значению нельзя
  верить по провенансу. Два таких случая: (а) `title_provenance: derived` — заголовок
  snowball-кандидата из URL-каналов реконструирован из позиционного артефакта (текст,
  вырезанный геометрией прямоугольника ссылки, или целая строка doc.md), а не назван
  источником; (б) `url_provenance: suspect` — OECD отдал ОДИН адрес нескольким разным
  документам, чей он на самом деле, из данных не выводится (заголовок и юрисдикция при
  этом верны — ошибочен только адрес, и добыча по нему скачает чужой документ).
  Оба требуют явного ключа (`title:`/`source_url:`); взять правильное значение неоткуда,
  кроме WebSearch по документу, — который для батч-каналов и так обязателен (политика
  triage-channel-policy).
- Дефолты (сводка apply напечатает, во что развернулись; RAG/фасеты видят полные значения):
  `authority` — из doc_type: legislation→binding_law, regulation→regulation,
  report/academic_paper→report, guidance/framework/national_strategy→soft_law,
  technical_standard/technical_spec→voluntary_standard. Исключения (draft!) — задавать явно.
  `track` — jurisdiction в pipeline/config/target_entities.yaml (сегодня только me)→target-entity,
  think_tank/academia→research-papers, иначе intl-xperience.
- `relations` — если связь с другим документом реестра видна уже сейчас (`implements`/`cites`/…),
  указать сразу: второго прохода по документу не будет (pre-wave требование graph-v2).
- `source_format` — поддерживает `html`/`docx`/`xlsx` помимо `pdf`; опущенный ключ резолвится
  подсказкой кандидата (колонка `format_hint` в таблице ждущих), а без неё — дефолтом `pdf`
  (сводка эхнёт «по дефолту: …», когда сработала подсказка); сверить с квотой форматов волны.
- `official_alt_url` (опционально) — вторая ступень лестницы добычи (`https?://`); суждение об
  официальности зеркала вносит куратор сам, коннекторы этого знания не несут (накопленные
  зеркала недобываемых видны в секции ниже — `alternate_urls`, но перенос в это поле не автомат).
- Непустой `supersedes` в строке = НОВАЯ РЕДАКЦИЯ документа, уже лежащего в корпусе (тот же
  URL — нормальное состояние для законов), а НЕ дубль: `reject` здесь потерял бы редакцию.
  Ребро `supersedes` в meta.yaml проставит `apply` сам — вписывать его в `relations` руками
  не нужно (дубля не будет, но и смысла нет).

Заполняя `reject`:
- `reject_kind` (опционально; значения — enum `schema.RejectionKind`) разводит два разных
  отказа. Дефолт (ключ опущен) — «содержательно не подходит», решение окончательное.
  Второе значение помечает «нужен, но недобываем» (WAF/мёртвая ссылка): такой кандидат
  уходит НЕ в терминальный отказ, а в отдельную секцию этого worksheet, и `run_pipeline.py
  --recheck` периодически пробует его URL — обстоятельства меняются, а заметить это иначе
  некому.
- `action: revive` возвращает недобываемого кандидата в ждущие (снимает отказ и probe-поля).
  Только явное решение человека: dedup по-прежнему не воскрешает отклонённых сам.
"""

_UNACQUIRABLE_SECTION_HEADER = """\

## Недобываемые — ждут смены обстоятельств

Отклонены как «нужен, но недобываем», не как «не нужен». `probe_finding` — что увидел
последний `run_pipeline.py --recheck`; `acquirable:` означает, что канал открылся —
кандидата пора вернуть в ждущие решением `action: revive`.
"""


def unacquirable_candidates(
    candidates: list[schema.CandidateRecord], records: list[schema.SourceRecord]
) -> list[schema.CandidateRecord]:
    """Кандидаты, отклонённые как «нужен, но недобываем» (spec post-acquisition-lifecycle §5).

    Вычисляется реконсиляцией по ``rejected_kind`` — как и всё остальное в этом
    модуле — отдельного хранимого «статуса очереди» нет. Самые давно не пробованные
    первыми: тот же порядок, в котором их берёт recheck, — куратор видит список в
    его логике.

    ``records`` (spec discovery-acquire-seam-hardening §4, Г3) — реконсиляция с
    реестром, симметрично ``pending_candidates``: кандидат, чья пара ``(URL,
    supersedes)`` уже представлена в реестре (``registered_pairs``), из секции
    выбывает. До этого спека эта очередь была ЕДИНСТВЕННОЙ в слое кандидатов, не
    выводимой реконсиляцией — admit недобываемого кандидата напрямую (код
    разрешает молча; штатный revive→admit требует двух батчей, и срезать угол
    естественно при ``probe_finding: acquirable``) оставлял его НАВСЕГДА видимым
    здесь, хотя документ уже в корпусе. Кандидат без URL реконсиляции не поддаётся —
    остаётся видимым (тот же безопасный дефолт, что у ``pending_candidates``).
    """
    registered = registered_pairs(records)
    due: list[schema.CandidateRecord] = []
    for cand in candidates:
        if cand.rejected_kind is not schema.RejectionKind.unacquirable:
            continue
        url = cand.normalized_url or (dedup.normalize_url(cand.source_url) if cand.source_url else None)
        if url is not None and (url, cand.supersedes) in registered:
            continue
        due.append(cand)
    due.sort(key=lambda c: (c.probe_checked is not None, c.probe_checked or dt.date.min, c.raw_hash))
    return due


def _md_cell(value: str) -> str:
    """Обезвредить значение ячейки markdown-таблицы (spec discovery-acquire-seam-
    hardening §12): title/anchor из недоверенных источников (реестры, анкоры снежного
    кома) могут естественно нести ``|`` — без экранирования строка рвётся по колонкам.
    Применяется на ВСЕ интерполируемые ячейки обеих таблиц ``render_worksheet``, не
    только очевидно небезопасные.

    Перевод строки схлопывается в пробел (найдено живьём на боевом store 2026-07-27,
    после бэкфилла подсказок формата): два кандидата ``aiforgood`` несут ``\\n`` внутри
    ``title`` (титул стандарта ITU-T, перенесённый в исходном каталоге), и строка
    таблицы разваливалась на ДВЕ физические — хуже, чем от ``|``: тот добавляет колонку,
    а разрыв строки превращает хвост ячейки в отдельную псевдо-строку и сбивает разбор
    worksheet человеком и агентом. Экранировать перевод строки в GFM-таблице нечем —
    ячейка однострочна по грамматике, поэтому единственный корректный ответ — схлопнуть."""
    flattened = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return flattened.replace("|", "\\|")


def _worksheet_contract() -> dict[str, Any]:
    """Контракт формата решений КАК ДАННЫЕ (spec triage-intake-hardening §2) — то же
    содержание, что шапка-проза ``_WORKSHEET_HEADER``, но машинно-проверяемое: `defaults`/
    `vocab` эмитируются из живых источников истины, не литералом (то же правило, что для
    help-текстов CLI и SKILL.md — иначе дрейф между текстом и кодом)."""
    return {
        "required": list(_ADMIT_REQUIRED),
        "conditional": {
            "|".join(_CONDITIONAL_ADMIT_FIELDS): "обязательны, если пусты у кандидата",
        },
        "actions": list(_ACTIONS),
        "defaults": {"authority_by_doc_type": dict(_AUTHORITY_BY_DOC_TYPE)},
        "vocab": {
            "axes": sorted(schema.load_vocab("axes")),
            "doc_types": sorted(schema.load_vocab("doc_types")),
            "authority": sorted(schema.load_vocab("authority")),
        },
    }


def _candidate_payload(cand: schema.CandidateRecord) -> dict[str, Any]:
    data = cand.model_dump(mode="json", exclude_none=True)
    data["missing"] = _missing_conditional_fields(cand)
    return data


def worksheet_payload(
    pending: list[schema.CandidateRecord],
    unacquirable: list[schema.CandidateRecord] | None = None,
    *,
    total: int | None = None,
) -> dict[str, Any]:
    """Структура worksheet (spec triage-intake-hardening §2) — общий источник данных для
    обоих рендереров (`--format json|md`): гарантирует один и тот же набор кандидатов
    (по ``raw_hash``) в обоих форматах, а не две расходящиеся реализации.

    ``total`` — размер пула ДО ``--limit`` (после ``--connector``, если задан); ``None`` —
    отбор не применялся, ``pending_total`` тогда равен ``len(pending)``."""
    return {
        "contract": _worksheet_contract(),
        "pending_total": total if total is not None else len(pending),
        "shown": len(pending),
        "candidates": [_candidate_payload(c) for c in pending],
        "unacquirable": [_candidate_payload(c) for c in (unacquirable or [])],
    }


def render_worksheet(
    pending: list[schema.CandidateRecord],
    unacquirable: list[schema.CandidateRecord] | None = None,
    *,
    total: int | None = None,
) -> str:
    """Markdown-таблица ждущих кандидатов + шапка-инструкция (spec §3, `missing`-колонка
    и `total`-аннотация — spec triage-intake-hardening §1/§2).

    ``unacquirable`` — вторая секция (spec post-acquisition-lifecycle §5): очередь
    ожидания обстоятельств. Пустая -> секция не печатается вовсе (шум в типовом
    прогоне не нужен). ``total`` — см. ``worksheet_payload``; при отборе (``--connector``/
    ``--limit``) печатает строку «показано N из M ждущих» — молчаливое усечение читалось
    бы как «это вся очередь».
    """
    payload = worksheet_payload(pending, unacquirable, total=total)
    missing_by_hash = {row["raw_hash"]: row["missing"] for row in payload["candidates"]}

    lines = [_WORKSHEET_HEADER, ""]
    if total is not None:
        lines.append(f"Показано {payload['shown']} из {payload['pending_total']} ждущих.\n")
    lines.append(
        "| raw_hash | title | issuer | jurisdiction | doc_date | supersedes | connector_id "
        "| native_tags/matched_query | source_url | format_hint | missing |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cand in pending:
        tags = ", ".join(cand.native_tags) if cand.native_tags else (cand.matched_query or "")
        missing = missing_by_hash.get(cand.raw_hash, [])
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _md_cell(cand.raw_hash[:12]),
                _md_cell(cand.title or ""),
                _md_cell(cand.issuer or ""),
                _md_cell(cand.jurisdiction or ""),
                _md_cell(cand.doc_date.isoformat() if cand.doc_date else ""),
                # непустое значение = редакция существующей записи корпуса, не дубль
                _md_cell(cand.supersedes or ""),
                _md_cell(cand.connector_id),
                _md_cell(tags),
                _md_cell(cand.source_url or ""),
                _md_cell(cand.native_format_hint.value if cand.native_format_hint else ""),
                _md_cell(", ".join(missing)),
            )
        )
    if unacquirable:
        lines.append(_UNACQUIRABLE_SECTION_HEADER)
        lines.append(
            "| raw_hash | title | issuer | probe_checked | probe_finding | source_url | alternate_urls |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for cand in unacquirable:
            alternates = ", ".join(getattr(cand, "alternate_source_urls", None) or [])
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    _md_cell(cand.raw_hash[:12]),
                    _md_cell(cand.title or ""),
                    _md_cell(cand.issuer or ""),
                    _md_cell(cand.probe_checked.isoformat() if cand.probe_checked else "—"),
                    _md_cell(cand.probe_finding or "—"),
                    _md_cell(cand.source_url or ""),
                    _md_cell(alternates or "—"),
                )
            )
    return "\n".join(lines) + "\n"


# Действия decisions.yaml. ``revive`` (spec post-acquisition-lifecycle §5) — обратный
# ход для недобываемых: обстоятельства сменились, кандидат снова в игре.
_ACTIONS = ("admit", "reject", "revive")

_ADMIT_REQUIRED = (
    "id",
    "entity_id",
    "issuer_type",
    "geo_scope",
    "doc_type",
    "admission",
)

# Условно-обязательные при admit (spec triage-intake-hardening §1): не входят в
# _ADMIT_REQUIRED, т.к. обязательны, ТОЛЬКО если их нет у кандидата — большинство
# каналов issuer/title/source_url дают, требовать их переписывания вручную было бы
# налогом на нормальный случай. ЕДИНСТВЕННОЕ определение — используется и дверью
# (_build_admit_record), и аннотацией `missing` worksheet (§2): разошедшиеся копии
# этой проверки — ровно класс дефекта Г-находок PR #54 (discovery-acquire-seam-hardening).
_CONDITIONAL_ADMIT_FIELDS = ("title", "issuer", "language", "source_url")


def _missing_conditional_fields(
    cand: schema.CandidateRecord, decision: dict[str, Any] | None = None
) -> list[str]:
    """Поля из ``_CONDITIONAL_ADMIT_FIELDS``, отсутствующие у кандидата (и, если передано
    решение, не заданные override'ом в нём). ``decision=None`` — аннотация worksheet ДО
    того, как решение написано (проверяется только кандидат).

    **Провенанс приравнивается к отсутствию** — одно правило на оба поля (spec
    triage-intake-hardening §3/§6): ``title_provenance: derived`` (реконструкция из
    позиционного артефакта, а не заголовок — в реестре станет меткой узла графа) и
    ``url_provenance: suspect`` (значение ``website`` делят записи OECD с разными
    заголовками — добыча по нему скачает ЧУЖОЙ документ). Значение есть, но верить ему
    нельзя, поэтому решение обязано задать поле явно. ``None``-провенанс у легаси-записей
    требования не поднимает.
    """
    unusable_provenance = {
        "title": cand.title_provenance is schema.TitleProvenance.derived,
        "source_url": cand.url_provenance is schema.UrlProvenance.suspect,
    }
    missing = []
    for name in _CONDITIONAL_ADMIT_FIELDS:
        if getattr(cand, name) is not None and not unusable_provenance.get(name, False):
            continue
        if decision is not None and decision.get(name) is not None:
            continue
        missing.append(name)
    return missing

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


def load_target_entity_jurisdictions(path: Path = TARGET_ENTITIES_CONFIG_PATH) -> tuple[str, ...]:
    """Юрисдикции трека ``target_entity`` — из ``pipeline/config/target_entities.yaml``,
    НЕ хардкод в коде (решение куратора 2026-07-25: список конфигурируем, но лениво —
    сегодня в нём только Черногория; уход от хардкода важнее самого списка переключателей)."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(raw["jurisdictions"])


def _default_track(
    jurisdiction: str | None,
    issuer_type: schema.IssuerType,
    *,
    target_entity_jurisdictions: tuple[str, ...] | None = None,
) -> schema.Track:
    """track — из jurisdiction/issuer_type. Приоритет: jurisdiction в списке target-юрисдикций
    (прецедент me-undp-aila-2025: igo-доклад о target-юрисдикции живёт в треке target_entity)
    -> target_entity, затем think_tank/academia -> research-papers (вторичная аналитика),
    иначе intl-xperience. ``target_entity_jurisdictions=None`` -> читается реальный
    трекаемый конфиг (инъекция — только для тестов, не продакшн-путь)."""
    jurisdictions = (
        target_entity_jurisdictions
        if target_entity_jurisdictions is not None
        else load_target_entity_jurisdictions()
    )
    if jurisdiction is not None and jurisdiction in jurisdictions:
        return schema.Track.target_entity
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
    ``["authority=soft_law", "track=target-entity"]`` для эха в сводке apply (куратор
    видит, во что развернулись опущенные поля)."""
    missing = [k for k in _ADMIT_REQUIRED if k not in decision]
    if missing:
        raise ValueError(f"admit: отсутствуют обязательные поля: {', '.join(missing)}")

    # Условно-обязательные (spec triage-intake-hardening §1): у кандидата может не быть
    # title/issuer/language/source_url (agora/aiforgood/oecd/snowball массово их не дают,
    # либо OECD-значение недостоверно — §6); в этом случае решение обязано задать ключ явно.
    # Ошибка называет ИМЕННО недостающее поле и ключ, который его чинит — а не долетает
    # до ValueError из глубины promote_candidate, где контекста решения уже нет.
    missing_conditional = _missing_conditional_fields(cand, decision)
    if missing_conditional:
        field_name = missing_conditional[0]
        # Поле может физически ПРИСУТСТВОВАТЬ, но быть непригодным по провенансу:
        # «у кандидата нет X» при видимом в worksheet значении читалось бы как баг,
        # а не как требование — поэтому причина называется отдельно.
        if field_name == "title" and cand.title is not None:
            raise ValueError(
                "admit: `title` кандидата — реконструкция из позиционного артефакта "
                f"(title_provenance: derived), а не заголовок: {cand.title!r}. "
                "Задайте настоящий заголовок ключом `title:` в решении"
            )
        if field_name == "source_url" and cand.source_url is not None:
            raise ValueError(
                "admit: `source_url` кандидата недостоверен (url_provenance: suspect) — "
                f"источник отдал этот адрес нескольким разным документам: {cand.source_url!r}. "
                "Проверьте адрес документа и задайте его ключом `source_url:` в решении"
            )
        raise ValueError(
            f"admit: у кандидата нет `{field_name}`, задайте его ключом `{field_name}:` в решении"
        )

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

    # Резолюция формата (spec discovery-acquire-seam-hardening §8, Г7): явный ключ
    # решения > подсказка кандидата (by construction у eurlex/URL-эвристика) >
    # молчаливый дефолт "pdf". Выведенное из подсказки значение эхается в defaulted
    # той же механикой, что authority/track — куратор видит, во что развернулась
    # подсказка, не только счёт документов.
    if "source_format" in decision:
        source_format = schema.SourceFormat(decision["source_format"])
    elif cand.native_format_hint is not None:
        source_format = cand.native_format_hint
        defaulted.append(f"source_format={source_format.value} (подсказка кандидата)")
    else:
        source_format = schema.SourceFormat.pdf

    # official_alt_url (spec discovery-acquire-seam-hardening §9, Г13): вторая
    # ступень лестницы через дверь промоушена — суждение об официальности зеркала
    # куратор вносит явно, ни один коннектор этого знания не несёт (rationale спека).
    official_alt_url = decision.get("official_alt_url")
    if official_alt_url is not None and not re.match(r"^https?://", official_alt_url):
        raise ValueError(f"admit: official_alt_url не похож на URL: {official_alt_url!r}")

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
        admission=schema.Admission.model_validate(decision["admission"]),
        source_format=source_format,
        official_alt_url=official_alt_url,
        topics=decision.get("topics"),
        g2ai_pattern=decision.get("g2ai_pattern"),
        summary=decision.get("summary"),
        relations=[schema.Relation.model_validate(r) for r in relations_raw] if relations_raw else None,
        language=decision.get("language"),
        title=decision.get("title"),
        issuer=decision.get("issuer"),
        source_url=decision.get("source_url"),
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
    candidates = store.load(root)
    outcomes: list[ApplyOutcome] = []
    store_changed = False

    for decision in decisions:
        raw_hash_key = str(decision.get("raw_hash") or "")
        action = decision.get("action")

        if not raw_hash_key or action not in _ACTIONS:
            outcomes.append(
                ApplyOutcome(
                    raw_hash=raw_hash_key,
                    action=str(action),
                    ok=False,
                    detail=f"raw_hash обязателен, action должен быть одним из: {', '.join(_ACTIONS)}",
                )
            )
            continue

        try:
            cand = _resolve_candidate(raw_hash_key, candidates)
        except ValueError as exc:
            outcomes.append(ApplyOutcome(raw_hash=raw_hash_key, action=action, ok=False, detail=str(exc)))
            continue

        if action == "revive":
            # Реанимация недобываемого (spec post-acquisition-lifecycle §5): ЯВНОЕ
            # решение человека, а не автоматика — невоскрешение отклонённых через
            # dedup при этом не ослабляется ни на бит.
            if cand.rejected_reason is None and cand.rejected_kind is None:
                outcomes.append(
                    ApplyOutcome(
                        raw_hash=cand.raw_hash, action=action, ok=True,
                        detail="не был отклонён — уже в ждущих (без изменений)",
                    )
                )
                continue
            if dry_run:
                outcomes.append(
                    ApplyOutcome(
                        raw_hash=cand.raw_hash, action=action, ok=True,
                        detail=f"план: вернуть в ждущие (был: {cand.rejected_reason})",
                    )
                )
                continue
            cand.rejected_reason = None
            cand.rejected_kind = None
            cand.probe_checked = None
            cand.probe_finding = None
            store_changed = True
            outcomes.append(
                ApplyOutcome(raw_hash=cand.raw_hash, action=action, ok=True, detail="возвращён в ждущие")
            )
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
            try:
                # Ключ решения (`reject_kind`) и поле кандидата (`rejected_kind`) названы
                # по-разному намеренно: язык ДЕЙСТВИЯ vs хранимое СОСТОЯНИЕ — та же пара,
                # что `action: reject` -> `rejected_reason`. Опущенный ключ оставляет None
                # (== легаси == «содержательно не подходит»), а не пишет дефолт в store.
                kind = (
                    schema.RejectionKind(decision["reject_kind"])
                    if decision.get("reject_kind") is not None
                    else None
                )
            except ValueError as exc:
                outcomes.append(
                    ApplyOutcome(raw_hash=cand.raw_hash, action=action, ok=False, detail=str(exc))
                )
                continue
            suffix = f" [{kind.value}]" if kind is not None else ""
            if dry_run:
                outcomes.append(
                    ApplyOutcome(
                        raw_hash=cand.raw_hash, action=action, ok=True,
                        detail=f"план: отклонить{suffix} ({reason})",
                    )
                )
                continue
            cand.rejected_reason = reason
            cand.rejected_kind = kind
            store_changed = True
            outcomes.append(
                ApplyOutcome(raw_hash=cand.raw_hash, action=action, ok=True, detail=f"отклонён{suffix}")
            )
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
        store.save(candidates, root)

    return ApplySummary(outcomes=outcomes, dry_run=dry_run)
