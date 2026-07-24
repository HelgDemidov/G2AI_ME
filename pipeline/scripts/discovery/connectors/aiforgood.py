"""discovery/connectors/aiforgood.py — ITU AI Standards Exchange (aiforgood.itu.int) registry-коннектор.

Spec `docs/pipeline/discovery/tech_specs/aiforgood-standards/spec.md`. Третий экземпляр
архетипа `registry`: живой paginated JSON поверх WordPress `admin-ajax.php` — ближе по
форме к `eurlex.py` (живой источник, без DuckDB), но требует пагинации (в отличие от
EUR-Lex, где вся выборка приходит одним SPARQL-запросом) — ближе к `agora.py` по объёму
курсорной работы, без bulk-дампа/файлового кэша. Регистрируется в ядре при импорте
(см. ``discovery/connectors/__init__.py``).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core import schema
from core.env import REPO_ROOT
from discovery import dedup
from discovery.base import ConnectorCursor, DiscoverResult

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_aiforgood.yaml"
STANDARDS_BODIES_PATH = REPO_ROOT / "pipeline" / "vocab" / "vocab_standards_bodies.yaml"
CONNECTOR_ID = "aiforgood"

# group id (aiforgood.itu.int-специфичный, "gx0" и т.п.) -> entity_id справочника §3.
# Живьём подтверждено 2026-07-24 (get_groups). Группа, не входящая ни сюда, ни в
# exclude_groups конфига — НЕИЗВЕСТНАЯ организация (появилась после написания спека):
# коннектор её пропускает с диагностикой, а не угадывает entity_id из HTML-текста
# группы ("ITU-T <strong>(654)</strong>") — см. discover_aiforgood().
GROUP_ID_TO_ENTITY = {
    "gx0": "itu-t",
    "gx1091": "itu-r",
    "gx1141": "ietf",
    "gx8102": "u4ssc",
    "gx1043": "etsi",
    "gx1193": "tta",
}

RETRY_SCHEDULE = (1.0, 4.0, 15.0, 60.0)  # копия core/openrouter.py (принцип agora/eurlex)


@dataclass(frozen=True)
class AiforgoodConfig:
    """Разобранный ``pipeline/config/discovery_aiforgood.yaml`` (спек §4)."""

    enabled: bool
    ajax_endpoint: str
    topic: str
    exclude_groups: tuple[str, ...]
    exclude_status_substrings: tuple[str, ...]
    user_agent: str
    crawl_delay_seconds: float
    page_size: int
    timeout_seconds: float


def load_config(path: Path = CONFIG_PATH) -> AiforgoodConfig:
    """Разобрать ``discovery_aiforgood.yaml`` — плоский dict -> типизированный ``AiforgoodConfig``."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AiforgoodConfig(
        enabled=bool(raw["enabled"]),
        ajax_endpoint=str(raw["ajax_endpoint"]),
        topic=str(raw["topic"]),
        exclude_groups=tuple(raw["exclude_groups"]),
        exclude_status_substrings=tuple(raw["exclude_status_substrings"]),
        user_agent=str(raw["user_agent"]),
        crawl_delay_seconds=float(raw["crawl_delay_seconds"]),
        page_size=int(raw["page_size"]),
        timeout_seconds=float(raw["timeout_seconds"]),
    )


# --- §4: транспорт — GET admin-ajax.php + retry/backoff (принцип core/openrouter.py) ---


def fetch_json(
    params: dict[str, str], *, endpoint: str, user_agent: str, timeout: float
) -> dict[str, Any]:
    """GET-запрос к ``admin-ajax.php`` + retry/backoff (спек §4). Обычный REST-эндпоинт:
    ошибки транспортные (HTTP-код), не in-band-в-200 как у OpenRouter — retry-лестница
    зеркалит ``eurlex.fetch_sparql`` (нет ``InbandError``-ветки).

    ``user_agent`` — нейтральная строка конфига (OQ1 спека): ``robots.txt`` сайта явно
    запрещает ``ClaudeBot``/``GPTBot``/… — коннектор идентифицирует себя как research-бот,
    не как ИИ-краулер.
    """
    query = urllib.parse.urlencode(params)
    url = f"{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    reason = ""
    total_attempts = len(RETRY_SCHEDULE) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                body = exc.read().decode("utf-8", "replace")
                raise RuntimeError(f"aiforgood HTTP {exc.code}: {body[:500]}") from exc
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = str(exc)
        if attempt == total_attempts:
            break
        delay = RETRY_SCHEDULE[attempt - 1]
        print(f"попытка {attempt}/{total_attempts} через {delay:.0f}s: {reason}", file=sys.stderr)
        time.sleep(delay)
    raise RuntimeError(f"aiforgood: исчерпаны попытки ({total_attempts}) — {reason}")


# --- §1/§4: get_groups (источник истины по составу базы, не хардкод) ---


def get_groups(
    config: AiforgoodConfig, *, fetch: Callable[..., dict[str, Any]] = fetch_json
) -> list[dict[str, Any]]:
    """``action=get_groups`` -> список организаций топика (спек §1/§4). Форма живьём
    подтверждена 2026-07-24: ``{"success": true, "data": [{"id": "gx0", "text": "...",
    "data": {"total": 654}}, ...]}``. Если ITU добавит новую организацию — коннектор
    увидит её сам на следующем прогоне (не хардкодим список групп)."""
    payload = fetch(
        {"action": "get_groups", "topic": config.topic},
        endpoint=config.ajax_endpoint,
        user_agent=config.user_agent,
        timeout=config.timeout_seconds,
    )
    return list(payload.get("data") or [])


def group_total(group: dict[str, Any]) -> int:
    """``get_groups()``-запись -> заявленное число записей группы (``data.data.total``)."""
    return int((group.get("data") or {}).get("total") or 0)


# --- §4: get_standards — постраничный обход одной группы ---


def get_standards_page(
    config: AiforgoodConfig,
    *,
    group_id: str,
    index: int,
    fetch: Callable[..., dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Одна страница ``action=get_standards`` (спек §4). Форма живьём подтверждена
    2026-07-24: ``{"standards": [...], "totalCount": int, "facets": [...]}``, ровно
    ``page_size`` записей на страницу (сайт хардкодит 10, не конфигурируемо)."""
    return fetch(
        {"action": "get_standards", "topic": config.topic, "group": group_id, "index": str(index)},
        endpoint=config.ajax_endpoint,
        user_agent=config.user_agent,
        timeout=config.timeout_seconds,
    )


def paginate_group(
    config: AiforgoodConfig,
    *,
    group_id: str,
    fetch: Callable[..., dict[str, Any]] = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Обойти ВСЕ страницы одной группы до исчерпания ``totalCount`` (спек §4). Вежливый
    краулинг: ``sleep(crawl_delay_seconds)`` между запросами пагинации (не перед первым) —
    ``robots.txt`` требует ``Crawl-delay: 10``, и это же лечит остаточный риск §1 (bulk-
    выкачка может затриггерить rate/behavior-челлендж F5, вежливый краулинг — не браузер).
    """
    records: list[dict[str, Any]] = []
    index = 0
    total: int | None = None
    while total is None or index < total:
        if index > 0:
            sleep(config.crawl_delay_seconds)
        page = get_standards_page(config, group_id=group_id, index=index, fetch=fetch)
        batch = page.get("standards") or []
        if not batch:
            break
        records.extend(batch)
        total = int(page.get("totalCount") or len(records))
        index += len(batch)
    return records


# --- §4: курсор — множество виденных id_value (как seen-CELEX у eurlex) ---


def diff_cursor(
    all_ids: list[str], cursor: ConnectorCursor | None
) -> tuple[set[str], ConnectorCursor]:
    """Новые (не виденные) ``id_value`` + новый курсор = объединение старых и текущих
    (спек §4). Множество СТРОГО растёт — правка/исчезновение записи в живом индексе не
    выбрасывает её id из seen (тот же принцип, что ``eurlex.diff_cursor``)."""
    seen = set((cursor or {}).get("seen_ids") or [])
    fresh_ids = {i for i in all_ids if i not in seen}
    new_seen = sorted(seen | set(all_ids))
    return fresh_ids, {"seen_ids": new_seen}


# --- §3: справочник организаций (issuer-лукап; аналог build_graph.load_jurisdictions) ---


def load_standards_bodies(path: Path = STANDARDS_BODIES_PATH) -> dict[str, dict[str, str]]:
    """``vocab_standards_bodies.yaml`` -> ``{entity_id: {"kind": ..., "full_name": ...}}``."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(k): dict(v) for k, v in raw.items()}


# --- §4: отсевы (data-quality, НЕ relevance-суждение) ---


def is_excluded_status(status: str | None, exclude_substrings: tuple[str, ...]) -> bool:
    """``standard_status`` содержит одну из ``exclude_substrings`` (регистронезависимо) —
    нет стабильного текста для акквизиции (черновики, спек §1)."""
    if not status:
        return False
    lowered = status.lower()
    return any(sub.lower() in lowered for sub in exclude_substrings)


def _valid_url(url: str | None) -> bool:
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


# --- §4: маппинг записи -> CandidateRecord ---


def _map_record(
    record: dict[str, Any], *, entity_id: str, issuer_full_name: str
) -> schema.CandidateRecord | None:
    """Одна запись ``standards[]`` -> ``CandidateRecord`` (маппинг §4). ``None`` — пропуск
    (диагностика у вызывающей стороны, не исключение) — data-quality отсев, НЕ relevance."""
    id_value = record.get("id_value")
    if id_value is None:
        return None

    standard_name = (record.get("standard_name") or "").strip()
    standard_title = (record.get("standard_title") or "").strip()
    title: str | None
    if standard_name and standard_title:
        title = f"{standard_name}: {standard_title}"
    else:
        title = standard_name or standard_title or None
    if not title:
        return None

    source_url = record.get("standard_url")
    if not _valid_url(source_url):
        return None
    assert source_url is not None  # для mypy — _valid_url уже отсеяла None/пустое

    status = record.get("standard_status")
    std_type = record.get("standard_type")
    native_tags = [f"ITU AI Standards Exchange: {status or 'unknown'}"]
    if std_type:
        native_tags.append(f"type: {std_type}")

    summary = (record.get("standard_summary") or "").strip()
    native_summary = summary[: schema.CANDIDATE_SUMMARY_MAX] if summary and summary != "-" else None

    native_id = str(id_value)
    canonical = "|".join([native_id, standard_name, str(status)])
    raw_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return schema.CandidateRecord(
        title=title,
        issuer=issuer_full_name,
        jurisdiction=None,
        doc_date=None,
        language=None,
        source_url=source_url,
        native_summary=native_summary,
        native_id=native_id,
        native_tags=native_tags,
        connector_id=CONNECTOR_ID,
        retrieved_at=dt.date.today(),
        raw_hash=raw_hash,
        normalized_url=dedup.normalize_url(source_url),
    )


# --- §4: discover_aiforgood() top-level ---


def discover_aiforgood(
    cursor: ConnectorCursor | None,
    *,
    config: AiforgoodConfig | None = None,
    fetch: Callable[..., dict[str, Any]] = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
    bodies: dict[str, dict[str, str]] | None = None,
) -> DiscoverResult:
    """``Connector.discover()`` для aiforgood (спек §4): обойти группы (кроме
    ``exclude_groups`` и неизвестных — не в ``GROUP_ID_TO_ENTITY``) -> пагинировать ->
    отфильтровать черновики -> замаппить -> отфильтровать по seen-id-курсору.

    ``fetch``/``sleep`` инжектируем — тесты подменяют фейками, сеть/реальные паузы в CI
    не участвуют. ``bodies`` — справочник §3 (по умолчанию читается с диска).
    """
    cfg = config or load_config()
    org_bodies = bodies if bodies is not None else load_standards_bodies()
    groups = get_groups(cfg, fetch=fetch)

    candidates: list[schema.CandidateRecord] = []
    skipped_draft = 0
    skipped_no_title_or_url = 0
    skipped_unknown_group = 0
    excluded_group_count = 0

    for group in groups:
        group_id = group.get("id")
        if not group_id or group_id in cfg.exclude_groups:
            excluded_group_count += 1
            continue
        entity_id = GROUP_ID_TO_ENTITY.get(group_id)
        if entity_id is None or entity_id not in org_bodies:
            skipped_unknown_group += 1
            continue
        issuer_full_name = org_bodies[entity_id]["full_name"]

        records = paginate_group(cfg, group_id=group_id, fetch=fetch, sleep=sleep)
        for record in records:
            if is_excluded_status(record.get("standard_status"), cfg.exclude_status_substrings):
                skipped_draft += 1
                continue
            cand = _map_record(record, entity_id=entity_id, issuer_full_name=issuer_full_name)
            if cand is None:
                skipped_no_title_or_url += 1
                continue
            candidates.append(cand)

    all_ids = [c.native_id for c in candidates if c.native_id]
    fresh_ids, new_cursor = diff_cursor(all_ids, cursor)
    fresh = [c for c in candidates if c.native_id in fresh_ids]

    status_label = "no_new" if cursor is not None and not fresh else "fetched"
    diagnostics = {
        "status": status_label,
        "found": len(candidates),
        "fresh": len(fresh),
        "excluded_groups": excluded_group_count,
        "skipped_unknown_group": skipped_unknown_group,
        "skipped_draft": skipped_draft,
        "skipped_no_title_or_url": skipped_no_title_or_url,
    }
    return DiscoverResult(candidates=fresh, cursor=new_cursor, diagnostics=diagnostics)
