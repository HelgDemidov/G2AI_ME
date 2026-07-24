"""discovery/connectors/oecd.py — OECD.AI Policy Navigator registry-коннектор.

Spec `docs/pipeline/discovery/tech_specs/discovery-oecd/spec.md`. Четвёртый экземпляр
архетипа `registry`, ближайший родственник `aiforgood.py` по форме (живой paginated
JSON, seen-id-курсор, без DuckDB) — плюс два свойства, специфичных этому источнику:
недокументированный backend (`api.oecdai.org`, найден в HTML фронтенда, не объявлен
официально) требует shape-гейта честной деградации (§3), а сырые данные несут Faker.js-
порченную демографию и PII редакторов OECD, которые НИКОГДА не должны попасть в
`CandidateRecord`/`candidates.yaml` (§1 барьеры). Регистрируется в ядре при импорте
(см. ``discovery/connectors/__init__.py``).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core import fsio
from core.env import REPO_ROOT
from discovery.base import ConnectorCursor

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_oecd.yaml"
CACHE_DIR = REPO_ROOT / "pipeline" / "discovery_cache" / "oecd"
SNAPSHOT_PATH = CACHE_DIR / "policy-initiatives-latest.json"
CONNECTOR_ID = "oecd"

RETRY_SCHEDULE = (1.0, 4.0, 15.0, 60.0)  # копия core/openrouter.py (принцип aiforgood/eurlex)


@dataclass(frozen=True)
class OecdConfig:
    """Разобранный ``pipeline/config/discovery_oecd.yaml`` (спек §4)."""

    enabled: bool
    endpoint: str
    user_agent: str
    crawl_delay_seconds: float
    timeout_seconds: float
    include_categories: tuple[str, ...]
    probe_category: str
    probe_initiative_types: tuple[str, ...]


def load_config(path: Path = CONFIG_PATH) -> OecdConfig:
    """Разобрать ``discovery_oecd.yaml`` — плоский dict -> типизированный ``OecdConfig``."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OecdConfig(
        enabled=bool(raw["enabled"]),
        endpoint=str(raw["endpoint"]),
        user_agent=str(raw["user_agent"]),
        crawl_delay_seconds=float(raw["crawl_delay_seconds"]),
        timeout_seconds=float(raw["timeout_seconds"]),
        include_categories=tuple(raw["include_categories"]),
        probe_category=str(raw["probe_category"]),
        probe_initiative_types=tuple(raw["probe_initiative_types"]),
    )


# --- §3: транспорт — GET policy-initiatives?page=N + retry/backoff (принцип aiforgood/eurlex) ---


def fetch_json(
    page: int, *, endpoint: str, user_agent: str, timeout: float
) -> dict[str, Any]:
    """Одна страница ``GET {endpoint}?page=N`` + retry/backoff (спек §3). Обычный REST-
    эндпоинт: ошибки транспортные (HTTP-код), не in-band-в-200 как у OpenRouter — retry-
    лестница зеркалит ``aiforgood.fetch_json``/``eurlex.fetch_sparql``.

    ``user_agent`` — нейтральная строка конфига: `robots.txt` на api-хосте отсутствует
    (404, живая проба §1), но вежливость — наш собственный выбор, не требование сайта.
    """
    url = f"{endpoint}?page={page}"
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
                raise RuntimeError(f"oecd HTTP {exc.code}: {body[:500]}") from exc
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = str(exc)
        if attempt == total_attempts:
            break
        delay = RETRY_SCHEDULE[attempt - 1]
        print(f"попытка {attempt}/{total_attempts} через {delay:.0f}s: {reason}", file=sys.stderr)
        time.sleep(delay)
    raise RuntimeError(f"oecd: исчерпаны попытки ({total_attempts}) — {reason}")


# --- §3: full-scan пагинация + shape-гейт честной деградации ---


def _check_shape(page: dict[str, Any]) -> None:
    """Backend недокументирован (найден в HTML фронтенда, нигде официально не объявлен,
    §1) — может исчезнуть/сменить форму без предупреждения. Страница 1 обязана нести
    непустой ``data`` И положительный ``total`` И ``lastPage`` >= 1; нарушение -> громкий
    отказ (спек §3), не тихие «0 кандидатов» умершего backend'а."""
    data = page.get("data")
    total = page.get("total")
    last_page = page.get("lastPage")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"oecd: backend изменил форму/деградировал — пустой data: {page!r}")
    if not isinstance(total, int) or total <= 0:
        raise RuntimeError(f"oecd: backend изменил форму/деградировал — total={total!r}")
    if not isinstance(last_page, int) or last_page < 1:
        raise RuntimeError(f"oecd: backend изменил форму/деградировал — lastPage={last_page!r}")


def fetch_all_pages(
    config: OecdConfig,
    *,
    fetch: Callable[..., dict[str, Any]] = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Полный обход ``page=1..lastPage`` (спек §3) — без early-stop по seen-id (rationale:
    редакционная публикация со СТАРЫМ id может всплыть на любой глубине списка; полнота
    важнее 5 минут вежливого обхода). ``sleep`` — между страницами, НЕ перед первой
    (конвенция ``aiforgood.paginate_group``). Расхождение суммы полученного с заявленным
    ``total`` — тоже отказ (backend соврал о своей форме на середине обхода)."""
    first = fetch(1, endpoint=config.endpoint, user_agent=config.user_agent, timeout=config.timeout_seconds)
    _check_shape(first)
    records: list[dict[str, Any]] = list(first["data"])
    last_page = int(first["lastPage"])
    total = int(first["total"])

    for page_num in range(2, last_page + 1):
        sleep(config.crawl_delay_seconds)
        page = fetch(
            page_num, endpoint=config.endpoint, user_agent=config.user_agent,
            timeout=config.timeout_seconds,
        )
        batch = page.get("data")
        if not isinstance(batch, list):
            raise RuntimeError(f"oecd: backend изменил форму на странице {page_num}: {page!r}")
        records.extend(batch)

    if len(records) != total:
        raise RuntimeError(
            f"oecd: заявлен total={total}, реально получено {len(records)} — "
            "backend изменил форму/данные посреди обхода"
        )
    return records


# --- §3: снапшот сырья — страховка от исчезновения недокументированного backend'а ---


def save_snapshot(records: list[dict[str, Any]], *, path: Path = SNAPSHOT_PATH) -> None:
    """Полный сырой массив записей плоским JSON (спек §3) — атомарная перезапись.
    Пишется и при ``--dry-run`` (кэш-артефакт, не store — прецедент snowball
    ``.citations.yaml``). НЕСЁТ PII редакторов OECD (§1) — gitignored, наружу не версионируется."""
    fsio.atomic_write_text(path, json.dumps(records, ensure_ascii=False, indent=2))


# --- §3: курсор — множество виденных native_id (как seen-CELEX у eurlex/aiforgood) ---


def diff_cursor(
    all_ids: list[str], cursor: ConnectorCursor | None
) -> tuple[set[str], ConnectorCursor]:
    """Новые (не виденные) ``id`` + новый курсор = объединение старых и текущих (спек §3).
    Множество СТРОГО растёт — правка/удаление записи в живом индексе не выбрасывает её id
    из seen (тот же принцип, что ``eurlex.diff_cursor``/``aiforgood.diff_cursor``)."""
    seen = set((cursor or {}).get("seen_ids") or [])
    fresh_ids = {i for i in all_ids if i not in seen}
    new_seen = sorted(seen | set(all_ids))
    return fresh_ids, {"seen_ids": new_seen}
