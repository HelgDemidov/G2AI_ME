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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.env import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_oecd.yaml"
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
