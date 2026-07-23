"""discovery/connectors/eurlex.py — EUR-Lex/CELLAR (живой SPARQL) registry-коннектор.

Spec `docs/pipeline/discovery/tech_specs/discovery-eurlex/spec.md`. Второй экземпляр
архетипа `registry` после AGORA — и первый, где источник живой запрашиваемый индекс
(SPARQL), а не одноразовый bulk-дамп: без DuckDB/registry_store (§2 спека), курсор —
множество виденных CELEX, а не version-гейт (§4). Регистрируется в ядре при импорте
(см. ``discovery/connectors/__init__.py``).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.env import REPO_ROOT
from discovery.base import ConnectorCursor

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_eurlex.yaml"
CONNECTOR_ID = "eurlex"

RETRY_SCHEDULE = (1.0, 4.0, 15.0, 60.0)  # копия core/openrouter.py (спек §2) — тот же
                                          # принцип, локальный маленький цикл без нового модуля


@dataclass(frozen=True)
class EurlexConfig:
    """Разобранный ``pipeline/config/discovery_eurlex.yaml`` (спек §7)."""

    enabled: bool
    sparql_endpoint: str
    eurovoc_concepts: tuple[str, ...]
    expression_language: str
    result_limit: int
    timeout_seconds: float


def load_config(path: Path = CONFIG_PATH) -> EurlexConfig:
    """Разобрать ``discovery_eurlex.yaml`` — плоский dict -> типизированный ``EurlexConfig``."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EurlexConfig(
        enabled=bool(raw["enabled"]),
        sparql_endpoint=str(raw["sparql_endpoint"]),
        eurovoc_concepts=tuple(raw["eurovoc_concepts"]),
        expression_language=str(raw["expression_language"]),
        result_limit=int(raw["result_limit"]),
        timeout_seconds=float(raw["timeout_seconds"]),
    )


# --- §2: SPARQL-транспорт (retry/backoff — принцип core/openrouter.py) ---


def fetch_sparql(query: str, *, endpoint: str, timeout: float) -> dict[str, Any]:
    """GET-запрос к SPARQL-эндпоинту + retry/backoff (спек §2). CELLAR — обычный
    SPARQL-эндпоинт: ошибки транспортные (HTTP-код), не in-band-в-200 как у OpenRouter —
    retry-лестница проще ``core/openrouter.chat_request`` (нет ``InbandError``-ветки)."""
    params = urllib.parse.urlencode({"query": query, "format": "application/sparql-results+json"})
    url = f"{endpoint}?{params}"
    reason = ""
    total_attempts = len(RETRY_SCHEDULE) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                body = exc.read().decode("utf-8", "replace")
                raise RuntimeError(f"EUR-Lex SPARQL HTTP {exc.code}: {body[:500]}") from exc
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = str(exc)
        if attempt == total_attempts:
            break
        delay = RETRY_SCHEDULE[attempt - 1]
        print(f"попытка {attempt}/{total_attempts} через {delay:.0f}s: {reason}", file=sys.stderr)
        time.sleep(delay)
    raise RuntimeError(f"EUR-Lex SPARQL: исчерпаны попытки ({total_attempts}) — {reason}")


# --- §4: курсор — множество виденных CELEX (не version-гейт, как у AGORA) ---


def diff_cursor(
    all_ids: list[str], cursor: ConnectorCursor | None
) -> tuple[set[str], ConnectorCursor]:
    """Новые (не виденные) id + новый курсор = объединение старых и текущих (спек §4).

    Работает на голых CELEX-строках, не на ``CandidateRecord`` — идемпотентность курсора
    не зависит от формы маппинга. Множество СТРОГО растёт (никогда не уменьшается) —
    правка/исчезновение работы в живом индексе не выбрасывает её CELEX из seen (§Вне скоупа).
    """
    seen = set((cursor or {}).get("seen_celex") or [])
    fresh_ids = {i for i in all_ids if i not in seen}
    new_seen = sorted(seen | set(all_ids))
    return fresh_ids, {"seen_celex": new_seen}
