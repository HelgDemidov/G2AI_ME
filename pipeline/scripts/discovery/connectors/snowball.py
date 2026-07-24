"""discovery/connectors/snowball.py — backward-snowballing по собственному корпусу.

Spec `docs/pipeline/discovery/tech_specs/discovery-snowball/spec.md`. Пятый архетип
(`ConnectorKind.snowball`) — единственный, читающий не внешний источник, а уже принятые
документы корпуса (`raw.*`/`doc.md`): гиперлинк-аннотации raw.pdf, href raw.html,
напечатанные URL doc.md (§2), плюс opt-in LLM-стадия текстовых цитат без URL (§5).
Регистрируется в ядре при импорте (см. ``discovery/connectors/__init__.py``).

Этот коммит — только конфиг (§3 спека): типизированный ``SnowballConfig`` + ``load_config``.
Экстракторы/маппинг/курсор/регистрация коннектора — последующие коммиты.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.env import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_snowball.yaml"
CONNECTOR_ID = "snowball"


@dataclass(frozen=True)
class SourceFilter:
    """Какие документы корпуса майнить (спек §3). Пустые кортежи — разрешающие (без фильтра)."""

    tracks: tuple[str, ...]
    target_fit: tuple[str, ...]
    include_doc_ids: tuple[str, ...]
    exclude_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class UrlFilter:
    """Какие найденные URL отсеивать (спек §3). Пустые кортежи — ничего не режем."""

    exclude_domains: tuple[str, ...]
    exclude_url_substrings: tuple[str, ...]


@dataclass(frozen=True)
class EmitConfig:
    """Тумблеры экстракторов (спек §2/§5) — независимо включаемые/выключаемые каналы."""

    pdf_annotations: bool
    html_hrefs: bool
    printed_urls: bool
    text_citations: bool


@dataclass(frozen=True)
class SnowballConfig:
    """Разобранный ``pipeline/config/discovery_snowball.yaml`` (спек §3)."""

    enabled: bool
    source_filter: SourceFilter
    url_filter: UrlFilter
    emit: EmitConfig
    max_candidates: int | None
    citations_model: str


def _validate_max_candidates(value: Any) -> int | None:
    """Sanity-чек `max_candidates` (спек §3): ``None`` — без капа; иначе целое >= 0.

    Отклоняет отрицательные/нецелые/строковые значения ДО старта майнинга (fail-fast
    в конфиге, не на середине прогона) — ``bool`` явно исключён (``isinstance(True, int)``
    истинно в Python, но булево значение здесь не осмысленно как кап).
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"max_candidates: ожидалось целое >= 0 или null, получено {value!r}")
    if value < 0:
        raise ValueError(f"max_candidates: ожидалось целое >= 0, получено {value!r}")
    return value


def _tuple_of_str(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(str(v) for v in (raw.get(key) or []))


def load_config(path: Path = CONFIG_PATH) -> SnowballConfig:
    """Разобрать ``discovery_snowball.yaml`` — плоский dict -> типизированный ``SnowballConfig``."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    source_filter_raw: dict[str, Any] = raw.get("source_filter") or {}
    url_filter_raw: dict[str, Any] = raw.get("url_filter") or {}
    emit_raw: dict[str, Any] = raw.get("emit") or {}

    return SnowballConfig(
        enabled=bool(raw["enabled"]),
        source_filter=SourceFilter(
            tracks=_tuple_of_str(source_filter_raw, "tracks"),
            target_fit=_tuple_of_str(source_filter_raw, "target_fit"),
            include_doc_ids=_tuple_of_str(source_filter_raw, "include_doc_ids"),
            exclude_doc_ids=_tuple_of_str(source_filter_raw, "exclude_doc_ids"),
        ),
        url_filter=UrlFilter(
            exclude_domains=_tuple_of_str(url_filter_raw, "exclude_domains"),
            exclude_url_substrings=_tuple_of_str(url_filter_raw, "exclude_url_substrings"),
        ),
        emit=EmitConfig(
            pdf_annotations=bool(emit_raw.get("pdf_annotations", True)),
            html_hrefs=bool(emit_raw.get("html_hrefs", True)),
            printed_urls=bool(emit_raw.get("printed_urls", True)),
            text_citations=bool(emit_raw.get("text_citations", False)),
        ),
        max_candidates=_validate_max_candidates(raw.get("max_candidates")),
        citations_model=str(raw["citations_model"]),
    )
