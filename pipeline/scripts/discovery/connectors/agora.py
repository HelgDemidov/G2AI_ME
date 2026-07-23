"""discovery/connectors/agora.py — ETO AGORA (Zenodo bulk) registry-коннектор.

Spec `docs/pipeline/discovery/tech_specs/discovery-agora/spec.md`. Первый реальный
экземпляр архетипа `registry`: fetch (Zenodo API, версионный курсор, zip-кэш,
DuckDB-ingestion — этот слой) + SQL-фильтр (все не-US + US-проба по узкой оси
agentic_g2ai) и маппинг в CandidateRecord — следующий слой того же модуля.
Регистрируется в ядре при импорте (см. ``discovery/connectors/__init__.py``).
"""
from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.env import REPO_ROOT
from discovery import registry_store
from discovery.base import ConnectorCursor

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_agora.yaml"
TRIAGE_CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "triage.yaml"
CACHE_DIR = REPO_ROOT / "pipeline" / "discovery_cache" / "agora"
CONNECTOR_ID = "agora"

_CONCEPT_RECID_RE = re.compile(r"zenodo\.(\d+)$")


@dataclass(frozen=True)
class AgoraConfig:
    """Разобранный ``pipeline/config/discovery_agora.yaml`` (спек §4)."""

    enabled: bool
    zenodo_doi: str
    non_us_include_all: bool
    us_probe_limit: int
    us_probe_min_year: int | None
    us_probe_match_terms: tuple[str, ...]


def load_config(path: Path = CONFIG_PATH) -> AgoraConfig:
    """Разобрать ``discovery_agora.yaml`` — плоский dict -> типизированный ``AgoraConfig``."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    probe = raw["us_axis_probe"]
    return AgoraConfig(
        enabled=bool(raw["enabled"]),
        zenodo_doi=str(raw["zenodo_doi"]),
        non_us_include_all=bool(raw["non_us"]["include_all"]),
        us_probe_limit=int(probe["limit"]),
        us_probe_min_year=probe.get("min_year"),
        us_probe_match_terms=tuple(probe["match_terms"]),
    )


def frontier_year(path: Path = TRIAGE_CONFIG_PATH) -> int:
    """``frontier_year`` из ``triage.yaml`` — единый источник истины планки фронтира.

    Прецедент — скилл directed-search уже читает то же значение оттуда; ``discovery_agora.yaml``
    не дублирует число (``min_year: null`` -> резолвится через эту функцию, спек §4).
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return int(raw["frontier_year"])


def resolve_min_year(config: AgoraConfig) -> int:
    """``us_probe_min_year`` override, если задан, иначе ``frontier_year`` из triage.yaml."""
    if config.us_probe_min_year is not None:
        return config.us_probe_min_year
    return frontier_year()


def _concept_recid(doi: str) -> str:
    """Zenodo concept DOI (``10.5281/zenodo.<recid>``) -> concept recid для ``/versions/latest``."""
    match = _CONCEPT_RECID_RE.search(doi)
    if not match:
        raise ValueError(f"не удалось извлечь Zenodo concept recid из DOI: {doi!r}")
    return match.group(1)


def fetch_latest_metadata(doi: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Резолвить concept DOI -> метаданные ТЕКУЩЕЙ версии записи Zenodo (без скачивания zip).

    Живьём проверено 2026-07-23: ``GET /api/records/{concept_recid}/versions/latest``
    отдаёт HTTP 301 с JSON-телом ``{"location": ...}``; ``urllib`` следует редиректу
    сам и возвращает полную запись целевой версии (``id``/``metadata.version``/``files``).
    """
    url = f"https://zenodo.org/api/records/{_concept_recid(doi)}/versions/latest"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def cursor_from_metadata(record: dict[str, Any]) -> ConnectorCursor:
    """Zenodo record JSON -> непрозрачный курсор коннектора (спек §3): версия/id/md5 файла."""
    files = record.get("files") or []
    md5 = files[0]["checksum"] if files else None
    return {"zenodo_version": record["metadata"]["version"], "record_id": record["id"], "md5": md5}


def download_url_from_metadata(record: dict[str, Any]) -> str:
    """Прямая ссылка на содержимое единственного файла записи (``agora.zip``)."""
    files = record.get("files") or []
    if not files:
        raise ValueError(f"запись Zenodo {record.get('id')} не содержит файлов")
    return str(files[0]["links"]["self"])


def download_zip(download_url: str, dest: Path, *, timeout: float = 300.0) -> None:
    """Скачать zip-архив (спек §3.3). Кэш идемпотентен — вызывающая сторона не зовёт
    повторно на уже скачанную версию (см. discover(), commit 4)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(download_url, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


def ingest_dump(
    zip_path: Path, *, source_version: str, db_path: Path = registry_store.DEFAULT_DB_PATH
) -> None:
    """Распаковать zip и загрузить ``documents.csv``+``authorities.csv`` в ``registry.duckdb``.

    CSV лежат в подпапке ``agora/`` ВНУТРИ архива (проверено живьём на реальном дампе,
    не в корне zip) — распаковка идёт во временную папку рядом с zip-кэшем, не в его корень.
    """
    extract_dir = zip_path.parent / f"_extract-{source_version}"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    conn = registry_store.connect(db_path)
    try:
        registry_store.ingest_csv(
            conn,
            schema="agora",
            table="documents_raw",
            csv_path=extract_dir / "agora" / "documents.csv",
            source_version=source_version,
        )
        registry_store.ingest_csv(
            conn,
            schema="agora",
            table="authorities_raw",
            csv_path=extract_dir / "agora" / "authorities.csv",
            source_version=source_version,
        )
    finally:
        conn.close()
