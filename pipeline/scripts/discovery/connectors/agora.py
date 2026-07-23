"""discovery/connectors/agora.py — ETO AGORA (Zenodo bulk) registry-коннектор.

Spec `docs/pipeline/discovery/tech_specs/discovery-agora/spec.md`. Первый реальный
экземпляр архетипа `registry`: fetch (Zenodo API, версионный курсор, zip-кэш,
DuckDB-ingestion — этот слой) + SQL-фильтр (все не-US + US-проба по узкой оси
agentic_g2ai) и маппинг в CandidateRecord — следующий слой того же модуля.
Регистрируется в ядре при импорте (см. ``discovery/connectors/__init__.py``).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core import schema
from core.env import REPO_ROOT
from discovery import dedup, registry_store
from discovery.base import ConnectorCursor, DiscoverResult

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


# --- §4: гибрид-фильтр (все не-US + US-проба по узкой оси agentic_g2ai) ---

_ROW_FIELDS = (
    "agora_id",
    "official_name",
    "casual_name",
    "link",
    "authority",
    "jurisdiction",
    "activity",
    "activity_date",
    "proposed_date",
    "short_summary",
    "tags",
    "machine_flag",
    "haystack",
    "probe_reason",
)

_URL_RE = re.compile(r"^https?://")


def _hybrid_query(match_terms: tuple[str, ...], *, min_year: int, limit: int) -> tuple[str, list[Any]]:
    """SQL к ``registry.duckdb`` — join юрисдикции + два непересекающихся среза (спек §4).

    ``non_us``: весь хвост без ранжирования. ``us_probe``: US-документы с годом >=
    ``min_year`` И >=1 совпадением по границе слова в ``haystack`` (Tags+Official
    name+Short summary), top-``limit`` по (число совпадений, свежесть). Плейсхолдеры
    ``?`` биндятся ПОЗИЦИОННО по порядку появления в тексте запроса — порядок
    ``params`` ниже обязан совпадать (case_sum использован ОДИН раз, не дважды).
    """
    patterns = [rf"\b{term}\b" for term in match_terms]
    case_sum = " + ".join(
        "CASE WHEN regexp_matches(haystack, ?, 'i') THEN 1 ELSE 0 END" for _ in patterns
    )
    columns = ", ".join(
        [
            'agora_id', 'official_name', 'casual_name', 'link', 'authority', 'jurisdiction',
            'activity', 'activity_date', 'proposed_date', 'short_summary', 'tags',
            'machine_flag', 'haystack', 'probe_reason',
        ]
    )
    query = f"""
    WITH joined AS (
        SELECT
            d."AGORA ID" AS agora_id,
            d."Official name" AS official_name,
            d."Casual name" AS casual_name,
            d."Link to document" AS link,
            d."Authority" AS authority,
            COALESCE(a."Jurisdiction", '') AS jurisdiction,
            d."Most recent activity" AS activity,
            d."Most recent activity date" AS activity_date,
            d."Proposed date" AS proposed_date,
            d."Short summary" AS short_summary,
            d."Tags" AS tags,
            d."Summaries and tags may include unreviewed machine output" AS machine_flag,
            COALESCE(d."Tags", '') || ' ' || COALESCE(d."Official name", '')
                || ' ' || COALESCE(d."Short summary", '') AS haystack
        FROM agora.documents_raw d
        LEFT JOIN agora.authorities_raw a ON d."Authority" = a."Name"
    ),
    non_us AS (
        SELECT *, 'non_us' AS probe_reason FROM joined WHERE jurisdiction != 'United States'
    ),
    us_scored AS (
        SELECT *, ({case_sum}) AS match_count FROM joined
        WHERE jurisdiction = 'United States' AND year(activity_date) >= ?
    ),
    us_probe AS (
        SELECT
            agora_id, official_name, casual_name, link, authority, jurisdiction, activity,
            activity_date, proposed_date, short_summary, tags, machine_flag, haystack,
            'us_axis_probe' AS probe_reason
        FROM us_scored
        WHERE match_count > 0
        ORDER BY match_count DESC, activity_date DESC
        LIMIT ?
    )
    SELECT {columns} FROM non_us
    UNION ALL
    SELECT {columns} FROM us_probe
    """
    params: list[Any] = list(patterns) + [min_year, limit]
    return query, params


def _matched_terms(haystack: str, match_terms: tuple[str, ...]) -> list[str]:
    """Термины ``match_terms``, реально совпавшие в ``haystack`` (та же граница слова, что SQL)."""
    return [t for t in match_terms if re.search(rf"\b{re.escape(t)}\b", haystack, re.IGNORECASE)]


def _map_row(row: dict[str, Any], *, match_terms: tuple[str, ...]) -> schema.CandidateRecord | None:
    """Строка гибрид-фильтра -> ``CandidateRecord`` (маппинг §5). ``None`` — пропуск (диагностика
    у вызывающей стороны), не исключение: невалидный ``source_url`` не должен ронять батч."""
    source_url = row["link"]
    if not source_url or not _URL_RE.match(source_url):
        return None

    title = row["official_name"] or row["casual_name"]
    jurisdiction = row["jurisdiction"] or None
    doc_date = row["activity_date"] or row["proposed_date"]

    native_summary = None
    if row["machine_flag"] is False and row["short_summary"]:
        native_summary = row["short_summary"][: schema.CANDIDATE_SUMMARY_MAX]

    tags: list[str] = []
    if row["tags"]:
        tags.extend(t.strip() for t in row["tags"].split(";") if t.strip())
    if row["activity"]:
        tags.append(f"AGORA activity: {row['activity']}")
    if row["probe_reason"] == "us_axis_probe":
        matched = _matched_terms(row["haystack"] or "", match_terms)
        tags.append(f"us-axis-probe (matched: {', '.join(matched)})")

    canonical = "|".join(str(row[field]) for field in _ROW_FIELDS)
    raw_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return schema.CandidateRecord(
        title=title,
        issuer=row["authority"],
        jurisdiction=jurisdiction,
        doc_date=doc_date,
        language=None,
        source_url=source_url,
        native_summary=native_summary,
        native_id=str(row["agora_id"]),
        native_tags=tags or None,
        connector_id=CONNECTOR_ID,
        retrieved_at=dt.date.today(),
        raw_hash=raw_hash,
        normalized_url=dedup.normalize_url(source_url),
    )


def select_and_map_candidates(
    conn: Any, *, match_terms: tuple[str, ...], min_year: int, limit: int
) -> tuple[list[schema.CandidateRecord], int]:
    """Прогнать гибрид-фильтр (§4) и замаппить строки в ``CandidateRecord`` (§5).

    Возвращает ``(кандидаты, пропущено_из_за_невалидного_url)`` — пропуск не роняет батч.
    """
    query, params = _hybrid_query(match_terms, min_year=min_year, limit=limit)
    rows = conn.execute(query, params).fetchall()
    dict_rows = [dict(zip(_ROW_FIELDS, row, strict=True)) for row in rows]

    candidates: list[schema.CandidateRecord] = []
    skipped = 0
    for dict_row in dict_rows:
        cand = _map_row(dict_row, match_terms=match_terms)
        if cand is None:
            skipped += 1
            continue
        candidates.append(cand)
    return candidates, skipped


# --- §3/§6: discover() top-level + Connector protocol ---


def discover_agora(
    cursor: ConnectorCursor | None,
    *,
    config: AgoraConfig | None = None,
    fetch_metadata: Callable[[str], dict[str, Any]] = fetch_latest_metadata,
    download: Callable[[str, Path], None] = download_zip,
    cache_dir: Path = CACHE_DIR,
    db_path: Path = registry_store.DEFAULT_DB_PATH,
) -> DiscoverResult:
    """``Connector.discover()`` для AGORA (спек §3): version-гейт -> fetch+cache+ingest ->
    SQL-фильтр -> маппинг. ``fetch_metadata``/``download`` инжектируемы — тесты подменяют
    их фейками, сеть в CI не участвует."""
    cfg = config or load_config()
    record = fetch_metadata(cfg.zenodo_doi)
    new_cursor = cursor_from_metadata(record)

    if cursor is not None and cursor.get("zenodo_version") == new_cursor["zenodo_version"]:
        return DiscoverResult(candidates=[], cursor=cursor, diagnostics={"status": "unchanged"})

    zip_path = cache_dir / f"agora-{new_cursor['zenodo_version']}.zip"
    if not zip_path.exists():
        download(download_url_from_metadata(record), zip_path)
    ingest_dump(zip_path, source_version=str(new_cursor["zenodo_version"]), db_path=db_path)

    conn = registry_store.connect(db_path)
    try:
        min_year = resolve_min_year(cfg)
        candidates, skipped = select_and_map_candidates(
            conn, match_terms=cfg.us_probe_match_terms, min_year=min_year, limit=cfg.us_probe_limit
        )
    finally:
        conn.close()

    diagnostics = {"status": "fetched", "found": len(candidates), "skipped_invalid_url": skipped}
    return DiscoverResult(candidates=candidates, cursor=new_cursor, diagnostics=diagnostics)


@dataclass(frozen=True)
class AgoraConnector:
    """Реализация протокола ``Connector`` (спек §0/§6) — первый экземпляр архетипа `registry`."""

    id: str = CONNECTOR_ID
    kind: schema.ConnectorKind = schema.ConnectorKind.registry
    enabled: bool = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        return discover_agora(cursor)
