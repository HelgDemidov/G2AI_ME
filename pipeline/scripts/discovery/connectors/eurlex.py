"""discovery/connectors/eurlex.py — EUR-Lex/CELLAR (живой SPARQL) registry-коннектор.

Spec `docs/pipeline/discovery/tech_specs/discovery-eurlex/spec.md`. Второй экземпляр
архетипа `registry` после AGORA — и первый, где источник живой запрашиваемый индекс
(SPARQL), а не одноразовый bulk-дамп: без DuckDB/registry_store (§2 спека), курсор —
множество виденных CELEX, а не version-гейт (§4). Регистрируется в ядре при импорте
(см. ``discovery/connectors/__init__.py``).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
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
from discovery import dedup, registry
from discovery.base import ConnectorCursor, DiscoverResult

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


# --- производный язык: единый источник — expression_language конфига ---

_LANGUAGE_AUTHORITY = "http://publications.europa.eu/resource/authority/language"

# EU authority alpha-3 (значения expression_language/cdm:expression_uses_language) ->
# ISO 639-1 alpha-2 (CandidateRecord.language / URL-сегмент eur-lex.europa.eu /
# LANG()-фильтр SPARQL). Живьём сверено с authority/language SPARQL-таблицей CELLAR
# (2026-07-23, skos:exactMatch -> id.loc.gov/vocabulary/iso639-1/<код>) — НЕ по памяти:
# первые-2-буквы труncation не работает (EST -> "et", не "ES" — это испанский код).
# Скоуп — ровно 24 официальных языка ЕС: EUR-Lex физически не публикует ни на каких
# других (см. чартер §2 "24 языка, вкл. et/hr" — это и есть исчерпывающий список).
_EXPRESSION_LANGUAGE_TO_ISO = {
    "BUL": "bg", "HRV": "hr", "CES": "cs", "DAN": "da", "NLD": "nl", "ENG": "en",
    "EST": "et", "FIN": "fi", "FRA": "fr", "DEU": "de", "ELL": "el", "HUN": "hu",
    "GLE": "ga", "ITA": "it", "LAV": "lv", "LIT": "lt", "MLT": "mt", "POL": "pl",
    "POR": "pt", "RON": "ro", "SLK": "sk", "SLV": "sl", "SPA": "es", "SWE": "sv",
}


def resolve_iso_language(expression_language: str) -> str:
    """``expression_language``-конфиг (EU authority alpha-3) -> ISO 639-1 alpha-2 —
    ЕДИНЫЙ источник для трёх мест, которые раньше были захардкожены на ``"en"``
    независимо друг от друга (URL-сегмент, ``CandidateRecord.language``, SPARQL
    LANG()-фильтр издателя): смена ``expression_language`` в конфиге раньше молча НЕ
    доходила ни до одного из них — рассинхрон конфиг/код, найденный куратором.

    Неизвестный код -> ``ValueError`` (не молчаливая порча значения): в отличие от
    ``decode_celex_type``/``concept_label`` (pre-signal триажу, безобидная ошибка),
    ``language`` становится частью ``CandidateRecord.language`` -> ``SourceRecord.language``
    -> реальные решения слоя знаний (фасеты/retrieval/OCR-языки) — неверный, но
    похожий на правду код здесь опаснее явного отказа.
    """
    iso = _EXPRESSION_LANGUAGE_TO_ISO.get(expression_language.upper())
    if iso is None:
        known = ", ".join(sorted(_EXPRESSION_LANGUAGE_TO_ISO))
        raise ValueError(
            f"неизвестный expression_language {expression_language!r} — "
            f"добавьте в _EXPRESSION_LANGUAGE_TO_ISO или используйте один из: {known}"
        )
    return iso


# --- §3: SPARQL-запрос (широкий тег-фильтр — тип решает триаж, не коннектор) ---


def build_query(config: EurlexConfig) -> str:
    """Построить SPARQL-запрос §3 из конфига: VALUES-список EuroVoc-концептов + язык
    выражения + safety-cap лимит. Никакого CELEX-сектора/типа в запросе — коннектор
    не судит тип документа (§0), это pre-signal триажу (см. ``decode_celex_type``)."""
    if not config.eurovoc_concepts:
        raise ValueError("eurovoc_concepts пуст — нечего искать")
    values = "\n".join(f"    <{c}>" for c in config.eurovoc_concepts)
    language_uri = f"{_LANGUAGE_AUTHORITY}/{config.expression_language}"
    iso_lang = resolve_iso_language(config.expression_language)
    return f"""
PREFIX cdm:  <http://publications.europa.eu/ontology/cdm#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?celex ?date ?title ?authorLabel ?concept WHERE {{
  VALUES ?concept {{
{values}
  }}
  ?work cdm:work_is_about_concept_eurovoc ?concept .
  ?work cdm:resource_legal_id_celex ?celex .
  OPTIONAL {{ ?work cdm:work_date_document ?date . }}
  OPTIONAL {{
    ?expr cdm:expression_belongs_to_work ?work .
    ?expr cdm:expression_uses_language <{language_uri}> .
    ?expr cdm:expression_title ?title .
  }}
  OPTIONAL {{
    ?work cdm:work_created_by_agent ?agent .
    ?agent skos:prefLabel ?authorLabel . FILTER(LANG(?authorLabel) = "{iso_lang}")
  }}
}} ORDER BY DESC(?date) LIMIT {config.result_limit}
"""


def _binding_value(row: dict[str, Any], var: str) -> str | None:
    binding = row.get(var)
    return str(binding["value"]) if binding else None


def parse_bindings(sparql_json: dict[str, Any]) -> list[dict[str, str | None]]:
    """SPARQL-JSON -> список сырых строк (celex/date/title/authorLabel/concept),
    ДО группировки по CELEX. Отсутствующая OPTIONAL-переменная -> None, не KeyError."""
    bindings = sparql_json.get("results", {}).get("bindings", [])
    return [
        {
            "celex": _binding_value(row, "celex"),
            "date": _binding_value(row, "date"),
            "title": _binding_value(row, "title"),
            "authorLabel": _binding_value(row, "authorLabel"),
            "concept": _binding_value(row, "concept"),
        }
        for row in bindings
    ]


def group_by_celex(rows: list[dict[str, str | None]]) -> dict[str, dict[str, Any]]:
    """Схлопнуть строки (co-authored работа -> N строк, multi-concept матч -> тоже N
    строк) в одну запись на CELEX (спек §3/§5): первая непустая дата/заголовок,
    авторы склеены по первому появлению без дублей, концепты — множество."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        celex = row.get("celex")
        if not celex:
            continue
        entry = grouped.setdefault(
            celex, {"date": None, "title": None, "authors": [], "concepts": set()}
        )
        if row.get("date") and not entry["date"]:
            entry["date"] = row["date"]
        if row.get("title") and not entry["title"]:
            entry["title"] = row["title"]
        author = row.get("authorLabel")
        if author and author not in entry["authors"]:
            entry["authors"].append(author)
        concept = row.get("concept")
        if concept:
            entry["concepts"].add(concept)
    return grouped


# --- §5.1: CELEX-декод (pre-signal ТИПА для триажа, best-effort, НЕ вердикт) ---

_SECTOR_LABELS = {
    "3": "legal act",
    "5": "preparatory act",
    "6": "case-law",
    "4": "international agreement",
    "7": "national transposition",
}

_TYPE_LABELS = {
    "R": "Regulation",
    "L": "Directive",
    "D": "Decision",
    "H": "Recommendation",
    "G": "Resolution/guideline",
    "M": "merger notification",
    "A": "opinion",
    "C": "communication/notice",
    "DC": "communication/notice",
    "PC": "proposal",
    "SC": "staff working document",
}

_CELEX_TYPE_RE = re.compile(r"^\d\d{4}([A-Z]{1,2})\d")


def decode_celex_type(celex: str) -> str:
    """CELEX -> читаемый pre-signal типа (спек §5.1). Неизвестный сектор/тип -> сырьё
    (не краш) — это подсказка триажу, не гейт (§0): точность здесь не нужна."""
    sector_label = _SECTOR_LABELS.get(celex[:1], "other") if celex else "other"
    match = _CELEX_TYPE_RE.match(celex)
    type_code = match.group(1) if match else ""
    type_label = _TYPE_LABELS.get(type_code, type_code or "unknown")
    return f"EUR-Lex: {sector_label}, {type_label}"


# --- концепт EuroVoc -> человекочитаемый ярлык (native_tags) ---

_CONCEPT_LABELS = {
    "http://eurovoc.europa.eu/3030": "artificial intelligence",
    "http://eurovoc.europa.eu/c_3dfe52ca": "machine learning",
    "http://eurovoc.europa.eu/c_65b9cd79": "artificial neural network",
    "http://eurovoc.europa.eu/c_67092197": "natural language processing",
    "http://eurovoc.europa.eu/c_df93fd35": "text and data mining",
    "http://eurovoc.europa.eu/c_5a195ffd": "smart technology",
    "http://eurovoc.europa.eu/3293": "cybernetics",
}


def concept_label(uri: str) -> str:
    """URI EuroVoc-концепта -> человекочитаемый ярлык. Неизвестный концепт (напр.
    добавленный в конфиг куратором позже) -> последний сегмент URI, не краш — тот
    же принцип 'unknown -> raw', что у ``decode_celex_type``."""
    return _CONCEPT_LABELS.get(uri, uri.rsplit("/", 1)[-1])


# --- §5: маппинг сгруппированного результата -> CandidateRecord ---


def _build_source_url(celex: str, iso_lang: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/{iso_lang.upper()}/TXT/HTML/?uri=CELEX:{celex}"


def _decode_date(raw: str | None) -> dt.date | None:
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _map_group(
    celex: str, entry: dict[str, Any], *, iso_lang: str
) -> schema.CandidateRecord | None:
    """Одна сгруппированная запись -> ``CandidateRecord``. ``None`` — пропуск: без
    непустого заголовка на ``iso_lang`` кандидат непромоутим (``title`` обязателен) —
    data-quality отсев (§0/§5), НЕ relevance-суждение о типе документа.

    ``iso_lang`` — РЕЗОЛЬВНУТЫЙ (``resolve_iso_language``) код, тот же, что ушёл в
    SPARQL-запрос (§3) — источник ``language`` записи и URL-сегмента, ни то ни другое
    больше не хардкожено на ``"en"`` независимо от конфига."""
    title = entry.get("title")
    if not title:
        return None

    source_url = _build_source_url(celex, iso_lang)
    authors: list[str] = entry.get("authors") or []
    issuer = " / ".join(authors) if authors else None
    doc_date = _decode_date(entry.get("date"))
    concepts: set[str] = entry.get("concepts") or set()
    concept_names = sorted(concept_label(c) for c in concepts)

    native_tags = [decode_celex_type(celex)]
    if concept_names:
        native_tags.append("EuroVoc: " + ", ".join(concept_names))

    canonical = "|".join([celex, str(doc_date), title, issuer or ""])
    raw_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return schema.CandidateRecord(
        title=title,
        issuer=issuer,
        jurisdiction="European Union",
        doc_date=doc_date,
        language=iso_lang,
        source_url=source_url,
        native_summary=None,
        native_id=celex,
        native_tags=native_tags,
        rights=schema.Rights.cc_by,
        connector_id=CONNECTOR_ID,
        retrieved_at=dt.date.today(),
        raw_hash=raw_hash,
        normalized_url=dedup.normalize_url(source_url),
    )


def map_rows_to_candidates(
    rows: list[dict[str, str | None]], *, iso_lang: str = "en",
) -> tuple[list[schema.CandidateRecord], int]:
    """Сгруппировать строки по CELEX (§3) и замаппить в ``CandidateRecord`` (§5).

    ``iso_lang`` — РЕЗОЛЬВНУТЫЙ (``resolve_iso_language``) ISO 639-1 код; вызывающая
    сторона (``discover_eurlex``) резолвит его из ``config.expression_language`` и
    передаёт сюда. Дефолт ``"en"`` — для прямых вызовов этой функции с фикстурой без
    полного конфига (обратная совместимость существующих тестов).

    Возвращает ``(кандидаты, пропущено_без_заголовка)`` — пропуск не роняет батч.
    Порядок кандидатов следует порядку первого появления CELEX во входных строках
    (``ORDER BY DESC(?date)`` запроса — детерминизм от сервера, не от Python).
    """
    grouped = group_by_celex(rows)
    candidates: list[schema.CandidateRecord] = []
    skipped = 0
    for celex, entry in grouped.items():
        cand = _map_group(celex, entry, iso_lang=iso_lang)
        if cand is None:
            skipped += 1
            continue
        candidates.append(cand)
    return candidates, skipped


# --- discover_eurlex() top-level ---


def discover_eurlex(
    cursor: ConnectorCursor | None,
    *,
    config: EurlexConfig | None = None,
    fetch: Callable[..., dict[str, Any]] = fetch_sparql,
) -> DiscoverResult:
    """``Connector.discover()`` для EUR-Lex (спек §3/§4): построить запрос -> выполнить
    -> распарсить -> сгруппировать+замаппить -> отфильтровать по seen-CELEX-курсору.
    ``fetch`` инжектируем — тесты подменяют фейком, сеть в CI не участвует."""
    cfg = config or load_config()
    iso_lang = resolve_iso_language(cfg.expression_language)
    query = build_query(cfg)
    sparql_json = fetch(query, endpoint=cfg.sparql_endpoint, timeout=cfg.timeout_seconds)
    rows = parse_bindings(sparql_json)
    candidates, skipped = map_rows_to_candidates(rows, iso_lang=iso_lang)

    all_ids = [c.native_id for c in candidates if c.native_id]
    fresh_ids, new_cursor = diff_cursor(all_ids, cursor)
    fresh = [c for c in candidates if c.native_id in fresh_ids]

    status = "no_new" if cursor is not None and not fresh else "fetched"
    diagnostics = {
        "status": status,
        "found": len(candidates),
        "fresh": len(fresh),
        "skipped_no_title": skipped,
    }
    return DiscoverResult(candidates=fresh, cursor=new_cursor, diagnostics=diagnostics)


@dataclass
class EurlexConnector:
    """Реализация протокола ``Connector`` (спек §0/§6) — второй экземпляр архетипа
    `registry`. НЕ ``frozen`` — симметрично ``AgoraConnector`` (Protocol требует
    settable-атрибуты, даже если ничего не переприсваивается)."""

    id: str = CONNECTOR_ID
    kind: schema.ConnectorKind = schema.ConnectorKind.registry
    enabled: bool = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        return discover_eurlex(cursor)


# Регистрация при импорте (чартер §4.3 «манифест», спек §6): `enabled` — из конфига,
# не хардкод. Срабатывает один раз за интерпретатор — по факту импорта этого модуля
# (см. `discovery/connectors/__init__.py` + `discover.py`).
registry.register(EurlexConnector(enabled=load_config().enabled))
