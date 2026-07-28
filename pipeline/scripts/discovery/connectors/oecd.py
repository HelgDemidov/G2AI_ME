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

import datetime as dt
import hashlib
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

from core import fsio, schema
from core.env import REPO_ROOT
from discovery import dedup, registry
from discovery.base import ConnectorCursor, DiscoverResult

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


# --- §2/§3: фильтр объёма — гибрид (полные policy-категории + узкая проба из "projects") ---


def in_scope(record: dict[str, Any], config: OecdConfig) -> bool:
    """Гибридный фильтр §2 (решение куратора 2026-07-25): категория целиком в
    ``include_categories``, ИЛИ категория — ``probe_category`` И ``initiativeType.name``
    в ``probe_initiative_types``. Список include, не exclude (rationale): новая категория
    в таксономии OECD пройдёт мимо честно (счётчик ``skipped_out_of_scope`` у вызывающей
    стороны), не втянется молча."""
    category = record.get("category")
    if category in config.include_categories:
        return True
    if category != config.probe_category:
        return False
    itype = record.get("initiativeType")
    itype_name = itype.get("name") if isinstance(itype, dict) else None
    return itype_name in config.probe_initiative_types


def _valid_url(url: str | None) -> bool:
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


def _record_title(record: dict[str, Any]) -> str:
    return str(record.get("englishName") or record.get("originalName") or "").strip()


def ambiguous_websites(records: list[dict[str, Any]]) -> set[str]:
    """Значения ``website``, которые делят записи с РАЗНЫМИ заголовками (спек
    triage-intake-hardening §6) — для них владельца из данных определить НЕЧЕМ.

    Дефект в экспорте OECD.AI, не в нашем коде: 49 значений на 104 записи снапшота
    (замер 2026-07-28). Похоже на протяжку значения в их конвейере (id соседние), но
    **направление непостоянно** — `kb.se` утёк от «KB Lab» и вверх, и вниз по id,
    поэтому эвристика «владелец — меньший id» ЛОЖНА (проверено: она назначала владельцем
    kb.se финский «Tampere Pulse»). Единственный надёжный вывод — «этому значению нельзя
    верить для ВСЕЙ группы», поэтому помечается вся группа, а не «лишние» в ней.

    Совпадение заголовков (одна инициатива, заведённая дважды — живой случай: канадский
    Voluntary Code, фламандский AI Action Plan, G7 ×2) группу НЕ порочит: там общий URL
    законен. Сравнение по ``dedup.normalized_title`` — та же нормализация, что у
    стратегии 2 дедупа, чтобы «AI Action Plan» и «AI  Action  Plan» не разошлись.
    """
    by_site: dict[str, set[str]] = {}
    for record in records:
        website = record.get("website")
        if not _valid_url(website):
            continue
        title = _record_title(record)
        if not title:
            continue
        by_site.setdefault(str(website), set()).add(dedup.normalized_title(title))
    return {site for site, titles in by_site.items() if len(titles) > 1}


def _source_url(record: dict[str, Any]) -> str | None:
    """``website``, иначе первый валидный ``relevantUrls[0]`` (спек §3). ``relevantFiles``
    (OECD-hosted PDF) сознательно НЕ фолбэк — rehost, не официальный первоисточник
    (rationale)."""
    website = record.get("website")
    if _valid_url(website):
        return website
    urls = record.get("relevantUrls")
    if isinstance(urls, list) and urls and _valid_url(urls[0]):
        return str(urls[0])
    return None


# --- §3: маппинг записи -> CandidateRecord (Faker/PII-барьеры, §1) ---


def _map_record(
    record: dict[str, Any], *, ambiguous_sites: frozenset[str] | set[str] = frozenset()
) -> schema.CandidateRecord | None:
    """Одна запись ``data[]`` -> ``CandidateRecord`` (маппинг §3). ``None`` — пропуск
    (диагностика у вызывающей стороны), не исключение — data-quality отсев, не relevance.

    ``ambiguous_sites`` (спек triage-intake-hardening §6) — значения ``website``, которым
    нельзя верить (см. ``ambiguous_websites``); считается вызывающей стороной ОДИН раз по
    полному списку записей, потому что свойство групповое и по одной записи не выводится.

    **Faker-барьер (§1):** из ``gaiinCountry``/``intergovernmentalOrganisation`` читается
    ТОЛЬКО ``.name`` — оба объекта несут порченные Faker.js-поля (демография countries;
    у intergovernmentalOrganisation живьём подтверждено то же самое — ``website``/
    ``description`` вида ``"http://demetrius.info"``/``"Temporibus rerum cupiditate."``,
    не настоящие данные организации). **PII-барьер (§1):** ``createdByEmail``/
    ``publishedByEmail``/``*ByName`` и т.п. не читаются НИГДЕ в этой функции — живые
    адреса @oecd.org не должны попасть в версионируемый ``candidates.yaml``.
    """
    native_id = record.get("id")
    if native_id is None:
        return None
    native_id_str = str(native_id)

    english_name = (record.get("englishName") or "").strip()
    original_name = (record.get("originalName") or "").strip()
    title = english_name or original_name or None
    if not title:
        return None

    source_url = _source_url(record)
    if source_url is None:
        return None

    country = record.get("gaiinCountry")
    igo = record.get("intergovernmentalOrganisation")
    jurisdiction: str | None = None
    if isinstance(country, dict):
        jurisdiction = country.get("name")
    elif isinstance(igo, dict):
        jurisdiction = igo.get("name")

    issuer = record.get("responsibleOrganisation")
    issuer = issuer if isinstance(issuer, str) and issuer.strip() else None

    summary = (record.get("description") or "").strip()
    native_summary = summary[: schema.CANDIDATE_SUMMARY_MAX] if summary else None

    category = record.get("category")
    itype = record.get("initiativeType")
    itype_name = itype.get("name") if isinstance(itype, dict) else None
    extent_binding = record.get("extentBinding")
    start_year = record.get("startYear")

    native_tags: list[str] = []
    if category:
        native_tags.append(f"category: {category}")
    if itype_name:
        native_tags.append(f"type: {itype_name}")
    if extent_binding:
        native_tags.append(f"binding: {extent_binding}")
    # ``start_year`` в native_tags БОЛЬШЕ НЕ дублируется (решение куратора 2026-07-28):
    # год стал структурным полем ``doc_year``, а native_tags остаются рубриками источника
    # в ЕГО таксономии — держать одно значение в двух местах значит дать им разойтись.

    canonical = "|".join([native_id_str, english_name, str(record.get("updatedAt"))])
    raw_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return schema.CandidateRecord(
        title=title,
        # Реестр отдаёт заголовок КАК заголовок (spec triage-intake-hardening §3).
        title_provenance=schema.TitleProvenance.stated,
        issuer=issuer,
        jurisdiction=jurisdiction,
        doc_date=None,   # источник даёт только год — фабриковать 1 января нельзя
        doc_year=int(start_year) if start_year else None,
        language=None,
        source_url=source_url,
        native_summary=native_summary,
        native_id=native_id_str,
        native_tags=native_tags or None,
        connector_id=CONNECTOR_ID,
        retrieved_at=dt.date.today(),
        raw_hash=raw_hash,
        normalized_url=dedup.normalize_url(source_url),
        native_format_hint=dedup.format_hint_from_url(source_url),
        url_provenance=(
            schema.UrlProvenance.suspect
            if record.get("website") in ambiguous_sites
            else schema.UrlProvenance.stated
        ),
    )


# --- §3: discover_oecd() top-level ---


def discover_oecd(
    cursor: ConnectorCursor | None,
    *,
    config: OecdConfig | None = None,
    fetch: Callable[..., dict[str, Any]] = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
    snapshot_path: Path = SNAPSHOT_PATH,
) -> DiscoverResult:
    """``Connector.discover()`` для oecd (спек §3): полный обход -> снапшот сырья ->
    фильтр §2 -> маппинг -> отсев по seen-id-курсору.

    ``fetch``/``sleep`` инжектируем — тесты подменяют фейками, сеть/реальные паузы в CI
    не участвуют. Снапшот пишется ВСЕГДА (даже при ``--dry-run`` у вызывающего CLI) —
    кэш-артефакт-страховка, не store."""
    cfg = config or load_config()
    records = fetch_all_pages(cfg, fetch=fetch, sleep=sleep)
    save_snapshot(records, path=snapshot_path)

    candidates: list[schema.CandidateRecord] = []
    skipped_out_of_scope = 0
    # Комбинированный счётчик (принцип aiforgood.skipped_no_title_or_url): _map_record
    # отдаёт None как единый data-quality сигнал (нет id/title/URL), не разложенный по
    # причине — реконструировать причину постфактум у вызывающей стороны было бы
    # дублированием той же логики, что уже внутри _map_record, и рисковало бы разойтись
    # с ней при будущей правке.
    skipped_unmappable = 0
    # Свойство ГРУППОВОЕ (значение website делят записи с разными заголовками), поэтому
    # считается один раз по ПОЛНОМУ списку — до фильтра области видимости: запись вне
    # нашей области всё равно доказывает, что значение спорное.
    ambiguous_sites = ambiguous_websites(records)

    for record in records:
        if not in_scope(record, cfg):
            skipped_out_of_scope += 1
            continue
        cand = _map_record(record, ambiguous_sites=ambiguous_sites)
        if cand is None:
            skipped_unmappable += 1
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
        "skipped_out_of_scope": skipped_out_of_scope,
        "skipped_unmappable": skipped_unmappable,
        # Не отсев: кандидат допущен, но его URL требует проверки куратором при admit
        # (заголовок/издатель/юрисдикция у него ВЕРНЫЕ — теряется только адрес).
        "url_suspect": sum(
            1 for c in candidates if c.url_provenance is schema.UrlProvenance.suspect
        ),
    }
    return DiscoverResult(candidates=fresh, cursor=new_cursor, diagnostics=diagnostics)


@dataclass
class OecdConnector:
    """Реализация протокола ``Connector`` (спек §0/§3) — четвёртый экземпляр архетипа
    `registry`. НЕ ``frozen`` — симметрично ``AgoraConnector``/``EurlexConnector``/
    ``AiforgoodConnector`` (Protocol требует settable-атрибуты)."""

    id: str = CONNECTOR_ID
    kind: schema.ConnectorKind = schema.ConnectorKind.registry
    enabled: bool = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        return discover_oecd(cursor)


# Регистрация при импорте (чартер §4.3 «манифест», спек §3): `enabled` — из конфига,
# не хардкод. Срабатывает один раз за интерпретатор — по факту импорта этого модуля
# (см. `discovery/connectors/__init__.py` + `discover.py`).
registry.register(OecdConnector(enabled=load_config().enabled))
