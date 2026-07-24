"""discovery/connectors/snowball.py — backward-snowballing по собственному корпусу.

Spec `docs/pipeline/discovery/tech_specs/discovery-snowball/spec.md`. Пятый архетип
(`ConnectorKind.snowball`) — единственный, читающий не внешний источник, а уже принятые
документы корпуса (`raw.*`/`doc.md`): гиперлинк-аннотации raw.pdf, href raw.html,
напечатанные URL doc.md (§2), плюс opt-in LLM-стадия текстовых цитат без URL (§5).
Регистрируется в ядре при импорте (см. ``discovery/connectors/__init__.py``).

Коммит 1 — конфиг (§3 спека): типизированный ``SnowballConfig`` + ``load_config``.
Коммит 2 — экстрактор PDF-аннотаций (§2.1/§2.4): группировка/склейка по ``uri``,
crop anchor-текста, санитизация URL, отсев самоссылок/уже-в-корпусе.
Маппинг/курсор/регистрация коннектора — последующие коммиты.
"""
from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pdfplumber
import yaml

from core import schema
from core.env import REPO_ROOT
from discovery.dedup import normalize_url

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_snowball.yaml"
CONNECTOR_ID = "snowball"

# §2.4 шаг 4: срез хвостовой пунктуации ДО остальных проверок санитизации.
_TAIL_PUNCT = ").,;»”’\"'"
# §2.4 шаг 4: URL короче этого — мусор (напр. живой пример GAIRI "http://a").
_MIN_URL_LENGTH = 12


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


# --- §2/§2.4: общий выход всех экстракторов, до маппинга в CandidateRecord (коммит 4) ---


@dataclass(frozen=True)
class RawLink:
    """Один найденный URL-носитель — общий формат для всех экстракторов §2 (до маппинга)."""

    url: str
    anchor: str
    page_number: int | None = None  # только для §2.1 (pdf); None у html/md-экстракторов


# --- §2.4 шаг 4: санитизация одного URL (переиспользуется всеми экстракторами §2/§5) ---


def sanitize_url(url: str | None) -> str | None:
    """Санитизировать один URL-кандидат (спек §2.4 шаг 4). ``None`` — мусор/отсев:
    не-http(s) схема (тем же путём отсеиваются ``mailto:``/``javascript:`` — §2.2/§2.3),
    в хосте нет точки, итоговая длина < ``_MIN_URL_LENGTH`` (живой пример GAIRI:
    ``http://a``). Хвостовая пунктуация СРЕЗАЕТСЯ (правится), не отсеивает саму находку.
    """
    if not url:
        return None
    stripped = url.strip().rstrip(_TAIL_PUNCT)
    if not stripped:
        return None
    parsed = urlsplit(stripped)
    if parsed.scheme not in ("http", "https"):
        return None
    if "." not in parsed.netloc:
        return None
    if len(stripped) < _MIN_URL_LENGTH:
        return None
    return unicodedata.normalize("NFC", stripped)


# --- §2.4 шаги 5/6: отсев самоссылок и уже-в-корпусе (общая функция для всех экстракторов) ---


def is_self_or_corpus_link(
    normalized: str, *, source_url: str, records: list[schema.SourceRecord]
) -> bool:
    """Находка — ссылка на сам документ-источник (шаг 5) или на любой другой документ,
    уже принятый в корпус (шаг 6, дешёвая проверка по ``records`` — не полагаемся на
    полноту ``candidates.yaml``). ``normalized`` — уже прогнанный через ``dedup.normalize_url``."""
    if normalize_url(source_url) == normalized:
        return True
    return any(normalize_url(r.source_url) == normalized for r in records)


# --- §2.1/§2.4: гиперлинк-аннотации raw.pdf ---


def _quad_to_top_rect(
    quad: tuple[float, float, float, float, float, float, float, float], page_height: float
) -> tuple[float, float, float, float]:
    """Один квадрилатераль ``QuadPoints`` (8 чисел, PDF-пространство: y растёт вверх,
    порядок вершин у реальных производителей ненадёжен — берём min/max, не вершину
    по позиции) -> (x0, top, x1, bottom) в системе pdfplumber (top растёт вниз от
    верха страницы), зеркало формулы ``pdfplumber.page.Page.annots`` (``_invert_box``)."""
    xs = quad[0::2]
    ys = quad[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, page_height - y1, x1, page_height - y0


def _annotation_rects(annot: dict[str, Any], page_height: float) -> list[tuple[float, float, float, float]]:
    """Аннотация -> список (x0, top, x1, bottom) — по одному на квадрилатераль, если
    у аннотации есть ``QuadPoints`` (спек §2.4 шаг 3: кроп по каждому квадру отдельно,
    НЕ по общему rect — общий rect многострочной аннотации захватывает посторонний
    текст между строками); иначе — единственный rect самой аннотации (частый случай
    на реальных PDF-генераторах — живая сверка 2026-07-24: ни одна аннотация корпуса
    не несёт ``QuadPoints``, каждая строка обёрнутой ссылки — отдельная аннотация)."""
    quad_points = ((annot.get("data") or {}).get("QuadPoints")) or []
    if not quad_points:
        return [(annot["x0"], annot["top"], annot["x1"], annot["bottom"])]
    rects: list[tuple[float, float, float, float]] = []
    for i in range(0, len(quad_points) - 7, 8):
        quad = tuple(float(v) for v in quad_points[i : i + 8])
        rects.append(_quad_to_top_rect(quad, page_height))  # type: ignore[arg-type]
    return rects or [(annot["x0"], annot["top"], annot["x1"], annot["bottom"])]


def group_by_uri(annots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Аннотации ОДНОЙ страницы, несущие ``uri`` -> сгруппированы по ``uri``, каждая
    группа отсортирована по порядку чтения ``(top, x0)`` (спек §2.4 шаги 1-2). Порядок
    ВХОДНОГО списка не влияет на результат — только геометрия (hypothesis-тест)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annot in annots:
        uri = annot.get("uri")
        if uri:
            groups[uri].append(annot)
    for uri in groups:
        groups[uri].sort(key=lambda a: (a["top"], a["x0"]))
    return dict(groups)


def extract_pdf_annotation_links(raw_path: Path) -> list[RawLink]:
    """Извлечь ссылки из гиперлинк-аннотаций ``raw.pdf`` (спек §2.1/§2.4). Сканы без
    текст-слоя не несут аннотаций by construction — просто ноль находок, отдельный
    детект не нужен. URL уже прогнан через ``sanitize_url`` (мусорные находки отсеяны
    здесь же — self-link/corpus-link отсев делает вызывающая сторона, ей нужен контекст
    документа-источника, которого у чистого экстрактора нет)."""
    links: list[RawLink] = []
    with pdfplumber.open(raw_path) as pdf:
        for page in pdf.pages:
            annots = [a for a in (page.annots or []) if a.get("uri")]
            if not annots:
                continue
            page_height = page.height
            for uri, group in group_by_uri(annots).items():
                clean_url = sanitize_url(uri)
                if clean_url is None:
                    continue
                anchor_parts: list[str] = []
                for annot in group:
                    for rect in _annotation_rects(annot, page_height):
                        x0, top, x1, bottom = rect
                        x0c, x1c = max(0.0, x0), min(page.width, x1)
                        topc, botc = max(0.0, top), min(page.height, bottom)
                        if x1c <= x0c or botc <= topc:
                            continue
                        text = (page.crop((x0c, topc, x1c, botc)).extract_text() or "").strip()
                        if text:
                            anchor_parts.append(text)
                anchor = " ".join(" ".join(anchor_parts).split())
                links.append(RawLink(url=clean_url, anchor=anchor, page_number=page.page_number))
    return links
