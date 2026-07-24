"""discovery/connectors/snowball.py — backward-snowballing по собственному корпусу.

Spec `docs/pipeline/discovery/tech_specs/discovery-snowball/spec.md`. Пятый архетип
(`ConnectorKind.snowball`) — единственный, читающий не внешний источник, а уже принятые
документы корпуса (`raw.*`/`doc.md`): гиперлинк-аннотации raw.pdf, href raw.html,
напечатанные URL doc.md (§2), плюс opt-in LLM-стадия текстовых цитат без URL (§5).
Регистрируется в ядре при импорте (см. ``discovery/connectors/__init__.py``).

Коммит 1 — конфиг (§3 спека): типизированный ``SnowballConfig`` + ``load_config``.
Коммит 2 — экстрактор PDF-аннотаций (§2.1/§2.4): группировка/склейка по ``uri``,
crop anchor-текста, санитизация URL, отсев самоссылок/уже-в-корпусе.
Коммит 3 — экстракторы href raw.html и напечатанных URL doc.md (§2.2/§2.3).
Коммит 4 — маппинг в CandidateRecord, pre-signal, курсор/fingerprint,
регистрация коннектора в ядре (§3/§4).
Коммит 5 — CLI-подкоманда `snowball` + `orchestrate.connectors_override` (§3).
Коммит 6 — LLM-стадия текстовых цитат без URL, opt-in `emit.text_citations` (§5).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import pdfplumber
import yaml
from lxml import html as lxml_html

from convert.converters import was_ocr_normalized
from core import fsio, openrouter, schema
from core.env import REPO_ROOT
from discovery import registry
from discovery.base import ConnectorCursor, DiscoverResult
from discovery.dedup import normalize_url

CONFIG_PATH = REPO_ROOT / "pipeline" / "config" / "discovery_snowball.yaml"
CONNECTOR_ID = "snowball"
LEADS_FILENAME = ".snowball_leads.yaml"
_CITATIONS_CACHE_FILENAME = ".citations.yaml"

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
    citations_model_fallback: str | None


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
        citations_model_fallback=(
            str(raw["citations_model_fallback"]) if raw.get("citations_model_fallback") else None
        ),
    )


# --- §2/§2.4: общий выход всех экстракторов, до маппинга в CandidateRecord (коммит 4) ---


@dataclass(frozen=True)
class RawLink:
    """Один найденный URL-носитель — общий формат для всех экстракторов §2 (до маппинга)."""

    url: str
    anchor: str
    page_number: int | None = None  # только для §2.1 (pdf); None у html/md-экстракторов
    ocr_text_url: bool = False  # §2.3: URL пришёл из OCR-нормализованного текста — риск искажения


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


# --- §2.2: href из raw.html ---


def extract_html_href_links(raw_path: Path, *, source_url: str) -> list[RawLink]:
    """Извлечь ссылки из ``<a href>`` ``raw.html`` (спек §2.2). Относительные href
    резолвятся ``urljoin`` от ``source_url`` документа-источника (сам HTML своего URL не
    знает); чистые фрагменты-якоря (``#...``) отсеиваются ДО резолюции — после urljoin
    они неотличимы от самоссылки на документ и молча растворились бы в self-link фильтре
    (коммит 4), но явный отсев здесь дешевле и честнее по намерению. ``mailto:``/
    ``javascript:`` отсеиваются той же ``sanitize_url`` (не-http(s) схема — §2.4), без
    отдельного спецкейса. Конвертер (``include_links=False``) не участвует — читаем
    исходный raw.html сами."""
    tree = lxml_html.fromstring(raw_path.read_bytes())
    links: list[RawLink] = []
    for a in tree.iter("a"):
        href = a.get("href")
        if not href or href.startswith("#"):
            continue
        clean_url = sanitize_url(urljoin(source_url, href))
        if clean_url is None:
            continue
        anchor = " ".join((a.text_content() or "").split())
        links.append(RawLink(url=clean_url, anchor=anchor))
    return links


# --- §2.3: напечатанные URL в doc.md (все форматы, единственный канал для сканов) ---

_PRINTED_URL_RE = re.compile(r'https?://[^\s)\]>"«»]+')


def extract_printed_urls(doc_md_path: Path, *, ocr_normalized: bool = False) -> list[RawLink]:
    """Извлечь URL, напечатанные прямо в тексте ``doc.md`` (спек §2.3) — покрывает ВСЕ
    форматы (pdf/html/docx/xlsx) и единственный канал для сканов (аннотаций нет, текст —
    единственный носитель). Контекст — строка doc.md с находкой (её же вызывающая сторона
    урезает до ``CANDIDATE_SUMMARY_MAX`` при маппинге, коммит 4). ``ocr_normalized`` —
    решение вызывающей стороны (``converters.was_ocr_normalized(raw_path)``, читается
    один раз на документ, не здесь) -> помечает КАЖДУЮ находку этого документа
    ``ocr_text_url=True`` (известный класс OCR-искажений цифр/диакритики)."""
    text = doc_md_path.read_text(encoding="utf-8")
    links: list[RawLink] = []
    for line in text.splitlines():
        for match in _PRINTED_URL_RE.finditer(line):
            clean_url = sanitize_url(match.group(0))
            if clean_url is None:
                continue
            context = " ".join(line.split())
            links.append(RawLink(url=clean_url, anchor=context, ocr_text_url=ocr_normalized))
    return links


# --- §3: source_filter/url_filter — применяются К СПИСКУ ДОКУМЕНТОВ / К ОДНОЙ находке ---


def apply_source_filter(
    records: list[schema.SourceRecord], source_filter: SourceFilter
) -> list[schema.SourceRecord]:
    """Отфильтровать документы корпуса, подлежащие майнингу (спек §3). Каждый непустой
    компонент — независимое сужение (AND); пустой компонент — разрешает всё (§3: никаких
    жёстких дефолтов). ``include_doc_ids`` — allowlist (если непуст, режет ДО остальных)."""
    result = records
    if source_filter.include_doc_ids:
        result = [r for r in result if r.id in source_filter.include_doc_ids]
    if source_filter.exclude_doc_ids:
        result = [r for r in result if r.id not in source_filter.exclude_doc_ids]
    if source_filter.tracks:
        result = [r for r in result if r.track.value in source_filter.tracks]
    if source_filter.target_fit:
        result = [
            r
            for r in result
            if r.relevance is not None and r.relevance.target_fit.value in source_filter.target_fit
        ]
    return result


def is_url_filtered(url: str, url_filter: UrlFilter) -> bool:
    """Находка отсеивается url_filter'ом (спек §3) — домен ИЛИ подстрока URL в чёрном
    списке. Пустые списки — ничего не режем (§3: разрешающие дефолты)."""
    host = urlsplit(url).netloc.lower()
    if any(host == d.lower() or host.endswith("." + d.lower()) for d in url_filter.exclude_domains):
        return True
    return any(sub in url for sub in url_filter.exclude_url_substrings)


# --- §4: pre-signal matched_vocab_tags — лексическое пересечение, НЕ вердикт ---

_VOCAB_SOURCES = ("vocab_topics.yaml", "vocab_g2ai_patterns.yaml")


def _load_vocab_terms() -> list[tuple[str, str]]:
    """``(оригинальный-ключ, ключ-с-пробелами)`` из vocab_topics/vocab_g2ai_patterns —
    источник истины для pre-сигнала (спек §3), не инлайновый список (skill-content-vs-
    source-of-truth дисциплина)."""
    terms: list[tuple[str, str]] = []
    for name in _VOCAB_SOURCES:
        raw: dict[str, Any] = yaml.safe_load((schema.VOCAB_DIR / name).read_text(encoding="utf-8"))
        for key in (raw.get("terms") or {}):
            terms.append((key, key.replace("-", " ")))
    return terms


def match_vocab_tags(text: str, vocab_terms: list[tuple[str, str]]) -> list[str]:
    """Ключи словаря, чья space-форма встречается в ``text`` (регистронезависимо) —
    дешёвый pre-сигнал (спек §4), НЕ триажный вердикт."""
    lowered = text.lower()
    return [key for key, spaced in vocab_terms if spaced and spaced in lowered]


# --- §4: маппинг RawLink -> CandidateRecord ---


def _fallback_title(url: str) -> str:
    """Anchor пуст (напр. иконка-ссылка без текста, спек §2.1) -> последний осмысленный
    сегмент пути URL, иначе домен."""
    parts = urlsplit(url)
    segment = parts.path.rstrip("/").rsplit("/", 1)[-1]
    return segment or parts.netloc


def map_link(
    link: RawLink,
    *,
    source_record: schema.SourceRecord,
    location_kind: str,
    vocab_terms: list[tuple[str, str]],
) -> schema.CandidateRecord:
    """``RawLink`` -> ``CandidateRecord`` (спек §4). ``location_kind`` — какой экстрактор
    породил находку (``"pdf"``/``"html"``/``"md"``) — определяет форму ``native_id``,
    вызывающая сторона знает это по построению (какой экстрактор вызван), а не сама
    находка (§2/§2.3: ``page_number`` есть только у pdf-находок)."""
    normalized = normalize_url(link.url)
    anchor = link.anchor.strip()
    title = anchor or _fallback_title(link.url)
    native_summary = anchor[: schema.CANDIDATE_SUMMARY_MAX] if anchor else None
    host = urlsplit(link.url).netloc
    native_tags = [f"domain: {host}", f"source: {source_record.id}"]
    if link.ocr_text_url:
        native_tags.append("ocr-text-url")
    if location_kind == "pdf" and link.page_number is not None:
        native_id = f"{source_record.id}#p{link.page_number}"
    else:
        native_id = f"{source_record.id}#{location_kind}"
    raw_hash = hashlib.sha256(f"snowball|{normalized}".encode("utf-8")).hexdigest()
    matched = match_vocab_tags(anchor, vocab_terms)

    return schema.CandidateRecord(
        title=title,
        source_url=link.url,
        native_summary=native_summary,
        native_id=native_id,
        native_tags=native_tags,
        matched_vocab_tags=matched or None,
        connector_id=CONNECTOR_ID,
        retrieved_at=dt.date.today(),
        raw_hash=raw_hash,
        normalized_url=normalized,
    )


# --- §4: курсор — fingerprint по документу (sha256 raw + sha256 doc.md) ---


def document_fingerprint(rec: schema.SourceRecord, root: Path) -> str:
    """``sha256(sha256_raw | sha256_doc_md)`` (спек §4) — меняется, если поменялся ЛИБО
    оригинал (``.state.yaml`` пересчитывает hash при передобыче), ЛИБО конвертация
    (``doc.md``); отсутствующая часть — литерал ``"-"`` (нет ``.state.yaml``/`doc.md`
    ещё не сгенерирован)."""
    state_path = schema.state_file(rec, root)
    raw_sha = "-"
    if state_path.exists():
        state = schema.load_state(state_path)
        raw_sha = state.sha256 or "-"
    md_path = schema.md_file(rec, root)
    md_sha = "-"
    if md_path.exists():
        md_sha = hashlib.sha256(md_path.read_bytes()).hexdigest()
    return hashlib.sha256(f"{raw_sha}|{md_sha}".encode("utf-8")).hexdigest()


# --- §5: LLM-стадия текстовых цитат без URL (opt-in `emit.text_citations`) ---

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)  # группа 1 = уровень, 2 = текст
# Стоп-набор секций-кандидатов (спек §5) — EN + региональные (сербский/черногорский
# латиница) варианты «Литература»/«Извори». «further resources»/«further reading»
# добавлены по живой находке мини-A/B (2026-07-24): sg-imda-mgf-agentic-2026 несёт
# единственный настоящий цитатный раздел корпуса именно под этим заголовком («Annex A:
# Further resources») — без слов «references»/«bibliography» вовсе; без этой записи
# детектор проходил бы мимо него целиком.
_CITATION_HEADING_WORDS = (
    "references", "bibliography", "sources", "endnotes", "notes", "reference list",
    "further resources", "further reading",
    "литература", "извори",
)
_HEADING_NUMBER_PREFIX_RE = re.compile(r"^[\d.]+\s*")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_DENSE_BLOCK_MIN_RUN = 3
_DENSE_BLOCK_MIN_LINE_LEN = 20

CITATION_PROMPT_TEMPLATE = """The following text is a references/bibliography section \
extracted verbatim from a policy document. Extract every distinct citation.

Output a JSON object with exactly one key "citations", whose value is an array of \
objects, each with fields:
- "title": string, the work's title EXACTLY as printed in the text below (a verbatim \
  substring of it — do not paraphrase, translate, or reformat)
- "issuer": string or null — publisher/organization/author, if identifiable
- "year": integer or null
- "url": string or null — only if a URL is printed for THIS specific citation

Output ONLY the JSON object — no other text, no markdown code fence.

Text:
{section_text}"""

CITATION_MAX_TOKENS = 4000


def _is_citation_heading(heading_text: str) -> bool:
    """Живая находка мини-A/B (2026-07-24, sg-imda-mgf-agentic-2026): чистый substring-
    матч ложно ловил `#### 1.2.1 Sources of risk` (обычный содержательный заголовок,
    НЕ библиография) по слову «sources» — «Sources» само по себе однозначно, «Sources
    of X» уже нет. Разбор: ОДНОСЛОВНЫЕ кандидаты (sources/references/notes/…) требуют
    ТОЧНОГО совпадения всего заголовка (после среза ведущей нумерации `1.2.1 `/`6. ` и
    двоеточия) — многословные фразы (`further resources`/`reference list`/…)
    самодостаточно специфичны и остаются подстрокой."""
    stripped = _HEADING_NUMBER_PREFIX_RE.sub("", heading_text).strip().rstrip(":").lower()
    for word in _CITATION_HEADING_WORDS:
        if " " in word:
            if word in stripped:
                return True
        elif stripped == word:
            return True
    return False


def _find_dense_year_blocks(text: str) -> list[str]:
    """Плотные блоки строк с годом-паттерном — фолбэк, когда в документе нет
    заголовка-кандидата (спек §5: «плотные блоки строк с годом ... и признаками
    цитаты»). Порог длины строки отсеивает голые номера/даты в тексте документа."""
    blocks: list[str] = []
    run: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= _DENSE_BLOCK_MIN_LINE_LEN and _YEAR_RE.search(stripped):
            run.append(stripped)
        else:
            if len(run) >= _DENSE_BLOCK_MIN_RUN:
                blocks.append("\n".join(run))
            run = []
    if len(run) >= _DENSE_BLOCK_MIN_RUN:
        blocks.append("\n".join(run))
    return blocks


def find_citation_sections(text: str) -> list[str]:
    """Секции-кандидаты для LLM-экстракции (спек §5): текст между заголовком из
    стоп-набора и следующим заголовком ТОГО ЖЕ ИЛИ БОЛЕЕ ВЫСОКОГО уровня (не любого
    следующего — живая находка мини-A/B 2026-07-24: у sg-imda-mgf-agentic-2026
    настоящий цитатный раздел («## Annex A: Further resources») организован
    ВЛОЖЕННЫМИ подзаголовками («###### 1. Introduction to Agentic AI» и т.п.) —
    граница «любой следующий заголовок» обрывала бы секцию на первой же строке,
    теряя весь реальный контент; граница «того же/более высокого уровня» останавливается
    только на настоящей соседней секции, напр. «## Annex B»). Пусто -> LLM НЕ
    вызывается вовсе (дёшево до дорогого)."""
    headings = list(_HEADING_RE.finditer(text))
    sections: list[str] = []
    for i, h in enumerate(headings):
        level = len(h.group(1))
        if not _is_citation_heading(h.group(2)):
            continue
        start = h.end()
        end = len(text)
        for later in headings[i + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        body = text[start:end].strip()
        if body:
            sections.append(body)
    if sections:
        return sections
    return _find_dense_year_blocks(text)


def _build_citation_payload(model: str, section_text: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": CITATION_PROMPT_TEMPLATE.format(section_text=section_text)}],
        "response_format": {"type": "json_object"},
        "max_tokens": CITATION_MAX_TOKENS,
    }


def _normalize_for_verbatim(text: str) -> str:
    return " ".join(text.split()).lower()


def passes_verbatim_gate(title: str, section_text: str) -> bool:
    """Анти-галлюцинационный гейт (спек §5, философия witness-гейта cloud_ocr): title
    обязан присутствовать в секции подстрокой после нормализации пробелов/регистра —
    LLM не имеет права «дополнять» библиографию несуществующими записями."""
    if not title:
        return False
    return _normalize_for_verbatim(title) in _normalize_for_verbatim(section_text)


@dataclass(frozen=True)
class CitationLead:
    """Текстовая цитата БЕЗ URL — НЕ кандидат (чартер §9 Q3: неактируем добычей без
    source_url); сырьё мини-кампании directed-search, которая резолвит официальный
    URL и делает inject (спек §5)."""

    title: str
    issuer: str | None
    year: int | None
    source_doc_id: str
    context: str


def _citations_cache_path(md_path: Path) -> Path:
    return md_path.parent / _CITATIONS_CACHE_FILENAME


def _load_citations_cache(md_path: Path) -> dict[str, Any]:
    path = _citations_cache_path(md_path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_citations_cache(md_path: Path, cache: dict[str, Any]) -> None:
    fsio.atomic_write_text(
        _citations_cache_path(md_path), yaml.safe_dump(cache, allow_unicode=True, sort_keys=False)
    )


def _try_extract_raw_citations(
    section: str, model: str, call_model: Callable[[dict[str, Any]], dict[str, Any]]
) -> list[Any] | None:
    """Один вызов модели -> сырой список ``citations``, либо ``None`` при ОТКАЗЕ (пустой
    или синтаксически невалидный JSON-ответ). Живой класс дефекта (мини-A/B 2026-07-24,
    `deepseek-v4-flash`): модель на дефолтном reasoning-effort может сжечь ВЕСЬ
    `max_tokens` на скрытые reasoning-токены и вернуть пустой видимый контент —
    воспроизведено дважды из трёх живых прогонов. Пустой СПИСОК (модель честно ответила
    «цитат нет», валидный JSON) — НЕ отказ, fallback не запускает."""
    response = call_model(_build_citation_payload(model, section))
    raw_content = response["choices"][0]["message"].get("content") or ""
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    citations = parsed.get("citations")
    return citations if isinstance(citations, list) else []


def extract_text_citations(
    md_path: Path,
    *,
    doc_id: str,
    model: str,
    call_model: Callable[[dict[str, Any]], dict[str, Any]],
    fallback_model: str | None = None,
) -> tuple[list[RawLink], list[CitationLead]]:
    """LLM-стадия текстовых цитат без URL (спек §5, opt-in). Возвращает
    ``(URL-несущие находки как RawLink, лиды без URL)`` — URL-находки идут ДАЛЬШЕ по
    общему пути §4 (self-link/url_filter/max_candidates/маппинг), не спецкейсятся.

    ``call_model`` — инжектируемый транспорт (тесты подставляют фейк; продакшен —
    partial вокруг ``core.openrouter.chat_request`` с живым ``OPENROUTER_API_KEY``,
    тот же паттерн, что ``fetch``/``get_standards_page`` у aiforgood/eurlex). Кэш
    ``.citations.yaml`` — ключ sha256(секции): повторный прогон по тем же секциям
    НЕ зовёт ``call_model`` вовсе. ``fallback_model`` (решение куратора 2026-07-24,
    по факту живого мини-A/B: minimax — основная, gemini — резерв) — если ``model``
    отказал (см. ``_try_extract_raw_citations``), СРАЗУ повторный вызов на резервной
    модели для ЭТОЙ секции, до финального пустого результата; какая модель реально
    ответила — пишется в кэш (``entry["model"]``), не только номинальный конфиг."""
    text = md_path.read_text(encoding="utf-8")
    sections = find_citation_sections(text)
    if not sections:
        return [], []

    cache = _load_citations_cache(md_path)
    cache_dirty = False
    links: list[RawLink] = []
    leads: list[CitationLead] = []

    for section in sections:
        section_hash = hashlib.sha256(section.encode("utf-8")).hexdigest()
        entry = cache.get(section_hash)
        if entry is None:
            raw_citations = _try_extract_raw_citations(section, model, call_model)
            used_model = model
            if raw_citations is None and fallback_model is not None:
                raw_citations = _try_extract_raw_citations(section, fallback_model, call_model)
                used_model = fallback_model
            if raw_citations is None:
                raw_citations = []  # и основная, и резервная модель отказали — честный пустой результат

            verified: list[dict[str, Any]] = []
            for item in raw_citations:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                if not passes_verbatim_gate(title, section):
                    continue
                verified.append(
                    {"title": title, "issuer": item.get("issuer"), "year": item.get("year"), "url": item.get("url")}
                )
            entry = {"model": used_model, "citations": verified}
            cache[section_hash] = entry
            cache_dirty = True

        for item in entry["citations"]:
            url = item.get("url")
            clean_url = sanitize_url(url) if url else None
            if clean_url is not None:
                links.append(RawLink(url=clean_url, anchor=item["title"]))
            else:
                leads.append(
                    CitationLead(
                        title=item["title"],
                        issuer=item.get("issuer"),
                        year=item.get("year"),
                        source_doc_id=doc_id,
                        context=section[: schema.CANDIDATE_SUMMARY_MAX],
                    )
                )

    if cache_dirty:
        _save_citations_cache(md_path, cache)

    return links, leads


def save_leads(leads: list[dict[str, Any]], root: Path) -> None:
    """Записать ``sources/.snowball_leads.yaml`` (спек §5) — ПЕРЕЗАПИСЫВАЕТСЯ целиком
    каждым прогоном с ``--with-citations`` (не аппендится): лиды — сырьё СЛЕДУЮЩЕЙ
    directed-search мини-кампании, не постоянное состояние; известное ограничение —
    лиды документа, не пере-майненного в этом прогоне (курсор его пропустил), в
    файл не попадают, пока документ не изменится или курсор не будет сброшен."""
    fsio.atomic_write_text(
        root / LEADS_FILENAME, yaml.safe_dump(leads, allow_unicode=True, sort_keys=False)
    )


# --- §4: discover_snowball() top-level ---


def _default_call_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Продакшен-транспорт §5 — ленивое чтение ключа (спек: `emit.text_citations=false`
    по умолчанию, прогон без цитат не должен требовать `OPENROUTER_API_KEY` вовсе)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("нет OPENROUTER_API_KEY (см. .env / .env.example) — нужен для emit.text_citations")
    return openrouter.chat_request(payload, api_key=api_key)


def discover_snowball(
    cursor: ConnectorCursor | None,
    *,
    config: SnowballConfig | None = None,
    root: Path = schema.DEFAULT_SOURCES,
    records: list[schema.SourceRecord] | None = None,
    call_model: Callable[[dict[str, Any]], dict[str, Any]] = _default_call_model,
) -> DiscoverResult:
    """``Connector.discover()`` для snowball (спек §4): отфильтровать документы (§3) ->
    для каждого НЕизменившегося (по курсору) — скип; иначе прогнать включённые
    экстракторы (§2, + §5 LLM-стадия цитат, если ``emit.text_citations`` и документ не
    ``confidential``) -> отсеять самоссылки/уже-в-корпусе/url_filter -> замаппить ->
    применить ``max_candidates`` (если задан) -> обновить курсор ТОЛЬКО для документов,
    чьи находки не были урезаны капом (спек §3: недомайненный хвост добирается
    следующим прогоном).

    ``records`` — инжектируемый список документов корпуса (тесты подставляют фикстуры;
    по умолчанию читается с диска, как у остальных потребителей ``schema.load_records``).
    ``call_model`` — инжектируемый транспорт §5 (тесты подставляют фейк).
    """
    cfg = config or load_config()
    all_records = records if records is not None else schema.load_records(root)
    filtered = apply_source_filter(all_records, cfg.source_filter)
    vocab_terms = _load_vocab_terms()

    mined_before = dict((cursor or {}).get("mined") or {})
    mined_after = dict(mined_before)

    candidates: list[schema.CandidateRecord] = []
    all_leads: list[CitationLead] = []
    docs_scanned = 0
    docs_skipped_cursor = 0
    truncated_docs = 0
    truncated_candidates = 0
    filtered_self_or_corpus = 0
    filtered_by_url_filter = 0
    per_extractor = {"pdf_annotations": 0, "html_hrefs": 0, "printed_urls": 0, "text_citations": 0}
    cap_remaining = cfg.max_candidates

    for rec in filtered:
        raw_path = schema.raw_file(rec, root)
        md_path = schema.md_file(rec, root)
        if raw_path is None or not md_path.exists():
            continue  # документ ещё не добыт/не сконвертирован — просто нечего майнить

        fingerprint = document_fingerprint(rec, root)
        if mined_before.get(rec.id) == fingerprint:
            docs_skipped_cursor += 1
            continue
        docs_scanned += 1

        raw_links: list[tuple[RawLink, str]] = []
        if cfg.emit.pdf_annotations and raw_path.suffix == ".pdf":
            found = extract_pdf_annotation_links(raw_path)
            per_extractor["pdf_annotations"] += len(found)
            raw_links.extend((link, "pdf") for link in found)
        if cfg.emit.html_hrefs and raw_path.suffix == ".html":
            found = extract_html_href_links(raw_path, source_url=rec.source_url)
            per_extractor["html_hrefs"] += len(found)
            raw_links.extend((link, "html") for link in found)
        if cfg.emit.printed_urls:
            ocr_flag = raw_path.suffix == ".pdf" and was_ocr_normalized(raw_path)
            found = extract_printed_urls(md_path, ocr_normalized=ocr_flag)
            per_extractor["printed_urls"] += len(found)
            raw_links.extend((link, "md") for link in found)
        if cfg.emit.text_citations and rec.sensitivity != schema.Sensitivity.confidential:
            cite_links, cite_leads = extract_text_citations(
                md_path,
                doc_id=rec.id,
                model=cfg.citations_model,
                call_model=call_model,
                fallback_model=cfg.citations_model_fallback,
            )
            per_extractor["text_citations"] += len(cite_links)
            raw_links.extend((link, "citation") for link in cite_links)
            all_leads.extend(cite_leads)

        doc_candidates: list[schema.CandidateRecord] = []
        for link, kind in raw_links:
            normalized = normalize_url(link.url)
            if is_self_or_corpus_link(normalized, source_url=rec.source_url, records=all_records):
                filtered_self_or_corpus += 1
                continue
            if is_url_filtered(link.url, cfg.url_filter):
                filtered_by_url_filter += 1
                continue
            doc_candidates.append(
                map_link(link, source_record=rec, location_kind=kind, vocab_terms=vocab_terms)
            )

        doc_truncated = False
        if cap_remaining is not None and len(doc_candidates) > cap_remaining:
            truncated_candidates += len(doc_candidates) - cap_remaining
            doc_candidates = doc_candidates[:cap_remaining]
            doc_truncated = True
            cap_remaining = 0
        elif cap_remaining is not None:
            cap_remaining -= len(doc_candidates)

        candidates.extend(doc_candidates)

        if doc_truncated:
            truncated_docs += 1  # НЕ обновляем mined_after[rec.id] — хвост добирает следующий прогон
        else:
            mined_after[rec.id] = fingerprint

        if cap_remaining is not None and cap_remaining <= 0:
            break  # остальные документы этот прогон не трогает вовсе (курсор их не помнит)

    status = "no_new" if cursor is not None and not candidates else "fetched"
    diagnostics: dict[str, Any] = {
        "status": status,
        "docs_scanned": docs_scanned,
        "docs_skipped_cursor": docs_skipped_cursor,
        "found": sum(per_extractor.values()),
        "fresh": len(candidates),
        "filtered_self_or_corpus": filtered_self_or_corpus,
        "filtered_by_url_filter": filtered_by_url_filter,
        "per_extractor": dict(per_extractor),
        "truncated_docs": truncated_docs,
        "truncated_candidates": truncated_candidates,
        # §5: лиды (цитаты без URL) — сериализуемые dict, не dataclass (диагностика
        # должна пройти через ConnectorRunSummary как plain data, см. orchestrate.py).
        "leads": [
            {"title": lead.title, "issuer": lead.issuer, "year": lead.year,
             "source_doc_id": lead.source_doc_id, "context": lead.context}
            for lead in all_leads
        ],
    }
    return DiscoverResult(candidates=candidates, cursor={"mined": mined_after}, diagnostics=diagnostics)


@dataclass
class SnowballConnector:
    """Реализация протокола ``Connector`` (спек §1) — единственный архетип ``snowball``,
    источник — не внешний сервис, а уже принятый корпус. НЕ ``frozen`` — симметрично
    ``AgoraConnector``/``EurlexConnector``/``AiforgoodConnector`` (Protocol требует
    settable-атрибуты, даже если ничего не переприсваивается)."""

    id: str = CONNECTOR_ID
    kind: schema.ConnectorKind = schema.ConnectorKind.snowball
    enabled: bool = True
    config: SnowballConfig | None = None  # None -> discover_snowball грузит yaml сам;
    # непустой — CLI-подкоманда `snowball` даёт слитый конфиг (yaml + CLI-флаги, спек §3),
    # на один прогон, без мутации ни реестра, ни файла на диске (orchestrate.connectors_override).
    root: Path = schema.DEFAULT_SOURCES  # ОБЯЗАН совпадать с --root оркестратора (discover.py
    # передаёт его явно) — иначе коннектор молча сканирует боевой sources/ вместо тестового/
    # переданного корня (живой дефект, пойманный test_discover_cli.py при написании коммита 5).

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        return discover_snowball(cursor, config=self.config, root=self.root)


# Регистрация при импорте (чартер §4.3 «манифест», спек §1): `enabled` — из конфига,
# не хардкод. Срабатывает один раз за интерпретатор — по факту импорта этого модуля
# (см. `discovery/connectors/__init__.py` + `discover.py`).
registry.register(SnowballConnector(enabled=load_config().enabled))
