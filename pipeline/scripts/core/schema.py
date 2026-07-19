"""Схема метаданных записей корпуса (``meta.yaml``, corpus-layout-v2) + рендер frontmatter.

Структурная валидация — pydantic (типы, форматы, обязательность, стабильные enum).
Проверка принадлежности контролируемым словарям (doc_type / authority / topics /
g2ai_pattern) и ссылочной целостности ``relations`` вынесена в ``validate_sources.py``,
т.к. требует загрузки внешних vocab-файлов из ``pipeline/vocab/``.
"""
from __future__ import annotations

import datetime as _dt
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from core import fsio
from core.env import REPO_ROOT

# Каталог контролируемых словарей: pipeline/vocab/ (sibling каталога scripts/).
VOCAB_DIR = Path(__file__).resolve().parents[2] / "vocab"
# Корень дерева папок-документов (corpus-layout-v2). Единственный источник —
# env.REPO_ROOT; потребители (run_pipeline/corpus_index/build_graph) импортируют
# отсюда, не из validate_sources — зависимость «инструмент → валидатор ради
# константы» была неверна по направлению слоёв.
DEFAULT_SOURCES = REPO_ROOT / "sources"

# Внутренний id: kebab-slug минимум из двух сегментов, напр. ``sg-imda-mgf-agentic-2026``.
ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)+$"
# Слаг сущности (== папка под track): lowercase-kebab, допускает один сегмент (ee, oecd, anthropic).
ENTITY_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class IssuerType(str, Enum):
    """Тип издателя (структурный enum, стабильный)."""

    government = "government"
    igo = "igo"
    standards_body = "standards_body"
    think_tank = "think_tank"
    private_company = "private_company"
    academia = "academia"
    other = "other"


class GeoScope(str, Enum):
    """Географический охват документа."""

    national = "national"
    regional = "regional"
    international = "international"
    global_ = "global"  # 'global' — ключевое слово Python, отсюда суффикс


class Track(str, Enum):
    """Верхний аналитический раскол корпуса (== верхняя папка под ``sources/``; corpus-layout-v2)."""

    intl_xperience = "intl-xperience"
    montenegro = "montenegro"


class SourceFormat(str, Enum):
    """Формат первоисточника: диктует расширение ``raw.*``, классификацию добычи и конвертер
    (чартер ``convert/architecture.md`` §3.2)."""

    pdf = "pdf"
    html = "html"


class TranslationStatus(str, Enum):
    """Статус перевода (RU/ME — вторая фаза)."""

    not_started = "not_started"
    in_progress = "in_progress"
    done = "done"


class AcquisitionMethod(str, Enum):
    """Последний-известный канал добычи байтов.

    Подсказка, не жёсткий конфиг: оркестратор всё равно идёт по лестнице
    ``direct -> official_alt -> manual -> archive`` (см. ``source-acquisition-ladder/spec.md``).
    """

    direct = "direct"
    official_alt = "official_alt"
    manual = "manual"
    archive = "archive"


class Fidelity(str, Enum):
    """Честность добытых байтов относительно официального источника."""

    live = "live"
    rehost = "rehost"
    manual = "manual"
    archived_snapshot = "archived_snapshot"


class Sensitivity(str, Enum):
    """Чувствительность документа. Гейтит acquisition-лестницу: confidential -> archive недоступен."""

    normal = "normal"
    confidential = "confidential"


class Rights(str, Enum):
    """Режим прав на переиздание документа (закрытое множество).

    Захватывается best-effort DISCOVERY-коннектором, финализируется на Стадии 1
    триажа (см. ``source-relevance-triage``). Forward-looking метаданные для будущей
    публикации итогового пакета; шиппнутую acquisition-лестницу НЕ гейтит
    (та гейтится по ``sensitivity``).
    """

    ogl = "ogl"
    cc_by = "cc-by"
    public_domain = "public_domain"
    crown = "crown"
    unknown = "unknown"
    all_rights_reserved = "all_rights_reserved"


class TargetFit(str, Enum):
    """Тир целевого соответствия документа оси анализа (source-relevance-triage §2.2)."""

    primary = "primary"
    context = "context"
    background = "background"


class Axis(str, Enum):
    """Ось оценки target_fit (§2.3): узкая агентная vs широкий цифровой суверенитет."""

    agentic_g2ai = "agentic_g2ai"
    digital_sovereignty = "digital_sovereignty"


class AssessedStage(str, Enum):
    """Докуда дошла оценка: дешёвый триаж по метаданным vs подтверждение по тексту."""

    triage = "triage"
    confirmed = "confirmed"


class ConnectorKind(str, Enum):
    """Архетип discovery-коннектора, породившего кандидата (discovery/architecture.md §3).

    Общий стабильный enum: определяется здесь (розетка триажа), импортируется discovery-core.
    """

    registry = "registry"
    outlet_watcher = "outlet_watcher"
    directed_search = "directed_search"
    manual = "manual"


class RelationType(str, Enum):
    """Тип типизированного ребра графа документ->документ."""

    references = "references"
    cites = "cites"
    supersedes = "supersedes"
    superseded_by = "superseded_by"
    implements = "implements"
    amends = "amends"
    responds_to = "responds_to"
    related_to = "related_to"
    translation_of = "translation_of"


class Relation(BaseModel):
    """Ребро графа: связь текущей записи с другой по её ``id``."""

    model_config = ConfigDict(extra="forbid")

    type: RelationType
    target: str = Field(pattern=ID_PATTERN)


class Dates(BaseModel):
    """Гранулярные даты документа (все опциональны)."""

    model_config = ConfigDict(extra="forbid")

    published: _dt.date | None = None
    updated: _dt.date | None = None
    effective: _dt.date | None = None  # дата вступления в силу (для законов)
    retrieved: _dt.date | None = None  # дата скачивания
    last_checked: _dt.date | None = None  # свежесть: когда последний раз перепроверяли источник


class Relevance(BaseModel):
    """Вердикт триажа релевантности (source-relevance-triage). Присваивает ТОЛЬКО триаж."""

    model_config = ConfigDict(extra="forbid")

    target_fit: TargetFit
    axis: Axis
    assessed_stage: AssessedStage
    rationale: str = Field(min_length=1)
    assessed_date: _dt.date


class OperationalState(BaseModel):
    """Производное/операционное состояние документа — sidecar ``.state.yaml`` (corpus-layout-v2).

    Машиннописаное (пайплайн/ладдер), отдельно от курируемого ``meta.yaml``: целостность,
    канал добычи, статус процессов. Отсутствующий файл == пустое состояние (свежий документ).
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # stat-guard для sha256: needed_stages пересчитывает хэш ТОЛЬКО если size/mtime
    # разошлись с записанными здесь — иначе полное чтение raw на КАЖДОМ прогоне ради
    # «делать нечего». Старые .state.yaml без этих полей валидны (Optional) — первый
    # прогон бэкфиллит (см. run_pipeline._adopt_untracked_raw).
    raw_size: int | None = None
    raw_mtime_ns: int | None = None
    acquisition_method: AcquisitionMethod | None = None
    acquisition_checked: _dt.date | None = None
    fidelity: Fidelity | None = None
    retrieved_snapshot_date: _dt.date | None = None
    translation_status: TranslationStatus = TranslationStatus.not_started
    converter_name: str | None = None     # какой конвертер породил текущий doc.md
    converter_version: str | None = None  # его версия (реконсиляция реконверсии)
    # C1 (spec convert-hardening): авто-QA вместо ручного аудита каждого документа —
    # список строк-дефектов convert/lint.py (пустой = чисто); машиночитаемость нужна
    # worksheet'у батч-триажа (spec discovery-manual), флагованные документы видны
    # при Стадии 2. Старые .state.yaml без поля валидны (Field с default).
    lint_defects: list[str] = Field(default_factory=list)


class SourceRecord(BaseModel):
    """Курируемая запись документа (``meta.yaml``, corpus-layout-v2) — человек-источник истины.

    Операционное состояние (sha256/acquisition/…) — в ``OperationalState`` (``.state.yaml``);
    провенанс добычи — в ``CandidateRecord`` (``candidates.yaml``); пути выводятся из папки.
    Только Dublin-Core-библиография + минимум аналитики (topics/g2ai_pattern/summary/relevance).
    """

    model_config = ConfigDict(extra="forbid")

    # --- идентичность ---
    id: str = Field(pattern=ID_PATTERN)
    entity_id: str = Field(pattern=ENTITY_PATTERN)  # слаг сущности (== папка); для наций == iso2
    track: Track  # верхний раскол корпуса (== верхняя папка)
    # --- библиография (Dublin Core) ---
    title: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    issuer_type: IssuerType
    geo_scope: GeoScope
    # ISO 639-1 (2 буквы), где код существует; иначе ISO 639-3 (3 буквы) — напр.
    # черногорский 'cnr' не имеет 639-1-кода. Использовать 639-1, когда он есть
    # (`en`, не `eng`; `sr`, не `srp`) — 639-3 только для языков без 639-1.
    language: str = Field(pattern=r"^[a-z]{2,3}$")
    dates: Dates = Field(default_factory=Dates)
    doc_type: str = Field(min_length=1)      # словарь — validate_sources.py
    authority: str = Field(min_length=1)     # словарь — validate_sources.py
    source_url: str = Field(pattern=r"^https?://")            # официальный первоисточник
    official_alt_url: str | None = Field(default=None, pattern=r"^https?://")  # вход ладдера
    source_format: SourceFormat = SourceFormat.pdf  # расширение raw.*, классификация добычи, конвертер
    sensitivity: Sensitivity = Sensitivity.normal            # гейтит archive-ступень ладдера
    rights: Rights = Rights.unknown
    # --- аналитика (минимум, контент — EN) ---
    topics: list[str] = Field(default_factory=list)          # словарь
    g2ai_pattern: list[str] = Field(default_factory=list)    # словарь; только матрично-релевантные
    summary: str | None = None                               # 2–3 предложения, EN
    relations: list[Relation] = Field(default_factory=list)
    relevance: Relevance | None = None       # вердикт триажа; обязателен — правило validate_sources
    in_force: bool | None = None             # действует ли «живой» документ (взвешивание свежести)


class CandidateRecord(BaseModel):
    """Кандидат-источник из DISCOVERY (живёт в candidates.yaml, до допуска в реестр).

    Лёгкий пермиссивный (``extra="allow"``) upstream-кузен ``SourceRecord``: данные
    коннекторов разнородны и неполны. Не несёт ``relevance`` (discovery не оценивает)
    и не требует ``id`` (присваивается при промоушене). Контракт-розетка между
    DISCOVERY (writer) и триажем (reader); см. source-relevance-triage §3.
    """

    model_config = ConfigDict(extra="allow")

    # провенанс добычи (обязательно)
    connector_id: str = Field(min_length=1)
    connector_kind: ConnectorKind
    retrieved_at: _dt.date
    source_ref: str = Field(min_length=1)
    raw_hash: str = Field(min_length=1)
    # best-effort библиография (Optional — данные upstream неполны)
    title: str | None = None
    issuer: str | None = None
    jurisdiction: str | None = None
    source_url: str | None = Field(default=None, pattern=r"^https?://")
    doc_date: _dt.date | None = None
    reported_status: str | None = None
    language: str | None = None
    rights: Rights | None = None  # best-effort от коннектора; финализирует триаж
    sensitivity: Sensitivity | None = None  # best-effort; несётся в acquisition-гейт
    # passthrough-обогащение источника (Optional)
    native_summary: str | None = None
    native_id: str | None = None
    native_tags: list[str] = Field(default_factory=list)
    # дешёвый pre-signal (НЕ вердикт — target_fit присваивает только триаж)
    matched_query: str | None = None
    matched_vocab_tags: list[str] = Field(default_factory=list)
    # dedup-ключи (заполняет discovery)
    normalized_url: str | None = None
    content_hash: str | None = None
    # причина отказа (если триаж отклонил — кандидат остаётся в candidates.yaml)
    rejected_reason: str | None = None


def doc_dir(rec: SourceRecord, root: Path) -> Path:
    """Папка документа: ``<root>/<track>/<entity_id>/<id>/`` (corpus-layout-v2)."""
    return root / rec.track.value / rec.entity_id / rec.id


def raw_file(rec: SourceRecord, root: Path) -> Path | None:
    """Оригинал документа: ``raw.*`` в папке (ext глобится). Несколько raw.* -> ошибка."""
    matches = sorted(doc_dir(rec, root).glob("raw.*"))
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(f"{rec.id}: несколько raw.* в папке ({names})")
    return matches[0] if matches else None


def raw_target(rec: SourceRecord, root: Path, ext: str = "pdf") -> Path:
    """Путь для НОВОГО оригинала (пишущий близнец ``raw_file``): ``<doc_dir>/raw.<ext>``."""
    return doc_dir(rec, root) / f"raw.{ext}"


def md_file(rec: SourceRecord, root: Path) -> Path:
    """Конвертация: ``<doc_dir>/doc.md``."""
    return doc_dir(rec, root) / "doc.md"


def state_file(rec: SourceRecord, root: Path) -> Path:
    """Операционный sidecar: ``<doc_dir>/.state.yaml``."""
    return doc_dir(rec, root) / ".state.yaml"


def check_layout(meta_path: Path, rec: SourceRecord, seen_ids: set[str]) -> list[str]:
    """Чистые инварианты раскладки corpus-layout-v2: папка документа == ``id``;
    папка сущности == ``entity_id``; верхняя папка == ``track``; глобальная
    уникальность ``id`` (по ``seen_ids``). Пустой список = ок.

    Единственный источник этого знания — ``load_records`` (raise-режим) и
    ``validate_sources.validate_sources`` (collect-режим) вызывают одну и ту же
    проверку вместо двух дрейфующих копий. НЕ мутирует ``seen_ids`` — когда и
    как регистрировать проверенный id, решает вызывающая сторона.
    """
    errors: list[str] = []
    doc, entity, track = meta_path.parent, meta_path.parent.parent, meta_path.parent.parent.parent
    if doc.name != rec.id:
        errors.append(f"{meta_path}: папка '{doc.name}' != id '{rec.id}'")
    if entity.name != rec.entity_id:
        errors.append(f"{meta_path}: папка сущности '{entity.name}' != entity_id '{rec.entity_id}'")
    if track.name != rec.track.value:
        errors.append(f"{meta_path}: верхняя папка '{track.name}' != track '{rec.track.value}'")
    if rec.id in seen_ids:
        errors.append(f"{meta_path}: дубль id '{rec.id}'")
    return errors


def load_records(sources_root: Path) -> list[SourceRecord]:
    """Собрать записи корпуса обходом дерева ``sources/**/meta.yaml`` (строго, raises).

    Инварианты — см. ``check_layout``. Порядок — по ``id`` (детерминизм).
    Полную семантику (словари, relevance, relations) проверяет validate_sources.py.
    """
    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    for meta_path in sorted(sources_root.rglob("meta.yaml")):
        raw: Any = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        rec = SourceRecord.model_validate(raw)
        errors = check_layout(meta_path, rec, seen_ids)
        if errors:
            raise ValueError("\n".join(errors))
        seen_ids.add(rec.id)
        records.append(rec)
    records.sort(key=lambda r: r.id)
    return records


def load_candidates(candidates_path: Path) -> list[CandidateRecord]:
    """Загрузить и структурно провалидировать кандидатов ``candidates.yaml``.

    Слой кандидатов — отдельный файл (``sources/candidates.yaml``), наполняется
    DISCOVERY-коннекторами; триаж читает и промоутит допущенных (см. ``promote_candidate``).
    Пустой/новый файл -> пустой список.
    """
    raw: Any = yaml.safe_load(candidates_path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{candidates_path}: верхний уровень должен быть списком кандидатов")
    return [CandidateRecord.model_validate(item) for item in raw]


def load_state(state_path: Path) -> OperationalState:
    """Загрузить операционное состояние ``.state.yaml`` (отсутствует/пуст -> пустое состояние)."""
    if not state_path.exists():
        return OperationalState()
    raw: Any = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    if raw is None:
        return OperationalState()
    return OperationalState.model_validate(raw)


def save_state(state_path: Path, state: OperationalState) -> None:
    """Атомарно записать операционное состояние (машиннописаный файл, plain YAML + tmp->rename)."""
    payload = state.model_dump(mode="json", exclude_none=True)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    fsio.atomic_write_text(state_path, text)


def promote_candidate(
    cand: CandidateRecord,
    *,
    id: str,
    entity_id: str,
    track: Track,
    issuer_type: IssuerType,
    geo_scope: GeoScope,
    doc_type: str,
    authority: str,
    relevance: Relevance,
    source_format: SourceFormat = SourceFormat.pdf,
) -> SourceRecord:
    """Промоутнуть кандидата в курируемый ``SourceRecord`` (конверсия типа для ``meta.yaml``).

    Издательские/классификационные решения (``id``/``entity_id``/``track``/``issuer_type``/
    ``geo_scope``/``doc_type``/``authority``) и вердикт ``relevance`` — аргументы (решение
    триажа), не выводятся из кандидата. Обязательные поля, которых у кандидата может не быть
    (``title``/``issuer``/``language``/``source_url``), берутся из кандидата и обязаны
    присутствовать — иначе ``ValueError``. Провенанс добычи остаётся в ``candidates.yaml``
    (в ``meta.yaml`` НЕ копируется — corpus-layout-v2).
    """
    title, issuer, language, source_url = cand.title, cand.issuer, cand.language, cand.source_url
    missing = [
        name
        for name, val in (
            ("title", title),
            ("issuer", issuer),
            ("language", language),
            ("source_url", source_url),
        )
        if val is None
    ]
    if missing:
        raise ValueError(
            f"кандидат ({cand.connector_id}/{cand.source_ref}): "
            f"нельзя промоутить без полей: {', '.join(missing)}"
        )
    assert title is not None and issuer is not None
    assert language is not None and source_url is not None

    return SourceRecord(
        id=id,
        entity_id=entity_id,
        track=track,
        title=title,
        issuer=issuer,
        issuer_type=issuer_type,
        geo_scope=geo_scope,
        language=language,
        dates=Dates(published=cand.doc_date),
        doc_type=doc_type,
        authority=authority,
        source_url=source_url,
        source_format=source_format,
        rights=cand.rights or Rights.unknown,
        sensitivity=cand.sensitivity or Sensitivity.normal,
        relevance=relevance,
    )


def load_vocab(name: str, vocab_dir: Path = VOCAB_DIR) -> set[str]:
    """Множество допустимых терминов из ``pipeline/vocab/vocab_<name>.yaml``.

    Формат vocab-файла: верхний ключ ``terms`` -> маппинг ``термин: описание``.
    """
    path = vocab_dir / f"vocab_{name}.yaml"
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("terms"), dict):
        raise ValueError(f"{path}: ожидался маппинг с ключом 'terms'")
    return {str(term) for term in data["terms"]}


def render_frontmatter(rec: SourceRecord) -> str:
    """YAML-frontmatter для ``.md``, порождённый из записи реестра.

    Реестр — единый источник истины; frontmatter не редактируется вручную,
    а генерируется этой функцией (курируемое подмножество полей).
    """
    fields: dict[str, Any] = {
        "id": rec.id,
        "title": rec.title,
        "entity_id": rec.entity_id,
        "issuer": rec.issuer,
        "issuer_type": rec.issuer_type.value,
        "doc_type": rec.doc_type,
        "authority": rec.authority,
        "language": rec.language,
        "published": rec.dates.published.isoformat() if rec.dates.published else None,
        "source_url": rec.source_url,
        "g2ai_pattern": rec.g2ai_pattern,
        "topics": rec.topics,
    }
    present = {k: v for k, v in fields.items() if v not in (None, [], "")}
    body = yaml.safe_dump(present, allow_unicode=True, sort_keys=False)
    return f"---\n{body}---\n"
