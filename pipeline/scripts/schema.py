"""Схема метаданных записей ``sources.yaml`` (G2AI-корпус) + рендер frontmatter.

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

# Каталог контролируемых словарей: pipeline/vocab/ (sibling каталога scripts/).
VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab"

# Внутренний id: kebab-slug минимум из двух сегментов, напр. ``sg-imda-mgf-agentic-2026``.
ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)+$"


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


class Status(str, Enum):
    """Статус записи в пайплайне."""

    pending = "pending"
    downloaded = "downloaded"
    converted = "converted"
    verified = "verified"


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


class SourceRecord(BaseModel):
    """Одна запись реестра первоисточников (один документ корпуса)."""

    model_config = ConfigDict(extra="forbid")

    # --- идентичность ---
    id: str = Field(pattern=ID_PATTERN)
    # --- библиография ---
    title: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    issuer_type: IssuerType
    country: str | None = None
    country_iso2: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    geo_scope: GeoScope
    language: str = Field(pattern=r"^[a-z]{2}$")  # ISO 639-1
    dates: Dates = Field(default_factory=Dates)
    doc_version: str | None = None
    # --- классификация (принадлежность словарям проверяет validate_sources.py) ---
    doc_type: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    topics: list[str] = Field(default_factory=list)
    g2ai_pattern: list[str] = Field(default_factory=list)
    # --- связи (рёбра графа) ---
    relations: list[Relation] = Field(default_factory=list)
    # --- аналитика ---
    summary: str | None = None
    tech_basis: str | None = None
    # --- релевантность/актуальность (source-relevance-triage) ---
    relevance: Relevance | None = None  # обязательность в sources.yaml — правило validate_sources
    in_force: bool | None = None  # действует ли «живой» документ (взвешивание свежести)
    # --- провенанс ---
    source_url: str = Field(pattern=r"^https?://")
    official_alt_url: str | None = Field(default=None, pattern=r"^https?://")
    press_release_url: str | None = Field(default=None, pattern=r"^https?://")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_path: str | None = None
    md_path: str | None = None
    acquisition_method: AcquisitionMethod | None = None
    acquisition_checked: _dt.date | None = None
    fidelity: Fidelity | None = None
    retrieved_snapshot_date: _dt.date | None = None
    sensitivity: Sensitivity = Sensitivity.normal
    rights: Rights = Rights.unknown
    # --- пайплайн ---
    status: Status
    translation_status: TranslationStatus = TranslationStatus.not_started
    notes: str | None = None


def load_records(sources_path: Path) -> list[SourceRecord]:
    """Загрузить и структурно провалидировать записи реестра (raises на битой структуре).

    Полную валидацию (словари, уникальность id, relations) делает validate_sources.py.
    """
    raw: Any = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{sources_path}: верхний уровень должен быть списком записей")
    return [SourceRecord.model_validate(item) for item in raw]


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
        "country": rec.country,
        "issuer": rec.issuer,
        "issuer_type": rec.issuer_type.value,
        "doc_type": rec.doc_type,
        "authority": rec.authority,
        "language": rec.language,
        "published": rec.dates.published.isoformat() if rec.dates.published else None,
        "source_url": rec.source_url,
        "g2ai_pattern": rec.g2ai_pattern,
        "topics": rec.topics,
        "translation_status": rec.translation_status.value,
    }
    present = {k: v for k, v in fields.items() if v not in (None, [], "")}
    body = yaml.safe_dump(present, allow_unicode=True, sort_keys=False)
    return f"---\n{body}---\n"
