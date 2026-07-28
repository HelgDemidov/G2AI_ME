"""Схема метаданных записей корпуса (``meta.yaml``, corpus-layout-v2) + рендер frontmatter.

Структурная валидация — pydantic (типы, форматы, обязательность, стабильные enum).
Проверка принадлежности контролируемым словарям (doc_type / authority / topics /
g2ai_pattern) и ссылочной целостности ``relations`` вынесена в ``validate_sources.py``,
т.к. требует загрузки внешних vocab-файлов из ``pipeline/vocab/``.
"""
from __future__ import annotations

import datetime as _dt
import re
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """Верхний аналитический раскол корпуса (== верхняя папка под ``sources/``; corpus-layout-v2).

    ``target_entity`` (переименован из ``montenegro`` 2026-07-25): ПЕРВИЧНЫЙ объект анализа
    проекта — государство(-а), для которого(-ых) пишется итоговый пакет предложений (сегодня
    единственно Черногория; список — `pipeline/config/target_entities.yaml`, НЕ хардкод в коде
    — `discovery/manual.py::_default_track` читает его, не сравнивает jurisdiction с "me"
    напрямую). Переименован из странового имени в ролевое: расширение фокуса на новую
    юрисдикцию — правка конфига, не схемы; `entity_id` (== iso2 юрисдикции) внутри трека
    остаётся содержательным разграничителем МЕЖДУ несколькими target-сущностями, когда/если
    их станет больше одной.

    ``research_papers`` (2026-07-19): третья, ГЕОГРАФИЧЕСКИ-НЕЙТРАЛЬНАЯ линия — вторичная
    аналитическая литература (think tank/university research hub отчёты: WEF, CNAS, UNIDIR
    и т.п.), а не практика конкретного государства. Отличие от intl_xperience/target_entity:
    те — ПЕРВОИСТОЧНИКИ (что государство реально сделало/приняло), это — АНАЛИЗ О практиках
    (что исследователи ДУМАЮТ о них); классическое разделение primary/secondary sources.
    entity_id для этого трека — слаг ИЗДАТЕЛЯ (`wef`/`unidir`/`cnas`, не iso2 — geo_scope
    здесь typically global/international, гейт `geo_scope=national ⇒ entity_id==iso2`
    (`validate_sources.py`) не применяется). Суб-иерархия — по издателю (см. entity_id),
    НЕ по теме: `topics`/`g2ai_pattern` уже дают полную, многозначную тематическую
    классификацию через слой знаний (`doc_facets`/`topics_map`) — дублировать её жёсткой
    (одна тема на папку) файловой иерархией значило бы проиграть многотемным отчётам
    ничего не давая взамен (тот же вывод независимо подтверждён обзором практик
    коммерческих RAG-систем 2025-2026: иерархия retrieval живёт в метаданных документа/
    секции/чанка, а не в файловой системе — она там для провенанса и уникальности id).

    ``tech_standards`` (2026-07-24, спек `aiforgood-standards`): четвёртая линия —
    технические стандарты ИИ (ITU-T/ITU-R/U4SSC/IETF/ETSI/TTA), слой знания, на который
    опираются регуляции/стратегии остальных треков. entity_id — слаг ИЗДАЮЩЕЙ ОРГАНИЗАЦИИ
    (`itu-t`/`etsi`/`tta`/…), тот же паттерн, что `research_papers` — не страна, поэтому
    `geo_scope=international` для ВСЕХ записей (иначе `entity_id="tta"` сломал бы гейт
    `geo_scope=national ⇒ entity_id==iso2`). Классификация международный/национальный/
    отраслевой — НЕ это поле, а отдельный справочник `vocab_standards_bodies.yaml`
    (аналог `jurisdictions.yaml`), см. спек §3.
    """

    intl_xperience = "intl-xperience"
    target_entity = "target-entity"
    research_papers = "research-papers"
    tech_standards = "tech-standards"


class SourceFormat(str, Enum):
    """Формат первоисточника: диктует расширение ``raw.*``, классификацию добычи и конвертер
    (чартер ``convert/architecture.md`` §3.2)."""

    pdf = "pdf"
    html = "html"
    docx = "docx"
    xlsx = "xlsx"


class AcquisitionMethod(str, Enum):
    """Последний-известный канал добычи байтов.

    Подсказка, не жёсткий конфиг: оркестратор всё равно идёт по лестнице
    ``direct -> official_alt -> manual -> archive`` (см. ``source-acquisition-ladder/spec.md``).
    """

    direct = "direct"
    official_alt = "official_alt"
    browser = "browser"  # headless-browser-resolver: Puppeteer-core+Lightpanda, только expected=html (spec §5)
    manual = "manual"
    archive = "archive"


class Fidelity(str, Enum):
    """Честность добытых байтов относительно официального источника."""

    live = "live"
    rehost = "rehost"
    rendered = "rendered"  # прошло через JS-рендер headless-браузера, не сырые байты сервера (browser rung)
    manual = "manual"
    archived_snapshot = "archived_snapshot"


class Sensitivity(str, Enum):
    """Чувствительность документа. Гейтит acquisition-лестницу: confidential -> archive недоступен."""

    normal = "normal"
    confidential = "confidential"


def external_disclosure_allowed(sensitivity: Sensitivity | None) -> bool:
    """Единый предикат политики «confidential не раскрывается третьим сторонам»
    (spec acquire-convert-seam-hardening §5, В5).

    До этого спека политика была реализована 8 независимыми точками ветвления в
    7 модулях (лестница добычи/SPN×2/облачный OCR-VLM/эмбеддинг×3/snowball-цитаты) —
    булев близнец класса «строка в N копиях»: новый внешний touchpoint должен САМ
    вспомнить про гейт, а забытая копия не падает, а молча раскрывает. Единая точка
    не устраняет необходимость КАЖДОЙ точке её вызвать, но делает нарушение видимым
    (guard-тест: прямое сравнение с ``Sensitivity.confidential`` вне этого модуля —
    красный тест).

    ``None`` == ``normal``-семантика (best-effort записи слоя discovery, где
    ``CandidateRecord.sensitivity`` ещё не финализирован триажем) — раскрытие
    разрешено, симметрично дефолту ``SourceRecord.sensitivity = Sensitivity.normal``.

    Граница политики (важно для корректного применения): «третья сторона» — любой
    сервис, НЕ являющийся самим издателем документа (облачный API эмбеддинга/OCR/VLM,
    Wayback CDX/SavePageNow, LLM-стадия snowball). Условный GET recheck-контура к
    ОФИЦИАЛЬНОМУ URL издателя этим предикатом НЕ гейтится — третьих сторон там нет
    (spec post-acquisition-lifecycle §1: confidential сознательно остаётся в
    ротации (a), в отличие от archive-ступени лестницы и SPN)."""
    return sensitivity is not Sensitivity.confidential


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


class ConnectorKind(str, Enum):
    """Архетип discovery-коннектора, породившего кандидата (discovery/architecture.md §3).

    Общий стабильный enum: определяется здесь (розетка триажа), импортируется discovery-core.
    """

    registry = "registry"
    outlet_watcher = "outlet_watcher"
    directed_search = "directed_search"
    manual = "manual"
    snowball = "snowball"


class RejectionKind(str, Enum):
    """Природа отказа триажа (spec post-acquisition-lifecycle §5).

    Enum, а не свободная строка, по правилу проекта: код ВЕТВИТСЯ по значению —
    ``unacquirable`` открывает кандидату вторую жизнь (отдельная секция worksheet,
    популяция (c) recheck-контура), ``irrelevant`` терминален. Отсутствие поля
    (легаси-кандидаты, отклонённые до этого спека) читается как ``irrelevant``:
    «не нужен» — исторический дефолт единственного существовавшего отказа.
    """

    irrelevant = "irrelevant"          # содержательно не подходит — решение окончательное
    unacquirable = "unacquirable"      # нужен, но недобываем (WAF/мёртвая ссылка) — ждёт смены обстоятельств


class TitleProvenance(str, Enum):
    """Откуда у кандидата взялся ``title`` (spec triage-intake-hardening §3).

    Enum, а не bool, по правилу проекта: код ВЕТВИТСЯ по значению — ``derived``
    ОБЯЗЫВАЕТ admit-решение задать ``title`` явно, ``stated`` нет. Отсутствие поля
    (легаси-кандидаты) читается как «неизвестно» и требования не поднимает: молча
    заблокировать допуск записей, заведённых до этого спека, было бы хуже, чем
    пропустить их с прежним поведением.

    Живая мотивация: у snowball 117 заголовков из 227 (52%) непригодны — ``map_link``
    берёт либо текст, вырезанный ГЕОМЕТРИЕЙ прямоугольника гиперссылки (режет слова,
    склеивает колонки), либо ЦЕЛУЮ строку ``doc.md``, в которой встретился URL. Ни то,
    ни другое никогда не было заголовком, а ``title`` становится МЕТКОЙ УЗЛА графа —
    мусор, допущенный дверью, застывает структурой слоя знаний.
    """

    stated = "stated"      # пришёл КАК заголовок: поле реестра, --title, verbatim-гейт цитат
    derived = "derived"    # реконструкция из позиционного артефакта — требует явного title при admit


class UrlProvenance(str, Enum):
    """Достоверность ``source_url`` кандидата (spec triage-intake-hardening §6).

    Тот же механизм и то же правило, что у ``TitleProvenance``: код ВЕТВИТСЯ по значению
    (``suspect`` обязывает admit-решение задать ``source_url`` явно), отсутствие поля —
    «неизвестно», требования не поднимает.

    Живая мотивация: в экспорте OECD.AI одно значение ``website`` принадлежит нескольким
    записям с РАЗНЫМИ заголовками (49 URL на 104 записи снапшота) — испанская стратегия
    несёт ссылку на норвежский Helsedirektoratet, немецкий «AI Action Plan» — на
    французский La French Tech. Опасность именно в правдоподобии: URL настоящий, рабочий
    и тематически смежный, а заголовок/издатель/юрисдикция ВЕРНЫЕ, поэтому карточка
    проходит любую проверку формы — а добыча молча скачает документ другой страны.
    """

    stated = "stated"      # источник назвал URL именно этого документа
    suspect = "suspect"    # значение делят записи с разными заголовками — владельца не определить


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
    # Поля ``last_checked`` здесь БОЛЬШЕ НЕТ (spec post-acquisition-lifecycle §7):
    # ручная бухгалтерия свежести протухает по построению (заполнялась вручную в
    # 4 записях и не имела ни одного кодового читателя). Её работу выполняет
    # ``OperationalState.acquisition_checked`` — машиннописаный курсор recheck-контура.


class Admission(BaseModel):
    """Вердикт допуска в корпус (source-relevance-triage). Присваивает ТОЛЬКО триаж."""

    model_config = ConfigDict(extra="forbid")

    axis: str = Field(min_length=1)  # словарь — validate_sources.py (vocab_axes)
    rationale: str = Field(min_length=1)


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
    # spec ocr-eval-harness §8.1 (S1): sha256 raw ДО OCR-мутации (ocrmypdf переписывает
    # raw.pdf in-place, §2 convert-ocr — `sha256` выше отражает мутированный файл, не
    # издательский оригинал). Проставляется ОДИН РАЗ, РОВНО когда нормализация реально
    # происходит (converters._ocr_normalize, ветка NeedsOCR), НЕ ретроспективно по факту
    # `_was_ocr_normalized` — иначе уже мутированные документы получили бы неверный хэш.
    # ЖЁСТКИЙ КАВЕТАТ: для сканов, сконвертированных ДО появления этого поля (на момент
    # написания — единственный, me-crps-registration-law-2025), оригинал уже утрачен —
    # `original_sha256` останется None НАВСЕГДА; восстановление возможно только
    # пере-добычей документа от издателя (вне скоупа), не задним числом в коде.
    original_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    acquisition_method: AcquisitionMethod | None = None
    acquisition_checked: _dt.date | None = None
    fidelity: Fidelity | None = None
    retrieved_snapshot_date: _dt.date | None = None
    converter_name: str | None = None     # какой конвертер породил текущий doc.md
    converter_version: str | None = None  # его версия (реконсиляция реконверсии)
    # C1 (spec convert-hardening): авто-QA вместо ручного аудита каждого документа —
    # список строк-дефектов convert/lint.py (пустой = чисто). Старые .state.yaml без
    # поля валидны (Field с default).
    # Потребители (spec convert-knowledge-seam-hardening §3): ИМЕНА дефектов едут в
    # фасет `doc_facets.quality_flags` и оттуда — пометкой ⚑ в выдаче hsearch; полная
    # евидентность (числа расхождений и т.п.) остаётся здесь. Прежний докстринг обещал
    # читателя-worksheet, которого никогда не существовало (аудит шва, Б2): поле
    # писалось прилежно и не читалось ничем — документ с непогашенным расхождением
    # чисел был в выдаче неотличим от чистого.
    lint_defects: list[str] = Field(default_factory=list)
    # spec convert-cloud-tier §2.2: валидность кэша .cloudocr.md — по этим двум
    # полям (модель + sha256 raw НА МОМЕНТ облачного вызова, т.е. ПОСЛЕ ocrmypdf).
    cloud_ocr_model: str | None = None
    cloud_ocr_raw_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # --- контур времени (spec post-acquisition-lifecycle §7); все Optional/с дефолтом,
    # старые .state.yaml валидны без единой правки ---
    # Серверные валидаторы ОТВЕТА, verbatim: единственный дешёвый способ спросить
    # издателя «изменилось ли с прошлого раза» (условный GET). Захватываются ТОЛЬКО
    # на ступенях direct/official_alt (§2): archive отдаёт заголовки Wayback, а не
    # издателя, browser — синтетические, manual — вовсе без HTTP-ответа.
    etag: str | None = None
    http_last_modified: str | None = None
    # Счётчик ПОДРЯД-совпадений валидатора (инкремент при 304 либо 200 с совпавшим
    # ETag; сброс в 0 при смене/первичном бутстрапе). Без него правило «drift для
    # html только при смене СТАБИЛЬНОГО валидатора» нереализуемо: по одним лишь
    # etag/http_last_modified неотличимо «забутстрапили вчера» от «совпал пять раз».
    etag_confirms: int = 0
    # Findings recheck-контура — ОДНО поле-строка с префиксом (паттерн lint_defects,
    # §2): `drift:`/`link-rot:`/`resurrected:` + евидентность. Машина флагует, человек
    # читает и решает (§6); очищается успешной передобычей.
    recheck_finding: str | None = None
    # Дата ПОПЫТКИ проактивного снимка SavePageNow (§4) — best-effort страховка, не
    # гарантия: нужна идемпотентности (повторный прогон не дёргает SPN за уже добытое).
    snapshot_requested: _dt.date | None = None
    # Backoff недобытых (§5): когда лестница провалилась и почему. Пока свежее
    # RETRY_BACKOFF_DAYS — download не планируется (кроме --force/--only); успешная
    # добыча очищает оба поля. ⚠ ЯКОРЬ backoff'а, НЕ курсор ротации (b) — см.
    # acquisition_probe_checked ниже (spec discovery-acquire-seam-hardening §3, Г2):
    # только полноценная попытка ВСЕЙ лестницы вправе переустановить это поле.
    acquisition_failed: _dt.date | None = None
    acquisition_failure_reason: str | None = None
    # Курсор ротации популяции (b) recheck-контура (spec discovery-acquire-seam-
    # hardening §3, Г2) — дата ПОСЛЕДНЕГО probe, отдельно от `acquisition_failed`.
    # Прежде probe при неуспехе бампал сам `acquisition_failed` (единственное
    # доступное ему поле) — тем самым ЗАНОВО закрывал окно backoff при каждом
    # регулярном `--recheck`, хотя probe заведомо СЛАБЕЕ полной лестницы (один GET
    # по `source_url`: official_alt/browser/archive ему недостижимы). Три класса
    # документов, которые лестница добыла бы, оставались «недобываемы» навсегда —
    # и тем надёжнее, чем прилежнее куратор гоняет контур времени (живой репро
    # аудита). Разведение ролей: probe бампает ТОЛЬКО это поле (+ reason), полная
    # попытка лестницы (`_record_acquisition_failure`) — только `acquisition_failed`.
    # Optional/дефолт None — легаси `.state.yaml` без поля валидны, ротация (b)
    # фолбэчит на `acquisition_failed` (см. `recheck.due_records`).
    acquisition_probe_checked: _dt.date | None = None
    # Backoff отказов КОНВЕРТАЦИИ (spec acquire-convert-seam-hardening §8, В11) —
    # зеркало пары полей выше, но для стадии convert, у которой backoff не было
    # вовсе: патологический документ (напр. 2-часовой OCR_TIMEOUT) повторял бы
    # полную стоимость КАЖДЫЙ прогон, след оставался только в логе. Успешная
    # конвертация ИЛИ свежая добыча (новые байты = новый вход) очищают все три поля.
    convert_failed: _dt.date | None = None
    convert_failure_reason: str | None = None
    # Конвертер, на котором провалилась ПОСЛЕДНЯЯ попытка — "имя@версия" одним
    # значением (компактный контекст инвалидации: бамп версии конвертера пробивает
    # backoff немедленно, детерминированный отказ старой версии не предсказывает
    # исход новой).
    convert_failed_converter: str | None = None


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
    admission: Admission | None = None       # вердикт триажа; обязателен — правило validate_sources
    # Поля ``in_force`` здесь БОЛЬШЕ НЕТ (spec post-acquisition-lifecycle §7): ручной
    # bool «действует ли закон» на сотнях записей обречён протухнуть, и ни один
    # потребитель его не читал. Валидность ВЫВОДИТСЯ из рёбер ``supersedes``
    # (см. ``superseded_ids`` ниже и спек graph-v2), а не ведётся руками.


# Жёсткий предел native_summary: ~2-3 предложения. Не «мягкая рекомендация» — pydantic
# отказывает; ручной inject сокращает куратор, адаптер registry-коннектора обрезает
# ДО валидации (обязательство его спека).
CANDIDATE_SUMMARY_MAX = 600


class CandidateRecord(BaseModel):
    """Кандидат-источник из DISCOVERY (живёт в candidates.yaml, до допуска в реестр).

    Лёгкий пермиссивный (``extra="allow"``) upstream-кузен ``SourceRecord``: данные
    коннекторов разнородны и неполны. Не несёт ``relevance`` (discovery не оценивает)
    и не требует ``id`` (присваивается при промоушене). Контракт-розетка между
    DISCOVERY (writer) и триажем (reader); см. source-relevance-triage §3.

    Порядок полей = порядок дампа в candidates.yaml (человекочитаемость: сначала
    библиография во главе с ``title``, провенанс — внизу). Архетип коннектора НЕ
    хранится отдельным полем (слим 2026-07-21: было write-only дублирование) —
    он выводится из грамматики ``connector_id``: ``manual`` -> ручной канал,
    ``search:<кампания>`` -> directed_search, прочие id -> кодовые коннекторы
    (archetype — атрибут класса коннектора, см. ``discovery/base.py``). Указатель
    на нативную запись источника у registry-коннекторов — ``native_id`` (их спеки
    обязаны его заполнять; отдельного ``source_ref`` больше нет). Легаси-поля
    старых файлов (``connector_kind``/``source_ref``/``reported_status``)
    поглощаются ``extra="allow"`` без ошибок.
    """

    model_config = ConfigDict(extra="allow")

    # best-effort библиография (Optional — данные upstream неполны)
    title: str | None = None
    # Откуда взялся title (spec triage-intake-hardening §3). ``None`` у легаси-записей
    # = «неизвестно», требования явного title при admit не поднимает.
    title_provenance: TitleProvenance | None = None
    issuer: str | None = None
    jurisdiction: str | None = None
    doc_date: _dt.date | None = None
    # Год документа — сигнал триажа (планка `frontier_year`, `pipeline/config/triage.yaml`).
    # ОТДЕЛЬНОЕ поле, а не замена ``doc_date``: три источника из пяти дают только год
    # (oecd `startYear`), а два — настоящую дату с точностью до дня (agora/eurlex, 430
    # кандидатов на 2026-07-28, из них ровно 1 приходится на 1 января). Огрубление
    # ``doc_date`` до года было бы фабрикацией наоборот: `dates.published` кормит
    # `valid_from` -> интервалы валидности -> срезы `--as-of`, и акт, действующий с
    # 2024-12-01, попал бы в срезы, где его ещё не было. Год выводится ЕДИНСТВЕННОЙ
    # точкой — валидатором ниже, поэтому коннектор не может его забыть или вывести иначе.
    doc_year: int | None = Field(default=None, ge=1000, le=2999)
    language: str | None = None
    source_url: str | None = Field(default=None, pattern=r"^https?://")
    # Достоверность source_url (spec triage-intake-hardening §6). ``None`` у легаси-записей
    # = «неизвестно», требования явного source_url при admit не поднимает.
    url_provenance: UrlProvenance | None = None
    rights: Rights | None = None  # best-effort от коннектора; финализирует триаж
    sensitivity: Sensitivity | None = None  # best-effort; несётся в acquisition-гейт
    # passthrough-обогащение источника (Optional; None -> не пишется в YAML вовсе)
    native_summary: str | None = Field(default=None, max_length=CANDIDATE_SUMMARY_MAX)
    native_id: str | None = None
    native_tags: list[str] | None = None
    # дешёвый pre-signal (НЕ вердикт — ось присваивает только триаж)
    matched_query: str | None = None
    matched_vocab_tags: list[str] | None = None
    # Подсказка формата (spec discovery-acquire-seam-hardening §8, Г7) — НЕ вердикт:
    # формат ОБЪЯВЛЯЕТ куратор при триаже, подсказка лишь замещает молчаливый дефолт
    # "pdf" в admit-двери, когда коннектор знает формат by construction (eurlex строит
    # `/TXT/HTML/`-URL) либо по расширению хвоста URL (`dedup.format_hint_from_url`).
    # Ошибочная подсказка опаснее отсутствующей — см. design rationale спека.
    native_format_hint: SourceFormat | None = None
    # провенанс добычи (обязательно)
    connector_id: str = Field(min_length=1)
    retrieved_at: _dt.date
    raw_hash: str = Field(min_length=1)
    # Заявленная РЕДАКЦИЯ: doc-id записи корпуса, которую этот кандидат сознательно
    # заменяет (spec discovery-candidates-sharding §5). Идентичность кандидата — не URL,
    # а пара (URL-идентичность, редакция): новая редакция закона живёт на ТОМ ЖЕ URL
    # (задокументированный факт «движущейся цели»), и без дискриминатора dedup поглощал
    # бы её как дубль предшественника — цикл редакций был бы физически невозможен через
    # единственную дверь `inject`. Концептуально — различение work/expression из ELI:
    # URL адресует work, корпусу нужны expressions. Дефолт None => поведение всех
    # существующих путей не меняется ни на бит (ключи с None-компонентой эквивалентны
    # прежним). Заполняется ТОЛЬКО явным `inject --supersedes` (решение человека, не
    # автоматика); `promote_candidate` материализует его в ребро графа.
    supersedes: str | None = Field(default=None, pattern=ID_PATTERN)
    # doc-id записи, которой кандидат стал при допуске (spec candidate-identity-hardening §1).
    # Штампуется ``apply_decisions`` после успешного ``save_record`` и служит ЕДИНСТВЕННЫМ
    # ключом реконсиляции обеих очередей слоя. Штамп сверяется с реестром, а не принимается
    # на веру: удаление папки документа (документированный путь исправления ошибки допуска)
    # возвращает кандидата в ждущие само.
    admitted_as: str | None = Field(default=None, pattern=ID_PATTERN)
    # причина отказа (если триаж отклонил — кандидат остаётся в слое кандидатов)
    rejected_reason: str | None = None
    # Природа отказа (spec post-acquisition-lifecycle §5): None у легаси-записей ==
    # ``irrelevant``-семантика. ``unacquirable`` — «нужен, но недобываем»: кандидат
    # уходит не в терминальный отказ, а в очередь ожидания обстоятельств (WAF снимут,
    # появится зеркало), которую периодически пробует recheck-контур.
    rejected_kind: RejectionKind | None = None
    # Probe-поля популяции (c) recheck: когда последний раз пробовали URL недобываемого
    # кандидата и что увидели (``acquirable``/``blocked``/``dead`` + детали). Пишутся
    # ТОЛЬКО машиной через store load-mutate-save; курируемого смысла не несут.
    probe_checked: _dt.date | None = None
    probe_finding: str | None = None

    @model_validator(mode="after")
    def _derive_doc_year(self) -> "CandidateRecord":
        """``doc_year`` выводится из ``doc_date``, если коннектор не дал год явно.

        ЕДИНСТВЕННАЯ точка вывода (решение куратора 2026-07-28): источники разнородны —
        agora/eurlex/manual дают полную дату, oecd только ``startYear``, aiforgood и
        snowball не дают ничего, — но обрабатываться год обязан ОДИНАКОВО, иначе пятый
        коннектор выведет его шестым способом. Валидатор на модели, а не хелпер в
        `dedup`, ровно поэтому: забыть вызов невозможно, поле заполняется самим фактом
        конструирования записи.

        **Противоречие даты и года — ГРОМКИЙ отказ, не тихая резолюция.** Источник
        с годом-без-даты (oecd) остаётся хозяином своего значения, но если коннектор
        подал ОБА и они расходятся — это его баг, а не выбор между двумя правдами:
        молча предпочесть любое значило бы отдать `valid_from` (а через него срезы
        `--as-of`) на волю случайности. Сегодня ни один путь противоречия не порождает
        (замер 2026-07-28: 430 записей с обоими полями, расхождений 0), поэтому запрет
        ничего не ломает и закрывает класс на будущее.
        """
        if self.doc_date is not None:
            if self.doc_year is None:
                self.doc_year = self.doc_date.year
            elif self.doc_year != self.doc_date.year:
                raise ValueError(
                    f"doc_year={self.doc_year} противоречит doc_date={self.doc_date.isoformat()} "
                    "— коннектор обязан подать либо согласованные значения, либо только одно"
                )
        return self


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


STATE_SIDECAR_NAME = ".state.yaml"
"""Имя пер-документного операционного сайдкара — единственное определение (spec
acquire-convert-seam-hardening §1/В12): до этого спека литерал жил в 3 копиях
(здесь + ``convert/converters.py`` дважды), и расхождение при смене раскладки не
падало бы, а тихо читало ПУСТОЕ состояние (``load_state`` несуществующего пути
честно возвращает ``OperationalState()``) — конвертер решил бы, что кэш невалиден,
и заново заплатил бы за облачный вызов."""


def state_file(rec: SourceRecord, root: Path) -> Path:
    """Операционный sidecar: ``<doc_dir>/.state.yaml``."""
    return doc_dir(rec, root) / STATE_SIDECAR_NAME


def state_file_for_raw(raw: Path) -> Path:
    """Тот же сайдкар, адресованный от пути ``raw.*`` — для точек convert-слоя,
    у которых есть только файл raw, без записи/корня (``converters._capture_original_sha256``/
    ``_cached_or_call_cloud``/``_ocr_normalize``)."""
    return raw.parent / STATE_SIDECAR_NAME


STATE_DIRNAME = ".state"


def state_dir(root: Path = DEFAULT_SOURCES) -> Path:
    """Каталог операционного состояния КОРПУСА (в отличие от ``state_file`` выше —
    пер-документного): ``<root>/.state/``. Тот же концепт («операционное, не для
    курирования человеком»), что и ``.state.yaml``, — одно слово на двух уровнях
    вложенности.

    Владелец каталога — этот модуль (``core.schema``, куда переехало со дня
    введения понятия в ``discovery/store.py`` — knowledge-hardening §2: у каталога
    уже три писателя из ДВУХ слоёв — ``discovery/store.py`` курсоры,
    ``connectors/snowball.py`` лиды, ``graph/cite_mining.py`` dangling-цитаты —
    он перерос «владение» discovery). ИМЯ ФАЙЛА внутри принадлежит писателю: этот
    модуль не знает о существовании конкретных коннекторов/майнеров, а писатели не
    дублируют знание о раскладке.
    """
    return root / STATE_DIRNAME


def corpus_lock_path(root: Path = DEFAULT_SOURCES) -> Path:
    """Эксклюзивный лок мутаторов КОРПУСА (spec discovery-acquire-seam-hardening §2, Г1):
    ``<root>/.state/corpus_mutation.lock`` — рядом с ``state_dir``, тот же владелец раскладки.

    Держатели: ``run_pipeline.py`` (штатный прогон стадий и ``--recheck`` — оба
    писатели ``.state.yaml``) и mutating-подкоманды ``discover.py`` (``inject``/
    ``apply``/``discover``/``snowball``, все без ``--dry-run`` — писатели слоя
    кандидатов, ``sources/candidates/``). Прежняя конвенция «имя файла — писатель»
    (см. ``state_dir`` выше) для ОБЩЕГО лока не работает по построению: у лока
    ВСЕГДА больше одного держателя, имя фиксирует НАЗНАЧЕНИЕ (взаимоисключение
    мутаторов корпуса), а не единственного писателя. До этого спека лок
    (``run_pipeline._run_lock_path``) взаимоисключал только штатный прогон и
    ``--recheck`` — пара ``--recheck`` ↔ ``discover.py`` оставалась «принятым
    остаточным риском», который живой репро аудита показал недооценённым (окно —
    минуты-часы сетевой работы, а не секунды; исход — физическое удаление шарда
    кандидатов или откат целого apply-батча, не «потеря записи друг друга»).
    """
    return state_dir(root) / "corpus_mutation.lock"


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


def superseded_ids(records: list[SourceRecord]) -> set[str]:
    """id записей, которые ЗАМЕНЕНЫ другой записью корпуса (выведенная валидность).

    Нормализует ОБА направления ребра — ``supersedes`` (у преемника) и
    ``superseded_by`` (у предшественника): оба легальны в ``RelationType``, и запись,
    оформленная любым из них, обязана трактоваться одинаково. Единственное
    определение на двух потребителей (spec post-acquisition-lifecycle §1 — исключение
    из ротации recheck; spec graph-v2 §2 — фасет ``superseded``), поэтому расхождение
    между ними исключено by construction, а не дисциплиной.

    Дальний конец ребра не проверяется на существование — ссылочную целостность
    ``relations`` держит ``validate_sources.py``, дублировать её здесь незачем.
    """
    superseded: set[str] = set()
    for rec in records:
        for rel in rec.relations:
            if rel.type is RelationType.supersedes:
                superseded.add(rel.target)      # rec заменяет target
            elif rel.type is RelationType.superseded_by:
                superseded.add(rec.id)          # rec заменён target'ом
    return superseded


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


def save_record(rec: SourceRecord, root: Path) -> Path:
    """Записать курируемый ``meta.yaml`` в папку документа (создав её): ``doc_dir/meta.yaml``.

    Атомарно (``fsio.atomic_write_text``, создаёт недостающие каталоги). Существующий
    ``meta.yaml`` -> ``ValueError`` — перезапись курируемого файла запрещена (правки только
    руками/явным решением куратора), тот же принцип, что у ``.state.yaml`` не применяется
    (сравни с ``save_state``, машиннописаный sidecar перезаписывается свободно). Возвращает
    путь ``meta.yaml``. Человек остаётся источником решений — этот писатель лишь материализует
    аргументы ``promote_candidate`` на диск.
    """
    path = doc_dir(rec, root) / "meta.yaml"
    if path.exists():
        raise ValueError(f"{path}: meta.yaml уже существует — перезапись курируемого файла запрещена")
    payload = rec.model_dump(mode="json", exclude_none=True)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    fsio.atomic_write_text(path, text)
    return path


CONTROL_DECISION_KEYS = frozenset({"raw_hash", "action", "reason", "reject_kind", "hidden_fields"})
"""Служебные ключи решения триажа — адресуют САМО решение (кого/что делать), а не запись
корпуса, и в ``curated`` не попадают. Единственное определение (spec drop-cursors-and-
decision-overlay §3): ``manual._build_admit_record`` фильтрует по нему."""


def promote_candidate(cand: CandidateRecord, curated: dict[str, Any]) -> SourceRecord:
    """Промоутнуть кандидата в курируемый ``SourceRecord`` (конверсия типа для ``meta.yaml``).

    **Решение триажа ЕСТЬ курируемая часть записи** — не двадцать именованных параметров,
    в которые каждое новое поле схемы приходилось добавлять правкой (шесть версий такого
    накопления подряд: ``topics``/``g2ai_pattern``/``summary``/``relations`` -> ``language``
    -> ``supersedes`` -> ``official_alt_url`` -> ``title``/``issuer``/``source_url``).
    ``curated`` накладывается ПОВЕРХ того, что кандидат уже знает о документе, поэтому:

    * любое поле ``SourceRecord`` выразимо решением без единой строки кода;
    * ключ ``dates`` наконец выразим — до этого дверь жёстко ставила
      ``Dates(published=cand.doc_date)``, и 76% очереди дали бы корпус с пустой временнóй
      осью (``validity_known=False`` в графе);
    * опечатка в ключе — ГРОМКИЙ отказ (``SourceRecord`` объявлен ``extra="forbid"``), а не
      молча потерянное поле: для машинного судьи это разница между «не записалось» и «не
      записалось, и никто не узнал».

    Размен назван явно: статическая типизация вызова меняется на рантайм-валидацию. Вход и
    так приходит из YAML, где mypy никогда не работал, а pydantic отказывает громко; но
    поверхность записи расширяется с курируемого подмножества до всей записи. Компенсация —
    ``extra="forbid"``, ``pattern`` полей и постбатчевый ``validate_sources`` (уже есть).

    Из кандидата берутся ``title``/``issuer``/``language``/``source_url`` (любое ``curated``
    их перекрывает — у части каналов их нет вовсе либо значение недостоверно), ``rights``/
    ``sensitivity`` с прежними дефолтами и ``doc_date`` -> ``dates.published``. Провенанс
    добычи остаётся в слое кандидатов (в ``meta.yaml`` НЕ копируется — corpus-layout-v2).

    ``cand.supersedes`` — ЕДИНСТВЕННАЯ точка, где временнáя цепочка редакций материализуется
    в курируемое ядро: непустое значение добавляет ребро ``supersedes`` без дубля по ключу
    ``(type, target)`` (куратор, вписавший то же ребро руками, не получает двойного).
    Потребитель ребра — вывод валидности graph-v2.
    """
    base: dict[str, Any] = {
        "title": cand.title,
        "issuer": cand.issuer,
        "language": cand.language,
        "source_url": cand.source_url,
        "rights": cand.rights or Rights.unknown,
        "sensitivity": cand.sensitivity or Sensitivity.normal,
    }
    if cand.doc_date is not None:
        base["dates"] = {"published": cand.doc_date}
    data = {key: value for key, value in base.items() if value is not None}
    data.update(curated)

    relations = list(data.get("relations") or [])
    if cand.supersedes is not None:
        present = any(
            _relation_key(rel) == (RelationType.supersedes.value, cand.supersedes)
            for rel in relations
        )
        if not present:
            relations.append({"type": RelationType.supersedes.value, "target": cand.supersedes})
    if relations:
        data["relations"] = relations

    return SourceRecord.model_validate(data)


def _relation_key(rel: Any) -> tuple[str, str]:
    """``(тип, цель)`` ребра, пришедшего либо словарём из решения, либо объектом ``Relation``."""
    if isinstance(rel, Relation):
        return (rel.type.value, rel.target)
    return (str(rel["type"]), str(rel["target"]))


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


# Якорь ``^`` без re.MULTILINE — совпадает ТОЛЬКО в начале строки-документа, поэтому
# горизонтальная линейка ``---`` в теле (markdownify эмитит её для docx ``<hr>``)
# отдельным фронтматтером не станет.
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def strip_frontmatter(md: str) -> str:
    """Снять YAML-frontmatter в начале ``.md`` (если он есть) — обратная сторона
    ``render_frontmatter``.

    Живёт ЗДЕСЬ, а не в ``index.chunking`` (spec convert-knowledge-seam-hardening §6):
    пишущая и снимающая половины одной грамматики принадлежат одному модулю, иначе
    потребители «тела документа» из разных слоёв тянут зависимость вверх по конвейеру
    (``convert.lint`` -> ``index``) — тот же класс, что переезд ``state_dir``
    (knowledge-hardening §2). ``index.chunking`` реэкспортирует имя для совместимости.
    """
    return _FRONTMATTER_RE.sub("", md, count=1)
