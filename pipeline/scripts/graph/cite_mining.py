"""L1: детерминированный слой цитирования — spec graph-v2 §3.

Юридический корпус аномально хорошо цитируем ФОРМАЛЬНО: CELEX, «Службени лист ЦГ»,
ISO/IEC, NIST. Это домен-преимущество, которого generic-GraphRAG не видит — рёбра
``cites`` извлекаются регексами с высокой точностью и НУЛЁМ LLM.

Без кэша/сайдкара: экстракция детерминирована и дёшева (секунды на сотни документов),
поэтому она консистентна с «полной пересборкой графа каждый прогон». Контраст с дорогим
VLM-кэшем ``.figures.yaml`` — там кэш оправдан ценой вызова, здесь был бы лишним
состоянием.

Резолюция идентификатора в doc-id — двухканальная и обе честны: (1) автоматически из
``source_url``, но ТОЛЬКО когда URL несёт идентификатор БУКВАЛЬНО; (2) курируемый
справочник ``pipeline/vocab/identifiers.yaml`` (класс ``jurisdictions`` — справочник БЕЗ
гейта словаря). Нерезолвнутое не выдумывается, а уходит в отчёт лидов.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core import fsio, schema
from discovery import store

logger = logging.getLogger("cite_mining")

IDENTIFIERS_PATH = schema.VOCAB_DIR / "identifiers.yaml"
CITE_LEADS_FILENAME = "cite_leads.yaml"
"""Имя файла отчёта ВНУТРИ ``sources/.state/`` — каталогом владеет ``discovery/store.py``,
имя файла принадлежит писателю (симметрия ``snowball_leads.yaml``)."""


@dataclass(frozen=True)
class CitePattern:
    """Один формальный идентификатор: как найти и как канонизировать.

    Реестр (``_PATTERNS``) — паттерн ``_CONVERTERS``: новый идентификатор добавляется
    ЗАПИСЬЮ, а не правкой ядра. Осознанная граница v1: сектор-3 CELEX покрыт типами
    актов L/R/D/E — рекомендации (H) и C-серия придут той же записью, когда встретятся
    в корпусе.

    ``canonical`` возвращает СПИСОК: одна цитата законно несёт несколько актов
    («Службени лист ЦГ бр. 65/20, 146/21 и 4/24» — три разных акта одной строкой).
    """

    name: str
    regex: re.Pattern[str]
    canonical: Callable[[re.Match[str]], list[str]]


def _celex(m: re.Match[str]) -> list[str]:
    return [f"CELEX:{m.group(1).upper()}"]


# Тип акта -> литера сектора-3 CELEX. Естественно-языковая ссылка канонизируется в ТО ЖЕ
# пространство идентификаторов, что и компактная форма: «Regulation (EU) 2024/1689» и
# «32024R1689» дают один идентификатор, одно ребро и одну строку identifiers.yaml.
_EU_ACT_LETTER = {"regulation": "R", "directive": "L", "decision": "D"}
# Правдоподобный диапазон года акта ЕС (ЕЭС основано в 1957-м; верхняя граница — запас на
# будущие акты). Вне диапазона совпадение отбрасывается ЦЕЛИКОМ, а не «поправляется»:
# «Decision 768/2008/EC» (номер/год/юрисдикция) без гейта давал `CELEX:30768D2008` —
# синтаксически правдоподобный мусор. Рёбер он не строит (резолюция не совпадёт), но
# отравляет `cite_leads.yaml`, который куратор читает при discovery-кампаниях.
_EU_YEAR_MIN, _EU_YEAR_MAX = 1950, 2049
# Гейт двузначных лет: «Службени лист ЦГ» существует только с 2006-го (независимость),
# поэтому 2-значный год — всегда 20xx. Форма /99 под литерой «CG» существовать не может.
_SLCG_CENTURY = 2000


def _eu_act(m: re.Match[str]) -> list[str]:
    """«Regulation (EU) 2024/1689» / «Regulation (EEC) No 3922/91» -> CELEX.

    Две ловушки, обе найдены живой сверкой с корпусом (2026-07-26), обе дают не пустой
    результат, а НЕВЕРНУЮ связь — самый дорогой класс ошибки для юридического графа:

    1. **Порядок чисел зависит от формы.** С 2015 года ЕС нумерует ``ГОД/номер``, до
       того — ``No номер/ГОД``. Обе формы живут в корпусе одновременно (219 и 79
       упоминаний), так что ветка не гипотетическая.
    2. **Год в старой форме бывает двузначным** («No 3922/91» = 1991). Без разворота
       получался ``CELEX:30091R3922`` — синтаксически правдоподобный, семантически
       мусор. Двузначный год здесь всегда 19xx: четырёхзначная запись вошла в обиход
       к концу 1990-х, а сама форма «No N/YY» после этого не использовалась.
    """
    first, second = int(m.group("first")), int(m.group("second"))
    year, number = (second, first) if m.group("no") else (first, second)
    if year < 100:
        year += 1900
    if not _EU_YEAR_MIN <= year <= _EU_YEAR_MAX:
        return []   # см. `_EU_YEAR_MIN`: не гадать, а промолчать
    letter = _EU_ACT_LETTER[m.group("kind").lower()]
    return [f"CELEX:3{year:04d}{letter}{number:04d}"]


def _eu_act_slash(m: re.Match[str]) -> list[str]:
    """«Directive 2000/31/EC», «Directive 95/46/EC» -> CELEX.

    Третья живая форма ссылки на акт ЕС: БЕЗ скобочной юрисдикции после типа акта,
    зато с её суффиксом в конце. Порядок здесь всегда ``ГОД/номер`` (в отличие от
    формы ``No номер/ГОД``), год бывает двузначным. Замер на корпусе 2026-07-26: 40
    уникальных актов в этой форме не распознавались вовсе — треть всех EU-ссылок.
    """
    year, number = int(m.group("year")), int(m.group("number"))
    if year < 100:
        year += 1900
    if not _EU_YEAR_MIN <= year <= _EU_YEAR_MAX:
        return []   # «Decision 768/2008/EC» — номер на месте года; см. `_EU_YEAR_MIN`
    letter = _EU_ACT_LETTER[m.group("kind").lower()]
    return [f"CELEX:3{year:04d}{letter}{number:04d}"]


def _sluzbeni(m: re.Match[str]) -> list[str]:
    idents: list[str] = []
    for number, year in re.findall(r"(\d+)\s*/\s*(\d{2,4})", m.group("numbers")):
        full_year = int(year) if len(year) == 4 else _SLCG_CENTURY + int(year)
        idents.append(f"SLCG:{int(number)}/{full_year}")
    return idents


def _iso(m: re.Match[str]) -> list[str]:
    # Только именованные группы: смешение с позиционными уже давало сдвиг нумерации
    # (необязательная `/IEC` сама по себе — группа №1).
    ident = f"ISO/IEC {m.group('body')}" if m.group("iec") else f"ISO {m.group('body')}"
    if m.group("part"):
        ident += f"-{m.group('part')}"
    if m.group("year"):
        ident += f":{m.group('year')}"
    return [ident]


def _nist(m: re.Match[str]) -> list[str]:
    return [f"NIST SP 800-{m.group('sp')}"] if m.group("sp") else [f"NIST AI {m.group('ai')}"]


_PATTERNS: dict[str, CitePattern] = {
    # Совпадает и в голом виде, и внутри ELI/URL-форм (`...uri=CELEX:32024R1689`) —
    # отдельный URL-паттерн не нужен.
    "celex": CitePattern("celex", re.compile(r"\b(3\d{4}[LRDE]\d{4})\b"), _celex),
    # Живая калибровка по корпусу (2026-07-26): реальная форма несёт кавычку после «CG»
    # и СПИСОК актов одной строкой — «(„Službeni list CG", br. 65/20, 146/21 i 4/24)».
    "sluzbeni_list_cg": CitePattern(
        "sluzbeni_list_cg",
        re.compile(
            r"(?:Службени\s+лист\s+ЦГ|Slu[žz]beni\s+list\s+CG)[\"“”„»']*\s*,?\s*"
            r"(?:бр|br)\.?\s*"
            r"(?P<numbers>\d+\s*/\s*\d{2,4}(?:\s*(?:,|и|i)\s*\d+\s*/\s*\d{2,4})*)",
            re.IGNORECASE,
        ),
        _sluzbeni,
    ),
    # Естественно-языковая ссылка на акт ЕС — доминирующая форма в конвертированном
    # тексте (компактный CELEX в нём не встречается вовсе, 2026-07-26).
    "eu_act": CitePattern(
        "eu_act",
        re.compile(
            r"\b(?P<kind>Regulation|Directive|Decision)s?\s+\((?:EU|EC|EEC|Euratom)\)\s+"
            r"(?P<no>No\s+)?(?P<first>\d{1,4})\s*/\s*(?P<second>\d{1,4})\b"
        ),
        _eu_act,
    ),
    # Та же семья, что eu_act, но без скобочной юрисдикции: «Directive 2000/31/EC».
    # Отдельная запись, а не ветка в eu_act: реестр на то и реестр — форма своя,
    # разбор своей группы, ноль условий в общем коде.
    "eu_act_slash": CitePattern(
        "eu_act_slash",
        re.compile(
            r"\b(?P<kind>Directive|Decision|Regulation)s?\s+"
            r"(?P<year>\d{2,4})\s*/\s*(?P<number>\d{1,4})\s*/\s*(?:EC|EEC|EU|Euratom)\b"
        ),
        _eu_act_slash,
    ),
    "iso": CitePattern(
        "iso",
        re.compile(
            r"\bISO(?P<iec>/IEC)?\s+(?P<body>\d{3,5})"
            r"(?:-(?P<part>\d+))?(?::(?P<year>\d{4}))?\b"
        ),
        _iso,
    ),
    "nist": CitePattern(
        "nist",
        re.compile(r"\bNIST\s+(?:SP\s+800-(?P<sp>\d+[A-Za-z]?)|AI\s+(?P<ai>\d+(?:-\d+)?))\b"),
        _nist,
    ),
}


ALIAS_RULE = "alias"
"""Имя правила для попаданий курируемого алиас-канала — отличимо в ``CiteEdge.rule`` и
в отчёте лидов от машинных паттернов."""


def _alias_hits(text: str, aliases: dict[str, str]) -> list[tuple[str, str]]:
    """Попадания курируемых алиасов: literal + границы слов, БЕЗ регекс-обобщений.

    Матчинг регистронезависимый: алиас — имя собственное («EU AI Act», «GDPR»), его
    регистр плавает по вёрстке и страдает от OCR, а референт от этого не меняется.
    Границы слов дают флексии бесплатно («GDPR-om» в черногорском тексте), но не дают
    подстрок внутри слова.

    Пустой/пробельный алиас пропускается: строка ``"": doc-id`` от опечатки в YAML
    иначе совпала бы в КАЖДОМ документе корпуса.
    """
    hits: list[tuple[str, str]] = []
    for alias in sorted(aliases):
        if not alias.strip():
            continue
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE):
            hits.append((aliases[alias], ALIAS_RULE))
    return hits


def extract_identifiers(text: str, aliases: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """``(канонический идентификатор, имя правила)`` — все формальные цитаты текста.

    Порядок детерминирован (правила по имени, вхождения по позиции), дубли схлопнуты:
    один и тот же акт, упомянутый в документе десять раз, — одно ребро, не десять.

    ``aliases`` — курируемый канал (spec graph-hardening §2): дефолт ``None`` означает
    экстракцию ТОЛЬКО паттернами. Это не «забыли передать», а рабочий режим: голден-тест
    сторожит стабильность РЕГЕКСОВ и обязан не зависеть от файла, который куратор
    законно правит в любой момент.

    Алиасы применяются ПОСЛЕ паттернов: формальная цитата — более сильное свидетельство,
    поэтому при совпадении идентификатора правилом остаётся имя паттерна, а ``alias``
    достаётся только тому, чего паттерны не нашли.
    """
    seen: dict[str, str] = {}
    for name in sorted(_PATTERNS):
        pattern = _PATTERNS[name]
        for match in pattern.regex.finditer(text):
            for ident in pattern.canonical(match):
                seen.setdefault(ident, name)
    for ident, rule in _alias_hits(text, aliases or {}):
        seen.setdefault(ident, rule)
    return sorted(seen.items())


def _load_section(path: Path, section: str) -> dict[str, str]:
    """Секция справочника как ``dict[str, str]``; отсутствие файла/секции — пустой словарь
    (не ошибка: справочник наполняется по мере встречи идентификаторов, как
    ``jurisdictions.yaml``)."""
    if not path.exists():
        return {}
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    raw = data.get(section) or {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def load_identifiers(path: Path = IDENTIFIERS_PATH) -> dict[str, str]:
    """Курируемый справочник ``формальный идентификатор -> doc-id`` (канал резолюции)."""
    return _load_section(path, "identifiers")


def load_aliases(path: Path = IDENTIFIERS_PATH) -> dict[str, str]:
    """Курируемые алиасы ``строка в тексте -> канонический идентификатор`` (канал
    экстракции, spec graph-hardening §2).

    Живая мотивация: `oxford-insights-gairi-2025` цитирует корпусный EU AI Act пять раз
    и ТОЛЬКО алиасом («the EU AI Act»), `me-undp-aila-2025` — GDPR только акронимом.
    Формального идентификатора в этих ссылках нет вовсе, поэтому регекс-майнер их не
    видит, а связь между документами реальна.
    """
    return _load_section(path, "aliases")


# Якорная форма CELEX в URL: `?uri=CELEX:52026DC0577` (и `%3A`-энкодинг двоеточия у
# ссылок, скопированных куратором из браузера). Здесь грамматика НАМЕРЕННО шире границы
# прозаического реестра (сектор 3 + LRDE + 4 цифры): префикс `CELEX:` снимает
# неоднозначность формы ПОЛНОСТЬЮ, гадать сектор/литеру/длину не нужно — а 79% живых
# CELEX это сектор 5 (`52026DC0577`, подготовительные акты), плюс литеры H/B/C/G/Y и
# номера в 5 и 8 цифр. Расширять так ПРОЗАИЧЕСКИЙ регекс запрещено: там за recall
# платят ложными рёбрами, а ложное ребро в юридическом графе дороже пропущенного.
# Хвосты: `-YYYYMMDD` — консолидированная версия, `R(NN)` — corrigendum; оба входят в
# идентификатор, иначе поправка резолвилась бы в базовый акт (неверная связь).
_URL_CELEX_ANCHOR = re.compile(
    r"CELEX(?::|%3A)(?P<celex>\d{5}[A-Z]{1,3}\d{2,8}(?:R\(\d{2}\))?(?:-\d{8})?)",
    re.IGNORECASE,
)


def identifiers_from_url(url: str) -> list[str]:
    """CELEX-идентификаторы, которые URL несёт БУКВАЛЬНО, по якорю ``CELEX:``.

    Проверено по коду коннектора (``eurlex._build_source_url``): он строит
    ``…/TXT/HTML/?uri=CELEX:{celex}`` без энкодинга, поэтому якорь покрывает все его
    допуски; ``%3A``-ветка нужна ссылкам, вставленным вручную из браузера.
    """
    return [f"CELEX:{m.group('celex').upper()}" for m in _URL_CELEX_ANCHOR.finditer(url)]


def identifiers_from_urls(records: list[schema.SourceRecord]) -> dict[str, str]:
    """Авто-резолюция из ``source_url`` — ТОЛЬКО для URL, несущих идентификатор БУКВАЛЬНО.

    Два канала: якорь ``CELEX:`` (``identifiers_from_url``, широкая грамматика) и общий
    реестр паттернов (компактный CELEX без префикса и прочие формы в теле ссылки).

    ⚠ Проверено на собственном корпусе: у ``eu-ai-act-2024`` ``source_url`` — OJ-форма
    (`uri=OJ:L_202401689`), из которой литера типа акта (R/L) НЕ выводится надёжно.
    Здесь эвристика честно ПАСУЕТ (ни один канал не совпадает) вместо того, чтобы
    угадать CELEX и связать документы неверно. Такие случаи закрывает справочник.

    Несколько записей на один идентификатор — идентификатор пропускается целиком:
    угадывать, какая из них «та самая», хуже, чем не построить ребро.
    """
    found: dict[str, set[str]] = {}
    for rec in records:
        idents = identifiers_from_url(rec.source_url)
        idents += [ident for ident, _rule in extract_identifiers(rec.source_url)]
        for ident in idents:
            found.setdefault(ident, set()).add(rec.id)
    resolved: dict[str, str] = {}
    for ident, doc_ids in found.items():
        if len(doc_ids) == 1:
            resolved[ident] = next(iter(doc_ids))
        else:
            logger.warning(
                "  ⚠ идентификатор %s встречается в source_url нескольких записей (%s) — пропущен",
                ident, ", ".join(sorted(doc_ids)),
            )
    return resolved


@dataclass(frozen=True)
class CiteEdge:
    """Ребро L1: ``source_id`` цитирует ``target_id`` (оба — doc-id корпуса)."""

    source_id: str
    target_id: str
    identifier: str
    rule: str


@dataclass
class MiningResult:
    edges: list[CiteEdge] = field(default_factory=list)
    leads: list[dict[str, Any]] = field(default_factory=list)   # нерезолвнутые -> отчёт
    dangling: list[str] = field(default_factory=list)           # протухшие записи справочника


def _resolution_map(
    records: list[schema.SourceRecord], identifiers: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Слить оба канала резолюции; вернуть ``(идентификатор -> doc-id, dangling)``.

    Курируемый справочник ПОБЕЖДАЕТ URL-эвристику: человек знает лучше регекса.
    Запись справочника, чей doc-id отсутствует в реестре, — предупреждение и строка
    отчёта, но НЕ краш сборки: справочник без гейта, протухшая запись не должна ронять
    граф всего корпуса.
    """
    known = {rec.id for rec in records}
    resolved = identifiers_from_urls(records)
    dangling: list[str] = []
    for ident, doc_id in identifiers.items():
        if doc_id not in known:
            dangling.append(f"{ident} -> {doc_id} (нет такой записи в реестре)")
            logger.warning("  ⚠ identifiers.yaml: %s ссылается на несуществующий %s", ident, doc_id)
            continue
        resolved[ident] = doc_id
    return resolved, sorted(dangling)


def mine_corpus(
    records: list[schema.SourceRecord],
    root: Path,
    identifiers: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
) -> MiningResult:
    """Промайнить ``doc.md`` всего корпуса на формальные цитаты (spec graph-v2 §3).

    Документ без ``doc.md`` (ещё не сконвертирован) просто пропускается — майнинг
    реконсиляционен, как всё остальное: появится текст — появятся рёбра.
    Самоцитирование (документ упомянул собственный идентификатор) ребром не становится.

    ``identifiers``/``aliases`` — обе секции курируемого справочника; ``None`` читает их
    с диска, явный словарь (в т.ч. пустой) отключает чтение, чтобы тест был герметичен.
    """
    resolved, dangling = _resolution_map(
        records, identifiers if identifiers is not None else load_identifiers()
    )
    alias_map = aliases if aliases is not None else load_aliases()
    edges: list[CiteEdge] = []
    unresolved: dict[str, dict[str, Any]] = {}

    for rec in records:
        md = schema.md_file(rec, root)
        if not md.exists():
            continue
        for ident, rule in extract_identifiers(md.read_text(encoding="utf-8"), alias_map):
            target = resolved.get(ident)
            if target is None:
                lead = unresolved.setdefault(
                    ident, {"identifier": ident, "rule": rule, "cited_by": []}
                )
                lead["cited_by"].append(rec.id)
                continue
            if target == rec.id:
                continue  # самоцитирование — не ребро
            edges.append(CiteEdge(rec.id, target, ident, rule))

    edges.sort(key=lambda e: (e.source_id, e.target_id, e.identifier))
    return MiningResult(edges, [unresolved[k] for k in sorted(unresolved)], dangling)


def leads_path(root: Path) -> Path:
    """Отчёт нерезолвнутых цитат: ``<root>/.state/cite_leads.yaml``."""
    return store.state_dir(root) / CITE_LEADS_FILENAME


def save_leads(leads: list[dict[str, Any]], root: Path) -> None:
    """Перезаписать отчёт целиком (производный артефакт, как и сам граф).

    Лиды — СЫРЬЁ для discovery (документ, на который ссылается корпус, но которого у нас
    нет), а НЕ кандидаты: единственная дверь в слой кандидатов остаётся ``inject`` с
    человеком/протоколом. Формат — запись на YAML-документ через пустую строку, тот же
    приём человекочитаемости, что у ``snowball_leads.yaml``.
    """
    parts = [yaml.safe_dump([lead], allow_unicode=True, sort_keys=False) for lead in leads]
    fsio.atomic_write_text(leads_path(root), "\n".join(parts))
