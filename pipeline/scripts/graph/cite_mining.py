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
    """

    name: str
    regex: re.Pattern[str]
    canonical: Callable[[re.Match[str]], str]


def _celex(m: re.Match[str]) -> str:
    return f"CELEX:{m.group(1).upper()}"


def _sluzbeni(m: re.Match[str]) -> str:
    return f"SLCG:{int(m.group(1))}/{m.group(2)}"


def _iso(m: re.Match[str]) -> str:
    # Только именованные группы: смешение с позиционными уже давало сдвиг нумерации
    # (необязательная `/IEC` сама по себе — группа №1).
    ident = f"ISO/IEC {m.group('body')}" if m.group("iec") else f"ISO {m.group('body')}"
    if m.group("part"):
        ident += f"-{m.group('part')}"
    if m.group("year"):
        ident += f":{m.group('year')}"
    return ident


def _nist(m: re.Match[str]) -> str:
    return f"NIST SP 800-{m.group('sp')}" if m.group("sp") else f"NIST AI {m.group('ai')}"


_PATTERNS: dict[str, CitePattern] = {
    # Совпадает и в голом виде, и внутри ELI/URL-форм (`...uri=CELEX:32024R1689`) —
    # отдельный URL-паттерн не нужен.
    "celex": CitePattern("celex", re.compile(r"\b(3\d{4}[LRDE]\d{4})\b"), _celex),
    "sluzbeni_list_cg": CitePattern(
        "sluzbeni_list_cg",
        re.compile(
            r"(?:Службени\s+лист\s+ЦГ|Slu[žz]beni\s+list\s+CG)\s*,?\s*"
            r"(?:бр|br)\.?\s*(\d+)\s*/\s*(\d{2,4})",
            re.IGNORECASE,
        ),
        _sluzbeni,
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


def extract_identifiers(text: str) -> list[tuple[str, str]]:
    """``(канонический идентификатор, имя правила)`` — все формальные цитаты текста.

    Порядок детерминирован (правила по имени, вхождения по позиции), дубли схлопнуты:
    один и тот же акт, упомянутый в документе десять раз, — одно ребро, не десять.
    """
    seen: dict[str, str] = {}
    for name in sorted(_PATTERNS):
        pattern = _PATTERNS[name]
        for match in pattern.regex.finditer(text):
            seen.setdefault(pattern.canonical(match), name)
    return sorted(seen.items())


def load_identifiers(path: Path = IDENTIFIERS_PATH) -> dict[str, str]:
    """Курируемый справочник ``идентификатор -> doc-id``. Отсутствует — пустой (не ошибка:
    справочник наполняется по мере встречи идентификаторов, как ``jurisdictions.yaml``)."""
    if not path.exists():
        return {}
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    raw = data.get("identifiers") or {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def identifiers_from_urls(records: list[schema.SourceRecord]) -> dict[str, str]:
    """Авто-резолюция из ``source_url`` — ТОЛЬКО для URL, несущих идентификатор БУКВАЛЬНО.

    ⚠ Проверено на собственном корпусе: у ``eu-ai-act-2024`` ``source_url`` — OJ-форма
    (`uri=OJ:L_202401689`), из которой литера типа акта (R/L) НЕ выводится надёжно.
    Здесь эвристика честно ПАСУЕТ (регекс не совпадает) вместо того, чтобы угадать
    CELEX и связать документы неверно. Такие случаи закрывает справочник.

    Несколько записей на один идентификатор — идентификатор пропускается целиком:
    угадывать, какая из них «та самая», хуже, чем не построить ребро.
    """
    found: dict[str, set[str]] = {}
    for rec in records:
        for ident, _rule in extract_identifiers(rec.source_url):
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
) -> MiningResult:
    """Промайнить ``doc.md`` всего корпуса на формальные цитаты (spec graph-v2 §3).

    Документ без ``doc.md`` (ещё не сконвертирован) просто пропускается — майнинг
    реконсиляционен, как всё остальное: появится текст — появятся рёбра.
    Самоцитирование (документ упомянул собственный идентификатор) ребром не становится.
    """
    resolved, dangling = _resolution_map(
        records, identifiers if identifiers is not None else load_identifiers()
    )
    edges: list[CiteEdge] = []
    unresolved: dict[str, dict[str, Any]] = {}

    for rec in records:
        md = schema.md_file(rec, root)
        if not md.exists():
            continue
        for ident, rule in extract_identifiers(md.read_text(encoding="utf-8")):
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
