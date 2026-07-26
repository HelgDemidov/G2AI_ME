"""Построитель гетерогенного графа знаний G2AI-корпуса из курируемых ``meta.yaml`` (NetworkX).

Узлы разных типов (атрибут ``ntype``): ``document``, ``pattern`` (паттерн G2AI),
``topic`` (тема), ``issuer`` (издатель), ``jurisdiction`` (страна/блок).

Рёбра (атрибут ``etype``):
  - документ -> концепт: ``exemplifies`` (паттерн), ``about`` (тема),
    ``published_by`` (издатель), ``applies_to`` (юрисдикция) — «связь через общий
    узел-концепт», основа кросс-странового сравнения;
  - документ -> документ: типы из ``relations`` (references/implements/...) —
    родословная/цитирование;
  - страна -> блок: ``member_of`` (напр. sg -> asean) — из ``pipeline/vocab/jurisdictions.yaml``.

Экспорт GraphML (значения атрибутов — только скаляры, для Gephi/Cytoscape).
CLI печатает статистику и примеры запросов.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from core.schema import VOCAB_DIR, DEFAULT_SOURCES, GeoScope, RelationType, SourceRecord
from graph import cite_mining
from core.validate_sources import validate_sources

logger = logging.getLogger("build_graph")

JURISDICTIONS_PATH = VOCAB_DIR / "jurisdictions.yaml"


# --- идентификаторы узлов: префикс по типу, чтобы id разных типов не сталкивались ---
def _doc_node(rec_id: str) -> str:
    return f"doc:{rec_id}"


def _pattern_node(pattern: str) -> str:
    return f"pattern:{pattern}"


def _topic_node(topic: str) -> str:
    return f"topic:{topic}"


def _issuer_node(name: str) -> str:
    return f"issuer:{name}"


def _country_node(iso2: str) -> str:
    return f"country:{iso2}"


def _bloc_node(key: str) -> str:
    return f"bloc:{key}"


def load_jurisdictions(path: Path = JURISDICTIONS_PATH) -> dict[str, dict[str, Any]]:
    """``{bloc_key: {'label': str, 'members': set[str]}}`` из jurisdictions.yaml."""
    if not path.exists():
        return {}
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    blocs = data.get("blocs", {}) if isinstance(data, dict) else {}
    result: dict[str, dict[str, Any]] = {}
    for key, val in blocs.items():
        label = str(val.get("label", key))
        members = {str(m).lower() for m in val.get("members", [])}
        result[str(key)] = {"label": label, "members": members}
    return result


@dataclass(frozen=True)
class Validity:
    """Временнóй интервал действия документа — ВЫВЕДЕННЫЙ, не хранимый (spec graph-v2 §1).

    ``valid_to is None`` — открытый интервал (преемника нет). ``known=False`` — честная
    неопределённость: либо у документа нет ни одной даты начала, либо преемник есть, но
    сам без даты. Такой документ НИКОГДА не выпадает из среза молча — ``as_of`` включает
    его с пометкой (принцип «сомнение ⇒ проза» этого же проекта).
    """

    valid_from: _dt.date | None
    valid_to: _dt.date | None
    known: bool


def _valid_from(rec: SourceRecord) -> _dt.date | None:
    """Начало действия: ``dates.effective`` (для законов точнее) иначе ``dates.published``."""
    return rec.dates.effective or rec.dates.published


def validity_intervals(records: list[SourceRecord]) -> dict[str, Validity]:
    """Интервалы действия всех записей — чистая функция от реестра (spec graph-v2 §1).

    Правило: ребро ``supersedes``/``superseded_by`` (нормализуются к одному направлению
    «преемник →supersedes→ предшественник») закрывает интервал предшественника датой
    начала преемника. ``amends`` НЕ закрывает — поправка меняет документ, а не заменяет
    его. Несколько преемников → закрывает САМЫЙ РАННИЙ (+ warning: топология
    подозрительная, обычно это ошибка курирования, а не реальная развилка).

    Почему выводится, а не ведётся полями: ручная бухгалтерия действия закона на сотнях
    записей протухает по построению — тот же аргумент, что похоронил ``in_force``
    (spec post-acquisition-lifecycle §7). Пересборка графа бесплатна, поля — нет.
    """
    starts = {rec.id: _valid_from(rec) for rec in records}
    # set, не list: одна связь легально оформляется С ОБОИХ концов (supersedes у
    # преемника И superseded_by у предшественника — тот же паттерн, что нормализует
    # schema.superseded_ids). Без дедупа такое двустороннее оформление давало бы
    # ложный "несколько преемников (X, X)" на штатной, а не подозрительной топологии.
    successors: dict[str, set[str]] = {rec.id: set() for rec in records}
    for rec in records:
        for rel in rec.relations:
            if rel.type is RelationType.supersedes:
                successors.setdefault(rel.target, set()).add(rec.id)   # rec заменяет target
            elif rel.type is RelationType.superseded_by:
                successors.setdefault(rec.id, set()).add(rel.target)   # rec заменён target'ом

    intervals: dict[str, Validity] = {}
    for rec in records:
        heirs = successors.get(rec.id, set())
        heir_dates = sorted(d for d in (starts.get(h) for h in heirs) if d is not None)
        # Порог по числу ПРЕЕМНИКОВ, а не датированных дат: подозрительна сама топология
        # (обычно это ошибка курирования), и молчать о ней, пока у второго преемника нет
        # даты, значило бы прятать её ровно в самом мутном случае.
        if len(heirs) > 1:
            logger.warning(
                "  ⚠ %s: несколько преемников (%s) — интервал закрывает самый ранний из "
                "датированных (%s)",
                rec.id, ", ".join(sorted(heirs)),
                ", ".join(d.isoformat() for d in heir_dates) or "ни одного",
            )
        valid_to = heir_dates[0] if heir_dates else None
        start = starts[rec.id]
        # Преемник есть, но датировать закрытие нечем -> честное «не знаем», а не
        # молчаливый вывод «действует до сих пор».
        known = start is not None and (not heirs or valid_to is not None)
        if start is not None and valid_to is not None and valid_to < start:
            # Пустой интервал: документ исчезает из ВСЕХ as_of-срезов молча — ровно то,
            # что этот слой обещает не делать. Данные не «чиним» (угадывать, какая из
            # двух дат неверна, нечем), но кричим: почти всегда это ошибка курирования.
            logger.warning(
                "  ⚠ %s: преемник начинается (%s) РАНЬШЕ начала самого документа (%s) — "
                "интервал пуст, документ выпадет из любого среза as_of",
                rec.id, valid_to.isoformat(), start.isoformat(),
            )
        intervals[rec.id] = Validity(start, valid_to, known)
    return intervals


def _as_graphml_date(value: _dt.date | None) -> str:
    """GraphML не умеет ``None`` — неизвестная дата едет пустой строкой (writer падает
    на None, проверено; тот же приём, что у остальных скалярных атрибутов экспорта)."""
    return value.isoformat() if value is not None else ""


def as_of(graph: nx.MultiDiGraph, date: _dt.date) -> list[str]:
    """doc-узлы, действовавшие на ``date`` (spec graph-v2 §1).

    Границы: ``valid_from`` ВКЛЮЧИТЕЛЬНО (документ действует со дня вступления в силу),
    ``valid_to`` ИСКЛЮЧИТЕЛЬНО (в день начала преемника действует уже преемник) — иначе
    на стыке редакций обе версии оказались бы действующими одновременно.

    Документы с ``validity_known=False`` включаются ВСЕГДА: неизвестность — не повод
    молча удалить документ из среза, за который потом пишутся нормативные предложения.
    """
    selected: list[str] = []
    for node, data in graph.nodes(data=True):
        if data.get("ntype") != "document":
            continue
        if not data.get("validity_known", False):
            selected.append(str(node))
            continue
        start = data.get("valid_from") or ""
        end = data.get("valid_to") or ""
        iso = date.isoformat()
        if start and iso < start:
            continue
        if end and iso >= end:
            continue
        selected.append(str(node))
    return sorted(selected)


def build_graph(
    records: list[SourceRecord],
    jurisdictions: dict[str, dict[str, Any]] | None = None,
    cites: list[cite_mining.CiteEdge] | None = None,
) -> nx.MultiDiGraph:
    """Собрать гетерогенный ориентированный мультиграф из записей реестра.

    Чистая функция от ДАННЫХ: чтение ``doc.md`` ради L1-цитат живёт в
    ``cite_mining.mine_corpus`` — сюда его результат приходит готовым списком
    (``cites``). Каждое ребро несёт ``layer``: ``L0`` — курируемое (meta.yaml/vocab),
    ``L1`` — детерминированный автослой. Значение вводится сразу трёхзначным (L2 —
    будущая LLM-экстракция), чтобы она не потребовала миграции атрибутов.
    """
    jurisdictions = jurisdictions or {}
    graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def ensure(node_id: str, ntype: str, label: str, **attrs: str) -> None:
        if node_id not in graph:
            graph.add_node(node_id, ntype=ntype, label=label, **attrs)

    countries_seen: set[str] = set()
    intervals = validity_intervals(records)  # выведенная валидность (spec graph-v2 §1)

    # первый проход: документы и концепты
    for rec in records:
        doc = _doc_node(rec.id)
        validity = intervals[rec.id]
        ensure(
            doc,
            "document",
            rec.title,
            doc_type=rec.doc_type,
            authority=rec.authority,
            language=rec.language,
            entity=rec.entity_id,
            issuer=rec.issuer,
        )
        graph.nodes[doc]["valid_from"] = _as_graphml_date(validity.valid_from)
        graph.nodes[doc]["valid_to"] = _as_graphml_date(validity.valid_to)
        graph.nodes[doc]["validity_known"] = validity.known
        issuer = _issuer_node(rec.issuer)
        ensure(issuer, "issuer", rec.issuer, issuer_type=rec.issuer_type.value)
        graph.add_edge(doc, issuer, etype="published_by", layer="L0")

        for pattern in rec.g2ai_pattern:
            node = _pattern_node(pattern)
            ensure(node, "pattern", pattern)
            graph.add_edge(doc, node, etype="exemplifies", layer="L0")

        for topic in rec.topics:
            node = _topic_node(topic)
            ensure(node, "topic", topic)
            graph.add_edge(doc, node, etype="about", layer="L0")

        if rec.geo_scope is GeoScope.national:
            iso2 = rec.entity_id.lower()  # для наций entity_id == iso2 (даёт членство в блоках)
            countries_seen.add(iso2)
            node = _country_node(iso2)
            ensure(node, "jurisdiction", iso2.upper(), jlevel="country")
            graph.add_edge(doc, node, etype="applies_to", layer="L0")

    # member_of: страна -> блок (только для стран, встретившихся в корпусе)
    for key, info in jurisdictions.items():
        present = info["members"] & countries_seen
        if not present:
            continue
        bloc = _bloc_node(key)
        ensure(bloc, "jurisdiction", info["label"], jlevel="bloc")
        for iso2 in sorted(present):
            graph.add_edge(_country_node(iso2), bloc, etype="member_of", layer="L0")

    # второй проход: документ -> документ (relations; все doc-узлы уже созданы)
    for rec in records:
        doc = _doc_node(rec.id)
        for rel in rec.relations:
            graph.add_edge(doc, _doc_node(rel.target), etype=rel.type.value, layer="L0")

    # третий проход: L1 — детерминированные цитаты (spec graph-v2 §3). Отличимы от
    # курируемых по построению (layer/rule), а не по договорённости.
    for edge in cites or []:
        graph.add_edge(
            _doc_node(edge.source_id), _doc_node(edge.target_id),
            etype="cites", layer="L1", rule=edge.rule, identifier=edge.identifier,
        )

    return graph


def build_corpus_graph(
    records: list[SourceRecord], root: Path, *, write_leads: bool = True
) -> tuple[nx.MultiDiGraph, cite_mining.MiningResult]:
    """Граф корпуса «под ключ»: L0 из реестра + L1 из майнинга ``doc.md``.

    Единственная точка сшивания для ОБОИХ потребителей (CLI этого модуля и
    ``run_pipeline --graphml``) — иначе экспорт из оркестратора молча остался бы без
    L1-слоя, и два «одинаковых» графа расходились бы содержанием.

    ``write_leads=False`` — для вызывающих сторон, которым нельзя писать на диск
    (dry-run/тесты): майнинг чистый, побочный эффект ровно один и он отключаем.
    """
    mining = cite_mining.mine_corpus(records, root)
    graph = build_graph(records, load_jurisdictions(), cites=mining.edges)
    if write_leads:
        cite_mining.save_leads(mining.leads, root)
    return graph, mining


def docs_by_pattern(graph: nx.MultiDiGraph, pattern: str) -> list[str]:
    """Документы, демонстрирующие данный G2AI-паттерн (кросс-страновой кластер)."""
    node = _pattern_node(pattern)
    if node not in graph:
        return []
    return sorted(u for u, _, d in graph.in_edges(node, data=True) if d.get("etype") == "exemplifies")


def docs_in_bloc(graph: nx.MultiDiGraph, bloc_key: str) -> list[str]:
    """Документы стран — членов блока (напр. все документы стран ЕС)."""
    bloc = _bloc_node(bloc_key)
    if bloc not in graph:
        return []
    countries = [u for u, _, d in graph.in_edges(bloc, data=True) if d.get("etype") == "member_of"]
    docs: set[str] = set()
    for country in countries:
        docs.update(
            u for u, _, d in graph.in_edges(country, data=True) if d.get("etype") == "applies_to"
        )
    return sorted(docs)


def lineage(graph: nx.MultiDiGraph, doc_id: str, etype: str = "implements") -> list[str]:
    """Исходящие документ->документ связи данного типа (родословная)."""
    doc = _doc_node(doc_id)
    if doc not in graph:
        return []
    return sorted(v for _, v, d in graph.out_edges(doc, data=True) if d.get("etype") == etype)


def export_graphml(graph: nx.MultiDiGraph, path: Path) -> None:
    """Экспорт в GraphML (для Gephi/Cytoscape)."""
    nx.write_graphml(graph, path)


def summary(graph: nx.MultiDiGraph) -> str:
    """Текстовая сводка: узлы по типам, рёбра по типам."""
    ntypes: Counter[str] = Counter(str(d.get("ntype")) for _, d in graph.nodes(data=True))
    etypes: Counter[str] = Counter(str(d.get("etype")) for _, _, d in graph.edges(data=True))
    lines = [
        f"Узлов: {graph.number_of_nodes()}, рёбер: {graph.number_of_edges()}",
        "  по типам узлов:  " + ", ".join(f"{k}={v}" for k, v in sorted(ntypes.items())),
        "  по типам рёбер:  " + ", ".join(f"{k}={v}" for k, v in sorted(etypes.items())),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построитель графа знаний G2AI-корпуса")
    parser.add_argument("sources", nargs="?", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--graphml", type=Path, default=None, help="экспортировать граф в GraphML")
    parser.add_argument(
        "--as-of", dest="as_of", type=_dt.date.fromisoformat, default=None, metavar="YYYY-MM-DD",
        help="временнóй срез: какие документы действовали на эту дату (валидность выводится "
             "из рёбер supersedes, не хранится полем)",
    )
    args = parser.parse_args(argv)

    sources_path: Path = args.sources

    errors, records = validate_sources(sources_path)
    if errors:
        print("реестр невалиден — сначала исправьте (validate_sources.py):", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    graph, mining = build_corpus_graph(records, sources_path)
    print(summary(graph))
    if mining.edges:
        print(f"L1-цитат: {len(mining.edges)} (детерминированный автослой, ноль LLM)")
    if mining.leads:
        print(
            f"Цитаты без записи в корпусе: {len(mining.leads)} -> "
            f"{cite_mining.leads_path(sources_path)} (сырьё для discovery, НЕ кандидаты)"
        )
    for entry in mining.dangling:
        print(f"  ⚠ identifiers.yaml: {entry}")

    if args.as_of is not None:
        # Срез — самостоятельный режим вывода: печатаем ЕГО и выходим, чтобы примеры
        # запросов ниже (полный граф) не смешивались со срезом в одном выводе.
        docs = as_of(graph, args.as_of)
        print(f"\nДействовали на {args.as_of.isoformat()} ({len(docs)} из {len(records)}):")
        for node in docs:
            data = graph.nodes[node]
            mark = "" if data.get("validity_known") else "  ⚠ валидность неизвестна"
            span = f"{data.get('valid_from') or '?'} → {data.get('valid_to') or '…'}"
            print(f"  {node[len('doc:'):]}  [{span}]{mark}")
        if args.graphml is not None:
            export_graphml(graph.subgraph(docs).copy(), args.graphml)
            print(f"\nGraphML среза записан: {args.graphml}")
        return 0

    # примеры запросов
    patterns = sorted(n[len("pattern:"):] for n, d in graph.nodes(data=True) if d.get("ntype") == "pattern")
    if patterns:
        first = patterns[0]
        print(f"\nДокументы с паттерном '{first}':")
        for doc in docs_by_pattern(graph, first):
            print(f"  {doc}")

    blocs = sorted(n[len("bloc:"):] for n, d in graph.nodes(data=True) if d.get("jlevel") == "bloc")
    for bloc in blocs:
        docs = docs_in_bloc(graph, bloc)
        if docs:
            print(f"\nДокументы стран блока '{bloc}': {', '.join(docs)}")

    if args.graphml is not None:
        export_graphml(graph, args.graphml)
        print(f"\nGraphML записан: {args.graphml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
