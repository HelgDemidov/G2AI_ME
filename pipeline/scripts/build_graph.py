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
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from schema import GeoScope, SourceRecord, load_records
from validate_sources import DEFAULT_SOURCES, validate_sources

JURISDICTIONS_PATH = Path(__file__).resolve().parent.parent / "vocab" / "jurisdictions.yaml"


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


def build_graph(
    records: list[SourceRecord],
    jurisdictions: dict[str, dict[str, Any]] | None = None,
) -> nx.MultiDiGraph:
    """Собрать гетерогенный ориентированный мультиграф из записей реестра."""
    jurisdictions = jurisdictions or {}
    graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def ensure(node_id: str, ntype: str, label: str, **attrs: str) -> None:
        if node_id not in graph:
            graph.add_node(node_id, ntype=ntype, label=label, **attrs)

    countries_seen: set[str] = set()

    # первый проход: документы и концепты
    for rec in records:
        doc = _doc_node(rec.id)
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
        issuer = _issuer_node(rec.issuer)
        ensure(issuer, "issuer", rec.issuer, issuer_type=rec.issuer_type.value)
        graph.add_edge(doc, issuer, etype="published_by")

        for pattern in rec.g2ai_pattern:
            node = _pattern_node(pattern)
            ensure(node, "pattern", pattern)
            graph.add_edge(doc, node, etype="exemplifies")

        for topic in rec.topics:
            node = _topic_node(topic)
            ensure(node, "topic", topic)
            graph.add_edge(doc, node, etype="about")

        if rec.geo_scope is GeoScope.national:
            iso2 = rec.entity_id.lower()  # для наций entity_id == iso2 (даёт членство в блоках)
            countries_seen.add(iso2)
            node = _country_node(iso2)
            ensure(node, "jurisdiction", iso2.upper(), jlevel="country")
            graph.add_edge(doc, node, etype="applies_to")

    # member_of: страна -> блок (только для стран, встретившихся в корпусе)
    for key, info in jurisdictions.items():
        present = info["members"] & countries_seen
        if not present:
            continue
        bloc = _bloc_node(key)
        ensure(bloc, "jurisdiction", info["label"], jlevel="bloc")
        for iso2 in sorted(present):
            graph.add_edge(_country_node(iso2), bloc, etype="member_of")

    # второй проход: документ -> документ (relations; все doc-узлы уже созданы)
    for rec in records:
        doc = _doc_node(rec.id)
        for rel in rec.relations:
            graph.add_edge(doc, _doc_node(rel.target), etype=rel.type.value)

    return graph


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
    args = parser.parse_args(argv)

    sources_path: Path = args.sources
    if not sources_path.exists():
        print(f"файл не найден: {sources_path}", file=sys.stderr)
        return 2

    errors = validate_sources(sources_path)
    if errors:
        print("реестр невалиден — сначала исправьте (validate_sources.py):", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    graph = build_graph(load_records(sources_path), load_jurisdictions())
    print(summary(graph))

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
