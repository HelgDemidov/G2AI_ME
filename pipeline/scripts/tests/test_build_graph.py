"""Тесты построителя гетерогенного графа знаний."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from build_graph import (
    build_graph,
    docs_by_pattern,
    docs_in_bloc,
    export_graphml,
    lineage,
    load_jurisdictions,
    summary,
)
from schema import SourceRecord
from test_schema import valid_record

JUR: dict[str, dict[str, Any]] = {"asean": {"label": "ASEAN", "members": {"sg", "th"}}}


def make(**over: Any) -> SourceRecord:
    data = valid_record()
    data.update(over)
    return SourceRecord.model_validate(data)


def test_node_types_present() -> None:
    graph = build_graph([make()], JUR)
    assert graph.nodes["doc:sg-imda-mgf-agentic-2026"]["ntype"] == "document"
    assert "pattern:agent-governance-framework" in graph
    assert "topic:ai-governance" in graph
    assert "issuer:Infocomm Media Development Authority (IMDA)" in graph
    assert graph.nodes["country:sg"]["jlevel"] == "country"
    assert graph.nodes["bloc:asean"]["jlevel"] == "bloc"


def test_doc_concept_edges() -> None:
    graph = build_graph([make()], JUR)
    etypes = {d["etype"] for _, _, d in graph.out_edges("doc:sg-imda-mgf-agentic-2026", data=True)}
    assert {"exemplifies", "about", "published_by", "applies_to"} <= etypes


def test_shared_pattern_cluster() -> None:
    a = make()
    b = make(id="eu-ec-ai-act-2024", country="Germany", country_iso2="de")
    graph = build_graph([a, b], JUR)
    docs = docs_by_pattern(graph, "agent-governance-framework")
    assert set(docs) == {"doc:sg-imda-mgf-agentic-2026", "doc:eu-ec-ai-act-2024"}


def test_member_of_only_for_present_countries() -> None:
    graph = build_graph([make()], JUR)  # только sg присутствует
    assert graph.has_edge("country:sg", "bloc:asean")
    assert "country:th" not in graph  # th — член asean, но в корпусе не встречался


def test_docs_in_bloc() -> None:
    graph = build_graph([make()], JUR)
    assert docs_in_bloc(graph, "asean") == ["doc:sg-imda-mgf-agentic-2026"]


def test_relations_lineage() -> None:
    a = make(relations=[{"type": "implements", "target": "eu-ec-ai-act-2024"}])
    b = make(id="eu-ec-ai-act-2024")
    graph = build_graph([a, b], JUR)
    assert lineage(graph, "sg-imda-mgf-agentic-2026", "implements") == ["doc:eu-ec-ai-act-2024"]


def test_missing_pattern_returns_empty() -> None:
    graph = build_graph([make()], JUR)
    assert docs_by_pattern(graph, "no-such-pattern") == []


def test_graphml_roundtrip(tmp_path: Path) -> None:
    graph = build_graph([make()], JUR)
    path = tmp_path / "graph.graphml"
    export_graphml(graph, path)
    reloaded = nx.read_graphml(path)
    assert reloaded.number_of_nodes() == graph.number_of_nodes()
    assert reloaded.number_of_edges() == graph.number_of_edges()


def test_summary_mentions_counts() -> None:
    graph = build_graph([make()], JUR)
    text = summary(graph)
    assert "Узлов" in text and "document=1" in text


def test_load_real_jurisdictions() -> None:
    blocs = load_jurisdictions()
    assert "eu" in blocs and "asean" in blocs
    assert "sg" in blocs["asean"]["members"]
