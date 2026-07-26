"""Тесты построителя гетерогенного графа знаний."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from graph.build_graph import (
    build_graph,
    docs_by_pattern,
    docs_in_bloc,
    export_graphml,
    lineage,
    load_jurisdictions,
    main,
    summary,
)
from core.schema import SourceRecord
from tests.support import valid_record

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
    """docs_by_pattern возвращает bare doc-id, без doc:-префикса (knowledge-hardening §3)."""
    a = make()
    b = make(id="eu-ec-ai-act-2024", entity_id="de")
    graph = build_graph([a, b], JUR)
    docs = docs_by_pattern(graph, "agent-governance-framework")
    assert set(docs) == {"sg-imda-mgf-agentic-2026", "eu-ec-ai-act-2024"}


def test_member_of_only_for_present_countries() -> None:
    graph = build_graph([make()], JUR)  # только sg присутствует
    assert graph.has_edge("country:sg", "bloc:asean")
    assert "country:th" not in graph  # th — член asean, но в корпусе не встречался


def test_docs_in_bloc() -> None:
    """bare doc-id (knowledge-hardening §3)."""
    graph = build_graph([make()], JUR)
    assert docs_in_bloc(graph, "asean") == ["sg-imda-mgf-agentic-2026"]


def test_relations_lineage() -> None:
    """bare doc-id (knowledge-hardening §3)."""
    a = make(relations=[{"type": "implements", "target": "eu-ec-ai-act-2024"}])
    b = make(id="eu-ec-ai-act-2024")
    graph = build_graph([a, b], JUR)
    assert lineage(graph, "sg-imda-mgf-agentic-2026", "implements") == ["eu-ec-ai-act-2024"]


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


@pytest.mark.parametrize(
    "content",
    [None, "- список, а не отображение\n", "", "blocs: 42\n", "blocs:\n  asean: [1, 2]\n"],
)
def test_missing_or_malformed_jurisdictions_degrades_to_empty(
    tmp_path: Path, content: str | None
) -> None:
    """Справочник БЕЗ гейта (симметрия с identifiers.yaml, knowledge-hardening §4):
    отсутствие файла и любая порча формата обязаны дать пустой словарь, а не
    уронить сборку графа всего корпуса."""
    path = tmp_path / "jurisdictions.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert load_jurisdictions(path) == {}


def test_corrupted_bloc_entry_is_skipped_not_fatal(tmp_path: Path, caplog: Any) -> None:
    """Один повреждённый блок (не словарь) не должен ронять остальные блоки."""
    path = tmp_path / "jurisdictions.yaml"
    path.write_text(
        "blocs:\n  broken: not-a-mapping\n  asean:\n    label: ASEAN\n    members: [sg]\n",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="build_graph"):
        blocs = load_jurisdictions(path)
    assert "broken" not in blocs
    assert blocs["asean"]["members"] == {"sg"}
    assert "broken" in caplog.text


def test_main_nonexistent_root_is_empty_valid_graph(tmp_path: Path, capsys: Any) -> None:
    """Несуществующий корень — валидный пустой корпус (та же семантика, что у
    validate_sources/run_pipeline), а не «файл не найден» (лексика эпохи sources.yaml)."""
    missing = tmp_path / "does-not-exist"
    exit_code = main([str(missing)])
    assert exit_code == 0
    assert "Узлов: 0" in capsys.readouterr().out
