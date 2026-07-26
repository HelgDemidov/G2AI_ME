"""Примитивы экспансии для evidence-слоя — spec graph-v2 §6.

Контракт с будущим `analyze-evidence`: вход — doc-id из retrieval, выход — соседи с
типами рёбер и layer-тегами. Тесты держат именно контракт (что видно потребителю), а не
внутренности обхода.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from core.schema import SourceRecord
from graph.build_graph import build_graph
from graph.expand import Neighbor, expand, personalized_rank
from tests.support import valid_record


def _rec(doc_id: str, *, relations: list[dict[str, str]] | None = None,
         topics: list[str] | None = None, issuer: str | None = None) -> SourceRecord:
    data = valid_record()
    data["id"] = doc_id
    data["relations"] = relations or []
    if topics is not None:
        data["topics"] = topics
    if issuer is not None:
        data["issuer"] = issuer
    return SourceRecord.model_validate(data)


def _chain() -> nx.MultiDiGraph:
    """A --implements--> B --cites--> C (C цитирует ещё и D через L1)."""
    return build_graph([
        _rec("a-doc-2026", relations=[{"type": "implements", "target": "b-doc-2025"}]),
        _rec("b-doc-2025", relations=[{"type": "cites", "target": "c-doc-2024"}]),
        _rec("c-doc-2024"),
    ])


# --- expand ---


def test_expand_returns_direct_typed_neighbours() -> None:
    result = expand(_chain(), ["a-doc-2026"])
    assert result == [Neighbor("b-doc-2025", "implements", "L0", "out", 1)]


def test_expand_is_undirected_but_keeps_direction() -> None:
    """Цитирующий и цитируемый одинаково интересны досье, но «А цитирует Б» и «Б
    цитируется А» — разные утверждения, терять направление нельзя."""
    result = expand(_chain(), ["b-doc-2025"])
    assert {(n.doc_id, n.direction) for n in result} == {
        ("c-doc-2024", "out"), ("a-doc-2026", "in"),
    }


def test_expand_two_hops_records_distance() -> None:
    result = expand(_chain(), ["a-doc-2026"], hops=2)
    assert [(n.doc_id, n.hops) for n in result] == [("b-doc-2025", 1), ("c-doc-2024", 2)]


def test_expand_prefers_nearest_link() -> None:
    """Документ, достижимый и за 1, и за 2 шага, — сосед 1-шаговый."""
    graph = build_graph([
        _rec("a-doc-2026", relations=[
            {"type": "cites", "target": "b-doc-2025"},
            {"type": "cites", "target": "c-doc-2024"},
        ]),
        _rec("b-doc-2025", relations=[{"type": "cites", "target": "c-doc-2024"}]),
        _rec("c-doc-2024"),
    ])
    hops = {n.doc_id: n.hops for n in expand(graph, ["a-doc-2026"], hops=2)}
    assert hops == {"b-doc-2025": 1, "c-doc-2024": 1}


def test_expand_excludes_seeds() -> None:
    """Seed'ы уже в паке — дублировать их в расширении незачем."""
    assert all(n.doc_id != "a-doc-2026" for n in expand(_chain(), ["a-doc-2026"], hops=3))


def test_expand_does_not_traverse_concept_nodes() -> None:
    """У каждого документа есть тема и издатель — соседство «через topic» мгновенно
    выродилось бы в полкорпуса, сигнала там нет."""
    graph = build_graph([_rec("a-doc-2026"), _rec("b-doc-2025")])  # общие topics/issuer
    assert expand(graph, ["a-doc-2026"]) == []


def test_expand_filters_by_etype() -> None:
    result = expand(_chain(), ["b-doc-2025"], etypes={"cites"})
    assert [n.doc_id for n in result] == ["c-doc-2024"]


def test_expand_exposes_layer_of_each_link() -> None:
    """Потребитель обязан видеть, курируемая связь или автоматическая."""
    from graph.cite_mining import CiteEdge

    graph = build_graph(
        [_rec("a-doc-2026"), _rec("b-doc-2025")],
        cites=[CiteEdge("a-doc-2026", "b-doc-2025", "CELEX:32024R1689", "celex")],
    )
    assert [(n.doc_id, n.etype, n.layer) for n in expand(graph, ["a-doc-2026"])] == [
        ("b-doc-2025", "cites", "L1")
    ]


def test_expand_ignores_unknown_seed() -> None:
    assert expand(_chain(), ["no-such-doc"]) == []


# --- personalized PageRank ---


def test_ppr_ranks_connected_documents_above_isolated() -> None:
    graph = build_graph([
        _rec("a-doc-2026", relations=[{"type": "cites", "target": "b-doc-2025"}],
             topics=["ai-governance"], issuer="A"),
        _rec("b-doc-2025", topics=["ai-governance"], issuer="B"),
        _rec("z-doc-2020", topics=["procurement"], issuer="Z"),
    ])
    ranked = personalized_rank(graph, ["a-doc-2026"])
    ids = [r.doc_id for r in ranked]
    assert ids.index("b-doc-2025") < ids.index("z-doc-2020")


def test_ppr_uses_concepts_as_connective_tissue() -> None:
    """Именно то, чего не умеет expand: общая тема даёт мягкий сигнал близости там,
    где жёсткого ребра документ→документ нет вовсе."""
    graph = build_graph([
        _rec("a-doc-2026", topics=["ai-governance"], issuer="A"),
        _rec("b-doc-2025", topics=["ai-governance"], issuer="B"),
        _rec("z-doc-2020", topics=["procurement"], issuer="Z"),
    ])
    assert expand(graph, ["a-doc-2026"]) == []           # жёстких рёбер нет
    ranked = personalized_rank(graph, ["a-doc-2026"])
    ids = [r.doc_id for r in ranked]
    assert ids.index("b-doc-2025") < ids.index("z-doc-2020")


def test_ppr_excludes_seeds_and_respects_k() -> None:
    graph = _chain()
    ranked = personalized_rank(graph, ["a-doc-2026"], k=1)
    assert len(ranked) == 1 and ranked[0].doc_id != "a-doc-2026"


def test_ppr_returns_only_documents() -> None:
    ranked = personalized_rank(_chain(), ["a-doc-2026"], k=50)
    assert ranked and all(not r.doc_id.startswith(("topic:", "issuer:")) for r in ranked)


def test_ppr_without_valid_seed_is_empty_not_uniform() -> None:
    """Равномерный PageRank по всему корпусу был бы шумом, выданным за релевантность."""
    assert personalized_rank(_chain(), ["no-such-doc"]) == []


def test_ppr_is_deterministic() -> None:
    graph = _chain()
    assert personalized_rank(graph, ["a-doc-2026"]) == personalized_rank(graph, ["a-doc-2026"])


def test_ppr_on_empty_graph_is_empty() -> None:
    assert personalized_rank(nx.MultiDiGraph(), ["a-doc-2026"]) == []


# --- математика собственной степенной итерации (nx.pagerank требует scipy, которого нет) ---


def _all_scores(graph: nx.MultiDiGraph, seeds: list[str]) -> dict[str, float]:
    from graph.expand import DOC_PREFIX, _power_iteration, _undirected_projection

    projection = _undirected_projection(graph)
    seed_nodes = {DOC_PREFIX + s for s in seeds}
    share = 1.0 / len(seed_nodes)
    p = {n: (share if n in seed_nodes else 0.0) for n in projection.nodes()}
    return _power_iteration(projection, p, 0.85)


def test_ppr_conserves_mass() -> None:
    """Ранги обязаны оставаться распределением (сумма ≈ 1): утечка массы — самый
    типичный дефект самописной степенной итерации, и на ранжировании он виден не сразу."""
    scores = _all_scores(_chain(), ["a-doc-2026"])
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_ppr_conserves_mass_with_isolated_node() -> None:
    """Висячий узел (без рёбер) не должен уносить массу из системы — его доля
    возвращается по вектору персонализации."""
    graph = _chain()
    graph.add_node("doc:lonely-2020", ntype="document", label="lonely")
    scores = _all_scores(graph, ["a-doc-2026"])
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_ppr_seed_keeps_highest_mass() -> None:
    """Персонализация работает: масса концентрируется у seed'а, а не размазывается."""
    scores = _all_scores(_chain(), ["a-doc-2026"])
    assert max(scores, key=lambda n: scores[n]) == "doc:a-doc-2026"


def test_ppr_two_seeds_split_personalization() -> None:
    """Два seed'а — симметричный вклад: иначе первый из списка тихо доминировал бы."""
    graph = build_graph([_rec("a-doc-2026"), _rec("b-doc-2025"), _rec("z-doc-2020")])
    scores = _all_scores(graph, ["a-doc-2026", "b-doc-2025"])
    assert scores["doc:a-doc-2026"] == pytest.approx(scores["doc:b-doc-2025"], rel=1e-6)


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_expand_never_returns_duplicates(hops: int) -> None:
    result = expand(_chain(), ["a-doc-2026"], hops=hops)
    assert len({n.doc_id for n in result}) == len(result)


def test_expand_multiple_seeds(_unused: Any = None) -> None:
    result = expand(_chain(), ["a-doc-2026", "c-doc-2024"])
    assert {n.doc_id for n in result} == {"b-doc-2025"}  # общий сосед, seed'ы исключены
