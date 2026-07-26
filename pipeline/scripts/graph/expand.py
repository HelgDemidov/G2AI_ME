"""Примитивы графовой экспансии для evidence-слоя — spec graph-v2 §6.

``retrieve()`` остаётся ПЕРВИЧНЫМ каналом: граф не ищет, он расширяет уже найденное.
Контракт с будущим ``analyze-evidence``: на вход doc-id из retrieval, на выход —
ранжированные соседи с типами рёбер и layer-тегами (чтобы пакет доказательств мог
показать, ПОЧЕМУ документ приложен, а не только что он релевантен).

Два примитива намеренно разной природы, они дополняют друг друга:

* ``expand`` — точное типизированное соседство ТОЛЬКО по рёбрам документ→документ.
  Объяснимо построчно («цитирует», «реализует»), поэтому годится в текст досье.
  Через концепт-узлы не ходит: у каждого документа есть тема и издатель, и соседство
  «через topic» мгновенно вырождается в «полкорпуса» — сигнала там нет.
* ``personalized_rank`` — PPR по ВСЕМУ графу, где концепт-узлы работают связующей
  тканью: общая тема/паттерн/издатель дают мягкий сигнал близости, которого нет
  жёстким ребром. Ровно то, чего не умеет ``expand``, и ровно поэтому не заменяет его.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx

from graph.build_graph import DOC_PREFIX as DOC_PREFIX  # реэкспорт: единств. определение (§3)

PPR_ALPHA = 0.85     # стандартный damping PageRank; тюнинг — только по данным evidence-паков
PPR_MAX_ITER = 100
# Порог остановки по L1-невязке. Калиброван замером (2026-07-26, синтетика 1000 док):
# 1e-9 -> 458 мс, 1e-6 -> 335 мс, 1e-4 -> 201 мс, при ИДЕНТИЧНОМ top-20 во всех трёх.
# Нам нужен ПОРЯДОК соседей, а не точные веса: гнать итерации до девятого знака —
# платить вдвое за цифры, которые никто не читает.
PPR_TOL = 1e-4


def _doc_id(node: str) -> str:
    return node[len(DOC_PREFIX):] if node.startswith(DOC_PREFIX) else node


def _is_document(graph: nx.MultiDiGraph, node: str) -> bool:
    return bool(graph.nodes[node].get("ntype") == "document")


@dataclass(frozen=True)
class Link:
    """Одно ребро между парой документов: чем связаны, каким слоем, в какую сторону."""

    etype: str
    layer: str
    direction: str  # "out" — исходящее ребро от seed'а, "in" — входящее


@dataclass(frozen=True)
class Neighbor:
    """Сосед документа: ВСЕ связи с ним на ближайшем расстоянии.

    ``links`` — множественное не для красоты: пара документов законно связана
    несколькими рёбрами сразу (новая редакция и ``supersedes`` предшественника из
    курируемого L0, и ``cites`` его газетного номера из L1). Контракт с
    ``analyze-evidence`` — «показать, ПОЧЕМУ документ приложен»; одна причина из двух
    сделала бы досье тихо неполным, поэтому здесь список, а не скаляр.

    ⚠ Честная граница (knowledge-hardening §3): для ``hops == 1`` контракт полон —
    ``links`` несёт ВСЕ рёбра между seed'ом и соседом. Для ``hops >= 2`` он описывает
    только рёбра ПОСЛЕДНЕГО шага пути к этому соседу — промежуточный узел (и его
    собственные связи с обоих концов) в ``links`` не восстановим. «Почему приложен»
    в этом случае объясняет только последний сегмент цепочки, не её целиком; захват
    полного пути — решение спека analyze-evidence, не этого модуля.
    """

    doc_id: str
    links: tuple[Link, ...]
    hops: int

    @property
    def etypes(self) -> tuple[str, ...]:
        """Типы всех связей — частый случай «просто перечислить, чем связаны»."""
        return tuple(link.etype for link in self.links)


def expand(
    graph: nx.MultiDiGraph,
    doc_ids: list[str],
    hops: int = 1,
    etypes: set[str] | None = None,
) -> list[Neighbor]:
    """Типизированное соседство документов на расстоянии ``hops`` (spec graph-v2 §6).

    Обход НЕНАПРАВЛЕННЫЙ по смыслу (цитирующий и цитируемый одинаково интересны
    досье), но направление каждого ребра сохраняется в результате — «А цитирует Б» и
    «Б цитируется А» это разные утверждения, и терять их нельзя.

    Возвращаются только документы, которых нет среди ``doc_ids`` (seed'ы — уже
    найденное retrieval'ом, дублировать их в расширении незачем). Ближайшая связь
    выигрывает: документ, достижимый и за 1, и за 2 шага, попадает как 1-шаговый — но
    на своём ближайшем расстоянии он несёт ВСЕ параллельные рёбра пары, а не первое
    попавшееся (spec graph-hardening §5).

    ⚠ ``etypes`` гейтит ОБХОД, а не только выдачу: сосед за неподходящим первым ребром
    недостижим и на втором шаге. Семантика намеренная — «путь, целиком собранный из
    рёбер этих типов», — но неочевидная, поэтому названа здесь явно.
    """
    seeds = [DOC_PREFIX + d for d in doc_ids if DOC_PREFIX + d in graph]
    seed_set = set(seeds)
    seen: set[str] = set(seeds)
    links: dict[str, set[Link]] = {}
    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)

    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        edges = [(v, d, "out") for _, v, d in graph.out_edges(node, data=True)]
        edges += [(u, d, "in") for u, _, d in graph.in_edges(node, data=True)]
        for other, data, direction in edges:
            if not _is_document(graph, other):
                continue  # концепт-узлы соседство вырождают — см. docstring модуля
            etype = str(data.get("etype", ""))
            if etypes is not None and etype not in etypes:
                continue
            if other not in seen:
                seen.add(other)
                queue.append((other, depth + 1))
            if other in seed_set:
                continue
            # Копим связи только для БЛИЖАЙШЕГО расстояния: рёбра, найденные дальше,
            # описывают уже другой путь, а сосед в досье фигурирует один раз.
            if depths.setdefault(other, depth + 1) < depth + 1:
                continue
            links.setdefault(other, set()).add(
                Link(etype=etype, layer=str(data.get("layer", "")), direction=direction)
            )

    found = [
        Neighbor(
            doc_id=_doc_id(node),
            links=tuple(sorted(edge_links, key=lambda link: (link.etype, link.layer, link.direction))),
            hops=depths[node],
        )
        for node, edge_links in links.items()
    ]
    return sorted(found, key=lambda n: (n.hops, n.doc_id))


@dataclass(frozen=True)
class RankedDoc:
    doc_id: str
    score: float


def _power_iteration(
    projection: nx.Graph, personalization: dict[str, float], alpha: float
) -> dict[str, float]:
    """Personalized PageRank степенной итерацией — СВОЯ, а не ``nx.pagerank``.

    ``nx.pagerank`` в networkx 3.x требует **scipy** (проверено живьём: ImportError на
    первом же вызове), а чистый ``_pagerank_python`` — приватная функция, зависеть от
    которой этому проекту уже больно (урок версийного скоса приватного API
    ``markitdown``). Тянуть scipy ради одной функции на графе в сотни узлов — против
    той же логики, по которой отвергнут ``leidenalg`` ради ``python-igraph``.

    Классическая формулировка: ``r = alpha·Wᵀr + (1−alpha)·p``, масса висячих узлов
    (без рёбер) возвращается по вектору персонализации ``p`` — иначе она утекала бы из
    системы и суммарный ранг переставал быть распределением.
    """
    nodes = list(projection.nodes())
    rank = dict(personalization)
    weighted_degree = {
        n: sum(float(d.get("weight", 1.0)) for _, _, d in projection.edges(n, data=True))
        for n in nodes
    }
    for _ in range(PPR_MAX_ITER):
        nxt = dict.fromkeys(nodes, 0.0)
        dangling = 0.0
        for node, score in rank.items():
            degree = weighted_degree[node]
            if degree == 0.0:
                dangling += score
                continue
            for _, other, data in projection.edges(node, data=True):
                nxt[other] += score * float(data.get("weight", 1.0)) / degree
        leak = (1.0 - alpha) + alpha * dangling
        nxt = {n: alpha * nxt[n] + leak * personalization[n] for n in nodes}
        if sum(abs(nxt[n] - rank[n]) for n in nodes) < PPR_TOL:
            return nxt
        rank = nxt
    return rank


def _undirected_projection(graph: nx.MultiDiGraph) -> nx.Graph:
    """MultiDiGraph → взвешенный неориентированный Graph (вес = число параллельных рёбер).

    Направление для PPR вредно: рёбра построены документ→концепт, и на ориентированном
    графе «поток» из документа в тему уже никогда не вернулся бы в другие документы той
    же темы — вся связующая ткань оказалась бы мёртвой.
    """
    projection = nx.Graph()
    projection.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if u == v:
            continue
        weight = projection.get_edge_data(u, v, {}).get("weight", 0)
        projection.add_edge(u, v, weight=weight + 1)
    return projection


def personalized_rank(
    graph: nx.MultiDiGraph, doc_ids: list[str], k: int = 10, alpha: float = PPR_ALPHA
) -> list[RankedDoc]:
    """Personalized PageRank от seed-документов; выдаются только ДОКУМЕНТЫ (spec §6).

    Миллисекунды на нашем масштабе. Seed'ы исключаются из выдачи (они уже в паке), как
    и в ``expand``. Неизвестные seed'ы игнорируются; если не осталось ни одного —
    пустой результат, а не равномерный PageRank по всему корпусу (тот был бы шумом,
    выданным за релевантность).
    """
    seeds = [DOC_PREFIX + d for d in doc_ids if DOC_PREFIX + d in graph]
    if not seeds or graph.number_of_nodes() == 0:
        return []

    projection = _undirected_projection(graph)
    seed_set = set(seeds)
    share = 1.0 / len(seed_set)
    personalization = {node: (share if node in seed_set else 0.0) for node in projection.nodes()}
    scores = _power_iteration(projection, personalization, alpha)

    ranked = [
        RankedDoc(_doc_id(node), float(score))
        for node, score in scores.items()
        if node not in seed_set and node in graph.nodes and _is_document(graph, node)
    ]
    ranked.sort(key=lambda r: (-r.score, r.doc_id))  # ties по id — детерминизм выдачи
    return ranked[:k]
