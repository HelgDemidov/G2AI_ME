"""Выведенная временнáя валидность и срез `as_of` — spec graph-v2 §1.

Валидность НЕ хранится полями: интервалы выводятся из рёбер `supersedes`/`superseded_by`
на каждой сборке графа. Тесты — на чистой функции от записей плюс проверка, что атрибуты
доезжают до узлов и переживают GraphML-экспорт.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from core.schema import SourceRecord
from graph.build_graph import Validity, as_of, build_graph, export_graphml, validity_intervals
from tests.support import valid_record


def make(doc_id: str, *, published: str | None = None, effective: str | None = None,
         relations: list[dict[str, str]] | None = None) -> SourceRecord:
    data = valid_record()
    data["id"] = doc_id
    dates: dict[str, str] = {}
    if published:
        dates["published"] = published
    if effective:
        dates["effective"] = effective
    data["dates"] = dates
    data["relations"] = relations or []
    return SourceRecord.model_validate(data)


def _sup(target: str) -> dict[str, str]:
    return {"type": "supersedes", "target": target}


# --- правило вывода интервалов ---


def test_chain_of_editions_closes_intervals() -> None:
    """A ← B ← C: каждая предыдущая редакция закрывается датой начала следующей."""
    a = make("me-law-2024", published="2024-01-01")
    b = make("me-law-2025", published="2025-01-01", relations=[_sup("me-law-2024")])
    c = make("me-law-2026", published="2026-01-01", relations=[_sup("me-law-2025")])

    iv = validity_intervals([a, b, c])

    assert iv["me-law-2024"] == Validity(dt.date(2024, 1, 1), dt.date(2025, 1, 1), True)
    assert iv["me-law-2025"] == Validity(dt.date(2025, 1, 1), dt.date(2026, 1, 1), True)
    assert iv["me-law-2026"] == Validity(dt.date(2026, 1, 1), None, True)  # действующая — открытый


def test_effective_wins_over_published() -> None:
    """Для законов дата вступления в силу точнее даты публикации."""
    rec = make("me-law-2025", published="2025-01-01", effective="2025-06-01")
    assert validity_intervals([rec])["me-law-2025"].valid_from == dt.date(2025, 6, 1)


def test_reverse_edge_direction_is_normalised() -> None:
    """`superseded_by` у предшественника — та же связь, что `supersedes` у преемника;
    документ, оформленный «обратным» ребром, обязан закрыться так же."""
    old = make("me-law-2024", published="2024-01-01",
               relations=[{"type": "superseded_by", "target": "me-law-2025"}])
    new = make("me-law-2025", published="2025-01-01")
    assert validity_intervals([old, new])["me-law-2024"].valid_to == dt.date(2025, 1, 1)


def test_multiple_successors_earliest_closes_and_warns(caplog: Any) -> None:
    """Подозрительная топология (обычно ошибка курирования) — закрываем самым ранним,
    но молчать нельзя: человек должен увидеть, что развилка вообще есть."""
    old = make("me-law-2024", published="2024-01-01")
    early = make("me-law-2025", published="2025-01-01", relations=[_sup("me-law-2024")])
    late = make("me-law-2026", published="2026-01-01", relations=[_sup("me-law-2024")])

    with caplog.at_level("WARNING", logger="build_graph"):
        iv = validity_intervals([old, early, late])

    assert iv["me-law-2024"].valid_to == dt.date(2025, 1, 1)
    assert "несколько преемников" in caplog.text


def test_amends_does_not_close_interval() -> None:
    """Поправка МЕНЯЕТ документ, а не заменяет его — закрывать интервал нечем."""
    base = make("me-law-2024", published="2024-01-01")
    amendment = make("me-law-2025", published="2025-01-01",
                     relations=[{"type": "amends", "target": "me-law-2024"}])
    iv = validity_intervals([base, amendment])
    assert iv["me-law-2024"] == Validity(dt.date(2024, 1, 1), None, True)


# --- честная неопределённость ---


def test_missing_dates_give_unknown_validity() -> None:
    rec = make("me-law-2024")
    assert validity_intervals([rec])["me-law-2024"] == Validity(None, None, False)


def test_successor_without_date_marks_predecessor_unknown() -> None:
    """Преемник есть, датировать закрытие нечем — «не знаем», а НЕ молчаливое
    «действует до сих пор»: второе было бы выдумкой в пользу устаревшей редакции."""
    old = make("me-law-2024", published="2024-01-01")
    new = make("me-law-2025", relations=[_sup("me-law-2024")])
    iv = validity_intervals([old, new])
    assert iv["me-law-2024"].valid_to is None
    assert iv["me-law-2024"].known is False


# --- срез as_of ---


def _slice(records: list[SourceRecord], date: str) -> list[str]:
    return [n[len("doc:"):] for n in as_of(build_graph(records), dt.date.fromisoformat(date))]


def test_as_of_picks_edition_in_force() -> None:
    recs = [
        make("me-law-2024", published="2024-01-01"),
        make("me-law-2025", published="2025-01-01", relations=[_sup("me-law-2024")]),
    ]
    assert _slice(recs, "2024-06-01") == ["me-law-2024"]
    assert _slice(recs, "2025-06-01") == ["me-law-2025"]


@pytest.mark.parametrize(
    "date,expected",
    [
        ("2024-12-31", ["me-law-2024"]),   # последний день старой
        ("2025-01-01", ["me-law-2025"]),   # день стыка — действует уже преемник
    ],
)
def test_as_of_boundaries(date: str, expected: list[str]) -> None:
    """valid_from ВКЛЮЧИТЕЛЬНО, valid_to ИСКЛЮЧИТЕЛЬНО — иначе в день стыка редакций
    обе версии оказались бы действующими одновременно."""
    recs = [
        make("me-law-2024", published="2024-01-01"),
        make("me-law-2025", published="2025-01-01", relations=[_sup("me-law-2024")]),
    ]
    assert _slice(recs, date) == expected


def test_as_of_excludes_document_not_yet_in_force() -> None:
    recs = [make("me-law-2025", published="2025-01-01")]
    assert _slice(recs, "2024-06-01") == []


def test_as_of_always_includes_unknown_validity() -> None:
    """Неизвестность — не повод молча удалить документ из среза, за который потом
    пишутся нормативные предложения."""
    recs = [make("me-law-undated"), make("me-law-2025", published="2025-01-01")]
    assert _slice(recs, "1999-01-01") == ["me-law-undated"]


def test_as_of_ignores_non_document_nodes() -> None:
    graph = build_graph([make("me-law-2025", published="2025-01-01")])
    assert all(n.startswith("doc:") for n in as_of(graph, dt.date(2025, 6, 1)))


# --- атрибуты узлов и экспорт ---


def test_graph_nodes_carry_validity_attributes() -> None:
    recs = [
        make("me-law-2024", published="2024-01-01"),
        make("me-law-2025", published="2025-01-01", relations=[_sup("me-law-2024")]),
    ]
    graph = build_graph(recs)
    old = graph.nodes["doc:me-law-2024"]
    assert (old["valid_from"], old["valid_to"], old["validity_known"]) == ("2024-01-01", "2025-01-01", True)
    assert graph.nodes["doc:me-law-2025"]["valid_to"] == ""  # открытый интервал


# --- CLI-режим среза ---


def _write(root: Path, rec_data: dict[str, Any]) -> None:
    from tests.support import write_doc

    write_doc(root, rec_data, md="# doc\n")


def _cli_records(root: Path) -> None:
    from tests.support import valid_record

    old = valid_record() | {"id": "me-law-2024", "dates": {"published": "2024-01-01"}}
    new = valid_record() | {
        "id": "me-law-2026",
        "dates": {"published": "2026-01-01"},
        "relations": [{"type": "supersedes", "target": "me-law-2024"}],
    }
    _write(root, old)
    _write(root, new)


def test_cli_as_of_prints_slice(tmp_path: Path, capsys: Any) -> None:
    from graph.build_graph import main

    _cli_records(tmp_path)
    assert main([str(tmp_path), "--as-of", "2025-01-01"]) == 0
    out = capsys.readouterr().out
    assert "Действовали на 2025-01-01 (1 из 2)" in out
    assert "me-law-2024" in out and "me-law-2026" not in out.split("Действовали")[1]


def test_cli_as_of_marks_unknown_validity(tmp_path: Path, capsys: Any) -> None:
    from tests.support import valid_record
    from graph.build_graph import main

    _write(tmp_path, valid_record() | {"id": "me-law-undated", "dates": {}})
    assert main([str(tmp_path), "--as-of", "2025-01-01"]) == 0
    assert "валидность неизвестна" in capsys.readouterr().out


def test_cli_as_of_exports_slice_subgraph(tmp_path: Path) -> None:
    """`--as-of` вместе с `--graphml` экспортирует СРЕЗ, а не полный граф — иначе флаг
    молча давал бы файл, не соответствующий напечатанному списку."""
    from graph.build_graph import main

    _cli_records(tmp_path)
    out = tmp_path / "slice.graphml"
    assert main([str(tmp_path), "--as-of", "2025-01-01", "--graphml", str(out)]) == 0
    exported = nx.read_graphml(out)
    assert set(exported.nodes()) == {"doc:me-law-2024"}


def test_cli_reports_dangling_identifier(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    """Протухшая запись справочника видна куратору в выводе, а не только в логе."""
    from graph import cite_mining
    from graph.build_graph import main

    _cli_records(tmp_path)
    monkeypatch.setattr(cite_mining, "load_identifiers", lambda: {"CELEX:32024R1689": "no-such"})
    assert main([str(tmp_path)]) == 0
    assert "identifiers.yaml" in capsys.readouterr().out


def test_cli_reports_leads_and_l1_edges(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from tests.support import valid_record, write_doc
    from graph import cite_mining
    from graph.build_graph import main

    write_doc(tmp_path, valid_record() | {"id": "me-law-2025"},
              md="ссылается на Regulation (EU) 2024/1689 и ISO/IEC 42001:2023\n")
    monkeypatch.setattr(cite_mining, "load_identifiers", lambda: {"CELEX:32024R1689": "me-law-2025"})
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Цитаты без записи в корпусе: 1" in out  # ISO — лид; CELEX — самоцитирование


def test_cli_rejects_invalid_registry(tmp_path: Path, capsys: Any) -> None:
    from graph.build_graph import main

    (tmp_path / "intl-xperience" / "sg" / "broken").mkdir(parents=True)
    (tmp_path / "intl-xperience" / "sg" / "broken" / "meta.yaml").write_text("id: broken\n", encoding="utf-8")
    assert main([str(tmp_path)]) == 1


def test_graphml_export_survives_unknown_dates(tmp_path: Path) -> None:
    """GraphML-writer падает на None — неизвестная дата обязана ехать пустой строкой,
    иначе экспорт корпуса ломается на первой же записи без дат."""
    out = tmp_path / "graph.graphml"
    export_graphml(build_graph([make("me-law-undated")]), out)
    reloaded = nx.read_graphml(out)
    assert reloaded.nodes["doc:me-law-undated"]["valid_from"] == ""
    assert reloaded.nodes["doc:me-law-undated"]["validity_known"] is False
