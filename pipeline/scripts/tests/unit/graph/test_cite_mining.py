"""L1-слой цитирования — spec graph-v2 §3.

Юридический корпус формально цитируем, поэтому рёбра `cites` берутся регексами с нулём
LLM. Тесты держат три границы: что паттерн ловит, чего он НЕ ловит (ложное ребро в
юридическом графе дороже пропущенного), и что нерезолвнутое уходит в отчёт, а не
выдумывается.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, strategies as st

from core.schema import SourceRecord
from graph import cite_mining
from graph.build_graph import build_corpus_graph, build_graph
from tests.support import valid_record, write_doc


def _rec(doc_id: str, *, url: str | None = None) -> SourceRecord:
    data = valid_record()
    data["id"] = doc_id
    if url:
        data["source_url"] = url
    return SourceRecord.model_validate(data)


def _place(root: Path, doc_id: str, md: str, *, url: str | None = None) -> SourceRecord:
    data = valid_record()
    data["id"] = doc_id
    if url:
        data["source_url"] = url
    write_doc(root, data, md=md)
    return SourceRecord.model_validate(data)


# --- реестр паттернов: что ловим ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("см. Regulation (EU) 32024R1689 далее", "CELEX:32024R1689"),
        ("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689", "CELEX:32024R1689"),
        ("Службени лист ЦГ бр. 12/2025", "SLCG:12/2025"),
        ("Službeni list CG br. 12/2025", "SLCG:12/2025"),
        ("Sluzbeni list CG, br 7/2024", "SLCG:7/2024"),
        ("соответствует ISO/IEC 42001:2023", "ISO/IEC 42001:2023"),
        ("ISO/IEC 23894", "ISO/IEC 23894"),
        ("ISO/IEC 27001-2:2022", "ISO/IEC 27001-2:2022"),
        ("ISO 9001", "ISO 9001"),
        ("NIST SP 800-53", "NIST SP 800-53"),
        ("NIST SP 800-53A", "NIST SP 800-53A"),
        ("NIST AI 100-1", "NIST AI 100-1"),
    ],
)
def test_patterns_extract_canonical_identifier(text: str, expected: str) -> None:
    assert [i for i, _ in cite_mining.extract_identifiers(text)] == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "версия 32024 без литеры",          # CELEX без типа акта
        "42024R1689",                        # сектор 4 — не наш сектор-3
        "ISO 42",                            # слишком короткий номер
        "NIST SP 500-53",                    # не серия 800
        "Службени лист РС бр. 12/2025",      # другая юрисдикция
    ],
)
def test_patterns_reject_near_misses(text: str) -> None:
    """Ложное ребро в юридическом графе дороже пропущенного — границы важнее охвата."""
    assert cite_mining.extract_identifiers(text) == []


def test_repeated_mention_gives_one_identifier() -> None:
    """Акт, упомянутый десять раз, — одно ребро, не десять."""
    text = "32024R1689 ... снова 32024R1689 ... и ещё 32024R1689"
    assert cite_mining.extract_identifiers(text) == [("CELEX:32024R1689", "celex")]


def test_extraction_is_deterministic_across_rules() -> None:
    text = "NIST AI 100-1 и ISO/IEC 42001:2023 и 32024R1689"
    twice = [cite_mining.extract_identifiers(text) for _ in range(2)]
    assert twice[0] == twice[1]
    assert [r for _, r in twice[0]] == ["celex", "iso", "nist"]  # правила по имени


@given(
    year=st.integers(min_value=1000, max_value=9999),
    letter=st.sampled_from("LRDE"),
    number=st.integers(min_value=0, max_value=9999),
)
def test_celex_roundtrip_property(year: int, letter: str, number: int) -> None:
    """Любая валидная форма сектора-3 распознаётся и канонизируется без потерь."""
    ident = f"3{year}{letter}{number:04d}"
    assert cite_mining.extract_identifiers(f"текст {ident} текст") == [(f"CELEX:{ident}", "celex")]


@given(number=st.integers(min_value=100, max_value=99999))
def test_iso_number_lengths_property(number: int) -> None:
    assert cite_mining.extract_identifiers(f"ISO/IEC {number}") == [(f"ISO/IEC {number}", "iso")]


# --- резолюция ---


def test_resolution_from_source_url() -> None:
    """URL, несущий идентификатор БУКВАЛЬНО, резолвится автоматически."""
    rec = _rec("eu-act-2024", url="https://eur-lex.europa.eu/eli/reg/2024/1689/oj?uri=CELEX:32024R1689")
    assert cite_mining.identifiers_from_urls([rec]) == {"CELEX:32024R1689": "eu-act-2024"}


def test_oj_form_url_is_not_guessed() -> None:
    """Живой случай `eu-ai-act-2024`: из OJ-формы литера типа акта не выводится надёжно —
    эвристика обязана ПАСОВАТЬ, а не угадать CELEX и связать документы неверно."""
    rec = _rec("eu-ai-act-2024", url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689")
    assert cite_mining.identifiers_from_urls([rec]) == {}


def test_ambiguous_url_identifier_is_skipped(caplog: Any) -> None:
    """Один идентификатор в URL двух записей — ребро не строится: угадывать, какая из
    них «та самая», хуже, чем не связать."""
    recs = [
        _rec("a-doc-2024", url="https://x.example/CELEX:32024R1689"),
        _rec("b-doc-2024", url="https://y.example/CELEX:32024R1689"),
    ]
    with caplog.at_level("WARNING", logger="cite_mining"):
        assert cite_mining.identifiers_from_urls(recs) == {}
    assert "нескольких записей" in caplog.text


def test_identifiers_yaml_resolves_and_beats_url(tmp_path: Path) -> None:
    """Курируемый справочник побеждает URL-эвристику: человек знает лучше регекса."""
    cited = _place(tmp_path, "eu-act-2024", "# doc\n")
    citing = _place(tmp_path, "me-law-2025", "ссылается на 32024R1689\n")
    result = cite_mining.mine_corpus(
        [cited, citing], tmp_path, identifiers={"CELEX:32024R1689": "eu-act-2024"}
    )
    assert [(e.source_id, e.target_id, e.rule) for e in result.edges] == [
        ("me-law-2025", "eu-act-2024", "celex")
    ]


def test_dangling_identifier_reports_without_crashing(tmp_path: Path, caplog: Any) -> None:
    """Справочник без гейта: протухшая запись не должна стоить графа всего корпуса."""
    citing = _place(tmp_path, "me-law-2025", "ссылается на 32024R1689\n")
    with caplog.at_level("WARNING", logger="cite_mining"):
        result = cite_mining.mine_corpus(
            [citing], tmp_path, identifiers={"CELEX:32024R1689": "no-such-doc"}
        )
    assert result.edges == []
    assert result.dangling and "no-such-doc" in result.dangling[0]


def test_unresolved_citation_becomes_lead_not_edge(tmp_path: Path) -> None:
    """Документа нет в корпусе — это СЫРЬЁ для discovery, а не ребро и не кандидат."""
    citing = _place(tmp_path, "me-law-2025", "см. ISO/IEC 42001:2023 и NIST AI 100-1\n")
    result = cite_mining.mine_corpus([citing], tmp_path, identifiers={})
    assert result.edges == []
    assert [lead["identifier"] for lead in result.leads] == ["ISO/IEC 42001:2023", "NIST AI 100-1"]
    assert result.leads[0]["cited_by"] == ["me-law-2025"]


def test_self_citation_is_not_an_edge(tmp_path: Path) -> None:
    rec = _place(tmp_path, "eu-act-2024", "этот акт 32024R1689 сам о себе\n")
    result = cite_mining.mine_corpus([rec], tmp_path, identifiers={"CELEX:32024R1689": "eu-act-2024"})
    assert result.edges == []


def test_document_without_md_is_skipped(tmp_path: Path) -> None:
    """Майнинг реконсиляционен: нет текста — нет рёбер, не ошибка."""
    data = valid_record()
    write_doc(tmp_path, data)  # без doc.md
    rec = SourceRecord.model_validate(data)
    assert cite_mining.mine_corpus([rec], tmp_path, identifiers={}).edges == []


# --- layer-теги и отчёт ---


def test_l1_edges_are_distinguishable_from_curated(tmp_path: Path) -> None:
    """Отличимость L0/L1 — по построению (атрибуты рёбер), а не по договорённости."""
    cited = _place(tmp_path, "eu-act-2024", "# doc\n")
    citing = _place(tmp_path, "me-law-2025", "ссылается на 32024R1689\n")
    mining = cite_mining.mine_corpus(
        [cited, citing], tmp_path, identifiers={"CELEX:32024R1689": "eu-act-2024"}
    )
    graph = build_graph([cited, citing], cites=mining.edges)

    l1 = [d for _, _, d in graph.edges(data=True) if d.get("layer") == "L1"]
    assert len(l1) == 1
    assert l1[0]["etype"] == "cites" and l1[0]["rule"] == "celex"
    assert l1[0]["identifier"] == "CELEX:32024R1689"
    assert all(d.get("layer") in ("L0", "L1") for _, _, d in graph.edges(data=True))


def test_curated_edges_keep_l0_tag(tmp_path: Path) -> None:
    data = valid_record()
    data["relations"] = [{"type": "implements", "target": "eu-act-2024"}]
    rec = SourceRecord.model_validate(data)
    graph = build_graph([rec])
    curated = [d for _, _, d in graph.edges(data=True) if d.get("etype") == "implements"]
    assert curated and curated[0]["layer"] == "L0"


def test_leads_written_to_corpus_state_dir(tmp_path: Path) -> None:
    """Отчёт живёт в `.state/` рядом с лидами snowball — каталогом владеет store,
    имя файла принадлежит писателю."""
    citing = _place(tmp_path, "me-law-2025", "см. ISO/IEC 42001:2023\n")
    _graph, mining = build_corpus_graph([citing], tmp_path)

    path = cite_mining.leads_path(tmp_path)
    assert path == tmp_path / ".state" / "cite_leads.yaml"
    assert path.exists()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert [lead["identifier"] for lead in saved] == ["ISO/IEC 42001:2023"]
    assert mining.leads == saved


def test_leads_are_rewritten_not_appended(tmp_path: Path) -> None:
    """Отчёт — производный артефакт (как сам граф), а не накапливаемый журнал."""
    cite_mining.save_leads([{"identifier": "OLD", "rule": "iso", "cited_by": ["x"]}], tmp_path)
    cite_mining.save_leads([{"identifier": "NEW", "rule": "iso", "cited_by": ["y"]}], tmp_path)
    saved = yaml.safe_load(cite_mining.leads_path(tmp_path).read_text(encoding="utf-8"))
    assert [lead["identifier"] for lead in saved] == ["NEW"]


def test_build_corpus_graph_can_skip_disk_write(tmp_path: Path) -> None:
    citing = _place(tmp_path, "me-law-2025", "см. ISO/IEC 42001:2023\n")
    build_corpus_graph([citing], tmp_path, write_leads=False)
    assert not cite_mining.leads_path(tmp_path).exists()


def test_shipped_identifiers_vocab_is_wellformed() -> None:
    """Файл в репозитории обязан парситься даже пустым — иначе первая же сборка графа
    на свежем клоне падает."""
    assert isinstance(cite_mining.load_identifiers(), dict)
