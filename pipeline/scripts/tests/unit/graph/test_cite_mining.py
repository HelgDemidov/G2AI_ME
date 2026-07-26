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

from core import schema
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
        # Живая форма корпуса: кавычка после «CG» + двузначный год.
        ('о привредним друштвима („Službeni list CG", br. 65/20)', "SLCG:65/2020"),
        ("Regulation (EU) 2024/1689", "CELEX:32024R1689"),
        ("Directive (EU) 2016/680", "CELEX:32016L0680"),
        ("Decision (EU) 2020/1234", "CELEX:32020D1234"),
        # Старая форма: числа в ОБРАТНОМ порядке относительно новой.
        ("Regulation (EU) No 1025/2012", "CELEX:32012R1025"),
        # ...и год в ней бывает двузначным — без разворота выходил CELEX:30091R3922.
        ("Regulation (EEC) No 3922/91", "CELEX:31991R3922"),
        ("Regulation (EEC) No 339/93", "CELEX:31993R0339"),
        # Третья форма: без скобочной юрисдикции, зато с суффиксом в конце.
        ("Directive 2000/31/EC", "CELEX:32000L0031"),
        ("Directive 95/46/EC", "CELEX:31995L0046"),
        ("Decision 2000/520/EC", "CELEX:32000D0520"),
        ("Directive 2019/790/EU", "CELEX:32019L0790"),
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
        # Год-гейт: номер стоит на месте года — «поправить» такое нельзя, только молчать.
        "Decision 768/2008/EC",
        "Regulation (EU) 3922/91",           # OCR потерял «No» -> год 3922
        "Directive 1234/5678/EU",
    ],
)
def test_patterns_reject_near_misses(text: str) -> None:
    """Ложное ребро в юридическом графе дороже пропущенного — границы важнее охвата."""
    assert cite_mining.extract_identifiers(text) == []


def test_gazette_citation_lists_several_acts() -> None:
    """Одна строка законно несёт НЕСКОЛЬКО актов — живая форма корпуса; терять хвост
    списка значило бы молча потерять две связи из трёх."""
    text = 'о привредним друштвима („Službeni list CG", br. 65/20, 146/21 i 4/24)'
    assert [i for i, _ in cite_mining.extract_identifiers(text)] == [
        "SLCG:146/2021", "SLCG:4/2024", "SLCG:65/2020",
    ]


def test_three_eu_forms_collapse_to_one_identifier() -> None:
    """Три живые формы ссылки на ОДИН акт дают один идентификатор. Иначе одна связь
    рассыпалась бы на три записи справочника и три ребра к одному документу."""
    forms = ["Regulation (EU) 2016/679", "32016R0679"]
    idents = {cite_mining.extract_identifiers(f)[0][0] for f in forms}
    assert idents == {"CELEX:32016R0679"}
    slash = {cite_mining.extract_identifiers(f)[0][0]
             for f in ["Directive 2016/680/EU", "32016L0680"]}
    assert slash == {"CELEX:32016L0680"}


def test_slash_form_does_not_shadow_parenthesised_form() -> None:
    """Паттерны не должны перехватывать чужую форму: у «Directive (EU) 2016/680»
    порядок чисел тот же, но разбирает её eu_act, и результат обязан совпасть."""
    assert cite_mining.extract_identifiers("Directive (EU) 2016/680") == [
        ("CELEX:32016L0680", "eu_act")
    ]


def test_eu_act_and_compact_celex_share_identifier_space() -> None:
    """Естественно-языковая ссылка и компактный CELEX дают ОДИН идентификатор — иначе
    одна связь распалась бы на две записи справочника и два ребра."""
    natural = cite_mining.extract_identifiers("Regulation (EU) 2024/1689")
    compact = cite_mining.extract_identifiers("32024R1689")
    assert natural[0][0] == compact[0][0] == "CELEX:32024R1689"


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


# --- курируемый алиас-канал (spec graph-hardening §2) ---


_AI_ACT = {"EU AI Act": "CELEX:32024R1689"}


def test_alias_extracts_identifier_absent_from_text() -> None:
    """Живой случай `oxford-insights-gairi-2025`: акт цитируется ТОЛЬКО алиасом, формального
    идентификатора в тексте нет вовсе — регекс-майнер такую связь не видит в принципе."""
    text = "Components of the EU AI Act entered into force throughout 2025."
    assert cite_mining.extract_identifiers(text) == []                       # паттерны молчат
    assert cite_mining.extract_identifiers(text, _AI_ACT) == [("CELEX:32024R1689", "alias")]


def test_extract_identifiers_is_pattern_only_by_default() -> None:
    """Контракт голден-теста: дефолт БЕЗ алиасов — стабильность регексов не должна
    зависеть от файла, который куратор законно правит в любой момент."""
    text = "the EU AI Act and 32016R0679"
    assert cite_mining.extract_identifiers(text) == [("CELEX:32016R0679", "celex")]


@pytest.mark.parametrize(
    "text,found",
    [
        ("primjena GDPR-om u praksi", True),      # флексия — границы слов дают её даром
        ("usklađen sa GDPR.", True),
        ("(GDPR)", True),
        ("the gdpr regime", True),                # регистр плавает по вёрстке и OCR
        ("GDPRX", False),                         # подстрока внутри слова — не цитата
        ("pre-GDPRish", False),
    ],
)
def test_alias_word_boundaries(text: str, found: bool) -> None:
    hits = cite_mining.extract_identifiers(text, {"GDPR": "CELEX:32016R0679"})
    assert bool(hits) is found


def test_alias_does_not_match_other_jurisdictions_act() -> None:
    """⚠ Ловушка, подтверждённая живьём в `oxford-insights-gairi-2025`: голый «AI Act» —
    это ещё и корейский «Basic AI Act», и тайваньский. Различающая форма обязана
    промолчать на чужих актах, иначе канал строит ложные рёбра пачками."""
    for foreign in ["South Korea's Basic AI Act", "a draft bill of its AI Act",
                    "the comprehensive Australia AI Act"]:
        assert cite_mining.extract_identifiers(foreign, _AI_ACT) == []


def test_pattern_wins_rule_tag_over_alias() -> None:
    """Формальная цитата — более сильное свидетельство: при совпадении идентификатора
    правилом остаётся паттерн, алиасу достаётся только ненайденное."""
    text = "the EU AI Act, formally 32024R1689"
    assert cite_mining.extract_identifiers(text, _AI_ACT) == [("CELEX:32024R1689", "celex")]


def test_blank_alias_is_ignored() -> None:
    """Опечатка `"": doc-id` в YAML иначе совпала бы в КАЖДОМ документе корпуса."""
    assert cite_mining.extract_identifiers("любой текст", {"": "CELEX:32024R1689", "  ": "X"}) == []


def test_alias_hit_goes_through_normal_resolution(tmp_path: Path) -> None:
    """Алиас сам по себе ребра не строит — он даёт идентификатор, который проходит ту же
    резолюцию: есть запись справочника — ребро, нет — обычный лид."""
    cited = _place(tmp_path, "eu-ai-act-2024", "# акт\n")
    citing = _place(tmp_path, "oxford-2025", "following the EU AI Act closely\n")
    result = cite_mining.mine_corpus(
        [cited, citing], tmp_path,
        identifiers={"CELEX:32024R1689": "eu-ai-act-2024"}, aliases=_AI_ACT,
    )
    assert [(e.source_id, e.target_id, e.rule) for e in result.edges] == [
        ("oxford-2025", "eu-ai-act-2024", "alias")
    ]

    unresolved = cite_mining.mine_corpus([citing], tmp_path, identifiers={}, aliases=_AI_ACT)
    assert unresolved.edges == []
    assert [lead["identifier"] for lead in unresolved.leads] == ["CELEX:32024R1689"]


def test_explicit_empty_aliases_keeps_test_hermetic(tmp_path: Path) -> None:
    """Явный словарь (в т.ч. пустой) отключает чтение диска: правка курируемого файла
    не должна менять исход юнит-теста."""
    citing = _place(tmp_path, "oxford-2025", "following the EU AI Act closely\n")
    assert cite_mining.mine_corpus([citing], tmp_path, identifiers={}, aliases={}).leads == []


# --- резолюция ---


def test_resolution_from_source_url() -> None:
    """URL, несущий идентификатор БУКВАЛЬНО, резолвится автоматически."""
    rec = _rec("eu-act-2024", url="https://eur-lex.europa.eu/eli/reg/2024/1689/oj?uri=CELEX:32024R1689")
    assert cite_mining.identifiers_from_urls([rec]) == {"CELEX:32024R1689": "eu-act-2024"}


@pytest.mark.parametrize(
    "url,expected",
    [
        # Форма, которую строит сам коннектор eurlex (`_build_source_url`).
        ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:52026DC0577",
         ["CELEX:52026DC0577"]),
        # Сектор 5 + двухбуквенный тип; сектор 3 с литерой вне LRDE прозаической границы.
        ("https://x.example/?uri=CELEX:32026H0123", ["CELEX:32026H0123"]),
        ("https://x.example/?uri=CELEX:62012CJ0123", ["CELEX:62012CJ0123"]),
        # Номер длиннее четырёх цифр.
        ("https://x.example/?uri=CELEX:32024R123456", ["CELEX:32024R123456"]),
        # Скопировано из браузера: двоеточие процент-энкодировано, регистр «плавает».
        ("https://x.example/?uri=celex%3A32024r1689", ["CELEX:32024R1689"]),
        # Хвосты: консолидированная версия и corrigendum — часть идентификатора, иначе
        # поправка резолвилась бы в базовый акт (неверная связь).
        ("https://x.example/?uri=CELEX:02016R0679-20160504", ["CELEX:02016R0679-20160504"]),
        ("https://x.example/?uri=CELEX:32004L0018R(01)", ["CELEX:32004L0018R(01)"]),
        # Без якоря догадки не строятся — OJ-форма остаётся нерезолвнутой.
        ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689", []),
    ],
)
def test_url_anchor_reads_celex_verbatim(url: str, expected: list[str]) -> None:
    """Префикс `CELEX:` снимает неоднозначность формы полностью — сектор/литеры/длину
    гадать не нужно, поэтому URL-канал шире прозаической границы v1 сознательно."""
    assert cite_mining.identifiers_from_url(url) == expected


def test_url_anchor_grammar_stays_out_of_prose() -> None:
    """⚠ Расширение — ТОЛЬКО в URL-канале: в прозе за recall платят ложными рёбрами.
    Та же строка, что резолвится по якорю, из голого текста не извлекается."""
    assert cite_mining.extract_identifiers("документ 52026DC0577 в тексте") == []
    assert cite_mining.identifiers_from_url("?uri=CELEX:52026DC0577") == ["CELEX:52026DC0577"]


def test_sector_five_url_resolves_to_record() -> None:
    """Замер, ради которого правка и делалась: 79% живых eurlex-CELEX — сектор 5,
    и до якоря ни один из них не резолвился (граница v1 знала только сектор 3)."""
    rec = _rec("eu-com-2026", url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52026DC0577")
    assert cite_mining.identifiers_from_urls([rec]) == {"CELEX:52026DC0577": "eu-com-2026"}


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


def test_unreadable_doc_is_isolated_not_fatal(tmp_path: Path, caplog: Any) -> None:
    """Один битый doc.md (не-UTF-8, напр. после сбоя диска) не должен ронять граф
    всего корпуса — паттерн изоляции отказа тот же, что в run_pipeline/discovery/apply."""
    broken = _place(tmp_path, "broken-doc-2025", "placeholder\n")
    schema.md_file(broken, tmp_path).write_bytes(b"\xff\xfe not valid utf-8 \x80\x81")
    healthy = _place(tmp_path, "healthy-doc-2025", "см. ISO/IEC 42001:2023\n")

    with caplog.at_level("WARNING", logger="cite_mining"):
        result = cite_mining.mine_corpus([broken, healthy], tmp_path, identifiers={})

    assert [lead["identifier"] for lead in result.leads] == ["ISO/IEC 42001:2023"]
    assert "broken-doc-2025" in caplog.text


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
    """Отчёт живёт в `.state/` рядом с лидами snowball — каталогом владеет
    core.schema (knowledge-hardening §2), имя файла принадлежит писателю."""
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
    на свежем клоне падает. Обе секции — независимые словари строк."""
    for section in (cite_mining.load_identifiers(), cite_mining.load_aliases()):
        assert isinstance(section, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in section.items())


@pytest.mark.parametrize("content", [None, "- список, а не отображение\n", "", "identifiers: 42\n"])
def test_missing_or_malformed_vocab_degrades_to_empty(tmp_path: Path, content: str | None) -> None:
    """Справочник БЕЗ гейта: его отсутствие (свежий клон, tmp-корень) и любая порча
    формата обязаны дать пустой словарь, а не уронить сборку графа всего корпуса."""
    path = tmp_path / "identifiers.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert cite_mining.load_identifiers(path) == {}
    assert cite_mining.load_aliases(path) == {}


def test_shipped_aliases_point_at_known_identifier_space() -> None:
    """Алиас обязан давать КАНОНИЧЕСКИЙ идентификатор (то же пространство, что паттерны),
    а не doc-id: иначе он молча минует канал резолюции и никогда не станет ребром.

    Префиксы читаются из ``IDENTIFIER_PREFIXES`` (knowledge-hardening §4) — единого
    реестра рядом с ``_PATTERNS``, не хардкода в тесте."""
    for alias, ident in cite_mining.load_aliases().items():
        assert ident.startswith(cite_mining.IDENTIFIER_PREFIXES), f"{alias} -> {ident}"


# --- state_dir: владелец переехал в core.schema (knowledge-hardening §2) ---


def test_leads_path_matches_schema_state_dir(tmp_path: Path) -> None:
    """Путь отчёта не изменился при переезде владельца каталога discovery -> schema."""
    assert cite_mining.leads_path(tmp_path) == schema.state_dir(tmp_path) / "cite_leads.yaml"


def test_cite_mining_does_not_import_discovery() -> None:
    """graph — не должен зависеть от discovery ради path-хелпера: единственная
    межслойная зависимость (``from discovery import store``) снята переездом
    ``state_dir`` в ``core.schema``."""
    assert not hasattr(cite_mining, "store")
    source = Path(cite_mining.__file__).read_text(encoding="utf-8")
    assert "import discovery" not in source and "from discovery" not in source


# --- frontmatter не участвует в экстракции (spec convert-knowledge-seam-hardening §6) ---


def test_frontmatter_metadata_does_not_create_edges(tmp_path: Path) -> None:
    """Курируемые метаданные — не текст издателя: акт, названный только в ``title``
    frontmatter'а, ребра ``cites`` не даёт (штатная форма заголовка в реестрах ЕС —
    «Guidance on applying Regulation (EU) …»), а тот же акт в ТЕЛЕ — даёт."""
    cited = _place(tmp_path, "gdpr-2016", "# GDPR\n")
    citing = _place(tmp_path, "guidance-doc-2025", "Тело без формальных ссылок.\n")
    md = schema.md_file(citing, tmp_path)
    md.write_text(
        "---\ntitle: Guidance on applying Regulation (EU) 2016/679 to AI\n---\n\n"
        + md.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    identifiers = {"CELEX:32016R0679": "gdpr-2016"}

    result = cite_mining.mine_corpus([cited, citing], tmp_path, identifiers=identifiers, aliases={})
    assert result.edges == []

    md.write_text(
        md.read_text(encoding="utf-8") + "\nсогласно Regulation (EU) 2016/679\n", encoding="utf-8"
    )
    result = cite_mining.mine_corpus([cited, citing], tmp_path, identifiers=identifiers, aliases={})
    assert [(e.source_id, e.target_id) for e in result.edges] == [("guidance-doc-2025", "gdpr-2016")]
