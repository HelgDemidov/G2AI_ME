"""Тесты discovery/connectors/snowball.py — LLM-стадия текстовых цитат (spec discovery-
snowball §5, коммит 6). ``call_model`` инжектируется как callable (та же техника, что
``fetch``/``get_standards_page`` у aiforgood/eurlex и реальный ``core/test_openrouter.py`` —
живой сетевой вызов НЕ тестируется НИГДЕ в проекте для costed LLM-путей, прецедент
cloud_ocr/figures_vlm)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import schema
from discovery.connectors.snowball import (
    CitationLead,
    RawLink,
    _default_call_model,
    discover_snowball,
    extract_text_citations,
    find_citation_sections,
    passes_verbatim_gate,
    save_leads,
)
from tests.support import build_pdf, valid_record, write_doc

_EMPTY_PDF = build_pdf(lines=[])  # валидный PDF без аннотаций/текста — для документов,
# которым в этих тестах нужен только doc.md (raw.* обязателен, чтобы документ вообще
# домайнился, см. discover_snowball's "нечего майнить" скип), не сам PDF-контент.

# --- find_citation_sections: секции по заголовку ---


@pytest.mark.parametrize(
    "heading",
    ["References", "Bibliography", "Sources", "Endnotes", "Notes", "Reference list", "Литература", "Извори"],
)
def test_find_citation_sections_recognizes_each_stopword(heading: str) -> None:
    text = f"# Intro\n\nSome prose.\n\n# {heading}\n\nSmith, J. (2024). A Report. Gov Press.\n"
    sections = find_citation_sections(text)
    assert len(sections) == 1
    assert "Smith, J. (2024)" in sections[0]


def test_find_citation_sections_stops_at_next_heading_of_any_level() -> None:
    text = "# References\n\nSmith, J. (2024). A Report.\n\n# Appendix\n\nUnrelated appendix text.\n"
    sections = find_citation_sections(text)
    assert len(sections) == 1
    assert "Appendix" not in sections[0]
    assert "Unrelated" not in sections[0]


def test_find_citation_sections_no_heading_no_dense_block_yields_empty() -> None:
    text = "# Title\n\nJust some ordinary prose with no years or citations.\n"
    assert find_citation_sections(text) == []


def test_find_citation_sections_dense_year_block_fallback() -> None:
    text = "\n".join(
        [
            "# Title",
            "",
            "Smith, J. (2024). A Long Enough Report Title Here.",
            "Doe, A. (2023). Another Long Enough Citation Line.",
            "Lee, K. (2022). Yet Another Sufficiently Long Citation.",
            "",
            "Some unrelated short line.",
        ]
    )
    sections = find_citation_sections(text)
    assert len(sections) == 1
    assert "Smith, J. (2024)" in sections[0]
    assert "Doe, A. (2023)" in sections[0]
    assert "unrelated" not in sections[0].lower()


def test_find_citation_sections_dense_block_below_min_run_is_ignored() -> None:
    text = "Smith, J. (2024). A Long Enough Report Title Here.\nDoe, A. (2023). Another Long Citation.\n"
    assert find_citation_sections(text) == []


# --- passes_verbatim_gate ---


def test_verbatim_gate_accepts_exact_substring() -> None:
    section = "Smith, J. (2024). Cyber Security Strategy 2025. Gov Press."
    assert passes_verbatim_gate("Cyber Security Strategy 2025", section)


def test_verbatim_gate_case_and_whitespace_insensitive() -> None:
    section = "Smith, J. (2024).   CYBER   Security\nStrategy 2025. Gov Press."
    assert passes_verbatim_gate("cyber security strategy 2025", section)


def test_verbatim_gate_rejects_partial_superstring_not_soft_matched() -> None:
    """Title — почти совпадение, но несёт СЛОВО, которого в секции нет («Update») —
    гейт обязан отсеять, не принимать по мягкому/частичному порогу."""
    section = "Smith, J. (2024). Cyber Security Strategy 2025. Gov Press."
    assert not passes_verbatim_gate("Cyber Security Strategy 2025 Update", section)


def test_verbatim_gate_rejects_empty_title() -> None:
    assert not passes_verbatim_gate("", "Some section text (2024).")


def test_verbatim_gate_rejects_title_not_present_at_all() -> None:
    section = "Smith, J. (2024). Cyber Security Strategy 2025. Gov Press."
    assert not passes_verbatim_gate("Completely Unrelated Fabricated Title", section)


# --- extract_text_citations: fake call_model, verbatim gate, cache, two output sorts ---


class _FakeModel:
    def __init__(self, response_json: dict[str, Any]) -> None:
        self.response_json = response_json
        self.calls = 0

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        content = json.dumps(self.response_json)
        return {"choices": [{"message": {"content": content}}]}


_REFERENCES_MD = (
    "# References\n\n"
    "Smith, J. (2024). Cyber Security Strategy 2025. https://gov.example/strategy.pdf\n"
    "Doe, A. (2023). Undated Policy Report With No Link Anywhere.\n"
)


def test_extract_text_citations_no_sections_skips_model_entirely(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Title\n\nordinary prose, no citation section\n", encoding="utf-8")
    fake = _FakeModel({"citations": []})

    links, leads = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=fake)

    assert links == []
    assert leads == []
    assert fake.calls == 0


def test_extract_text_citations_splits_url_and_no_url_outputs(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(_REFERENCES_MD, encoding="utf-8")
    fake = _FakeModel(
        {
            "citations": [
                {
                    "title": "Cyber Security Strategy 2025",
                    "issuer": "Gov Press",
                    "year": 2024,
                    "url": "https://gov.example/strategy.pdf",
                },
                {
                    "title": "Undated Policy Report With No Link Anywhere",
                    "issuer": None,
                    "year": 2023,
                    "url": None,
                },
            ]
        }
    )

    links, leads = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=fake)

    assert links == [RawLink(url="https://gov.example/strategy.pdf", anchor="Cyber Security Strategy 2025")]
    assert len(leads) == 1
    assert leads[0] == CitationLead(
        title="Undated Policy Report With No Link Anywhere",
        issuer=None,
        year=2023,
        source_doc_id="doc-1",
        context=leads[0].context,
    )
    assert fake.calls == 1


def test_extract_text_citations_verbatim_gate_drops_fabricated_title(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(_REFERENCES_MD, encoding="utf-8")
    fake = _FakeModel({"citations": [{"title": "A Completely Fabricated Title Not In Text", "url": None}]})

    links, leads = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=fake)

    assert links == []
    assert leads == []


def test_extract_text_citations_malformed_json_yields_nothing_not_crash(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(_REFERENCES_MD, encoding="utf-8")

    def broken(payload: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "not valid json {{{"}}]}

    links, leads = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=broken)
    assert links == []
    assert leads == []


def test_extract_text_citations_cache_hit_skips_model_call(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(_REFERENCES_MD, encoding="utf-8")
    fake = _FakeModel(
        {"citations": [{"title": "Cyber Security Strategy 2025", "url": "https://gov.example/strategy.pdf"}]}
    )

    first_links, _ = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=fake)
    assert fake.calls == 1
    assert (tmp_path / ".citations.yaml").exists()

    second_links, _ = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=fake)
    assert fake.calls == 1  # второй прогон — кэш-хит, ноль обращений к модели
    assert second_links == first_links


# --- sensitivity gate: применяется на уровне discover_snowball (не extract_text_citations) ---


def test_confidential_document_skips_llm_stage_entirely(tmp_path: Path) -> None:
    data = valid_record() | {
        "id": "confidential-doc",
        "entity_id": "me",
        "track": "montenegro",
        "sensitivity": "confidential",
    }
    rec = schema.SourceRecord.model_validate(data)
    # raw.* обязателен, иначе документ скипается целиком ДО дохода до sensitivity-гейта —
    # тест доказывал бы гейт даже если бы он был сломан/отсутствовал.
    write_doc(tmp_path, data, raw=_EMPTY_PDF, md=_REFERENCES_MD, state={"sha256": "a" * 64})

    def boom(payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("LLM-стадия НЕ должна вызываться для confidential-документа")

    from discovery.connectors import snowball as sb

    cfg = sb.SnowballConfig(
        enabled=True,
        source_filter=sb.SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=()),
        url_filter=sb.UrlFilter(exclude_domains=(), exclude_url_substrings=()),
        emit=sb.EmitConfig(pdf_annotations=True, html_hrefs=True, printed_urls=True, text_citations=True),
        max_candidates=None,
        citations_model="test/model",
    )
    result = discover_snowball(None, config=cfg, root=tmp_path, records=[rec], call_model=boom)
    assert result.diagnostics["per_extractor"]["text_citations"] == 0
    assert result.diagnostics["leads"] == []


def test_normal_document_llm_stage_runs_when_emitted(tmp_path: Path) -> None:
    data = valid_record() | {"id": "normal-doc", "entity_id": "me", "track": "montenegro"}
    rec = schema.SourceRecord.model_validate(data)
    # raw.* обязателен — discover_snowball скипает документ БЕЗ него ещё до LLM-гейта
    # (тот же ранний "нечего майнить" скип, что у pdf/html/md-экстракторов).
    write_doc(tmp_path, data, raw=_EMPTY_PDF, md=_REFERENCES_MD, state={"sha256": "a" * 64})

    fake = _FakeModel(
        {"citations": [{"title": "Cyber Security Strategy 2025", "url": "https://gov.example/strategy.pdf"}]}
    )
    from discovery.connectors import snowball as sb

    cfg = sb.SnowballConfig(
        enabled=True,
        source_filter=sb.SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=()),
        url_filter=sb.UrlFilter(exclude_domains=(), exclude_url_substrings=()),
        emit=sb.EmitConfig(pdf_annotations=True, html_hrefs=True, printed_urls=True, text_citations=True),
        max_candidates=None,
        citations_model="test/model",
    )
    result = discover_snowball(None, config=cfg, root=tmp_path, records=[rec], call_model=fake)
    urls = {c.source_url for c in result.candidates}
    assert "https://gov.example/strategy.pdf" in urls
    assert fake.calls == 1


# --- save_leads ---


def test_save_leads_writes_yaml_list(tmp_path: Path) -> None:
    leads = [{"title": "X", "issuer": None, "year": 2024, "source_doc_id": "doc-1", "context": "..."}]
    save_leads(leads, tmp_path)
    path = tmp_path / ".snowball_leads.yaml"
    assert path.exists()
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded == leads


def test_save_leads_overwrites_not_appends(tmp_path: Path) -> None:
    save_leads([{"title": "First run lead"}], tmp_path)
    save_leads([{"title": "Second run lead"}], tmp_path)
    import yaml

    loaded = yaml.safe_load((tmp_path / ".snowball_leads.yaml").read_text(encoding="utf-8"))
    assert loaded == [{"title": "Second run lead"}]


def test_save_leads_empty_list_still_writes_file(tmp_path: Path) -> None:
    save_leads([], tmp_path)
    assert (tmp_path / ".snowball_leads.yaml").exists()


# --- дополнительное покрытие: дозаливка после первого замера --cov (coverage discipline,
# /tech-spec §Тестовое покрытие) — плотность до этого места пропускала несколько реальных
# веток, не только happy path. ---


def test_find_citation_sections_dense_block_flushes_at_end_of_text_without_trailing_short_line() -> None:
    """Плотный блок, доходящий до САМОГО КОНЦА текста (без завершающей короткой строки,
    закрывающей run изнутри цикла) — блок должен зафиксироваться post-loop флашем, не
    потеряться."""
    text = "\n".join(
        [
            "Smith, J. (2024). A Long Enough Report Title Here.",
            "Doe, A. (2023). Another Long Enough Citation Line.",
            "Lee, K. (2022). Yet Another Sufficiently Long Citation.",
        ]
    )
    sections = find_citation_sections(text)
    assert len(sections) == 1
    assert "Lee, K. (2022)" in sections[0]


def test_extract_text_citations_corrupt_cache_yaml_treated_as_empty(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(_REFERENCES_MD, encoding="utf-8")
    (tmp_path / ".citations.yaml").write_text("{{{not: valid: yaml", encoding="utf-8")

    fake = _FakeModel({"citations": []})
    links, leads = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=fake)

    assert links == []
    assert leads == []
    assert fake.calls == 1  # повреждённый кэш == отсутствующий, модель ВСЁ РАВНО вызывается


def test_extract_text_citations_non_dict_items_in_citations_array_are_skipped(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(_REFERENCES_MD, encoding="utf-8")
    fake = _FakeModel({"citations": ["not a dict", 42, None, {"title": "Cyber Security Strategy 2025", "url": None}]})

    links, leads = extract_text_citations(md_path, doc_id="doc-1", model="test/model", call_model=fake)

    assert links == []
    assert len(leads) == 1
    assert leads[0].title == "Cyber Security Strategy 2025"


def test_default_call_model_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _default_call_model({"model": "test/model", "messages": []})


def test_discover_snowball_mines_html_document_end_to_end(tmp_path: Path) -> None:
    """discover_snowball на HTML-документе (не только PDF, как остальные cursor-тесты) —
    href-экстрактор реально подключается через основной оркестрационный путь, не только
    напрямую через extract_html_href_links."""
    data = valid_record() | {
        "id": "html-e2e-doc",
        "entity_id": "me",
        "track": "montenegro",
        "source_url": "https://gov.example/page",
    }
    rec = schema.SourceRecord.model_validate(data)
    write_doc(
        tmp_path,
        data,
        raw=b'<html><body><a href="https://example.org/found-via-html">a link</a></body></html>',
        raw_ext="html",
        md="no printed urls in md",
        state={"sha256": "a" * 64},
    )
    from discovery.connectors import snowball as sb

    cfg = sb.SnowballConfig(
        enabled=True,
        source_filter=sb.SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=()),
        url_filter=sb.UrlFilter(exclude_domains=(), exclude_url_substrings=()),
        emit=sb.EmitConfig(pdf_annotations=True, html_hrefs=True, printed_urls=True, text_citations=False),
        max_candidates=None,
        citations_model="test/model",
    )
    result = discover_snowball(None, config=cfg, root=tmp_path, records=[rec])
    urls = {c.source_url for c in result.candidates}
    assert "https://example.org/found-via-html" in urls
    assert result.diagnostics["per_extractor"]["html_hrefs"] == 1
