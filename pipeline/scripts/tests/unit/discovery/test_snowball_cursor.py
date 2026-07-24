"""Тесты discovery/connectors/snowball.py — discover_snowball() целиком: курсор/fingerprint-
скип, реконсиляционный no-op, max_candidates урезание, диагностика (spec discovery-snowball
§3/§4, коммит 4)."""
from __future__ import annotations

from pathlib import Path

from core import schema
from discovery.connectors.snowball import (
    EmitConfig,
    SnowballConfig,
    SourceFilter,
    UrlFilter,
    discover_snowball,
)
from tests.support import build_pdf, valid_record, write_doc

_PERMISSIVE_FILTER = SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=())
_PERMISSIVE_URL_FILTER = UrlFilter(exclude_domains=(), exclude_url_substrings=())


def _config(*, max_candidates: int | None = None) -> SnowballConfig:
    return SnowballConfig(
        enabled=True,
        source_filter=_PERMISSIVE_FILTER,
        url_filter=_PERMISSIVE_URL_FILTER,
        emit=EmitConfig(pdf_annotations=True, html_hrefs=True, printed_urls=True, text_citations=False),
        max_candidates=max_candidates,
        citations_model="test/model",
    )


def _pdf_with_link(url: str, anchor: str) -> bytes:
    return build_pdf(
        lines=[(anchor, 50.0, 60.0, 12.0)],
        links=[(url, 50.0, 55.0, 300.0, 80.0)],
    )


def _seed_doc(root: Path, *, doc_id: str, raw_sha: str, links: list[tuple[str, str]]) -> schema.SourceRecord:
    data = valid_record() | {"id": doc_id, "entity_id": "me", "track": "montenegro"}
    rec = schema.SourceRecord.model_validate(data)
    raw_bytes = build_pdf(
        lines=[(anchor, 50.0 + i * 0, 60.0 + i * 60.0, 12.0) for i, (_, anchor) in enumerate(links)],
        links=[(url, 50.0, 55.0 + i * 60.0, 300.0, 80.0 + i * 60.0) for i, (url, _) in enumerate(links)],
    )
    write_doc(root, data, raw=raw_bytes, md=f"# {doc_id}\n\nNo printed URLs here.\n", state={"sha256": raw_sha})
    return rec


def test_document_without_raw_or_doc_md_is_skipped_not_errored(tmp_path: Path) -> None:
    data = valid_record() | {"id": "no-raw-doc", "entity_id": "me", "track": "montenegro"}
    write_doc(tmp_path, data)  # только meta.yaml — ни raw, ни doc.md
    rec = schema.SourceRecord.model_validate(data)
    result = discover_snowball(None, config=_config(), root=tmp_path, records=[rec])
    assert result.candidates == []
    assert result.diagnostics["docs_scanned"] == 0


def test_first_run_mines_document_and_records_fingerprint(tmp_path: Path) -> None:
    rec = _seed_doc(tmp_path, doc_id="first-run-doc", raw_sha="a" * 64, links=[("https://example.org/a", "Doc A")])
    result = discover_snowball(None, config=_config(), root=tmp_path, records=[rec])
    assert len(result.candidates) == 1
    assert result.candidates[0].source_url == "https://example.org/a"
    assert rec.id in result.cursor["mined"]
    assert result.diagnostics["docs_scanned"] == 1
    assert result.diagnostics["docs_skipped_cursor"] == 0


def test_second_run_unchanged_corpus_is_noop(tmp_path: Path) -> None:
    rec = _seed_doc(tmp_path, doc_id="noop-doc", raw_sha="a" * 64, links=[("https://example.org/b", "Doc B")])
    first = discover_snowball(None, config=_config(), root=tmp_path, records=[rec])
    second = discover_snowball(first.cursor, config=_config(), root=tmp_path, records=[rec])
    assert second.candidates == []
    assert second.cursor == first.cursor
    assert second.diagnostics["docs_skipped_cursor"] == 1
    assert second.diagnostics["docs_scanned"] == 0


def test_changed_doc_md_triggers_remine(tmp_path: Path) -> None:
    data = valid_record() | {"id": "remine-doc", "entity_id": "me", "track": "montenegro"}
    rec = schema.SourceRecord.model_validate(data)
    raw_bytes = _pdf_with_link("https://example.org/c", "Doc C")
    write_doc(tmp_path, data, raw=raw_bytes, md="version one", state={"sha256": "a" * 64})
    first = discover_snowball(None, config=_config(), root=tmp_path, records=[rec])
    assert len(first.candidates) == 1

    write_doc(tmp_path, data, raw=raw_bytes, md="version TWO, changed", state={"sha256": "a" * 64})
    second = discover_snowball(first.cursor, config=_config(), root=tmp_path, records=[rec])
    # doc.md изменился -> fingerprint изменился -> пере-майнинг (та же ссылка снова найдена,
    # но она уже персистнута кросс-коннекторным dedup'ом на уровне оркестратора — здесь
    # discover_snowball() эмитит её заново, это ожидаемо для чистой функции коннектора)
    assert second.diagnostics["docs_scanned"] == 1
    assert second.diagnostics["docs_skipped_cursor"] == 0


def test_max_candidates_truncates_and_document_not_marked_mined(tmp_path: Path) -> None:
    rec = _seed_doc(
        tmp_path,
        doc_id="capped-doc",
        raw_sha="a" * 64,
        links=[("https://example.org/one", "Doc One"), ("https://example.org/two", "Doc Two")],
    )
    result = discover_snowball(None, config=_config(max_candidates=1), root=tmp_path, records=[rec])
    assert len(result.candidates) == 1
    assert result.diagnostics["truncated_docs"] == 1
    assert result.diagnostics["truncated_candidates"] == 1
    assert rec.id not in result.cursor["mined"]  # урезанный документ не считается domined


def test_max_candidates_not_exceeded_document_marked_mined_normally(tmp_path: Path) -> None:
    rec = _seed_doc(tmp_path, doc_id="under-cap-doc", raw_sha="a" * 64, links=[("https://example.org/x", "X")])
    result = discover_snowball(None, config=_config(max_candidates=5), root=tmp_path, records=[rec])
    assert len(result.candidates) == 1
    assert result.diagnostics["truncated_docs"] == 0
    assert rec.id in result.cursor["mined"]


def test_max_candidates_zero_emits_nothing_but_untouched_doc_not_truncated(tmp_path: Path) -> None:
    """Документ БЕЗ находок при max_candidates=0 — не «урезан» (нечего урезать), fingerprint
    фиксируется нормально; урезание — только когда реально что-то отброшено."""
    data = valid_record() | {"id": "zero-cap-doc", "entity_id": "me", "track": "montenegro"}
    rec = schema.SourceRecord.model_validate(data)
    raw_bytes = build_pdf(lines=[("no links on this page", 50.0, 60.0, 12.0)])
    write_doc(tmp_path, data, raw=raw_bytes, md="no urls here either", state={"sha256": "a" * 64})
    result = discover_snowball(None, config=_config(max_candidates=0), root=tmp_path, records=[rec])
    assert result.candidates == []
    assert result.diagnostics["truncated_docs"] == 0
    assert rec.id in result.cursor["mined"]


def test_self_link_and_corpus_link_are_excluded_end_to_end(tmp_path: Path) -> None:
    other_data = valid_record() | {
        "id": "other-existing-doc",
        "entity_id": "me",
        "track": "montenegro",
        "source_url": "https://example.org/other-doc",
    }
    other_rec = schema.SourceRecord.model_validate(other_data)

    self_data = valid_record() | {"id": "self-link-doc", "entity_id": "me", "track": "montenegro"}
    self_rec = schema.SourceRecord.model_validate(self_data)
    raw_bytes = build_pdf(
        lines=[("self", 50.0, 60.0, 12.0), ("corpus", 50.0, 120.0, 12.0), ("fresh", 50.0, 180.0, 12.0)],
        links=[
            (self_rec.source_url, 50.0, 55.0, 300.0, 80.0),
            (other_rec.source_url, 50.0, 115.0, 300.0, 140.0),
            ("https://example.org/genuinely-new", 50.0, 175.0, 300.0, 200.0),
        ],
    )
    write_doc(tmp_path, self_data, raw=raw_bytes, md="no urls in md", state={"sha256": "a" * 64})

    result = discover_snowball(
        None, config=_config(), root=tmp_path, records=[self_rec, other_rec]
    )
    urls = {c.source_url for c in result.candidates}
    assert urls == {"https://example.org/genuinely-new"}
    assert result.diagnostics["filtered_self_or_corpus"] == 2


def test_url_filter_excludes_matching_domain_end_to_end(tmp_path: Path) -> None:
    rec = _seed_doc(
        tmp_path,
        doc_id="url-filter-doc",
        raw_sha="a" * 64,
        links=[("https://blog.example.com/post", "Blog"), ("https://gov.example.org/law", "Law")],
    )
    cfg = _config()
    cfg = SnowballConfig(
        enabled=cfg.enabled,
        source_filter=cfg.source_filter,
        url_filter=UrlFilter(exclude_domains=("blog.example.com",), exclude_url_substrings=()),
        emit=cfg.emit,
        max_candidates=cfg.max_candidates,
        citations_model=cfg.citations_model,
    )
    result = discover_snowball(None, config=cfg, root=tmp_path, records=[rec])
    urls = {c.source_url for c in result.candidates}
    assert urls == {"https://gov.example.org/law"}
    assert result.diagnostics["filtered_by_url_filter"] == 1


def test_emit_toggle_disables_printed_urls_extractor(tmp_path: Path) -> None:
    data = valid_record() | {"id": "emit-toggle-doc", "entity_id": "me", "track": "montenegro"}
    rec = schema.SourceRecord.model_validate(data)
    raw_bytes = build_pdf(lines=[("no annotation links", 50.0, 60.0, 12.0)])
    write_doc(
        tmp_path,
        data,
        raw=raw_bytes,
        md="See https://example.org/printed-only for details.\n",
        state={"sha256": "a" * 64},
    )
    cfg_off = SnowballConfig(
        enabled=True,
        source_filter=_PERMISSIVE_FILTER,
        url_filter=_PERMISSIVE_URL_FILTER,
        emit=EmitConfig(pdf_annotations=True, html_hrefs=True, printed_urls=False, text_citations=False),
        max_candidates=None,
        citations_model="test/model",
    )
    result_off = discover_snowball(None, config=cfg_off, root=tmp_path, records=[rec])
    assert result_off.candidates == []

    result_on = discover_snowball(None, config=_config(), root=tmp_path, records=[rec])
    assert len(result_on.candidates) == 1
