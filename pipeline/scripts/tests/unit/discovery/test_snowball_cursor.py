"""Тесты discovery/connectors/snowball.py — discover_snowball() целиком: полная выдача,
кэш сырья, фильтры источника/URL, диагностика (spec discovery-snowball §3/§4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import schema
from discovery.connectors import snowball
from discovery.connectors.snowball import (
    EmitConfig,
    SnowballConfig,
    SourceFilter,
    UrlFilter,
    discover_snowball,
)
from tests.support import build_pdf, valid_record, write_doc

_PERMISSIVE_FILTER = SourceFilter(tracks=(), include_doc_ids=(), exclude_doc_ids=())
_PERMISSIVE_URL_FILTER = UrlFilter(exclude_domains=(), exclude_url_substrings=())


def _config() -> SnowballConfig:
    return SnowballConfig(
        enabled=True,
        source_filter=_PERMISSIVE_FILTER,
        url_filter=_PERMISSIVE_URL_FILTER,
        emit=EmitConfig(pdf_annotations=True, html_hrefs=True, printed_urls=True, text_citations=False),
        citations_model="test/model",
        citations_model_fallback=None,
    )


def _pdf_with_link(url: str, anchor: str) -> bytes:
    return build_pdf(
        lines=[(anchor, 50.0, 60.0, 12.0)],
        links=[(url, 50.0, 55.0, 300.0, 80.0)],
    )


def _seed_doc(root: Path, *, doc_id: str, raw_sha: str, links: list[tuple[str, str]]) -> schema.SourceRecord:
    data = valid_record() | {"id": doc_id, "entity_id": "me", "track": "target-entity"}
    rec = schema.SourceRecord.model_validate(data)
    raw_bytes = build_pdf(
        lines=[(anchor, 50.0 + i * 0, 60.0 + i * 60.0, 12.0) for i, (_, anchor) in enumerate(links)],
        links=[(url, 50.0, 55.0 + i * 60.0, 300.0, 80.0 + i * 60.0) for i, (url, _) in enumerate(links)],
    )
    write_doc(root, data, raw=raw_bytes, md=f"# {doc_id}\n\nNo printed URLs here.\n", state={"sha256": raw_sha})
    return rec


def test_document_without_raw_or_doc_md_is_skipped_not_errored(tmp_path: Path) -> None:
    data = valid_record() | {"id": "no-raw-doc", "entity_id": "me", "track": "target-entity"}
    write_doc(tmp_path, data)  # только meta.yaml — ни raw, ни doc.md
    rec = schema.SourceRecord.model_validate(data)
    result = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=tmp_path / "snowball_cache")
    assert result.candidates == []
    assert result.diagnostics["docs_scanned"] == 0


def test_first_run_mines_document(tmp_path: Path) -> None:
    rec = _seed_doc(tmp_path, doc_id="first-run-doc", raw_sha="a" * 64, links=[("https://example.org/a", "Doc A")])
    result = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=tmp_path / "snowball_cache")
    assert len(result.candidates) == 1
    assert result.candidates[0].source_url == "https://example.org/a"
    assert result.diagnostics["docs_scanned"] == 1


def test_second_run_returns_same_full_view(tmp_path: Path) -> None:
    """Коннектор — чистая функция от ИСТОЧНИКА (здесь источник = собственный корпус):
    повторный прогон отдаёт ТО ЖЕ, документ не «пропускается». Инвариант «повторный
    прогон — no-op» держит ``dedup`` оркестратора, и это проверено в test_orchestrate."""
    rec = _seed_doc(tmp_path, doc_id="noop-doc", raw_sha="a" * 64, links=[("https://example.org/b", "Doc B")])
    first = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=tmp_path / "snowball_cache")
    second = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=tmp_path / "snowball_cache")
    assert [c.raw_hash for c in second.candidates] == [c.raw_hash for c in first.candidates]
    assert second.diagnostics["docs_scanned"] == 1


def test_self_link_and_corpus_link_are_excluded_end_to_end(tmp_path: Path) -> None:
    other_data = valid_record() | {
        "id": "other-existing-doc",
        "entity_id": "me",
        "track": "target-entity",
        "source_url": "https://example.org/other-doc",
    }
    other_rec = schema.SourceRecord.model_validate(other_data)

    self_data = valid_record() | {"id": "self-link-doc", "entity_id": "me", "track": "target-entity"}
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

    result = discover_snowball(config=_config(), root=tmp_path, records=[self_rec, other_rec], cache_dir=tmp_path / "snowball_cache")
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
        citations_model=cfg.citations_model,
        citations_model_fallback=cfg.citations_model_fallback,
    )
    result = discover_snowball(config=cfg, root=tmp_path, records=[rec], cache_dir=tmp_path / "snowball_cache")
    urls = {c.source_url for c in result.candidates}
    assert urls == {"https://gov.example.org/law"}
    assert result.diagnostics["filtered_by_url_filter"] == 1


def test_emit_toggle_disables_printed_urls_extractor(tmp_path: Path) -> None:
    data = valid_record() | {"id": "emit-toggle-doc", "entity_id": "me", "track": "target-entity"}
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
        citations_model="test/model",
        citations_model_fallback=None,
    )
    result_off = discover_snowball(config=cfg_off, root=tmp_path, records=[rec], cache_dir=tmp_path / "snowball_cache")
    assert result_off.candidates == []

    result_on = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=tmp_path / "snowball_cache")
    assert len(result_on.candidates) == 1


# --- кэш СЫРЬЯ вместо курсора (spec drop-cursors-and-decision-overlay §2) ---


def test_cold_run_extracts_and_writes_cache(tmp_path: Path) -> None:
    rec = _seed_doc(tmp_path, doc_id="cold-doc", raw_sha="a" * 64, links=[("https://example.org/a", "Doc A")])
    cache_dir = tmp_path / "snowball_cache"

    result = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=cache_dir)

    assert result.diagnostics["cache_misses"] == 1
    assert result.diagnostics["cache_hits"] == 0
    assert snowball.raw_cache_path(rec.id, cache_dir).exists()


def test_warm_run_returns_same_output_without_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ключевое отличие кэша от курсора: подавляется РАБОТА, не выдача. Тёплый прогон
    отдаёт ТО ЖЕ, а не пустоту, — поэтому обогащение очереди продолжает течь."""
    rec = _seed_doc(tmp_path, doc_id="warm-doc", raw_sha="a" * 64, links=[("https://example.org/b", "Doc B")])
    cache_dir = tmp_path / "snowball_cache"
    first = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=cache_dir)

    def boom(*a: object, **kw: object) -> list[snowball.RawLink]:
        raise AssertionError("тёплый кэш не должен извлекать сырьё заново")

    monkeypatch.setattr(snowball, "extract_pdf_annotation_links", boom)
    monkeypatch.setattr(snowball, "extract_printed_urls", boom)
    second = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=cache_dir)

    assert second.diagnostics["cache_hits"] == 1
    assert [c.raw_hash for c in second.candidates] == [c.raw_hash for c in first.candidates]


def test_changed_doc_md_invalidates_only_its_own_cache_entry(tmp_path: Path) -> None:
    data = valid_record() | {"id": "invalidate-doc", "entity_id": "me", "track": "target-entity"}
    rec = schema.SourceRecord.model_validate(data)
    other = _seed_doc(tmp_path, doc_id="untouched-doc", raw_sha="b" * 64, links=[("https://example.org/x", "X")])
    raw_bytes = _pdf_with_link("https://example.org/c", "Doc C")
    write_doc(tmp_path, data, raw=raw_bytes, md="version one", state={"sha256": "a" * 64})
    cache_dir = tmp_path / "snowball_cache"
    discover_snowball(config=_config(), root=tmp_path, records=[rec, other], cache_dir=cache_dir)

    write_doc(tmp_path, data, raw=raw_bytes, md="version TWO, changed", state={"sha256": "a" * 64})
    second = discover_snowball(config=_config(), root=tmp_path, records=[rec, other], cache_dir=cache_dir)

    assert second.diagnostics["cache_misses"] == 1  # только изменившийся
    assert second.diagnostics["cache_hits"] == 1


def test_mapping_change_reaches_queue_on_warm_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Кэшируется ДОБЫЧА, никогда маппинг: правка ``map_link`` доезжает до всей очереди
    следующим прогоном на тёплом кэше, без бампа версий и миграций. Это же снимает нужду
    в ``Connector.version``, обязательном при кэшировании результата."""
    rec = _seed_doc(tmp_path, doc_id="mapping-doc", raw_sha="a" * 64, links=[("https://example.org/m", "Doc M")])
    cache_dir = tmp_path / "snowball_cache"
    discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=cache_dir)

    real_map = snowball.map_link

    def patched_map(link: snowball.RawLink, **kw: object) -> schema.CandidateRecord:
        cand = real_map(link, **kw)  # type: ignore[arg-type]
        cand.native_tags = [*(cand.native_tags or []), "new-mapping-field"]
        return cand

    monkeypatch.setattr(snowball, "map_link", patched_map)
    second = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=cache_dir)

    assert second.diagnostics["cache_hits"] == 1  # сырьё из кэша...
    assert "new-mapping-field" in (second.candidates[0].native_tags or [])  # ...маппинг свежий


def test_emit_toggle_invalidates_cache(tmp_path: Path) -> None:
    """``emit`` управляет ДОБЫЧЕЙ, поэтому входит в ключ кэша наравне с fingerprint: иначе
    прогон с выключенным каналом записал бы обеднённое сырьё, а следующий с включённым
    тихо отдал бы его же."""
    rec = _seed_doc(tmp_path, doc_id="emit-doc", raw_sha="a" * 64, links=[("https://example.org/e", "Doc E")])
    cache_dir = tmp_path / "snowball_cache"
    cfg_off = SnowballConfig(
        enabled=True,
        source_filter=_PERMISSIVE_FILTER,
        url_filter=_PERMISSIVE_URL_FILTER,
        emit=EmitConfig(pdf_annotations=False, html_hrefs=True, printed_urls=True, text_citations=False),
        citations_model="test/model",
        citations_model_fallback=None,
    )

    off = discover_snowball(config=cfg_off, root=tmp_path, records=[rec], cache_dir=cache_dir)
    on = discover_snowball(config=_config(), root=tmp_path, records=[rec], cache_dir=cache_dir)

    assert off.candidates == []
    assert on.diagnostics["cache_misses"] == 1  # смена канала — не попадание в кэш
    assert len(on.candidates) == 1
