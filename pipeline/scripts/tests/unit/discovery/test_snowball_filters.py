"""Тесты discover.py::_build_snowball_config_override (CLI-подкоманда `snowball`, spec
discovery-snowball §3, коммит 5) + индивидуальные emit-тумблеры pdf/html (printed_urls —
уже покрыт в test_snowball_cursor.py, коммит 4)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import discover
import pytest

from core import schema
from discovery.connectors import snowball
from tests.support import build_pdf, valid_record, write_doc


def _base_config() -> snowball.SnowballConfig:
    return snowball.SnowballConfig(
        enabled=True,
        source_filter=snowball.SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=()),
        url_filter=snowball.UrlFilter(exclude_domains=(), exclude_url_substrings=()),
        emit=snowball.EmitConfig(
            pdf_annotations=True, html_hrefs=True, printed_urls=True, text_citations=False
        ),
        max_candidates=None,
        citations_model="test/model",
        citations_model_fallback=None,
    )


def _args(**overrides: Any) -> argparse.Namespace:
    defaults = {
        "doc": None,
        "track": None,
        "tier": None,
        "exclude_domain": None,
        "with_citations": False,
        "max_candidates": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def _patched_load_config(monkeypatch: pytest.MonkeyPatch) -> snowball.SnowballConfig:
    base = _base_config()
    monkeypatch.setattr(snowball, "load_config", lambda path=snowball.CONFIG_PATH: base)
    return base


def test_no_flags_returns_yaml_as_is(_patched_load_config: snowball.SnowballConfig) -> None:
    merged = discover._build_snowball_config_override(_args())
    assert merged == _patched_load_config


def test_doc_flag_overrides_include_doc_ids(_patched_load_config: snowball.SnowballConfig) -> None:
    merged = discover._build_snowball_config_override(_args(doc=["me-crps-registration-law-2025"]))
    assert merged.source_filter.include_doc_ids == ("me-crps-registration-law-2025",)
    assert merged.source_filter.tracks == ()  # прочие поля source_filter не задеты


def test_track_flag_overrides_tracks(_patched_load_config: snowball.SnowballConfig) -> None:
    merged = discover._build_snowball_config_override(_args(track=["montenegro", "research-papers"]))
    assert merged.source_filter.tracks == ("montenegro", "research-papers")


def test_tier_flag_overrides_target_fit(_patched_load_config: snowball.SnowballConfig) -> None:
    merged = discover._build_snowball_config_override(_args(tier=["primary"]))
    assert merged.source_filter.target_fit == ("primary",)


def test_exclude_domain_flag_overrides_url_filter(_patched_load_config: snowball.SnowballConfig) -> None:
    merged = discover._build_snowball_config_override(_args(exclude_domain=["example.com"]))
    assert merged.url_filter.exclude_domains == ("example.com",)


def test_with_citations_flag_enables_text_citations(_patched_load_config: snowball.SnowballConfig) -> None:
    merged = discover._build_snowball_config_override(_args(with_citations=True))
    assert merged.emit.text_citations is True
    assert merged.emit.pdf_annotations is True  # прочие emit-поля не задеты


def test_max_candidates_flag_overrides_yaml_value(_patched_load_config: snowball.SnowballConfig) -> None:
    merged = discover._build_snowball_config_override(_args(max_candidates=25))
    assert merged.max_candidates == 25


def test_max_candidates_flag_zero_is_respected_not_treated_as_falsy(
    _patched_load_config: snowball.SnowballConfig,
) -> None:
    """``0`` — валидное значение капа («эмитировать ничего»), не «флаг не задан»
    (``args.max_candidates is None`` — правильная проверка, не bool-truthiness)."""
    merged = discover._build_snowball_config_override(_args(max_candidates=0))
    assert merged.max_candidates == 0


def test_cli_override_does_not_mutate_yaml_on_disk(tmp_path: Path) -> None:
    """CLI-override — эффект на один прогон; файл `discovery_snowball.yaml` на диске
    остаётся байт-в-байт нетронутым (спек §3)."""
    config_path = tmp_path / "discovery_snowball.yaml"
    config_path.write_text(
        "enabled: true\ncitations_model: test/model\nmax_candidates: null\n", encoding="utf-8"
    )
    before = config_path.read_bytes()

    import dataclasses

    real_load = snowball.load_config
    merged = dataclasses.replace(real_load(config_path), max_candidates=99)
    assert merged.max_candidates == 99
    assert config_path.read_bytes() == before


# --- emit-тумблеры: pdf_annotations / html_hrefs индивидуально (printed_urls — commit 4) ---


def test_emit_toggle_disables_pdf_annotations_extractor(tmp_path: Path) -> None:
    data = valid_record() | {"id": "emit-pdf-off-doc", "entity_id": "me", "track": "montenegro"}
    rec = schema.SourceRecord.model_validate(data)
    raw_bytes = build_pdf(
        lines=[("A link", 50.0, 60.0, 12.0)],
        links=[("https://example.org/pdf-annotation-link", 50.0, 55.0, 300.0, 80.0)],
    )
    write_doc(tmp_path, data, raw=raw_bytes, md="no printed urls in md text", state={"sha256": "a" * 64})

    cfg_off = snowball.SnowballConfig(
        enabled=True,
        source_filter=snowball.SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=()),
        url_filter=snowball.UrlFilter(exclude_domains=(), exclude_url_substrings=()),
        emit=snowball.EmitConfig(
            pdf_annotations=False, html_hrefs=True, printed_urls=True, text_citations=False
        ),
        max_candidates=None,
        citations_model="test/model",
        citations_model_fallback=None,
    )
    result = snowball.discover_snowball(None, config=cfg_off, root=tmp_path, records=[rec])
    assert result.candidates == []
    assert result.diagnostics["per_extractor"]["pdf_annotations"] == 0


def test_emit_toggle_disables_html_hrefs_extractor(tmp_path: Path) -> None:
    data = valid_record() | {
        "id": "emit-html-off-doc",
        "entity_id": "me",
        "track": "montenegro",
        "source_url": "https://gov.example/page",
    }
    rec = schema.SourceRecord.model_validate(data)
    write_doc(
        tmp_path,
        data,
        raw=b'<a href="https://example.org/html-href-link">a link</a>',
        raw_ext="html",
        md="no printed urls here",
        state={"sha256": "a" * 64},
    )

    cfg_off = snowball.SnowballConfig(
        enabled=True,
        source_filter=snowball.SourceFilter(tracks=(), target_fit=(), include_doc_ids=(), exclude_doc_ids=()),
        url_filter=snowball.UrlFilter(exclude_domains=(), exclude_url_substrings=()),
        emit=snowball.EmitConfig(
            pdf_annotations=True, html_hrefs=False, printed_urls=True, text_citations=False
        ),
        max_candidates=None,
        citations_model="test/model",
        citations_model_fallback=None,
    )
    result = snowball.discover_snowball(None, config=cfg_off, root=tmp_path, records=[rec])
    assert result.candidates == []
    assert result.diagnostics["per_extractor"]["html_hrefs"] == 0
