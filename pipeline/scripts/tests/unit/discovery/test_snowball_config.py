"""Тесты discovery/connectors/snowball.py — конфиг (spec discovery-snowball §3, коммит 1).

Экстракторы/маппинг/курсор/регистрация — последующие коммиты (см. spec «План коммитов»).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from core import schema
from discovery.connectors import snowball

# --- ConnectorKind.snowball ---


def test_connector_kind_snowball_round_trips_as_string() -> None:
    assert schema.ConnectorKind.snowball.value == "snowball"
    assert schema.ConnectorKind("snowball") is schema.ConnectorKind.snowball


# --- load_config: реальный трекаемый файл ---


def test_load_config_reads_real_tracked_config() -> None:
    """pipeline/config/discovery_snowball.yaml — настоящий трекаемый файл, не фикстура."""
    config = snowball.load_config()
    assert config.enabled is True
    assert config.source_filter.tracks == ()
    assert config.source_filter.target_fit == ()
    assert config.source_filter.include_doc_ids == ()
    assert config.source_filter.exclude_doc_ids == ()
    assert config.url_filter.exclude_domains == ()
    assert config.url_filter.exclude_url_substrings == ()
    assert config.emit.pdf_annotations is True
    assert config.emit.html_hrefs is True
    assert config.emit.printed_urls is True
    assert config.emit.text_citations is False
    assert config.max_candidates is None
    assert config.citations_model == "deepseek/deepseek-v4-flash"


# --- load_config: кастомный путь, дефолты вложенных секций ---


def _write_config(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "discovery_snowball.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_load_config_custom_path_full(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "enabled": False,
            "source_filter": {
                "tracks": ["montenegro"],
                "target_fit": ["primary"],
                "include_doc_ids": ["me-crps-registration-law-2025"],
                "exclude_doc_ids": ["eu-ai-act-2024"],
            },
            "url_filter": {
                "exclude_domains": ["example.com"],
                "exclude_url_substrings": ["/blog/"],
            },
            "emit": {
                "pdf_annotations": False,
                "html_hrefs": False,
                "printed_urls": False,
                "text_citations": True,
            },
            "max_candidates": 50,
            "citations_model": "test/model",
        },
    )
    config = snowball.load_config(path)
    assert config.enabled is False
    assert config.source_filter.tracks == ("montenegro",)
    assert config.source_filter.include_doc_ids == ("me-crps-registration-law-2025",)
    assert config.url_filter.exclude_domains == ("example.com",)
    assert config.emit.pdf_annotations is False
    assert config.emit.text_citations is True
    assert config.max_candidates == 50
    assert config.citations_model == "test/model"


def test_load_config_missing_nested_sections_default_to_permissive(tmp_path: Path) -> None:
    """Отсутствующие ``source_filter``/``url_filter``/``emit``/``max_candidates`` -> дефолты
    §3: пустые (разрешающие) фильтры, все emit-тумблеры включены кроме text_citations,
    без капа — консистентно с философией «нет жёстких дефолтов»."""
    path = _write_config(tmp_path, {"enabled": True, "citations_model": "x/y"})
    config = snowball.load_config(path)
    assert config.source_filter.tracks == ()
    assert config.url_filter.exclude_domains == ()
    assert config.emit.pdf_annotations is True
    assert config.emit.text_citations is False
    assert config.max_candidates is None


# --- max_candidates sanity-чек ---


@pytest.mark.parametrize(
    "value",
    [None, 0, 1, 50, 10_000],
)
def test_max_candidates_accepts_none_and_nonnegative_ints(tmp_path: Path, value: int | None) -> None:
    path = _write_config(tmp_path, {"enabled": True, "citations_model": "x/y", "max_candidates": value})
    config = snowball.load_config(path)
    assert config.max_candidates == value


@pytest.mark.parametrize(
    "value",
    [-1, -100, True, False, "5", 3.5, [1]],
)
def test_max_candidates_rejects_invalid_values(tmp_path: Path, value: Any) -> None:
    path = _write_config(tmp_path, {"enabled": True, "citations_model": "x/y", "max_candidates": value})
    with pytest.raises(ValueError, match="max_candidates"):
        snowball.load_config(path)
