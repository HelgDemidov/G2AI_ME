"""Тесты discovery/connectors/snowball.py — конфиг (spec discovery-snowball §3, коммит 1).

Экстракторы/маппинг/курсор/регистрация — последующие коммиты (см. spec «План коммитов»).
"""
from __future__ import annotations

import dataclasses
import inspect
import re
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


# --- orphan-ключ конфига (spec drop-relevance-tier §7; урок Г10 agora, PR #54) ---

_SNOWBALL_CONFIG_DATACLASSES = (
    snowball.SourceFilter,
    snowball.UrlFilter,
    snowball.EmitConfig,
    snowball.SnowballConfig,
)


def test_every_snowball_config_field_is_used_outside_load_config() -> None:
    """Урок orphan-конфига (agora, Г10: удалённый ``non_us_include_all`` не читался ни
    одной веткой) распространён на ``SnowballConfig`` — здесь вложенный (``source_filter``/
    ``url_filter``/``emit`` — отдельные dataclass'ы), поэтому скан обходит ВСЕ четыре секции,
    не копирует агоровский плоский тест буквально. Определения секций и ``load_config`` —
    единственные легитимные места, где поле упоминается ТОЛЬКО в парсинге/декларации;
    регрессия (новое поле без реального потребителя) должна ломать тест, а не проходить
    ревью молча. Честная граница (как у агоровского прецедента): текстовый скан, не AST —
    совпадение имени поля в прозе/докстроке даёт ложноотрицательный результат теоретически,
    но ни разу не наблюдалось на реальных полях этого модуля."""
    module_source = inspect.getsource(snowball)
    outside = module_source
    for dc in _SNOWBALL_CONFIG_DATACLASSES:
        outside = outside.replace(inspect.getsource(dc), "")
    outside = outside.replace(inspect.getsource(snowball.load_config), "")

    orphaned = [
        f"{dc.__name__}.{field.name}"
        for dc in _SNOWBALL_CONFIG_DATACLASSES
        for field in dataclasses.fields(dc)
        if not re.search(rf"\b{re.escape(field.name)}\b", outside)
    ]
    assert not orphaned, f"поля SnowballConfig не используются вне load_config: {orphaned}"


# --- load_config: реальный трекаемый файл ---


def test_load_config_reads_real_tracked_config() -> None:
    """pipeline/config/discovery_snowball.yaml — настоящий трекаемый файл, не фикстура."""
    config = snowball.load_config()
    assert config.enabled is True
    assert config.source_filter.tracks == ()
    assert config.source_filter.include_doc_ids == ()
    assert config.source_filter.exclude_doc_ids == ()
    assert config.url_filter.exclude_domains == ()
    assert config.url_filter.exclude_url_substrings == ()
    assert config.emit.pdf_annotations is True
    assert config.emit.html_hrefs is True
    assert config.emit.printed_urls is True
    assert config.emit.text_citations is False
    assert config.citations_model == "minimax/minimax-m3"
    assert config.citations_model_fallback == "google/gemini-3-flash-preview"


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
                "tracks": ["target-entity"],
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
            "citations_model": "test/model",
            "citations_model_fallback": "test/fallback-model",
        },
    )
    config = snowball.load_config(path)
    assert config.enabled is False
    assert config.source_filter.tracks == ("target-entity",)
    assert config.source_filter.include_doc_ids == ("me-crps-registration-law-2025",)
    assert config.url_filter.exclude_domains == ("example.com",)
    assert config.emit.pdf_annotations is False
    assert config.emit.text_citations is True
    assert config.citations_model == "test/model"
    assert config.citations_model_fallback == "test/fallback-model"


def test_load_config_missing_nested_sections_default_to_permissive(tmp_path: Path) -> None:
    """Отсутствующие ``source_filter``/``url_filter``/``emit`` -> дефолты
    §3: пустые (разрешающие) фильтры, все emit-тумблеры включены кроме text_citations,
    без капа — консистентно с философией «нет жёстких дефолтов»."""
    path = _write_config(tmp_path, {"enabled": True, "citations_model": "x/y"})
    config = snowball.load_config(path)
    assert config.source_filter.tracks == ()
    assert config.url_filter.exclude_domains == ()
    assert config.emit.pdf_annotations is True
    assert config.emit.text_citations is False
    assert config.citations_model_fallback is None


@pytest.mark.parametrize("raw_value", [None, ""])
def test_citations_model_fallback_falsy_yaml_value_is_none(tmp_path: Path, raw_value: str | None) -> None:
    """``citations_model_fallback: null`` (или пустая строка) -> ``None`` (fallback
    отключён) — не строка ``"None"``/пустая строка, которую попытались бы использовать
    как реальный слаг модели."""
    path = _write_config(
        tmp_path, {"enabled": True, "citations_model": "x/y", "citations_model_fallback": raw_value}
    )
    config = snowball.load_config(path)
    assert config.citations_model_fallback is None


