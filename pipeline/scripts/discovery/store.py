"""discovery/store.py — персист слоя кандидатов + курсоров (spec discovery-core §4).

Оба файла машиннописаные (комментарии не выживают перезапись) — тот же прецедент, что
у ``.state.yaml`` (corpus-layout-v2): производные/операционные артефакты, не курируемые
человеком напрямую.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core import fsio, schema

CANDIDATES_PATH = schema.DEFAULT_SOURCES / "candidates.yaml"
CURSORS_PATH = schema.DEFAULT_SOURCES / ".discovery_cursors.yaml"
"""Dot-файл — операционное состояние (курсоры), не данные; вне git по deny-default `/sources/**`."""


def load(path: Path = CANDIDATES_PATH) -> list[schema.CandidateRecord]:
    """Слой кандидатов; отсутствующий файл — пустой корпус кандидатов, не ошибка."""
    if not path.exists():
        return []
    return schema.load_candidates(path)


def save(candidates: list[schema.CandidateRecord], path: Path = CANDIDATES_PATH) -> None:
    """Атомарный полный перезапись store — не diff/append (список умещается в памяти)."""
    payload = [c.model_dump(mode="json", exclude_none=True) for c in candidates]
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    fsio.atomic_write_text(path, text)


def load_cursors(path: Path = CURSORS_PATH) -> dict[str, dict[str, Any]]:
    """``connector_id -> ConnectorCursor``; отсутствующий файл — пустой словарь (первый прогон)."""
    if not path.exists():
        return {}
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if raw is not None else {}


def save_cursors(cursors: dict[str, dict[str, Any]], path: Path = CURSORS_PATH) -> None:
    text = yaml.safe_dump(cursors, allow_unicode=True, sort_keys=False)
    fsio.atomic_write_text(path, text)
