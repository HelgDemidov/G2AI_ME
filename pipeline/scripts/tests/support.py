"""Общие фабрики тестовых данных: валидная запись корпуса + папка-документ на диске.

Выделены из test_schema.py при введении слоевой иерархии тестов (feature/repo-layout):
хелперы, разделяемые тестами разных слоёв, живут в support-модуле пакета tests,
а не импортируются из чужого тест-файла.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def valid_record() -> dict[str, Any]:
    """Минимально валидная запись (термины — из реальных словарей pipeline/vocab/)."""
    return {
        "id": "sg-imda-mgf-agentic-2026",
        "entity_id": "sg",
        "track": "intl-xperience",
        "title": "Model AI Governance Framework for Agentic AI",
        "issuer": "Infocomm Media Development Authority (IMDA)",
        "issuer_type": "government",
        "geo_scope": "national",
        "language": "en",
        "dates": {"published": "2026-05-20", "retrieved": "2026-07-15"},
        "doc_type": "framework",
        "authority": "soft_law",
        "topics": ["ai-governance", "agentic-ai"],
        "g2ai_pattern": ["agent-governance-framework"],
        "source_url": "https://example.org/doc.pdf",
        "relevance": {
            "target_fit": "primary",
            "axis": "agentic_g2ai",
            "assessed_stage": "confirmed",
            "rationale": "эталонный агентный G2AI-документ",
            "assessed_date": "2026-07-15",
        },
    }


def write_doc(
    root: Path,
    rec: dict[str, Any],
    *,
    raw: bytes | None = None,
    md: str | None = None,
    state: dict[str, Any] | None = None,
) -> Path:
    """Создать папку-документ sources/<track>/<entity>/<id>/ + meta.yaml (+ raw.pdf/doc.md/.state.yaml)."""
    d = root / str(rec["track"]) / str(rec["entity_id"]) / str(rec["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.yaml").write_text(yaml.safe_dump(rec, allow_unicode=True), encoding="utf-8")
    if raw is not None:
        (d / "raw.pdf").write_bytes(raw)
    if md is not None:
        (d / "doc.md").write_text(md, encoding="utf-8")
    if state is not None:
        (d / ".state.yaml").write_text(yaml.safe_dump(state, allow_unicode=True), encoding="utf-8")
    return d
