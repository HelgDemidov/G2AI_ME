"""Golden-самосверка конвертации: свежая конвертация каждого raw.* корпуса должна
байт-в-байт совпадать с телом уже лежащего на диске doc.md.

Хранимых голден-файлов нет: тела документов не версионируются в git (deny-default
.gitignore, копирайт), поэтому эталон — сам локальный doc.md. Требует локальный
корпус (sources/); в CI пропускается (@pytest.mark.corpus).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from convert import converters
from core import schema
from index.chunking import strip_frontmatter

pytestmark = pytest.mark.corpus


def _corpus_pairs() -> list[tuple[str, Path, Path, str]]:
    """(doc_id, raw, md, language) для записей с локальными raw.* и doc.md."""
    if not schema.DEFAULT_SOURCES.exists():
        return []
    out: list[tuple[str, Path, Path, str]] = []
    for rec in schema.load_records(schema.DEFAULT_SOURCES):
        raw = schema.raw_file(rec, schema.DEFAULT_SOURCES)
        md = schema.md_file(rec, schema.DEFAULT_SOURCES)
        if raw is not None and raw.exists() and md.exists():
            out.append((rec.id, raw, md, rec.language))
    return out


_PAIRS = _corpus_pairs()


@pytest.mark.parametrize(
    "doc_id,raw,md,language", _PAIRS, ids=[p[0] for p in _PAIRS]
)
def test_reconversion_matches_doc_md(
    doc_id: str, raw: Path, md: Path, language: str, tmp_path: Path
) -> None:
    conv = converters.resolve_converter(raw)
    out = tmp_path / "out.md"
    conv.convert(raw, out, language)
    expected = strip_frontmatter(md.read_text(encoding="utf-8")).lstrip("\n")
    assert out.read_text(encoding="utf-8") == expected
