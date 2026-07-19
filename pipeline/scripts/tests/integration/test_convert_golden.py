"""Golden-самосверка конвертации: свежая конвертация каждого raw.* корпуса + офлайн-
реинъекция фигур из кэша должна байт-в-байт совпадать с телом уже лежащего doc.md.

Хранимых голден-файлов нет: тела документов не версионируются в git (deny-default
.gitignore, копирайт), поэтому эталон — сам локальный doc.md. Требует локальный
корпус (sources/); в CI пропускается (@pytest.mark.corpus).

С convert-cloud-tier эталон — двухшаговая детерминированная функция (спек §7):
``doc.md = f(raw, .cloudocr.md, .figures.yaml, версия конвертера)`` — свежая
конвертация регенерирует ГОЛЫЕ маркеры фигур, а ``apply_figures_pass`` реинъецирует
их из тёплого кэша офлайн (ключ не нужен — ленивая проверка). Голден ОБЯЗАН быть
офлайн: любое касание сети (невалидный кэш -> реальный платный вызов, наблюдение
приёмки чекпоинта 1) — громкий отказ через monkeypatch ``chat_request``, а не
тихий счёт за конвертацию в тестовом прогоне.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from convert import converters, figures_vlm
from convert.cloud_ocr import ACTIVE_MODEL
from core import schema
from core.schema import SourceRecord
from index.chunking import strip_frontmatter

pytestmark = pytest.mark.corpus


def _corpus_entries() -> list[tuple[str, SourceRecord, Path, Path]]:
    """(doc_id, record, raw, md) для записей с локальными raw.* и doc.md."""
    if not schema.DEFAULT_SOURCES.exists():
        return []
    out: list[tuple[str, SourceRecord, Path, Path]] = []
    for rec in schema.load_records(schema.DEFAULT_SOURCES):
        raw = schema.raw_file(rec, schema.DEFAULT_SOURCES)
        md = schema.md_file(rec, schema.DEFAULT_SOURCES)
        if raw is not None and raw.exists() and md.exists():
            out.append((rec.id, rec, raw, md))
    return out


_ENTRIES = _corpus_entries()


@pytest.mark.parametrize(
    "doc_id,rec,raw,md", _ENTRIES, ids=[e[0] for e in _ENTRIES]
)
def test_reconversion_matches_doc_md(
    doc_id: str, rec: SourceRecord, raw: Path, md: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "core.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError(
                f"{doc_id}: голден коснулся сети — кэш (.cloudocr.md/.figures.yaml) "
                f"невалиден или отсутствует; реальный платный вызов в тестовом прогоне запрещён"
            )
        ),
    )
    conv = converters.resolve_converter(raw)
    out = tmp_path / "out.md"
    conv.convert(raw, out, rec.language, record=rec)
    figures_vlm.apply_figures_pass(out, raw, model=ACTIVE_MODEL)
    expected = strip_frontmatter(md.read_text(encoding="utf-8")).lstrip("\n")
    assert out.read_text(encoding="utf-8") == expected
