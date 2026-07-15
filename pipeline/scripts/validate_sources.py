"""Валидатор реестра ``sources.yaml`` G2AI-корпуса.

Проверяет: (1) структуру каждой записи через pydantic-схему; (2) принадлежность
``doc_type``/``authority``/``topics``/``g2ai_pattern`` контролируемым словарям;
(3) уникальность ``id``; (4) ссылочную целостность ``relations`` (цель — существующий id);
(5) наличие ``relevance`` (все записи sources.yaml — допущенные триажем; см.
source-relevance-triage).

Возвращает ненулевой код при ошибках — пригодно для pre-commit и CI.
Запуск::

    python3 validate_sources.py [путь_к_sources.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from schema import VOCAB_DIR, SourceRecord, load_vocab

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / "intl_xperience" / "countries" / "sources.yaml"


def validate_sources(sources_path: Path, vocab_dir: Path = VOCAB_DIR) -> list[str]:
    """Вернуть список сообщений об ошибках (пустой список = реестр валиден)."""
    errors: list[str] = []
    raw: Any = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return [f"{sources_path}: верхний уровень должен быть списком записей"]

    vocabs: dict[str, set[str]] = {
        "doc_type": load_vocab("doc_types", vocab_dir),
        "authority": load_vocab("authority", vocab_dir),
        "topics": load_vocab("topics", vocab_dir),
        "g2ai_pattern": load_vocab("g2ai_patterns", vocab_dir),
    }

    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        try:
            rec = SourceRecord.model_validate(item)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"запись #{i}: {loc}: {err['msg']}")
            continue

        if rec.id in seen_ids:
            errors.append(f"запись '{rec.id}': дубль id")
        seen_ids.add(rec.id)

        if rec.doc_type not in vocabs["doc_type"]:
            errors.append(f"запись '{rec.id}': doc_type '{rec.doc_type}' вне словаря")
        if rec.authority not in vocabs["authority"]:
            errors.append(f"запись '{rec.id}': authority '{rec.authority}' вне словаря")
        for topic in rec.topics:
            if topic not in vocabs["topics"]:
                errors.append(f"запись '{rec.id}': topic '{topic}' вне словаря")
        for pattern in rec.g2ai_pattern:
            if pattern not in vocabs["g2ai_pattern"]:
                errors.append(f"запись '{rec.id}': g2ai_pattern '{pattern}' вне словаря")

        if rec.relevance is None:
            errors.append(
                f"запись '{rec.id}': отсутствует relevance "
                "(обязателен для допущенной записи sources.yaml — прошла триаж)"
            )

        records.append(rec)

    for rec in records:
        for rel in rec.relations:
            if rel.target not in seen_ids:
                errors.append(
                    f"запись '{rec.id}': relation {rel.type.value} -> неизвестный id '{rel.target}'"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Валидация реестра sources.yaml G2AI-корпуса")
    parser.add_argument(
        "sources",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCES,
        help=f"путь к sources.yaml (по умолчанию {DEFAULT_SOURCES})",
    )
    parser.add_argument("--vocab-dir", type=Path, default=VOCAB_DIR, help="каталог vocab_*.yaml")
    args = parser.parse_args(argv)

    sources_path: Path = args.sources
    vocab_dir: Path = args.vocab_dir
    if not sources_path.exists():
        print(f"файл не найден: {sources_path}", file=sys.stderr)
        return 2

    errors = validate_sources(sources_path, vocab_dir)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"\n{len(errors)} ошибок(и)", file=sys.stderr)
        return 1
    print(f"OK: {sources_path} — валидно")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
