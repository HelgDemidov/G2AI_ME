"""Валидатор корпуса G2AI — дерево ``sources/**/meta.yaml`` (corpus-layout-v2).

Проверяет: (1) структуру каждой записи через pydantic-схему; (2) принадлежность
``doc_type``/``authority``/``topics``/``g2ai_pattern`` контролируемым словарям;
(3) уникальность ``id``; (4) ссылочную целостность ``relations`` (цель — существующий id);
(5) наличие ``relevance`` (каждая запись корпуса — допущенная триажем; см.
source-relevance-triage); (6) инварианты папок — ``schema.check_layout`` (папка
документа == ``id``, папка сущности == ``entity_id``, верхняя == ``track``);
(7) для ``geo_scope: national`` — ``entity_id`` имеет форму iso2 (2 буквы);
членство в конкретных блоках (``jurisdictions.yaml``) не проверяется — список
блоков заведомо неполон.

Возвращает ненулевой код при ошибках — пригодно для pre-commit и CI.
Запуск::

    python3 validate_sources.py [корень_sources]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from schema import DEFAULT_SOURCES, VOCAB_DIR, GeoScope, SourceRecord, check_layout, load_vocab

_ISO2_RE = re.compile(r"[a-z]{2}")


def validate_sources(
    sources_root: Path, vocab_dir: Path = VOCAB_DIR
) -> tuple[list[str], list[SourceRecord]]:
    """Вернуть (ошибки, успешно распарсенные записи); пустой список ошибок =
    корпус валиден. Обход ``sources/**/meta.yaml``.

    Возвращает и записи — вызывающая сторона (``run_pipeline``/``build_graph``)
    переиспользует их вместо повторного ``load_records()`` (двойной обход
    дерева на каждый прогон, включая no-op). Записи отсортированы по ``id`` —
    тот же порядок, что даёт ``schema.load_records``.
    """
    errors: list[str] = []
    if not sources_root.exists():
        return errors, []  # пустой/несуществующий корпус — валиден

    vocabs: dict[str, set[str]] = {
        "doc_type": load_vocab("doc_types", vocab_dir),
        "authority": load_vocab("authority", vocab_dir),
        "topics": load_vocab("topics", vocab_dir),
        "g2ai_pattern": load_vocab("g2ai_patterns", vocab_dir),
    }

    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    for meta_path in sorted(sources_root.rglob("meta.yaml")):
        rel_path = meta_path.relative_to(sources_root)
        loc_id = str(rel_path)
        try:
            raw: Any = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            rec = SourceRecord.model_validate(raw)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"{loc_id}: {loc}: {err['msg']}")
            continue
        except yaml.YAMLError as exc:
            errors.append(f"{loc_id}: YAML: {exc}")
            continue

        errors.extend(check_layout(rel_path, rec, seen_ids))
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

        if rec.geo_scope is GeoScope.national and not _ISO2_RE.fullmatch(rec.entity_id):
            errors.append(
                f"запись '{rec.id}': geo_scope=national требует entity_id формы iso2 "
                f"(2 буквы), получено '{rec.entity_id}'"
            )

        if rec.relevance is None:
            errors.append(
                f"запись '{rec.id}': отсутствует relevance "
                "(обязателен для допущенной записи — прошла триаж)"
            )

        records.append(rec)

    for rec in records:
        for rel in rec.relations:
            if rel.target not in seen_ids:
                errors.append(
                    f"запись '{rec.id}': relation {rel.type.value} -> неизвестный id '{rel.target}'"
                )
    records.sort(key=lambda r: r.id)
    return errors, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Валидация корпуса G2AI (sources/**/meta.yaml)")
    parser.add_argument(
        "sources",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCES,
        help=f"корень sources/ (по умолчанию {DEFAULT_SOURCES})",
    )
    parser.add_argument("--vocab-dir", type=Path, default=VOCAB_DIR, help="каталог vocab_*.yaml")
    args = parser.parse_args(argv)

    sources_path: Path = args.sources
    vocab_dir: Path = args.vocab_dir

    errors, _ = validate_sources(sources_path, vocab_dir)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"\n{len(errors)} ошибок(и)", file=sys.stderr)
        return 1
    print(f"OK: {sources_path} — валидно")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
