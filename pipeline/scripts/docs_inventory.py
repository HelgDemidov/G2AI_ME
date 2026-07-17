#!/usr/bin/env python3
"""Генератор инвентаризационной секции ROADMAP.md (guides/ROADMAP.md).

Сканирует docs/pipeline/**/{charters/*.md,tech_specs/*/spec.md}, извлекает первую
`Статус:`-строку каждого документа и перегенерирует таблицы между маркерами
AUTO-INVENTORY в целевом файле. Ручное (порядок блоков, колонка «Очередь»,
JIT-строки без файла) приходит из оверлея guides/roadmap.yaml — статусы там не
хранятся никогда (единственный источник статуса — сам документ). Таблица —
производный артефакт: перегенерация идемпотентна, `--check` сверяет без записи.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.env import REPO_ROOT

DOCS_ROOT = REPO_ROOT / "docs" / "pipeline"
DEFAULT_TARGET = DOCS_ROOT / "guides" / "ROADMAP.md"
DEFAULT_OVERLAY = DOCS_ROOT / "guides" / "roadmap.yaml"
BEGIN_MARK = "<!-- AUTO-INVENTORY:BEGIN -->"
END_MARK = "<!-- AUTO-INVENTORY:END -->"
NO_STATUS = "(нет Статус-строки)"
MAX_STATUS_LEN = 110  # длинные статусы (convert-ocr) усечь — таблица, не досье

_STATUS_RE = re.compile(r"^Статус:\s*(.+)$", re.MULTILINE)
_HEADER = "| Документ | Тип | Путь (`docs/pipeline/`) | Статус | Очередь |\n|---|---|---|---|---|"


@dataclass(frozen=True)
class DocRow:
    block: str   # верхняя папка блока (core/acquire/…)
    name: str    # колонка «Документ»: слаг папки спека либо «Архитектура <блок>»
    kind: str    # "спек" | "чартер"
    path: str    # относительный к docs/pipeline (или ручной текст у extra-строк)
    status: str
    queue: str | None = None  # None у сканированных строк -> берётся из overlay["queue"]


def extract_status(text: str) -> str:
    """Первая `Статус:`-строка документа; пробелы схлопнуты, длинное усечено."""
    m = _STATUS_RE.search(text)
    if m is None:
        return NO_STATUS
    status = " ".join(m.group(1).split())
    return status if len(status) <= MAX_STATUS_LEN else status[:MAX_STATUS_LEN].rstrip() + "…"


def scan(docs_root: Path) -> list[DocRow]:
    """Чартеры + спеки по конвенции раскладки docs (сортировка глоба — детерминизм)."""
    rows: list[DocRow] = []
    for p in sorted(docs_root.glob("*/charters/*.md")):
        block = p.parent.parent.name
        rows.append(
            DocRow(block, f"Архитектура {block}", "чартер",
                   p.relative_to(docs_root).as_posix(), extract_status(p.read_text(encoding="utf-8")))
        )
    for p in sorted(docs_root.glob("*/tech_specs/*/spec.md")):
        block = p.parent.parent.parent.name
        rows.append(
            DocRow(block, p.parent.name, "спек",
                   p.relative_to(docs_root).as_posix(), extract_status(p.read_text(encoding="utf-8")))
        )
    return rows


def _cell(s: str) -> str:
    """Экранировать `|` для markdown-таблицы."""
    return s.replace("|", "\\|")


def _extra_rows(overlay: dict[str, Any], block: str) -> list[DocRow]:
    return [
        DocRow(block, str(e["name"]), str(e.get("kind", "спек")),
               str(e.get("path", "—")), str(e.get("status", "")), str(e.get("queue", "—")))
        for e in overlay.get("extra") or []
        if e.get("block") == block
    ]


def render(rows: list[DocRow], overlay: dict[str, Any]) -> str:
    """Таблицы по блокам: порядок и заголовки — из оверлея; блоки вне оверлея — в хвост."""
    queue: dict[str, str] = {str(k): str(v) for k, v in (overlay.get("queue") or {}).items()}
    ordered = [(str(b["key"]), str(b["title"])) for b in overlay.get("blocks") or []]
    known = {key for key, _ in ordered}
    ordered += [(b, b.upper()) for b in sorted({r.block for r in rows} - known)]

    parts: list[str] = []
    for key, title in ordered:
        block_rows = [r for r in rows if r.block == key] + _extra_rows(overlay, key)
        if not block_rows:
            continue
        lines = [f"## {title}", "", _HEADER]
        for r in block_rows:
            path = f"`{r.path}`" if "/" in r.path else r.path
            q = r.queue if r.queue is not None else queue.get(r.name, "—")
            lines.append(
                f"| {_cell(r.name)} | {r.kind} | {path} | {_cell(r.status)} | {_cell(q)} |"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def replace_auto_section(target_text: str, body: str) -> str:
    """Заменить содержимое между AUTO-маркерами (сами маркеры сохраняются)."""
    begin, end = target_text.find(BEGIN_MARK), target_text.find(END_MARK)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError(f"в целевом файле нет корректной пары маркеров {BEGIN_MARK} … {END_MARK}")
    head = target_text[: begin + len(BEGIN_MARK)]
    return f"{head}\n\n{body}\n\n{target_text[end:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Перегенерация AUTO-секции ROADMAP.md")
    parser.add_argument("--docs-root", type=Path, default=DOCS_ROOT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--check", action="store_true",
                        help="только сверить (ненулевой код при расхождении), без записи")
    args = parser.parse_args(argv)

    overlay: Any = yaml.safe_load(args.overlay.read_text(encoding="utf-8")) or {}
    if not isinstance(overlay, dict):
        print(f"оверлей {args.overlay}: ожидался mapping", file=sys.stderr)
        return 2
    current = args.target.read_text(encoding="utf-8")
    try:
        desired = replace_auto_section(current, render(scan(args.docs_root), overlay))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if desired == current:
        print("ROADMAP актуален — без изменений")
        return 0
    if args.check:
        print(f"ROADMAP устарел: {args.target} расходится с документами/оверлеем", file=sys.stderr)
        return 1
    args.target.write_text(desired, encoding="utf-8")
    print(f"ROADMAP обновлён -> {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
