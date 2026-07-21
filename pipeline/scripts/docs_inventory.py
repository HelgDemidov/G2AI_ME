#!/usr/bin/env python3
"""Генератор инвентаризационной секции ROADMAP.md (roadmap/ROADMAP.md).

Сканирует docs/pipeline/**/{charters/*.md,tech_specs/*/spec.md}, извлекает первую
`Статус:`-строку каждого документа и перегенерирует ДВЕ секции целевого файла:
таблицы по блокам (маркеры AUTO-INVENTORY) и сквозную нумерованную очередь
(маркеры AUTO-QUEUE). Ручное (порядок блоков, колонка «Очередь», JIT-строки без
файла) приходит из оверлея roadmap/roadmap.yaml — статусы там не хранятся
никогда (единственный источник статуса — сам документ). Обе секции —
производный артефакт: перегенерация идемпотентна, `--check` сверяет без записи.

Колонка «Очередь» для спеков (`kind == "спек"`) самоисцеляется:
- терминальный статус (`реализовано…`/`❌…`) ВСЕГДА даёт «—», даже если в оверлее
  ещё осталось старое числовое значение — забытая правка `roadmap.yaml` после
  мерджа больше не может проявиться протухшим номером в таблице;
- спек без терминального статуса и без явного значения в оверлее получает
  автономер (максимум уже занятых номеров + 1) — не изображает знание
  истинного приоритета, просто честно ставит новое/нерешённое в конец
  очереди; куратор при желании переопределяет позицию в `roadmap.yaml`.

Сквозная очередь строится по ТЕМ ЖЕ разрешённым значениям (`_resolve_queue`),
поэтому таблица и нумерованный список не могут разойтись между собой.
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
DEFAULT_TARGET = DOCS_ROOT / "roadmap" / "ROADMAP.md"
DEFAULT_OVERLAY = DOCS_ROOT / "roadmap" / "roadmap.yaml"
BEGIN_MARK = "<!-- AUTO-INVENTORY:BEGIN -->"
END_MARK = "<!-- AUTO-INVENTORY:END -->"
QUEUE_BEGIN_MARK = "<!-- AUTO-QUEUE:BEGIN -->"
QUEUE_END_MARK = "<!-- AUTO-QUEUE:END -->"
NO_STATUS = "(нет Статус-строки)"
MAX_STATUS_LEN = 110  # длинные статусы (convert-ocr) усечь — таблица, не досье
TERMINAL_STATUS_PREFIXES = ("реализовано", "❌")  # спек закрыт -> очередь ему не нужна

_STATUS_RE = re.compile(r"^Статус:\s*(.+)$", re.MULTILINE)
_QUEUE_NUM_RE = re.compile(r"^\*\*(\d+)\*\*\s*(?:·\s*(.*))?$")
_HEADER = "| Документ | Тип | Путь (`docs/pipeline/`) | Статус | Очередь |\n|---|---|---|---|---|"


@dataclass(frozen=True)
class DocRow:
    block: str   # верхняя папка блока (core/acquire/…)
    name: str    # колонка «Документ»: слаг папки спека либо «Архитектура <блок>»
    kind: str    # "спек" | "чартер" | … (extra-строки: бэклог/гайды/документ)
    path: str    # относительный к docs/pipeline (или ручной текст у extra-строк)
    status: str
    queue: str | None = None  # None у сканированных строк -> берётся из overlay["queue"]


def is_terminal_status(status: str) -> bool:
    """Статус, при котором позиция в очереди больше не нужна (закрыт/отменён)."""
    return status.startswith(TERMINAL_STATUS_PREFIXES)


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


def _ordered_blocks(rows: list[DocRow], overlay: dict[str, Any]) -> list[tuple[str, str]]:
    """Порядок и заголовки блоков — из оверлея; блоки вне оверлея — в хвост (алфавит)."""
    ordered = [(str(b["key"]), str(b["title"])) for b in overlay.get("blocks") or []]
    known = {key for key, _ in ordered}
    ordered += [(b, b.upper()) for b in sorted({r.block for r in rows} - known)]
    return ordered


def _all_rows_in_order(rows: list[DocRow], overlay: dict[str, Any]) -> list[DocRow]:
    """Полный список строк (сканированные + extra) в порядке блоков рендера — единая
    основа и для таблиц, и для сквозной очереди, чтобы номера не могли разойтись."""
    out: list[DocRow] = []
    for key, _title in _ordered_blocks(rows, overlay):
        out += [r for r in rows if r.block == key] + _extra_rows(overlay, key)
    return out


def _leading_number(text: str) -> int | None:
    m = _QUEUE_NUM_RE.match(text)
    return int(m.group(1)) if m else None


def _resolve_queue(all_rows: list[DocRow], overlay: dict[str, Any]) -> list[str]:
    """Итоговое значение колонки «Очередь» на строку (индекс совпадает с all_rows).

    Не-«спек»-строки (чартер/бэклог/гайды/документ) — только явное значение
    (`row.queue` для extra либо `overlay["queue"]` по имени), иначе «—».
    «Спек»-строки: терминальный статус -> «—» БЕЗУСЛОВНО (даже если оверлей ещё
    хранит старое число — самоисцеление протухших записей); иначе явное
    значение, а при его отсутствии — автономер (макс. занятых + 1, по порядку
    строк, детерминированно).
    """
    queue_overlay: dict[str, str] = {str(k): str(v) for k, v in (overlay.get("queue") or {}).items()}
    explicit: list[str | None] = []
    for r in all_rows:
        raw = r.queue if r.queue is not None else queue_overlay.get(r.name)
        if r.kind != "спек":
            explicit.append(raw if raw is not None else "—")
        elif is_terminal_status(r.status):
            explicit.append("—")
        else:
            explicit.append(raw)  # None -> нужен автономер

    used = {n for v in explicit if v is not None for n in (_leading_number(v),) if n is not None}
    next_num = max(used, default=0) + 1

    resolved: list[str] = []
    for v in explicit:
        if v is not None:
            resolved.append(v)
        else:
            resolved.append(f"**{next_num}**")
            next_num += 1
    return resolved


def render(rows: list[DocRow], overlay: dict[str, Any]) -> str:
    """Таблицы по блокам (между маркерами AUTO-INVENTORY)."""
    all_rows = _all_rows_in_order(rows, overlay)
    resolved = _resolve_queue(all_rows, overlay)

    parts: list[str] = []
    for key, title in _ordered_blocks(rows, overlay):
        idx_rows = [(i, r) for i, r in enumerate(all_rows) if r.block == key]
        if not idx_rows:
            continue
        lines = [f"## {title}", "", _HEADER]
        for i, r in idx_rows:
            path = f"`{r.path}`" if "/" in r.path else r.path
            lines.append(
                f"| {_cell(r.name)} | {r.kind} | {path} | {_cell(r.status)} | {_cell(resolved[i])} |"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def render_queue_chain(rows: list[DocRow], overlay: dict[str, Any]) -> str:
    """Сквозная очередь (между маркерами AUTO-QUEUE) — нумерованный список,
    отсортированный по тем же разрешённым номерам, что и таблица (`_resolve_queue`)."""
    all_rows = _all_rows_in_order(rows, overlay)
    resolved = _resolve_queue(all_rows, overlay)

    numbered: list[tuple[int, str, str]] = []
    for r, q in zip(all_rows, resolved, strict=True):
        m = _QUEUE_NUM_RE.match(q)
        if m is None:
            continue
        numbered.append((int(m.group(1)), r.name, m.group(2) or ""))

    numbered.sort(key=lambda t: t[0])
    lines = [f"{n}. {name}" + (f" — {desc}" if desc else "") for n, name, desc in numbered]
    return "\n".join(lines) if lines else "(пусто)"


def _replace_marked_section(target_text: str, begin: str, end: str, body: str) -> str:
    """Заменить содержимое между парой маркеров (сами маркеры сохраняются)."""
    b, e = target_text.find(begin), target_text.find(end)
    if b < 0 or e < 0 or e < b:
        raise ValueError(f"в целевом файле нет корректной пары маркеров {begin} … {end}")
    head = target_text[: b + len(begin)]
    return f"{head}\n\n{body}\n\n{target_text[e:]}"


def replace_auto_section(target_text: str, body: str) -> str:
    """Заменить содержимое между маркерами AUTO-INVENTORY (обратная совместимость)."""
    return _replace_marked_section(target_text, BEGIN_MARK, END_MARK, body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Перегенерация AUTO-секций ROADMAP.md")
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
    rows = scan(args.docs_root)
    try:
        desired = _replace_marked_section(current, BEGIN_MARK, END_MARK, render(rows, overlay))
        desired = _replace_marked_section(
            desired, QUEUE_BEGIN_MARK, QUEUE_END_MARK, render_queue_chain(rows, overlay)
        )
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
