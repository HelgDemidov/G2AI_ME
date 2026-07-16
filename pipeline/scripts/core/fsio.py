"""Единая staging-политика для атомарной записи артефактов документа.

Staging-файлы именуются dot-префиксом (``.<name>.part``), а не суффиксом
(``<name>.part``) — чтобы читающие глобы вида ``raw.*`` (``schema.raw_file``)
их не матчили by construction, а не по дисциплине своевременной очистки.
"""
from __future__ import annotations

from pathlib import Path


def staging_path(target: Path) -> Path:
    """Скрытый staging-файл рядом с целью: ``.<name>.part`` — не матчится глобами вида ``raw.*``."""
    return target.parent / f".{target.name}.part"


def cleanup_staging(directory: Path) -> None:
    """Удалить осиротевшие ``.«*».part`` (останки упавших прогонов) — самовосстановление."""
    if not directory.exists():
        return
    for p in directory.glob(".*.part"):
        p.unlink(missing_ok=True)


def atomic_write_text(target: Path, text: str) -> None:
    """Атомарная запись текста: staging (tmp) -> rename. Сбой записи не трогает ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = staging_path(target)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
