"""Единая staging-политика для атомарной записи артефактов документа.

Staging-файлы именуются dot-префиксом (``.<name>.part``), а не суффиксом
(``<name>.part``) — чтобы читающие глобы вида ``raw.*`` (``schema.raw_file``)
их не матчили by construction, а не по дисциплине своевременной очистки.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
from collections.abc import Generator
from pathlib import Path


class AlreadyLocked(RuntimeError):
    """Другой процесс уже держит эксклюзивный лок этого файла (``exclusive_flock``)."""


@contextlib.contextmanager
def exclusive_flock(path: Path) -> Generator[None]:
    """Эксклюзивный неблокирующий ``flock`` на ``path`` — взаимоисключение писателей
    общего мутируемого состояния (spec acquire-convert-seam-hardening §3, В3).

    Семантика ``flock`` (не existence-файла, как ``git index.lock``): лок снимается
    ЯДРОМ при смерти процесса, держащего дескриптор, — класса «протухший лок,
    оставшийся после kill» не существует по построению (в отличие от lockfile-по-
    существованию, где падение процесса ДО unlink оставляет файл навсегда). Занято —
    ``AlreadyLocked``, не блокировка: вызывающая сторона (батч-прогон) не должна
    зависать в ожидании чужого прогона неопределённое время.

    Живая проба 2026-07-27: два независимых ``open``+``flock`` конфликтуют и ВНУТРИ
    одного процесса (per-open-file-description locking) — лок тестируем герметично,
    без запуска второго процесса. Содержимое файла не имеет значения (чистый маркер
    занятости) — открывается в режиме добавления, чтобы никогда не обрезать/не
    создавать гонку с чем-то, что могло бы читать этот файл параллельно.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyLocked(f"{path}: уже занято другим прогоном") from exc
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


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


def sha256_file(path: Path) -> str:
    """sha256 потоковым чтением (не грузит весь файл в память разом)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
