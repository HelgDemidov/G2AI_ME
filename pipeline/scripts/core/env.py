"""Минимальная загрузка .env в os.environ (без внешних зависимостей).

Существующие переменные окружения имеют приоритет (setdefault) — .env их не перетирает.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_dotenv(path: Path | None = None) -> None:
    """Прочитать KEY=VALUE из .env в os.environ (строки-комментарии и пустые пропускаются)."""
    env_path = path or REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
