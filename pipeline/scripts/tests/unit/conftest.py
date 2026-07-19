"""Герметичность unit-тестов от локального окружения разработчика.

На РАБОЧЕЙ машине в ``.env`` лежит настоящий рабочий ``OPENROUTER_API_KEY``
(см. CLAUDE.md) — без явной изоляции поведение ``convert.converters.
cloud_allowed`` (а через него — ``needed_stages``/маршрутизация ``_convert_pdf``)
недетерминированно менялось бы в зависимости от того, кто и где запускает тесты:
в CI ключа нет (гейт закрыт), локально — есть (гейт открыт), один и тот же тест
давал бы разные результаты. Autouse-фикстура фиксирует CI-базовый сценарий
(«ключа нет») для ВСЕХ unit-тестов по умолчанию; тесты, которым нужен открытый
гейт, переопределяют ``OPENROUTER_API_KEY`` в своём теле как обычно (monkeypatch
внутри теста применяется ПОСЛЕ фикстуры и имеет приоритет).

``OPENROUTER_API_KEY=""`` (не ``delenv``) — намеренно: ``core.env.load_dotenv``
использует ``os.environ.setdefault``, который сработал бы на ОТСУТСТВУЮЩЕМ ключе
и тихо подтянул бы настоящий из файла; пустая строка физически ПРИСУТСТВУЕТ в
``os.environ`` (``setdefault`` — no-op) и одновременно фальшива для гейта
(``not os.environ.get(...)``).
"""
from __future__ import annotations

from typing import Any

import pytest

from convert import converters


@pytest.fixture(autouse=True)
def _hermetic_cloud_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr(converters, "_CLOUD_DISABLED", False)
    monkeypatch.setattr(converters, "_CLOUD_KEY_WARNED", False)
