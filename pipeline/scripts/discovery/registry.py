"""discovery/registry.py — реестр коннекторов (паттерн `convert.converters._CONVERTERS`).

Добавить коннектор = новый модуль в `discovery/connectors/`, вызывающий `register()` при
импорте. Ноль правок этого файла (чартер §4.3) — доказано `test_registry.py` fake-коннектором.
"""
from __future__ import annotations

from discovery.base import Connector

CONNECTORS: dict[str, Connector] = {}


def register(connector: Connector) -> None:
    """Зарегистрировать коннектор под его ``id``. Дубль id — ошибка конфигурации, не runtime-факт."""
    if connector.id in CONNECTORS:
        raise ValueError(f"коннектор '{connector.id}' уже зарегистрирован")
    CONNECTORS[connector.id] = connector


def enabled_connectors(only: list[str] | None = None) -> list[Connector]:
    """Коннекторы к прогону: включённые (``enabled=True``), опционально сузить списком id.

    ``only`` с неизвестным id — ошибка (опечатка в CLI не должна тихо не запустить коннектор).
    """
    if only is not None:
        unknown = sorted(set(only) - CONNECTORS.keys())
        if unknown:
            raise ValueError(f"неизвестные коннекторы: {', '.join(unknown)}")
        candidates = [CONNECTORS[cid] for cid in only]
    else:
        candidates = list(CONNECTORS.values())
    return [c for c in candidates if c.enabled]
