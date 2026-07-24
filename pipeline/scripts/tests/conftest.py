"""Корневой conftest для ``pipeline/scripts/tests/`` (unit/ + integration/) — отдельно
от ``tests/unit/conftest.py`` (герметичность unit-путей, другая забота).

Умбрелла-маркер ``heavy`` (см. ``pyproject.toml``): физически присутствующий на ЭТОЙ
машине тулинг (bge-m3/ocrmypdf/soffice/mermaidx/Node+lightpanda) и растущий локальный
корпус (``sources/``) — общий класс тестов, которые НЕ скипаются локально без явного
фильтра (в отличие от CI/свежего клона, где их не из чего собрать); полный гейт по
умолчанию их исключает. Раньше это делалось растущим списком имён в команде
(``not model and not corpus and not ocr and ...``), продублированным в CLAUDE.md/CI/
скиллах — подтверждённый трижды (PR #26/#37/#40) класс дефекта «забыли добавить
новый маркер в одну из копий строки». Этот хук — структурный фикс: `heavy`
проставляется АВТОМАТИЧЕСКИ любому тесту, несущему один из ``_HEAVY_MARKERS``, — новый
маркер этого класса добавляется правкой ТОЛЬКО списка ниже, команда гейта (``-m "not
heavy"``) больше никогда не меняется. Индивидуальные маркеры (``-m ocr`` и т.п.)
работают как прежде — `heavy` ДОБАВЛЯЕТСЯ, не заменяет их.
"""
from __future__ import annotations

import pytest

_HEAVY_MARKERS = ("model", "corpus", "ocr", "libreoffice", "mermaid", "browser")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if any(item.get_closest_marker(name) is not None for name in _HEAVY_MARKERS):
            item.add_marker(pytest.mark.heavy)
