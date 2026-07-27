"""Реальные коннекторы (registry/outlet_watcher/directed_search/manual) — по одному модулю на id.

Каждый модуль вызывает ``discovery.registry.register()`` при импорте. Ядро (base/registry/
dedup/store/orchestrate) о конкретных коннекторах не знает — см. чартер §4.3. Манифест:
``_CONNECTOR_MODULES`` — greppable список того, что реально закодировано.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# Один элемент на коннектор — то же greppable перечисление, что раньше несла
# строка статических импортов, плюс изоляция отказа (см. ``_load_all``).
_CONNECTOR_MODULES = ("agora", "aiforgood", "eurlex", "oecd", "snowball")


def _load_all(names: tuple[str, ...] = _CONNECTOR_MODULES) -> None:
    """Импортировать каждый коннектор-модуль, изолируя отказ загрузки ОДНОГО от
    остальных (spec discovery-acquire-seam-hardening §7, Г6).

    Все пять коннекторов исполняют ``registry.register(...(enabled=load_config().enabled))``
    на импорт-тайме — опечатка в ЛЮБОМ из пяти ``discovery_*.yaml`` (файлы правятся
    руками по назначению — это тюнинг-конфиги) валила ВЕСЬ импорт пакета, а с ним и
    ``discover.py`` целиком, включая ``inject``/``worksheet``/``apply``, которым
    коннекторы не нужны вовсе. Тщательно выстроенная изоляция отказов оркестратора
    (``orchestrate.discover`` — упавший коннектор не рвёт прогон) обходилась этажом
    ниже, на импорт-тайме. Прецедент — stevedore (OpenStack): битый плагин
    репортится ``on_load_failure_callback`` и изолируется, не валя приложение.

    Незагруженный коннектор просто ОТСУТСТВУЕТ в реестре: ``discover`` его не
    гоняет (warning уже напечатан выше), ``--only <битый>`` даёт существующий
    громкий отказ «неизвестные коннекторы» (``registry.enabled_connectors``) —
    сломанность видна, но не заразна для остальных подкоманд.

    ``names`` — параметр, не захардкоженная константа внутри тела: тесты подставляют
    кортеж с фейковым битым именем модуля, герметично, без правки реальных конфигов.
    """
    for name in names:
        try:
            importlib.import_module(f"discovery.connectors.{name}")
        except Exception as exc:  # noqa: BLE001 — изоляция отказа загрузки, см. докстрока выше
            logger.warning("коннектор %s не загружен: %s", name, exc)


_load_all()
