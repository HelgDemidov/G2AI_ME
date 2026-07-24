"""Реальные коннекторы (registry/outlet_watcher/directed_search/manual) — по одному модулю на id.

Каждый модуль вызывает ``discovery.registry.register()`` при импорте. Ядро (base/registry/
dedup/store/orchestrate) о конкретных коннекторах не знает — см. чартер §4.3. Манифест: одна
строка импорта на коннектор, ниже — greppable список того, что реально закодировано.
"""
from discovery.connectors import agora  # noqa: F401 — регистрирует "agora" при импорте пакета
from discovery.connectors import eurlex  # noqa: F401 — регистрирует "eurlex" при импорте пакета
