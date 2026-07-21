"""Реальные коннекторы (registry/outlet_watcher/directed_search/manual) — по одному модулю на id.

Каждый модуль вызывает ``discovery.registry.register()`` при импорте. Ядро (base/registry/
dedup/store/orchestrate) о конкретных коннекторах не знает — см. чартер §4.3.
"""
