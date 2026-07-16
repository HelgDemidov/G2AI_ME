"""CLI-шим для обратной совместимости команд: реальный модуль — graph.build_graph.

НЕ импортировать из кода — только запуск как скрипта.
"""
from graph.build_graph import main

if __name__ == "__main__":
    raise SystemExit(main())
