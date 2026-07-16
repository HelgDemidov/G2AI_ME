"""CLI-шим для обратной совместимости команд: реальный модуль — index.vector_store.

НЕ импортировать из кода — только запуск как скрипта.
"""
from index.vector_store import main

if __name__ == "__main__":
    raise SystemExit(main())
