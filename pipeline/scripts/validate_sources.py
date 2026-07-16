"""CLI-шим для обратной совместимости команд: реальный модуль — core.validate_sources.

НЕ импортировать из кода — только запуск как скрипта.
"""
from core.validate_sources import main

if __name__ == "__main__":
    raise SystemExit(main())
