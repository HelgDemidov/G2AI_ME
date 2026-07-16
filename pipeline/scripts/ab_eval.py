"""CLI-шим для обратной совместимости команд: реальный модуль — index.ab_eval.

НЕ импортировать из кода — только запуск как скрипта.
"""
from index.ab_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
