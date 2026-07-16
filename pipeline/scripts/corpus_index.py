"""CLI-шим для обратной совместимости команд: реальный модуль — index.corpus_index.

НЕ импортировать из кода — только запуск как скрипта.
"""
from index.corpus_index import main

if __name__ == "__main__":
    raise SystemExit(main())
