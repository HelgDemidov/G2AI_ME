"""CLI-шим для обратной совместимости команд: реальный модуль — convert.ocr_eval.

НЕ импортировать из кода — только запуск как скрипта.
"""
from convert.ocr_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
