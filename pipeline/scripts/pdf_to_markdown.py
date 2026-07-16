"""CLI-шим для обратной совместимости команд: реальный модуль — convert.pdf_to_markdown.

НЕ импортировать из кода — только запуск как скрипта.
"""
import sys

from convert.pdf_to_markdown import convert

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
