"""Общий для pytest: гарантирует, что sibling-модули (schema, validate_sources)
импортируются из тестов независимо от каталога запуска."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
