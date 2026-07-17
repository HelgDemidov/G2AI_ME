"""Аналитический слой поверх corpus.db (chartер docs/pipeline/analyze/charters/architecture.md).

``retrieve()`` (retrieve.py) — единая точка гибридного (FTS+вектор, RRF) поиска с
фасетными фильтрами; фундамент для будущих evidence/matrix/draft-спеков.
"""
