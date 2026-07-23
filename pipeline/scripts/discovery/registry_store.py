"""discovery/registry_store.py — единый bronze-слой DuckDB для архетипа `registry`.

Spec `discovery-agora` §2 (решение куратора 2026-07-23, v2->v3): один DuckDB-файл
(`pipeline/discovery_cache/registry.duckdb`) обслуживает ВСЕ registry-коннекторы
(AGORA сейчас; EUR-Lex/OECD.AI/DPA — планируемые), не по файлу на коннектор.
Коннектор-агностично: этот модуль не знает о конкретных источниках, только
даёт SQL-движок + REPLACE-семантику загрузки. Staging-таблицы REPLACE на бамп
версии апстрима (не append-only) — историю версий несёт архивный кэш самого
коннектора (напр. version-именованный zip), не эта БД; реплей при пересборке
всегда идёт из архива, не из истории строк.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

from core.env import REPO_ROOT

DEFAULT_DB_PATH = REPO_ROOT / "pipeline" / "discovery_cache" / "registry.duckdb"

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """Схема/таблица подставляются в SQL как identifier (не value — ``?`` их не параметризует).

    Вызывающая сторона — всегда наш код (хардкод в connectors/*), не внешние данные, но
    дешёвая проверка формы закрывает класс SQL-инъекции по построению, а не по доверию.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"недопустимый SQL-идентификатор: {name!r}")
    return name


def connect(db_path: Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Открыть соединение с общей registry-БД (создав недостающие каталоги кэша)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def ingest_csv(
    conn: duckdb.DuckDBPyConnection,
    *,
    schema: str,
    table: str,
    csv_path: Path,
    source_version: str,
) -> None:
    """Загрузить CSV в ``<schema>.<table>`` (создав схему), REPLACE — не append.

    ``_source_version``/``_ingested_at`` — провенанс staging-снапшота (какая версия
    апстрима лежит в таблице СЕЙЧАС), не бизнес-поля источника. ``read_csv`` с
    ``auto_detect=true`` типизирует колонки (даты как ``DATE``, не ``VARCHAR``) и
    снимает UTF-8 BOM сам — живьём проверено на реальном дампе AGORA, никакой
    ручной ``utf-8-sig``-обработки не нужно (в отличие от stdlib ``csv``).
    """
    schema = _validate_identifier(schema)
    table = _validate_identifier(table)
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {schema}.{table} AS
        SELECT *, ? AS _source_version, current_date AS _ingested_at
        FROM read_csv(?, auto_detect=true)
        """,
        [source_version, str(csv_path)],
    )
