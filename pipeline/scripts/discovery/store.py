"""discovery/store.py — персист слоя кандидатов + курсоров (spec discovery-core §4).

Раскладка кандидатов — **шард на коннектор**: ``sources/candidates/<shard>.yaml``, где
``shard`` — санитизированный ``connector_id`` (spec discovery-candidates-sharding §1).
Монолит на 35+ тыс. строк потерял человекочитаемость, а агент, открывший его напрямую,
тащил весь объём в контекст; партиционирование по ``connector_id`` зеркалит две уже
существующие конвенции этого же кода (``sources/<track>/<entity>/<id>/`` и
``pipeline/discovery_cache/<connector>/``).

**Шарды — для человека, не для I/O (§2):** ``load``/``save`` работают полным списком в
памяти (это нужно кросс-коннекторному dedup и реконсиляции ``pending_candidates``),
поэтому у ``save`` нет понятия «шард текущего коннектора» — он всегда сериализует ВЕСЬ
список и пишет те шарды, чьи байты изменились. Следствие: мутация чужой записи
(``dedup._merge_provenance``, probe-поля recheck-контура) корректна автоматически —
вопрос «какие шарды переписать» исчезает как класс, а не решается dirty-трекингом.

Оба артефакта (кандидаты, курсоры) машиннописаные (комментарии не выживают перезапись) —
тот же прецедент, что у ``.state.yaml`` (corpus-layout-v2): производные/операционные
файлы, не курируемые человеком напрямую.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from core import fsio, schema

CANDIDATES_DIRNAME = "candidates"
LEGACY_CANDIDATES_FILENAME = "candidates.yaml"

CANDIDATES_DIR = schema.DEFAULT_SOURCES / CANDIDATES_DIRNAME
LEGACY_CANDIDATES_PATH = schema.DEFAULT_SOURCES / LEGACY_CANDIDATES_FILENAME
CURSORS_PATH = schema.DEFAULT_SOURCES / ".discovery_cursors.yaml"
"""Dot-файл — операционное состояние (курсоры), не данные; вне git по deny-default `/sources/**`."""

# Имя шарда: всё, что не буква/цифра/подчёркивание/дефис, схлопывается в "__".
# Санитизация закрывает три ФС-риска разом (§1): ":" грамматики connector_id
# (`search:<кампания>`), "/" в свободном тексте имени кампании (подкаталог/выход из
# каталога store) и ведущую точку (``Path.glob("*.yaml")`` матчит скрытые файлы —
# проверено эмпирически, поэтому шард `.foo.yaml` был бы прочитан как обычный).
_SHARD_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


def candidates_dir(root: Path = schema.DEFAULT_SOURCES) -> Path:
    """Каталог шардов слоя кандидатов под корнем корпуса.

    Единственная точка знания о раскладке store: потребители
    (``orchestrate``/``manual``/``discover``) передают КОРЕНЬ и не конструируют пути
    сами — раньше литерал ``"candidates.yaml"`` был продублирован в 4 файлах (§6).
    """
    return root / CANDIDATES_DIRNAME


def legacy_candidates_path(root: Path = schema.DEFAULT_SOURCES) -> Path:
    """Монолит ``sources/candidates.yaml`` — вход авто-миграции (§3), не цель записи."""
    return root / LEGACY_CANDIDATES_FILENAME


def shard_name(connector_id: str) -> str:
    """``connector_id`` -> имя файла шарда БЕЗ расширения (санитизация — см. ``_SHARD_UNSAFE_RE``)."""
    return _SHARD_UNSAFE_RE.sub("__", connector_id)


def shard_path(connector_id: str, root: Path = schema.DEFAULT_SOURCES) -> Path:
    """Путь шарда коннектора: ``<root>/candidates/<санитизированный id>.yaml``."""
    return candidates_dir(root) / f"{shard_name(connector_id)}.yaml"


def load(root: Path = schema.DEFAULT_SOURCES) -> list[schema.CandidateRecord]:
    """Слой кандидатов целиком; отсутствующий store — пустой корпус кандидатов, не ошибка.

    **Прецедентность миграции (§3): пока монолит существует — он ЕДИНСТВЕННЫЙ источник
    истины, шарды игнорируются.** Наивная конкатенация «монолит + шарды» ломалась в окне
    отказа: краш между записью шардов и удалением монолита давал бы КАЖДУЮ мигрированную
    запись дважды, а следующий ``save`` материализовал бы дубли в шардах навсегда (каскад:
    ``_resolve_candidate`` падает «raw_hash неоднозначен» на apply). С прецедентностью
    краш в любой точке безопасен — следующий прогон стартует с монолита заново.

    Порядок детерминирован: шарды по имени файла, внутри шарда — порядок записи.
    """
    legacy = legacy_candidates_path(root)
    if legacy.exists():
        return schema.load_candidates(legacy)
    records: list[schema.CandidateRecord] = []
    for shard in sorted(candidates_dir(root).glob("*.yaml")):
        records.extend(schema.load_candidates(shard))
    return records


def _dump_records(candidates: list[schema.CandidateRecord]) -> str:
    """Сериализация списка кандидатов в текст одного YAML-файла store.

    Каждый кандидат дампится отдельно, записи разделяются пустой строкой (YAML к ней
    безразличен, а человеку файл читать батчами при триаже) — порядок полей задаёт
    модель (``title`` первым, провенанс внизу; sort_keys=False).
    """
    parts = [
        yaml.safe_dump([c.model_dump(mode="json", exclude_none=True)],
                       allow_unicode=True, sort_keys=False)
        for c in candidates
    ]
    return "\n".join(parts)


def _partition(candidates: list[schema.CandidateRecord]) -> dict[str, list[schema.CandidateRecord]]:
    """Разложить полный список по имени шарда; коллизия санитизации — громкий отказ.

    Теоретическая коллизия (``search:x`` vs литеральный ``search__x``) падает
    ``ValueError``, а не сливает два источника в один файл молча (§1).
    """
    by_shard: dict[str, list[schema.CandidateRecord]] = {}
    origins: dict[str, str] = {}
    for cand in candidates:
        name = shard_name(cand.connector_id)
        known = origins.setdefault(name, cand.connector_id)
        if known != cand.connector_id:
            raise ValueError(
                f"коллизия имён шардов: connector_id {known!r} и {cand.connector_id!r} "
                f"санитизируются в один файл '{name}.yaml'"
            )
        by_shard.setdefault(name, []).append(cand)
    return by_shard


def save(candidates: list[schema.CandidateRecord], root: Path = schema.DEFAULT_SOURCES) -> None:
    """Полный перезапись store шардами (§2): партиционирование ВСЕГО списка по
    ``connector_id``, атомарная запись пофайлово, удаление опустевших шардов.

    Пишутся только шарды, чьи байты изменились — не оптимизация I/O (объёмы смешные),
    а гигиена: неизменившиеся шарды не получают ложный mtime-чурн (бэкапы,
    наблюдаемость «что тронул прогон»).

    Крашеустойчивость: атомарность пофайловая; краш между шардами оставляет часть
    старых — следующий ``save`` дописывает (та же самовосстанавливаемость, что везде в
    пайплайне). Монолит удаляется ПОСЛЕДНИМ шагом (§3, см. ``load`` — прецедентность):
    до этого момента он остаётся источником истины, поэтому окно краша не порождает
    дублей. Конкурентных писателей нет (single-user, один процесс — конституция проекта).
    """
    desired: dict[Path, str] = {
        candidates_dir(root) / f"{name}.yaml": _dump_records(records)
        for name, records in _partition(candidates).items()
    }

    for path, text in sorted(desired.items()):
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue  # байты не изменились — не трогаем mtime
        fsio.atomic_write_text(path, text)

    for existing in sorted(candidates_dir(root).glob("*.yaml")):
        if existing not in desired:
            existing.unlink()  # коннектор больше не даёт записей — шард опустел

    legacy_candidates_path(root).unlink(missing_ok=True)  # авто-миграция завершена


def load_cursors(path: Path = CURSORS_PATH) -> dict[str, dict[str, Any]]:
    """``connector_id -> ConnectorCursor``; отсутствующий файл — пустой словарь (первый прогон)."""
    if not path.exists():
        return {}
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if raw is not None else {}


def save_cursors(cursors: dict[str, dict[str, Any]], path: Path = CURSORS_PATH) -> None:
    text = yaml.safe_dump(cursors, allow_unicode=True, sort_keys=False)
    fsio.atomic_write_text(path, text)
