"""Идемпотентный оркестратор G2AI-пайплайна: end-to-end по записям ``sources.yaml``.

Модель — РЕКОНСИЛЯЦИЯ (desired-state), а не хранимый флаг статуса: нужная работа
выводится из фактического состояния файловой системы (есть ли raw + совпадает ли
sha256; есть ли/свежий ли .md; синхронен ли frontmatter), поэтому повторный запуск
идемпотентен по построению и самовосстанавливается (удалили файл — стадия
переиграется). Ничего не переписывается в сам `sources.yaml` (он — ручной источник истины).

Стадии на документ: download → convert → frontmatter. Затем корпусный index
(FTS5 + опц. векторы). Отказ одного документа не прерывает батч.

Практики (актуально на июль 2026, right-sized — без Airflow/Prefect/Dagster, они серверные
и избыточны для ~100-200 документов на слабом железе): идемпотентность+инкрементальность,
ретраи с backoff (в curl), quality-gate (валидация реестра + sha256 + непустой вывод),
атомарная запись (tmp→rename), наблюдаемость (логи + сводка), dry-run.

CLI::

    run_pipeline.py [sources.yaml] [--only ID] [--force] [--dry-run]
                    [--no-download] [--embed] [--graphml PATH] [--db PATH]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import acquisition
import build_graph
import corpus_index
import schema
import validate_sources
import vector_store
from chunking import strip_frontmatter
from embed import get_embedder
from pdf_to_markdown import convert as pdf_convert

logger = logging.getLogger("run_pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]
# Браузероподобный UA: WAF-ы гос. сайтов часто блокируют не-браузерные UA (см. CLAUDE.md).
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


class Stage(str, Enum):
    download = "download"
    convert = "convert"
    frontmatter = "frontmatter"


@dataclass
class DocResult:
    doc_id: str
    done: list[Stage] = field(default_factory=list)
    up_to_date: bool = False
    error: str | None = None


# --- пути и хеши ---
def _raw_file(rec: schema.SourceRecord, root: Path) -> Path | None:
    return root / rec.raw_path if rec.raw_path else None


def _md_file(rec: schema.SourceRecord, root: Path) -> Path | None:
    return root / rec.md_path if rec.md_path else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- реконсиляция (чистая логика) ---
def _compose_md(rec: schema.SourceRecord, current_md: str) -> str:
    """Желаемое содержимое .md = свежий frontmatter из реестра + тело (без старого frontmatter)."""
    body = strip_frontmatter(current_md).lstrip("\n")
    return schema.render_frontmatter(rec) + "\n" + body


def needed_stages(rec: schema.SourceRecord, root: Path, *, force: bool = False) -> list[Stage]:
    """Какие стадии нужны документу по фактическому состоянию ФС."""
    stages: list[Stage] = []
    raw = _raw_file(rec, root)
    md = _md_file(rec, root)

    if raw is not None:
        if force or not raw.exists():
            stages.append(Stage.download)
        elif rec.sha256 and _sha256(raw) != rec.sha256:
            stages.append(Stage.download)  # файл повреждён/изменился

    if md is not None:
        stale = raw is not None and raw.exists() and md.exists() and raw.stat().st_mtime > md.stat().st_mtime
        if force or Stage.download in stages or not md.exists() or stale:
            stages.append(Stage.convert)

    if md is not None:
        if Stage.convert in stages:
            stages.append(Stage.frontmatter)
        elif md.exists():
            current = md.read_text(encoding="utf-8")
            if _compose_md(rec, current) != current:
                stages.append(Stage.frontmatter)  # frontmatter разошёлся с реестром

    return stages


# --- исполнители стадий (side-effect, атомарная запись) ---
def _do_download(
    rec: schema.SourceRecord,
    root: Path,
    *,
    sources_path: Path,
    pause: float,
    interactive: bool = False,
    watch_dir: Path | None = None,
) -> None:
    """Скачивание через acquisition-лестницу (direct -> official_alt; см. acquisition.py).

    Не резюмируется между попытками (без ``curl -C -``): переключение лестницы
    между разными URL на один и тот же файл сделало бы резюм небезопасным
    (докачка "не с того" ответа), а лестница и так не кеширует известный блок
    между прогонами (§5 спека) — каждая попытка полная и честная.

    ``interactive`` (=есть ``--only``, решение №2 при реализации спека): если
    лестница упирается в блок, единственный документ в прогоне — живая сессия
    с пользователем, поэтому запускаем синхронный 1-клик watch-folder путь.
    В батч-прогоне (``interactive=False``) блок просто репортится как отказ
    документа — батч не прерывается (существующее поведение, ничего не меняли).

    После успеха точечно обновляет ``acquisition_method``/``acquisition_checked``/
    ``fidelity`` в ``sources.yaml`` (round-trip-safe, см. ``persist_acquisition_state``)
    и печатает сводку правок — не молча (решение 2026-07-15, см. спек).
    """
    raw = _raw_file(rec, root)
    if raw is None:
        raise RuntimeError("нет raw_path в записи")
    if not rec.source_url:
        raise RuntimeError("нет source_url для скачивания")
    if shutil.which("curl") is None:
        raise RuntimeError("curl не найден в PATH")
    raw.parent.mkdir(parents=True, exist_ok=True)
    part = raw.parent / (raw.name + ".part")  # атомарный (rename) staging-файл
    try:
        result = acquisition.run_ladder(rec, part, user_agent=USER_AGENT)
    except acquisition.AcquisitionBlocked as exc:
        if not interactive:
            raise
        logger.info("  %s: %s", rec.id, exc)
        logger.info("  открываю в браузере и жду файл (папка: %s)…", watch_dir or acquisition.default_watch_dir())
        result = acquisition.acquire_manually(rec, part, watch_dir=watch_dir)
    if rec.sha256 and _sha256(part) != rec.sha256:
        raise RuntimeError(f"sha256 не совпал (ожидался {rec.sha256[:12]}…)")
    part.replace(raw)
    changed = acquisition.persist_acquisition_state(
        sources_path, rec.id,
        acquisition_method=result.method, fidelity=result.fidelity, checked=_dt.date.today(),
    )
    if changed:
        summary = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in changed.items())
        logger.info("  sources.yaml обновлён (%s): %s", rec.id, summary)
    logger.info("  добыто %s: метод=%s fidelity=%s", rec.id, result.method.value, result.fidelity.value)
    if pause > 0:
        time.sleep(pause)


def _do_convert(rec: schema.SourceRecord, root: Path) -> None:
    raw = _raw_file(rec, root)
    md = _md_file(rec, root)
    if raw is None or not raw.exists():
        raise RuntimeError("нет raw-файла для конвертации")
    if md is None:
        raise RuntimeError("нет md_path в записи")
    md.parent.mkdir(parents=True, exist_ok=True)
    tmp = md.parent / (md.name + ".tmp")
    pdf_convert(str(raw), str(tmp))
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("конвертация дала пустой файл")
    tmp.replace(md)


def _do_frontmatter(rec: schema.SourceRecord, root: Path) -> bool:
    """Синхронизировать frontmatter .md с реестром. Возвращает True, если файл изменён."""
    md = _md_file(rec, root)
    if md is None or not md.exists():
        raise RuntimeError("нет .md для синхронизации frontmatter")
    current = md.read_text(encoding="utf-8")
    desired = _compose_md(rec, current)
    if desired == current:
        return False
    tmp = md.parent / (md.name + ".tmp")
    tmp.write_text(desired, encoding="utf-8")
    tmp.replace(md)
    return True


# --- оркестрация ---
def process_docs(
    records: list[schema.SourceRecord],
    root: Path,
    *,
    sources_path: Path,
    force: bool,
    dry_run: bool,
    no_download: bool,
    pause: float,
    interactive: bool = False,
    watch_dir: Path | None = None,
) -> tuple[list[DocResult], bool]:
    """Прогнать документы по стадиям. Возвращает результаты и флаг «что-то изменилось».

    ``interactive`` включает синхронный 1-клик watch-folder путь для manual-блоков
    (осмысленно только для одно-документных прогонов — ``main()`` включает его
    именно тогда, когда задан ``--only``).
    """
    results: list[DocResult] = []
    changed = False
    for rec in records:
        res = DocResult(rec.id)
        stages = needed_stages(rec, root, force=force)
        if not stages:
            res.up_to_date = True
            logger.info("• %s: актуально", rec.id)
            results.append(res)
            continue
        logger.info("• %s: %s%s", rec.id, "→".join(s.value for s in stages), " [dry-run]" if dry_run else "")
        for stage in stages:
            try:
                if stage is Stage.download:
                    if no_download:
                        raise RuntimeError("нужен download, но задан --no-download (скачайте raw вручную)")
                    if not dry_run:
                        _do_download(
                            rec, root, sources_path=sources_path, pause=pause,
                            interactive=interactive, watch_dir=watch_dir,
                        )
                elif stage is Stage.convert:
                    if not dry_run:
                        _do_convert(rec, root)
                else:
                    if not dry_run:
                        _do_frontmatter(rec, root)
                res.done.append(stage)
                if not dry_run:
                    changed = True
            except Exception as exc:  # noqa: BLE001 — изоляция отказа документа
                res.error = f"{stage.value}: {exc}"
                logger.error("  ✗ %s: %s", rec.id, res.error)
                break  # остальные стадии этого документа пропускаем
        results.append(res)
    return results, changed


def rebuild_index(sources_path: Path, db_path: Path, *, embed: bool) -> str:
    """Пересобрать корпусный индекс: FTS5 (всегда) + векторы (если embed). Требует токенизатор bge-m3."""
    from bge_tokenizer import token_counter  # ленивый импорт: модель-зависимо

    try:
        counter = token_counter()
    except FileNotFoundError as exc:
        return f"пропущен (нет токенизатора bge-m3: {exc})"
    chunks = corpus_index.chunks_from_corpus(sources_path, counter)
    conn = corpus_index.create_db(db_path)
    corpus_index.index_chunks(conn, chunks)
    conn.close()
    status = f"FTS: {len(chunks)} чанков"
    if embed:
        embedder = get_embedder("bge")
        import sqlite3

        conn = sqlite3.connect(db_path)
        ids, texts = vector_store.chunk_texts(conn)
        vector_store.store_vectors(conn, ids, embedder.embed(texts), embedder.name)
        conn.close()
        status += f"; векторы: {len(ids)} ({embedder.name})"
    return status


def _report(results: list[DocResult]) -> int:
    up = sum(r.up_to_date for r in results)
    failed = [r for r in results if r.error]
    processed = [r for r in results if r.done and not r.error]
    logger.info(
        "Итог: %d документ(ов) | актуально: %d | обработано: %d | ошибок: %d",
        len(results), up, len(processed), len(failed),
    )
    for res in failed:
        logger.info("  ✗ %s — %s", res.doc_id, res.error)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Идемпотентный оркестратор G2AI-пайплайна")
    parser.add_argument("sources", nargs="?", type=Path, default=validate_sources.DEFAULT_SOURCES)
    parser.add_argument("--db", type=Path, default=corpus_index.DEFAULT_DB)
    parser.add_argument("--only", default=None, help="обработать только документ с этим id")
    parser.add_argument("--force", action="store_true", help="переиграть все стадии независимо от состояния")
    parser.add_argument("--dry-run", action="store_true", help="только показать план, без изменений")
    parser.add_argument("--no-download", action="store_true", help="не скачивать (raw добавляются вручную)")
    parser.add_argument("--embed", action="store_true", help="также пересобрать векторы (медленно)")
    parser.add_argument("--graphml", type=Path, default=None, help="экспортировать граф в GraphML")
    parser.add_argument("--pause", type=float, default=1.0, help="пауза между скачиваниями, сек")
    parser.add_argument(
        "--watch-dir", type=Path, default=None,
        help="папка для ручного (manual) watch-folder пути; по умолчанию — системная папка загрузок",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.sources.exists():
        logger.error("файл не найден: %s", args.sources)
        return 2

    # quality-gate: реестр обязан быть валиден
    errors = validate_sources.validate_sources(args.sources)
    if errors:
        logger.error("реестр невалиден (%d) — исправьте перед прогоном:", len(errors))
        for err in errors:
            logger.error("  %s", err)
        return 1

    records = schema.load_records(args.sources)
    if args.only:
        records = [r for r in records if r.id == args.only]
        if not records:
            logger.error("документ с id %r не найден", args.only)
            return 2

    # Синхронный manual watch-folder путь — только осмыслен для одно-документного
    # прогона (--only): пользователь реально сидит и ждёт клика (§6 спека, решение №2).
    results, changed = process_docs(
        records, REPO_ROOT,
        sources_path=args.sources,
        force=args.force, dry_run=args.dry_run, no_download=args.no_download, pause=args.pause,
        interactive=bool(args.only), watch_dir=args.watch_dir,
    )

    # корпусный индекс: пересобираем, если что-то изменилось / нет БД / --force
    if args.dry_run:
        logger.info("Индекс: dry-run, не трогаем")
    elif changed or args.force or not args.db.exists():
        logger.info("Индекс: %s", rebuild_index(args.sources, args.db, embed=args.embed))
    else:
        logger.info("Индекс: без изменений, пересборка не нужна")

    if args.graphml is not None and not args.dry_run:
        graph = build_graph.build_graph(records, build_graph.load_jurisdictions())
        build_graph.export_graphml(graph, args.graphml)
        logger.info("GraphML: %s", args.graphml)

    return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())
