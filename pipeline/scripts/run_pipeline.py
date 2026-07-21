"""Идемпотентный оркестратор G2AI-пайплайна: end-to-end по дереву корпуса ``sources/``.

Модель — РЕКОНСИЛЯЦИЯ (desired-state), а не хранимый флаг статуса: нужная работа
выводится из фактического состояния файловой системы (есть ли raw + совпадает ли
sha256; есть ли/свежий ли .md; синхронен ли frontmatter), поэтому повторный запуск
идемпотентен по построению и самовосстанавливается (удалили файл — стадия
переиграется). Курируемые ``meta.yaml`` не переписываются (человек — источник истины);
машина пишет только операционный сайдкар ``.state.yaml``.

Стадии на документ: download → convert → figures → frontmatter (figures —
VLM-пасс фигур, spec convert-cloud-tier §5, только для документов с необра-
ботанными маркерами и открытым облачным гейтом). Затем корпусный index
(FTS5 + опц. векторы). Отказ одного документа не прерывает батч.

Практики (актуально на июль 2026, right-sized — без Airflow/Prefect/Dagster, они серверные
и избыточны для ~100-200 документов на слабом железе): идемпотентность+инкрементальность,
ретраи с backoff (в curl), quality-gate (валидация реестра + sha256 + непустой вывод),
атомарная запись (tmp→rename), наблюдаемость (логи + сводка), dry-run.

CLI::

    run_pipeline.py [sources_root] [--only ID] [--force] [--dry-run]
                    [--no-download] [--embed] [--graphml PATH] [--db PATH]
                    [--no-cloud] [--vlm-model MODEL]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pdfplumber

from acquire import acquisition
from convert import cloud_ocr, converters, figures_vlm, lint
from graph import build_graph
from index import corpus_index
from core import fsio
from core import schema
from core import validate_sources
from core.env import load_dotenv
from index import vector_store
from index.chunking import strip_frontmatter
from index.embed import (
    DEFAULT_BACKEND,
    DEFAULT_CLOUD_DIMS,
    DEFAULT_CLOUD_MODEL,
    OnnxBgeEmbedder,
    get_embedder,
)

logger = logging.getLogger("run_pipeline")

# Браузероподобный UA: WAF-ы гос. сайтов часто блокируют не-браузерные UA (см. CLAUDE.md).
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


class Stage(str, Enum):
    download = "download"
    convert = "convert"
    figures = "figures"
    frontmatter = "frontmatter"


@dataclass
class DocResult:
    doc_id: str
    done: list[Stage] = field(default_factory=list)
    up_to_date: bool = False
    error: str | None = None


# --- пути и хеши (пути выводятся из папки-документа: schema.raw_file/md_file/state_file) ---
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
    """Какие стадии нужны документу по фактическому состоянию ФС (пути выводятся из папки).

    Целостность raw — дешёвым stat-guard'ом: sha256 пересчитывается (полное чтение
    файла) ТОЛЬКО если ``size``/``mtime_ns`` разошлись с записанными в
    ``.state.yaml`` — иначе на КАЖДОМ прогоне читались бы гигабайты raw ради «делать
    нечего». Честная оговорка: guard доверяет mtime — подмена файла с подделкой
    mtime+size его обойдёт, но это уже модель угроз, не защита от случайной порчи;
    ``--force`` всегда пересчитывает.
    """
    stages: list[Stage] = []
    raw = schema.raw_file(rec, root)          # существующий raw.* или None
    md = schema.md_file(rec, root)            # doc.md (путь; может не существовать)
    state = schema.load_state(schema.state_file(rec, root))

    if force or raw is None:
        stages.append(Stage.download)
    elif state.sha256:
        st = raw.stat()
        stat_matches = st.st_size == state.raw_size and st.st_mtime_ns == state.raw_mtime_ns
        if not stat_matches and _sha256(raw) != state.sha256:
            stages.append(Stage.download)     # файл повреждён/изменился vs записанный sha

    stale = False
    if raw is not None and raw.exists() and md.exists():
        stale = raw.stat().st_mtime > md.stat().st_mtime
    converter_changed = False
    if raw is not None and md.exists():
        conv = converters.resolve_converter(raw)   # UnsupportedFormat => planning-отказ (изолирован)
        converter_changed = (state.converter_name, state.converter_version) != (conv.name, conv.version)
    # ФС-реконсиляция §6.4 спека convert-cloud-tier: удаление .cloudocr.md — ЕДИНСТВЕННЫЙ
    # способ инвалидации (отдельного флага/CLI нет) — сайдкар пропал, а state всё ещё
    # помнит облачную модель => следующий прогон обязан пересчитать (Stage.convert
    # заново вызовет _cached_or_call_cloud, та увидит cache-мисс и позвонит в облако).
    cloudocr_cache_missing = (
        raw is not None and raw.exists() and state.cloud_ocr_model is not None
        and not cloud_ocr.cache_path(raw).exists()
    )
    if force or Stage.download in stages or not md.exists() or stale or converter_changed or cloudocr_cache_missing:
        stages.append(Stage.convert)

    # Порядок стадий: convert -> figures -> frontmatter (spec §5). Свежая конвертация
    # ВСЕГДА регенерирует голые маркеры (позиции/id пересчитываются заново) — figures
    # планируется безусловно following convert, тем же паттерном, что frontmatter
    # ниже. Без свежей конвертации — desired-state самовосстановление: документ,
    # уже несущий необработанные маркеры (напр. первый прогон после апгрейда на
    # convert-cloud-tier), должен доехать до figures БЕЗ форсированной реконверсии.
    cloud_ok = converters.cloud_allowed(rec)
    current_md_text: str | None = None
    if Stage.convert in stages:
        if cloud_ok:
            stages.append(Stage.figures)
    elif md.exists():
        current_md_text = md.read_text(encoding="utf-8")
        if cloud_ok and figures_vlm.has_bare_markers(current_md_text):
            stages.append(Stage.figures)

    if Stage.convert in stages or Stage.figures in stages:
        stages.append(Stage.frontmatter)
    elif md.exists():
        current = current_md_text if current_md_text is not None else md.read_text(encoding="utf-8")
        if _compose_md(rec, current) != current:
            stages.append(Stage.frontmatter)  # frontmatter разошёлся с реестром

    return stages


def _adopt_untracked_raw(rec: schema.SourceRecord, root: Path) -> None:
    """Обеспечить, что существующий raw отслеживается sha256 + stat-guard'ом
    (``raw_size``/``raw_mtime_ns``). Покрывает два случая:

    (а) raw добыт вручную (``--no-download``) — единственный писатель sha был у
    ``_do_download``; без усыновления повреждение такого файла оставалось бы
    невидимым навсегда.
    (б) ``.state.yaml`` старого формата (sha есть, guard-полей ещё нет —
    добавлены этим спеком): бэкфиллит их, но ТОЛЬКО если текущее содержимое
    подтверждённо совпадает с уже записанным sha (одноразовая верификация при
    миграции) — иначе рассинхрон/порча тихо получили бы «благословение» без
    проверки, и guard начал бы доверять непроверенному файлу навсегда.

    Идемпотентно. ``acquisition_method``/``fidelity`` не трогает — канал добычи
    неизвестен изначально, человек фиксирует его сам при желании.
    """
    raw = schema.raw_file(rec, root)
    if raw is None:
        return
    state_path = schema.state_file(rec, root)
    state = schema.load_state(state_path)
    st = raw.stat()
    if state.sha256 is None:
        state.sha256 = _sha256(raw)
        state.raw_size = st.st_size
        state.raw_mtime_ns = st.st_mtime_ns
        schema.save_state(state_path, state)
        logger.info("  %s: усыновлён ручной raw, sha зафиксирован", rec.id)
    elif state.raw_size is None or state.raw_mtime_ns is None:
        if _sha256(raw) == state.sha256:
            state.raw_size = st.st_size
            state.raw_mtime_ns = st.st_mtime_ns
            schema.save_state(state_path, state)


# --- исполнители стадий (side-effect, атомарная запись) ---
def _do_download(
    rec: schema.SourceRecord,
    root: Path,
    *,
    pause: float,
    interactive: bool = False,
    watch_dir: Path | None = None,
) -> None:
    """Скачивание через acquisition-лестницу (direct -> official_alt; см. acquisition.py).

    Цель — ``<doc_dir>/raw.<ext>``, расширение из ``rec.source_format`` (pdf/html;
    OCR-путь для сканов — будущее, бэклог #4). Не резюмируется между попытками
    (без ``curl -C -``); лестница не кеширует блок.

    ``interactive`` (=есть ``--only``): при блоке ЖИВОГО документа — синхронный 1-клик
    watch-folder путь. В батче (``interactive=False``) блок репортится как отказ
    документа (батч не прерывается). Мёртвый URL -> archive (автоматически, оба режима).

    Скачивание идёт во временный staging-файл (``fsio.staging_path`` — dot-префикс,
    невидим для глоба ``raw.*``); при ЛЮБОМ отказе (в т.ч. пробрасываемый наверх
    batch-блок) staging убирается в ``finally`` — challenge-тело/огрызок никогда не
    остаётся под именем, которое ``schema.raw_file`` мог бы принять за оригинал.
    При успехе — single-raw финализация: прежние ``raw.*`` (иной канал/формат)
    удаляются перед публикацией нового, чтобы в папке не оказалось двух оригиналов.

    После успеха пишет операционное состояние (sha256/acquisition_method/fidelity/
    checked) в ``.state.yaml`` (машиннописаный sidecar, corpus-layout-v2).
    """
    if not rec.source_url:
        raise RuntimeError("нет source_url для скачивания")
    if shutil.which("curl") is None:
        raise RuntimeError("curl не найден в PATH")
    raw = schema.raw_target(rec, root, ext=rec.source_format.value)
    raw.parent.mkdir(parents=True, exist_ok=True)
    part = fsio.staging_path(raw)
    try:
        try:
            result = acquisition.run_ladder(rec, part, user_agent=USER_AGENT)
        except acquisition.AcquisitionBlocked as exc:
            if not interactive:
                raise
            logger.info("  %s: %s", rec.id, exc)
            logger.info(
                "  открываю в браузере и жду файл (папка: %s)…",
                watch_dir or acquisition.default_watch_dir(),
            )
            result = acquisition.acquire_manually(rec, part, watch_dir=watch_dir)
        except acquisition.AcquisitionDead as exc:
            logger.info("  %s: %s", rec.id, exc)
            logger.info("  ищу снимок в Wayback…")
            result = acquisition.fetch_from_archive(rec, part, user_agent=USER_AGENT)
        for old in schema.doc_dir(rec, root).glob("raw.*"):
            if old != raw:
                old.unlink()  # смена канала/формата -> заменяем оригинал целиком
        part.replace(raw)
    finally:
        part.unlink(missing_ok=True)  # после успешного replace part не существует — no-op;
        # при любом исключении (в т.ч. пробрасываемом AcquisitionBlocked) убирает огрызок
    state_path = schema.state_file(rec, root)
    state = schema.load_state(state_path)
    st = raw.stat()
    state.sha256 = _sha256(raw)
    state.raw_size = st.st_size
    state.raw_mtime_ns = st.st_mtime_ns
    state.acquisition_method = result.method
    state.fidelity = result.fidelity
    state.acquisition_checked = _dt.date.today()
    state.retrieved_snapshot_date = result.retrieved_snapshot_date
    schema.save_state(state_path, state)
    logger.info("  добыто %s: метод=%s fidelity=%s (.state.yaml обновлён)", rec.id, result.method.value, result.fidelity.value)
    if pause > 0:
        time.sleep(pause)


def _raw_text(raw: Path, fmt: str) -> str | None:
    """Дешёвый pdfplumber-проход для C1-линта (паттерн ``converters._detect_scan``) —
    конвертация редка (раз на документ), секунды приемлемы. html -> None:
    trafilatura срезает boilerplate, ratio raw-vs-md было бы неинформативно
    (spec convert-hardening §5). Диагностический проход — падение на нём (напр.
    edge-case pdfminer-флуктуация) не должно ронять УЖЕ успешную конвертацию,
    поэтому отказ тихо даёт None (text-loss просто не проверяется на этом
    документе), а не пропагирует исключение (§6: lint никогда не роняет конвертацию).

    Возвращает ПОЛНЫЙ текст (не только длину): на OCR-ветке это ЖЕ tesseract-слой,
    который служит независимым свидетелем witness-линта (spec convert-cloud-tier
    §3) — переиспользуется вызывающей стороной вместо второго pdfplumber-прохода.
    """
    if fmt != "pdf":
        return None
    try:
        with pdfplumber.open(raw) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:  # noqa: BLE001 — диагностический проход, см. docstring
        logger.debug("не удалось извлечь текст raw для C1-линта: %s", raw, exc_info=True)
        return None


def _do_convert(rec: schema.SourceRecord, root: Path) -> None:
    raw = schema.raw_file(rec, root)
    md = schema.md_file(rec, root)
    if raw is None or not raw.exists():
        raise RuntimeError("нет raw-файла для конвертации")
    conv = converters.resolve_converter(raw)
    md.parent.mkdir(parents=True, exist_ok=True)
    tmp = fsio.staging_path(md)
    conv.convert(raw, tmp, rec.language, record=rec)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("конвертация дала пустой файл")
    tmp.replace(md)

    # C1 (spec convert-hardening): авто-QA вместо ручного аудита каждого документа —
    # никогда не роняет конвертацию, только сигналит (лог + машиночитаемый state).
    md_text = md.read_text(encoding="utf-8")
    raw_text = _raw_text(raw, conv.name)
    defects = lint.lint_conversion(
        md_text,
        raw_text_chars=len(raw_text) if raw_text is not None else None,
        fmt=conv.name,
    )

    state_path = schema.state_file(rec, root)
    state = schema.load_state(state_path)
    # OCR-путь мутирует raw IN-PLACE — пересчитать ДО witness-гейта (§3 спека
    # convert-cloud-tier сверяет ТЕКУЩИЙ sha256 raw с тем, что зафиксировал облачный
    # вызов при конвертации; устаревшая пара sha/model — это фолбэк-путь ЭТОГО
    # прогона, а не облачный vintage, witness тут неприменим).
    raw_sha256 = _sha256(raw)
    if raw_text is not None and state.cloud_ocr_model is not None and state.cloud_ocr_raw_sha256 == raw_sha256:
        # doc.md ЭТОГО прогона — подтверждённо облачный вывод (spec §3): raw_text —
        # тот же tesseract-слой, что служит witness. Никакой сети/токенов.
        defects.extend(lint.witness_checks(raw_text, strip_frontmatter(md_text)))

    for defect in defects:
        logger.warning("  ⚠ %s: convert-lint — %s", rec.id, defect)

    state.converter_name, state.converter_version = conv.name, conv.version
    state.lint_defects = defects
    # OCR-путь (convert-ocr) мутирует raw IN-PLACE (один PDF-файл на документ, без
    # сайдкара .ocr.pdf) — sha256/размер/mtime обязаны обновиться здесь, иначе
    # следующий stat-guard (needed_stages) увидит расхождение со старой записью и
    # решит, что raw «повреждён», затребовав передобычу поверх уже нормализованного
    # файла. Пересчёт безвреден и для не-OCR форматов (raw не менялся — sha совпадёт).
    st = raw.stat()
    state.sha256 = raw_sha256
    state.raw_size = st.st_size
    state.raw_mtime_ns = st.st_mtime_ns
    schema.save_state(state_path, state)


def _do_figures(rec: schema.SourceRecord, root: Path) -> None:
    """VLM-пасс фигур (spec convert-cloud-tier §5) — идемпотентен по построению
    (figures_vlm.apply_figures_pass), гейт (cloud_allowed) уже применён в
    needed_stages при планировании этой стадии."""
    raw = schema.raw_file(rec, root)
    md = schema.md_file(rec, root)
    if raw is None or not raw.exists():
        raise RuntimeError("нет raw-файла для фигурного пасса")
    if not md.exists():
        raise RuntimeError("нет doc.md для фигурного пасса")
    figures_vlm.apply_figures_pass(md, raw, model=cloud_ocr.ACTIVE_MODEL)


def _do_frontmatter(rec: schema.SourceRecord, root: Path) -> bool:
    """Синхронизировать frontmatter doc.md с реестром. Возвращает True, если файл изменён."""
    md = schema.md_file(rec, root)
    if not md.exists():
        raise RuntimeError("нет doc.md для синхронизации frontmatter")
    current = md.read_text(encoding="utf-8")
    desired = _compose_md(rec, current)
    if desired == current:
        return False
    fsio.atomic_write_text(md, desired)
    return True


# --- оркестрация ---
def process_docs(
    records: list[schema.SourceRecord],
    root: Path,
    *,
    force: bool,
    dry_run: bool,
    no_download: bool,
    pause: float,
    interactive: bool = False,
    watch_dir: Path | None = None,
) -> list[DocResult]:
    """Прогнать документы по стадиям. Возвращает результаты по каждому документу.

    ``interactive`` включает синхронный 1-клик watch-folder путь для manual-блоков
    (осмысленно только для одно-документных прогонов — ``main()`` включает его
    именно тогда, когда задан ``--only``).

    Изоляция отказа охватывает и ПЛАНИРОВАНИЕ (staging-чистку + усыновление
    неотслеженного raw + ``needed_stages``), не только исполнение стадий: битый
    ``.state.yaml`` или папка с несколькими ``raw.*`` (``schema.raw_file`` кидает
    ``ValueError``) роняют только этот документ, а не весь батч.

    Усыновление (``_adopt_untracked_raw``) пропускается при ``dry_run`` — оно
    ПИШЕТ ``.state.yaml`` (посчитанный sha256), а dry-run обязан быть no-op;
    staging-чистка (garbage, не значимое состояние) выполняется в обоих режимах.

    Не отслеживает «что-то изменилось» (раньше — in-run флаг ``changed``):
    решение о пересборке индекса теперь реконсилируется по ``corpus_index.
    corpus_fingerprint`` в ``main()`` ПОСЛЕ вызова этой функции (конвертация
    меняет mtime ``doc.md``) — а не по эфемерному флагу, теряемому при крахе.
    """
    results: list[DocResult] = []
    for rec in records:
        res = DocResult(rec.id)
        try:
            fsio.cleanup_staging(schema.doc_dir(rec, root))  # останки упавшего прогона — самовосстановление
            if not dry_run:  # усыновление ПИШЕТ .state.yaml — dry-run обязан быть no-op
                _adopt_untracked_raw(rec, root)  # ручной/старого формата raw — под контролем целостности
            stages = needed_stages(rec, root, force=force)
        except Exception as exc:  # noqa: BLE001 — изоляция отказа документа (планирование)
            res.error = f"planning: {exc}"
            logger.error("  ✗ %s: %s", rec.id, res.error)
            results.append(res)
            continue
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
                            rec, root, pause=pause,
                            interactive=interactive, watch_dir=watch_dir,
                        )
                elif stage is Stage.convert:
                    if not dry_run:
                        _do_convert(rec, root)
                elif stage is Stage.figures:
                    if not dry_run:
                        _do_figures(rec, root)
                else:
                    if not dry_run:
                        _do_frontmatter(rec, root)
                res.done.append(stage)
            except Exception as exc:  # noqa: BLE001 — изоляция отказа документа
                res.error = f"{stage.value}: {exc}"
                logger.error("  ✗ %s: %s", rec.id, res.error)
                break  # остальные стадии этого документа пропускаем
        results.append(res)
    return results


def rebuild_index(
    sources_path: Path,
    db_path: Path,
    *,
    embed: bool,
    force: bool = False,
    embed_backend: str = DEFAULT_BACKEND,
) -> str:
    """Пересобрать корпусный индекс: FTS5 (инкрементально по изменённым ``doc.md``,
    либо полностью при ``force``) + векторы (если embed; бэкенд — ``embed_backend``,
    дефолт облачный, spec embed-api-first §4). Требует токенизатор bge-m3 (чанковка
    остаётся на нём при любом эмбеддере).

    ``corpus_fingerprint``/``chunk_max_tokens`` пишутся в ``index_meta`` атомарно с
    чанками (см. ``corpus_index.index_corpus`` / ``index_chunks``) — реконсиляция
    пересборки в ``main`` полагается на этот отпечаток. Ветка «нет токенизатора»
    намеренно НЕ трогает индекс: следующий прогон (когда модель появится) честно
    доиндексирует по нетронутому отпечатку — самовосстановление по построению.
    Отказ векторной стадии (облако после ретраев/нет ключа) НЕ трогает FTS-часть —
    она уже закоммичена к этому моменту; исключение уходит в ``main`` (репорт +
    ненулевой exit-код).
    """
    from index.bge_tokenizer import EMBED_MAX_TOKENS, token_counter  # ленивый импорт: модель-зависимо

    try:
        counter = token_counter()
    except FileNotFoundError as exc:
        return f"пропущен (нет токенизатора bge-m3: {exc})"
    conn = corpus_index.create_db(db_path)
    status = corpus_index.index_corpus(conn, sources_path, counter, EMBED_MAX_TOKENS, force=force)
    conn.close()
    if embed:
        load_dotenv()  # облачному бэкенду нужен OPENROUTER_API_KEY из .env
        embedder = get_embedder(embed_backend)
        conn = sqlite3.connect(db_path)
        vector_store.check_chunk_budget(conn, embedder.max_tokens)
        # sensitivity-гейт (spec embed-api-first §3.3): облачный бэкенд не эмбеддит
        # чанки, все носители которых confidential; локальный — без фильтра
        exclude = (
            vector_store.confidential_doc_ids(conn) if embed_backend == "openrouter" else None
        )
        if exclude:
            all_pending, _ = vector_store.chunk_hashes(conn, not_embedded_for=embedder.name)
            hashes, texts = vector_store.chunk_hashes(
                conn, not_embedded_for=embedder.name, exclude_all_carriers_in=exclude
            )
            skipped = len(all_pending) - len(hashes)
        else:
            hashes, texts = vector_store.chunk_hashes(conn, not_embedded_for=embedder.name)
            skipped = 0
        if hashes:  # эмбеддим только НОВЫЕ хэши (правка 1 документа != пере-embed всего корпуса)
            # чекпоинтинг батчами — обрыв теряет ≤1 батч (spec embed-local-swap §5)
            vector_store.embed_and_store(conn, embedder, hashes, texts)
        removed = vector_store.gc_vectors(conn, embedder.name)
        conn.close()
        status += f"; векторы: +{len(hashes)} ({embedder.name}), GC {removed}"
        if skipped:
            status += (
                f"; {skipped} чанков только-confidential пропущены облачным эмбеддером"
                " (локальный прогон: --embed-backend bge)"
            )
    return status


def _read_index_fingerprint(db_path: Path) -> str | None:
    """Прочитать ``corpus_fingerprint`` уже собранного индекса. ``None``, если БД
    ещё нет (не создаём пустой файл ради чтения — ``sqlite3.connect`` иначе
    сделал бы это сам) или ключ отсутствует (индекс собран без него/устарел)."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        return corpus_index.read_meta(conn, "corpus_fingerprint")
    finally:
        conn.close()


def _needs_index_rebuild(sources_path: Path, db_path: Path, *, force: bool) -> tuple[bool, str]:
    """Решить, нужна ли пересборка индекса (реконсиляция по глобальному fingerprint,
    а не по in-run флагу), и вернуть посчитанный текущий отпечаток. Отпечаток —
    быстрый гейт «есть ли работа вообще»; саму пересборку (полную или инкрементальную
    по ``doc_state``) и запись нового отпечатка делает ``rebuild_index``/``index_corpus``."""
    current_fp = corpus_index.corpus_fingerprint(sources_path)
    stored_fp = _read_index_fingerprint(db_path)
    return (force or stored_fp != current_fp), current_fp


def _embed_namespace(backend: str) -> str:
    """Неймспейс векторов бэкенда БЕЗ конструирования эмбеддера: облачному нужен
    API-ключ, локальному — скачанные файлы модели, а для ПОДСЧЁТА недостающих
    векторов достаточно идентификатора (тот же, что ``embed_and_store`` пишет в
    ``vectors.model``)."""
    if backend == "openrouter":
        return f"{DEFAULT_CLOUD_MODEL}@{DEFAULT_CLOUD_DIMS}"
    return OnnxBgeEmbedder.name


def _report_unembedded(db_path: Path, backend: str) -> None:
    """Сводка отставания векторного слоя: сколько чанков не видны векторному каналу.

    Дефолтный прогон сознательно НЕ эмбеддит (облако = кредиты + сеть, явное
    действие куратора — spec embed-api-first §4), но молчать об отставании нельзя:
    документ без векторов теряет кросс-язычный/перефразированный retrieval,
    оставаясь видимым только FTS-каналу. Подсказка, не ошибка. Confidential-only
    чанки облачный добор пропустит и отчитается сам (сегодня в корпусе их нет —
    sensitivity латентна)."""
    if not db_path.exists():
        return
    namespace = _embed_namespace(backend)
    conn = sqlite3.connect(db_path)
    try:
        missing = vector_store.unembedded_count(conn, namespace)
    finally:
        conn.close()
    if missing:
        logger.info(
            "Векторы: %d чанков без эмбеддинга (%s) — добор: vector_store.py embed-corpus либо --embed",
            missing, namespace,
        )


def scan_fallback_counts(records: list[schema.SourceRecord], root: Path) -> tuple[int, int]:
    """``(n_fallback, n_confidential)`` среди ОТСКАНИРОВАННЫХ документов без
    успешного облачного OCR (spec ocr-eval-harness §8.3, S5). Скан определяется
    метаданными ocrmypdf (``converters._was_ocr_normalized`` — та же проверка,
    что ветвит ``_convert_pdf``), не повторной детекцией ``NeedsOCR``. OCR-путь
    существует только для PDF (``rec.source_format`` — курируемый источник
    истины, не переоткрытие по расширению файла) — живой прогон на реальном
    корпусе поймал `PdfminerException` на `eu-ai-act-2024` (raw.html) ДО того,
    как этот гейт появился: `_was_ocr_normalized` безусловно открывает файл
    через `pdfplumber`, что валится на не-PDF.

    - ``fallback`` — ``cloud_allowed(rec)`` было True, но ``cloud_ocr_model`` в
      ``.state.yaml`` так и не проставился: облако ДОЛЖНО было отработать и не
      смогло (сеть/лимиты/ретраи исчерпаны) — неожиданный отказ.
    - ``confidential`` — ``cloud_allowed(rec)`` False ИМЕННО из-за
      ``sensitivity`` — намеренная политика, не сбой.

    Остальные причины ``cloud_allowed=False`` (``--no-cloud`` этого прогона,
    отсутствующий ключ) НЕ считаются здесь: ``--no-cloud`` — явный выбор
    куратора текущего прогона, а отсутствие ключа уже даёт собственный warning
    внутри ``cloud_allowed`` (один раз за прогон) — дублировать нечего.

    Чистая функция: без сети, состояние читается с диска (``.state.yaml``)."""
    fallback = confidential = 0
    for rec in records:
        if rec.source_format is not schema.SourceFormat.pdf:
            continue  # OCR-путь существует только для PDF; _was_ocr_normalized падает на html/docx/xlsx
        raw = schema.raw_file(rec, root)
        if raw is None or not raw.exists() or not converters._was_ocr_normalized(raw):
            continue  # born-digital либо ещё не сконвертирован — не скан
        state = schema.load_state(schema.state_file(rec, root))
        if state.cloud_ocr_model is not None:
            continue  # облако отработало
        if converters.cloud_allowed(rec):
            fallback += 1
        elif rec.sensitivity is schema.Sensitivity.confidential:
            confidential += 1
    return fallback, confidential


def _report_scan_fallback(records: list[schema.SourceRecord], root: Path) -> None:
    """Сводка фолбэк-OCR-пути — симметрично ``_report_unembedded`` (PR #25):
    молчать об этом классе отставания нельзя, иначе единичный warning внутри
    ``_cached_or_call_cloud`` прокрутится незамеченным в логе батч-прогона на
    сотнях сканов."""
    fallback, confidential = scan_fallback_counts(records, root)
    if fallback:
        logger.warning(
            "OCR: %d скан(ов) на локальном пути из-за отказа облака (сеть/лимиты/ретраи исчерпаны) "
            "— проверьте .state.yaml/lint_defects",
            fallback,
        )
    if confidential:
        logger.info(
            "OCR: %d confidential-скан(ов) намеренно на локальном пути (sensitivity-гейт)",
            confidential,
        )


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
    parser.add_argument("sources", nargs="?", type=Path, default=schema.DEFAULT_SOURCES)
    parser.add_argument("--db", type=Path, default=corpus_index.DEFAULT_DB)
    parser.add_argument("--only", default=None, help="обработать только документ с этим id")
    parser.add_argument("--force", action="store_true", help="переиграть все стадии независимо от состояния")
    parser.add_argument("--dry-run", action="store_true", help="только показать план, без изменений")
    parser.add_argument("--no-download", action="store_true", help="не скачивать (raw добавляются вручную)")
    parser.add_argument(
        "--embed", action="store_true",
        help="также пересобрать векторы (облачный API: дёшево и быстро; --embed-backend bge — локально/медленно)",
    )
    parser.add_argument(
        "--embed-backend", choices=["openrouter", "bge"], default=DEFAULT_BACKEND,
        help="бэкенд эмбеддинга для --embed: openrouter — production-дефолт, bge — локальный фолбэк",
    )
    parser.add_argument("--graphml", type=Path, default=None, help="экспортировать граф в GraphML")
    parser.add_argument("--pause", type=float, default=1.0, help="пауза между скачиваниями, сек")
    parser.add_argument(
        "--watch-dir", type=Path, default=None,
        help="папка для ручного (manual) watch-folder пути; по умолчанию — системная папка загрузок",
    )
    parser.add_argument(
        "--no-cloud", action="store_true",
        help="отключить облачный OCR/figures (spec convert-cloud-tier §6.3) — офлайн-режим, поведение до спека",
    )
    parser.add_argument(
        "--vlm-model", default=None,
        help="override облачной модели для OCR/figures (эскалация для критичного документа, §6.4); "
             "единая модель на оба пути",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    converters.set_cloud_disabled(args.no_cloud)
    if args.vlm_model:
        cloud_ocr.ACTIVE_MODEL = args.vlm_model

    # quality-gate: реестр обязан быть валиден (пустой/несуществующий корень — валиден)
    errors, records = validate_sources.validate_sources(args.sources)
    if errors:
        logger.error("реестр невалиден (%d) — исправьте перед прогоном:", len(errors))
        for err in errors:
            logger.error("  %s", err)
        return 1

    if args.only:
        records = [r for r in records if r.id == args.only]
        if not records:
            logger.error("документ с id %r не найден", args.only)
            return 2

    # Синхронный manual watch-folder путь — только осмыслен для одно-документного
    # прогона (--only): пользователь реально сидит и ждёт клика (§6 спека, решение №2).
    results = process_docs(
        records, args.sources,
        force=args.force, dry_run=args.dry_run, no_download=args.no_download, pause=args.pause,
        interactive=bool(args.only), watch_dir=args.watch_dir,
    )

    # Сводка фолбэк-OCR-пути (S5, spec ocr-eval-harness §8.3) — читает ТОЛЬКО
    # .state.yaml с диска, не зависит от индекса/сети; безусловно (в т.ч. dry-run,
    # в отличие от _report_unembedded ниже, завязанной на corpus.db).
    _report_scan_fallback(records, args.sources)

    # корпусный индекс: реконсилируется по fingerprint (не по in-run флагу —
    # краш/прерывание между конвертацией и пересборкой не должны оставлять индекс
    # устаревшим навсегда). fp считается ПОСЛЕ process_docs — конвертация меняет
    # mtime doc.md.
    index_error: str | None = None
    if args.dry_run:
        logger.info("Индекс: dry-run, не трогаем")
    else:
        needs_rebuild, _ = _needs_index_rebuild(args.sources, args.db, force=args.force)
        if needs_rebuild:
            try:
                logger.info(
                    "Индекс: %s",
                    rebuild_index(
                        args.sources, args.db,
                        embed=args.embed, force=args.force, embed_backend=args.embed_backend,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — изоляция отказа стадии индекса:
                # FTS-часть закоммичена ДО векторной (порядок в rebuild_index), отказ
                # облака после ретраев её не рвёт; репорт + ненулевой exit, как у
                # прочих стадий (spec embed-api-first §4)
                index_error = str(exc)
                logger.error("  ✗ индекс: %s", index_error)
        else:
            logger.info("Индекс: актуален (fingerprint совпадает)")
        _report_unembedded(args.db, args.embed_backend)

    if args.graphml is not None and not args.dry_run:
        graph = build_graph.build_graph(records, build_graph.load_jurisdictions())
        build_graph.export_graphml(graph, args.graphml)
        logger.info("GraphML: %s", args.graphml)

    rc = _report(results)
    return 1 if index_error else rc


if __name__ == "__main__":
    raise SystemExit(main())
