#!/usr/bin/env python3
"""CLI дискавери-слоя G2AI-пайплайна: генерация кандидатов источников.

Чартер `docs/pipeline/discovery/charters/architecture.md`; спек discovery-core §5
(`discover`) + discovery-manual (`inject`/`worksheet`/`apply`).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from core import fsio, schema, validate_sources
from core.env import load_dotenv
from discovery import connectors, manual, store  # noqa: F401 — connectors: манифест реальных коннекторов (§4.3)
from discovery.orchestrate import DiscoverySummary, discover

if TYPE_CHECKING:  # только аннотации — рантайм-импорт ленивый, см. ниже
    from discovery.connectors import snowball


def _print_summary(summary: DiscoverySummary) -> None:
    for c in summary.connectors:
        if c.error is not None:
            print(f"  ✗ {c.connector_id}: ошибка — {c.error}")
        else:
            print(f"  {c.connector_id}: найдено {c.found} | свежих {c.fresh} | слито {c.merged}")
    mode = " (dry-run, ничего не записано)" if summary.dry_run else ""
    print(f"Итого: {summary.total_fresh} новых кандидат(ов){mode}")


def _cmd_discover(args: argparse.Namespace) -> int:
    summary = discover(only=args.only, root=args.root, dry_run=args.dry_run)
    _print_summary(summary)
    return 1 if summary.failed else 0


def _cmd_inject(args: argparse.Namespace) -> int:
    try:
        cand, is_new = manual.inject(
            url=args.url,
            title=args.title,
            issuer=args.issuer,
            language=args.language,
            jurisdiction=args.jurisdiction,
            date=args.date,
            summary=args.summary,
            kind=schema.ConnectorKind(args.kind),
            campaign=args.campaign,
            query=args.query,
            rights=schema.Rights(args.rights) if args.rights else None,
            sensitivity=schema.Sensitivity(args.sensitivity) if args.sensitivity else None,
            supersedes=args.supersedes,
            root=args.root,
        )
    except ValueError as exc:
        print(f"✗ {exc}")
        return 1
    if is_new:
        edition = f" (редакция, заменяет {cand.supersedes})" if cand.supersedes else ""
        print(f"добавлен кандидат: raw_hash={cand.raw_hash[:12]} title={cand.title!r}{edition}")
        return 0
    status = f"уже отклонён ранее: {cand.rejected_reason}" if cand.rejected_reason else "уже есть"
    print(f"кандидат уже присутствует ({status}): raw_hash={cand.raw_hash[:12]}")
    return 0


def _cmd_worksheet(args: argparse.Namespace) -> int:
    """spec triage-intake-hardening §2: ``--connector``/``--limit`` — отбор над списками
    ДО рендера (worksheet read-only, лока не берёт, как и раньше).

    ``--limit N`` режет ОБЕ секции до N независимо: «дай партию» относится и к ждущим, и к
    недобываемым — иначе ``--limit 20`` при сотне недобываемых отдавал бы 20 + 100, то есть
    не партию. Полные размеры снимаются ПОСЛЕ ``--connector``, но ДО ``--limit`` — они
    отвечают на «сколько ещё в том, что я запросил», а не «сколько во всей очереди»."""
    candidates = store.load(args.root)
    records = schema.load_records(args.root)
    pending = manual.pending_candidates(candidates, records)
    unacquirable = manual.unacquirable_candidates(candidates, records)

    if args.connector:
        pending = [c for c in pending if c.connector_id in args.connector]
        unacquirable = [c for c in unacquirable if c.connector_id in args.connector]
    pending_total = len(pending)
    unacquirable_total = len(unacquirable)
    if args.limit is not None:
        pending = pending[: args.limit]
        unacquirable = unacquirable[: args.limit]

    if args.format == "json":
        payload = manual.worksheet_payload(
            pending, unacquirable, total=pending_total, unacquirable_total=unacquirable_total
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = manual.render_worksheet(
            pending, unacquirable, total=pending_total, unacquirable_total=unacquirable_total
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"worksheet: {len(pending)} ждущих кандидат(ов) -> {args.out}")
    else:
        print(text)
    if unacquirable:
        shown_of = (
            f"{len(unacquirable)} из {unacquirable_total}"
            if len(unacquirable) < unacquirable_total
            else str(len(unacquirable))
        )
        print(f"(+ {shown_of} недобываемых кандидатов — ждут смены обстоятельств)")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    raw: Any = yaml.safe_load(args.decisions_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print(f"✗ {args.decisions_file}: верхний уровень decisions-файла должен быть списком")
        return 1

    summary = manual.apply_decisions(raw, root=args.root, dry_run=args.dry_run)
    for outcome in summary.outcomes:
        mark = "✓" if outcome.ok else "✗"
        print(f"  {mark} {outcome.raw_hash[:12]} [{outcome.action}]: {outcome.detail}")

    mode = " (dry-run, ничего не записано)" if summary.dry_run else ""
    print(f"Итого: {len(summary.outcomes)} решени(й), {len(summary.errors)} ошибок{mode}")

    admitted = any(o.ok and o.action == "admit" for o in summary.outcomes)
    if not summary.dry_run and admitted:
        print("Следующий шаг: pipeline/scripts/run_pipeline.py (скачивание/конвертация/индекс)")
        # Слабое место apply (spec vocab-axes, rationale): опечатка в словарном поле
        # (axis/doc_type/authority/...) материализуется в meta.yaml незамеченной до
        # следующего запуска validate_sources (CI/run_pipeline). Гейт сразу после батча
        # ловит её здесь, а не постфактум.
        vocab_errors, _ = validate_sources.validate_sources(args.root)
        if vocab_errors:
            print(f"⚠ реестр после apply невалиден ({len(vocab_errors)}) — исправьте перед run_pipeline:")
            for err in vocab_errors:
                print(f"  {err}")
            return 1

    return 1 if summary.errors else 0


def _build_snowball_config_override(args: argparse.Namespace) -> snowball.SnowballConfig:
    """yaml (``snowball.load_config()``) + CLI-флаги -> слитый конфиг НА ОДИН прогон
    (спек discovery-snowball §3). Заданный флаг ЗАМЕЩАЕТ соответствующее поле yaml
    целиком (не сливается поэлементно); незаданный (``None``/пустой список) — yaml как
    есть. Файл на диске не трогается."""
    from discovery.connectors import snowball  # ленивый: см. _cmd_snowball

    base = snowball.load_config()
    source_filter = base.source_filter
    if args.doc:
        source_filter = dataclasses.replace(source_filter, include_doc_ids=tuple(args.doc))
    if args.track:
        source_filter = dataclasses.replace(source_filter, tracks=tuple(args.track))

    url_filter = base.url_filter
    if args.exclude_domain:
        url_filter = dataclasses.replace(url_filter, exclude_domains=tuple(args.exclude_domain))

    emit = base.emit
    if args.with_citations:
        emit = dataclasses.replace(emit, text_citations=True)

    return dataclasses.replace(
        base,
        source_filter=source_filter,
        url_filter=url_filter,
        emit=emit,
    )


def _cmd_snowball(args: argparse.Namespace) -> int:
    # Импорт коннектора ЛЕНИВЫЙ, внутри его собственной подкоманды — иначе изоляция
    # отказа загрузки (`connectors._load_all`, spec discovery-acquire-seam-hardening §7,
    # Г6) обходится ровно так же, как раньше обходилась изоляция оркестратора: манифест
    # ловит битый `discovery_snowball.yaml` и пишет warning, а следующий же модульный
    # `from discovery.connectors import snowball` в этом файле поднимает его заново и
    # роняет ВЕСЬ CLI, включая inject/worksheet/apply, которым snowball не нужен вовсе
    # (проверено живьём на ревью PR #54: битый snowball-конфиг -> exit 1 и worksheet не
    # отрендерен, битый agora-конфиг -> exit 0 и worksheet на месте). Здесь громкий отказ
    # уместен и остаётся: подкоманду `snowball` без её конфига исполнить нечем.
    from discovery.connectors import snowball

    merged_config = _build_snowball_config_override(args)
    if merged_config.emit.text_citations:
        load_dotenv()  # §5 LLM-стадии нужен OPENROUTER_API_KEY из .env (зеркало run_pipeline)
    connector = snowball.SnowballConnector(config=merged_config, root=args.root)
    summary = discover(root=args.root, dry_run=args.dry_run, connectors_override=[connector])
    _print_summary(summary)

    # §5: лиды (цитаты без URL) — перезаписываются целиком каждым прогоном с
    # emit.text_citations включённым; --dry-run ничего не пишет на диск (симметрично
    # шардам кандидатов), отключённая стадия файл не трогает вовсе.
    if not args.dry_run and merged_config.emit.text_citations and summary.connectors:
        leads: list[dict[str, Any]] = summary.connectors[0].diagnostics.get("leads") or []  # type: ignore[assignment]
        snowball.save_leads(leads, args.root)
        if leads:
            print(f"snowball: {len(leads)} лид(ов) без URL -> {snowball.leads_path(args.root)}")

    return 1 if summary.failed else 0


# Мутирующие подкоманды берут общий корпусный лок (spec discovery-acquire-seam-
# hardening §2, Г1) — тот же файл, что и run_pipeline.py (штатный прогон/--recheck):
# `inject` мутирует всегда (нет `--dry-run`); `apply`/`discover`/`snowball` — когда
# НЕ dry-run (dry-run обязан быть no-op и не мешать живому прогону, конвенция
# run_pipeline). `worksheet` — read-only, лока не берёт никогда.
_MUTATING_ALWAYS = frozenset({"inject"})
_MUTATING_UNLESS_DRY_RUN = frozenset({"apply", "discover", "snowball"})


def _needs_corpus_lock(args: argparse.Namespace) -> bool:
    if args.cmd in _MUTATING_ALWAYS:
        return True
    if args.cmd in _MUTATING_UNLESS_DRY_RUN:
        return not args.dry_run
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DISCOVERY: генератор кандидатов источников")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_discover = sub.add_parser("discover", help="прогнать enabled-коннекторы")
    p_discover.add_argument(
        "--only", nargs="+", default=None, metavar="ID", help="ограничиться этими id коннекторов"
    )
    p_discover.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_discover.add_argument("--dry-run", action="store_true", help="сводка без записи store")
    p_discover.set_defaults(func=_cmd_discover)

    p_inject = sub.add_parser("inject", help="ручной/directed-search кандидат")
    p_inject.add_argument("--url", required=True)
    p_inject.add_argument("--title", required=True)
    p_inject.add_argument("--issuer", required=True)
    p_inject.add_argument("--language", required=True)
    p_inject.add_argument("--jurisdiction", default=None)
    p_inject.add_argument("--date", type=dt.date.fromisoformat, default=None, metavar="YYYY-MM-DD")
    p_inject.add_argument("--summary", default=None)
    p_inject.add_argument(
        "--kind",
        choices=[schema.ConnectorKind.manual.value, schema.ConnectorKind.directed_search.value],
        default="manual",
    )
    p_inject.add_argument("--campaign", default=None, help="обязателен при --kind directed_search")
    p_inject.add_argument("--query", default=None, help="обязателен при --kind directed_search")
    p_inject.add_argument("--rights", default=None, choices=[r.value for r in schema.Rights])
    p_inject.add_argument(
        "--sensitivity", default=None, choices=[s.value for s in schema.Sensitivity]
    )
    p_inject.add_argument(
        "--supersedes", default=None, metavar="DOC_ID",
        help="doc-id записи корпуса, которую этот кандидат ЗАМЕНЯЕТ (новая редакция на том же "
             "URL); предшественник обязан существовать в реестре",
    )
    p_inject.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_inject.set_defaults(func=_cmd_inject)

    p_worksheet = sub.add_parser("worksheet", help="таблица ждущих кандидатов (реконсиляция)")
    p_worksheet.add_argument("--out", type=Path, default=None, help="дефолт — stdout")
    p_worksheet.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_worksheet.add_argument(
        "--connector", action="append", default=None,
        help="ограничить коннектором (флаг повторяем для нескольких)",
    )
    p_worksheet.add_argument("--limit", type=int, default=None, help="показать не более N ждущих")
    p_worksheet.add_argument(
        "--format", choices=("json", "md"), default="json",
        help="json (дефолт, машинный контракт решений как данные) | md (человеческие таблицы)",
    )
    p_worksheet.set_defaults(func=_cmd_worksheet)

    p_apply = sub.add_parser("apply", help="применить batch-решения triage (promote/reject)")
    p_apply.add_argument("decisions_file", type=Path)
    p_apply.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_apply.add_argument("--dry-run", action="store_true", help="план без записи store/meta.yaml")
    p_apply.set_defaults(func=_cmd_apply)

    p_snowball = sub.add_parser(
        "snowball", help="backward-snowballing по собственному корпусу (пятый архетип)"
    )
    p_snowball.add_argument("--doc", nargs="+", default=None, metavar="ID", help="сузить до этих doc_id")
    p_snowball.add_argument("--track", nargs="+", default=None, metavar="TRACK")
    p_snowball.add_argument("--exclude-domain", nargs="+", default=None, metavar="DOMAIN")
    p_snowball.add_argument(
        "--with-citations", action="store_true", help="включить LLM-стадию текстовых цитат (§5)"
    )
    p_snowball.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_snowball.add_argument("--dry-run", action="store_true", help="сводка без записи store")
    p_snowball.set_defaults(func=_cmd_snowball)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if _needs_corpus_lock(args):
        try:
            with fsio.exclusive_flock(schema.corpus_lock_path(args.root)):
                return int(args.func(args))
        except fsio.AlreadyLocked as exc:
            # Текст исключения несёт ПУТЬ лока — без него куратор, гоняющий несколько
            # корпусов (`--root`), не видит, какой именно занят.
            print(f"✗ другой прогон run_pipeline/discover работает с этим корпусом ({exc})")
            return 1
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
