#!/usr/bin/env python3
"""CLI дискавери-слоя G2AI-пайплайна: генерация кандидатов источников.

Чартер `docs/pipeline/discovery/charters/architecture.md`; спек discovery-core §5
(`discover`) + discovery-manual (`inject`/`worksheet`/`apply`).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
from pathlib import Path
from typing import Any

import yaml

from core import schema, validate_sources
from discovery import connectors, manual, store  # noqa: F401 — connectors: манифест реальных коннекторов (§4.3)
from discovery.connectors import snowball
from discovery.orchestrate import DiscoverySummary, discover


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
            root=args.root,
        )
    except ValueError as exc:
        print(f"✗ {exc}")
        return 1
    if is_new:
        print(f"добавлен кандидат: raw_hash={cand.raw_hash[:12]} title={cand.title!r}")
        return 0
    status = f"уже отклонён ранее: {cand.rejected_reason}" if cand.rejected_reason else "уже есть"
    print(f"кандидат уже присутствует ({status}): raw_hash={cand.raw_hash[:12]}")
    return 0


def _cmd_worksheet(args: argparse.Namespace) -> int:
    candidates = store.load(args.root / "candidates.yaml")
    records = schema.load_records(args.root)
    pending = manual.pending_candidates(candidates, records)
    text = manual.render_worksheet(pending)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"worksheet: {len(pending)} ждущих кандидат(ов) -> {args.out}")
    else:
        print(text)
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
    base = snowball.load_config()
    source_filter = base.source_filter
    if args.doc:
        source_filter = dataclasses.replace(source_filter, include_doc_ids=tuple(args.doc))
    if args.track:
        source_filter = dataclasses.replace(source_filter, tracks=tuple(args.track))
    if args.tier:
        source_filter = dataclasses.replace(source_filter, target_fit=tuple(args.tier))

    url_filter = base.url_filter
    if args.exclude_domain:
        url_filter = dataclasses.replace(url_filter, exclude_domains=tuple(args.exclude_domain))

    emit = base.emit
    if args.with_citations:
        emit = dataclasses.replace(emit, text_citations=True)

    max_candidates = base.max_candidates if args.max_candidates is None else args.max_candidates

    return dataclasses.replace(
        base,
        source_filter=source_filter,
        url_filter=url_filter,
        emit=emit,
        max_candidates=max_candidates,
    )


def _cmd_snowball(args: argparse.Namespace) -> int:
    merged_config = _build_snowball_config_override(args)
    connector = snowball.SnowballConnector(config=merged_config, root=args.root)
    summary = discover(root=args.root, dry_run=args.dry_run, connectors_override=[connector])
    _print_summary(summary)

    # §5: лиды (цитаты без URL) — перезаписываются целиком каждым прогоном с
    # emit.text_citations включённым; --dry-run ничего не пишет на диск (симметрично
    # candidates.yaml/cursors), отключённая стадия файл не трогает вовсе.
    if not args.dry_run and merged_config.emit.text_citations and summary.connectors:
        leads: list[dict[str, Any]] = summary.connectors[0].diagnostics.get("leads") or []  # type: ignore[assignment]
        snowball.save_leads(leads, args.root)
        if leads:
            print(f"snowball: {len(leads)} лид(ов) без URL -> {args.root / snowball.LEADS_FILENAME}")

    return 1 if summary.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DISCOVERY: генератор кандидатов источников")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_discover = sub.add_parser("discover", help="прогнать enabled-коннекторы")
    p_discover.add_argument(
        "--only", nargs="+", default=None, metavar="ID", help="ограничиться этими id коннекторов"
    )
    p_discover.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_discover.add_argument("--dry-run", action="store_true", help="сводка без записи store/cursors")
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
    p_inject.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_inject.set_defaults(func=_cmd_inject)

    p_worksheet = sub.add_parser("worksheet", help="таблица ждущих кандидатов (реконсиляция)")
    p_worksheet.add_argument("--out", type=Path, default=None, help="дефолт — stdout")
    p_worksheet.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
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
    p_snowball.add_argument(
        "--tier", nargs="+", default=None, metavar="TIER", choices=[t.value for t in schema.TargetFit]
    )
    p_snowball.add_argument("--exclude-domain", nargs="+", default=None, metavar="DOMAIN")
    p_snowball.add_argument(
        "--with-citations", action="store_true", help="включить LLM-стадию текстовых цитат (§5)"
    )
    p_snowball.add_argument("--max-candidates", type=int, default=None, metavar="N")
    p_snowball.add_argument("--root", type=Path, default=schema.DEFAULT_SOURCES)
    p_snowball.add_argument("--dry-run", action="store_true", help="сводка без записи store/cursors")
    p_snowball.set_defaults(func=_cmd_snowball)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
