#!/usr/bin/env python3
"""CLI дискавери-слоя G2AI-пайплайна: генерация кандидатов источников.

Чартер `docs/pipeline/discovery/charters/architecture.md`; спек discovery-core §5
(`discover`) + discovery-manual (`inject`/`worksheet`/`apply`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from core import schema
from discovery import manual
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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
