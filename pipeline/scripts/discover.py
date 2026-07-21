#!/usr/bin/env python3
"""CLI дискавери-слоя G2AI-пайплайна: генерация кандидатов источников.

Чартер `docs/pipeline/discovery/charters/architecture.md`; спек discovery-core §5.
Подкоманда `discover` (прогон коннекторов) — этот спек. `inject`/`worksheet`/`apply`
резервированы за спеком discovery-manual — добавляются в этот же файл теми же subparsers.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core import schema
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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
