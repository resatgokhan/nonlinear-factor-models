"""Run with python3 -m bcd_lab.finance; finance YAML is a separate catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from bcd_lab.finance.config import load_config
from bcd_lab.finance.report import write_report
from bcd_lab.finance.runner import run_finance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Fit and evaluate rolling covariance estimators")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--period-limit", type=int)
    run.add_argument("--methods", nargs="+")
    run.add_argument("--overwrite", action="store_true")
    report = commands.add_parser("report", help="Regenerate metrics and plots from saved results")
    report.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "report":
        write_report(args.output_dir)
        return
    spec = load_config(args.config)
    if args.period_limit is not None:
        if args.period_limit < 1:
            parser.error("--period-limit must be positive")
        spec["window"]["periods"] = args.period_limit
    if args.methods:
        unknown = set(args.methods) - set(spec["methods"])
        if unknown:
            parser.error(f"Unknown methods: {sorted(unknown)}")
        spec["methods"] = {name: spec["methods"][name] for name in args.methods}
    output = args.output_dir or Path("results/finance") / spec["name"]
    run_finance(spec, output, overwrite=args.overwrite)
    for row in write_report(output):
        print(row)


if __name__ == "__main__":
    main()
