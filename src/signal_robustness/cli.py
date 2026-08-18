"""Command-line interface for synthetic and caller-supplied studies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from signal_robustness.study import run_study
from signal_robustness.synthetic import SyntheticSpec, generate_synthetic_prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-robustness",
        description="Run time-ordered probability evaluation and an aggregate viewer.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="run the seeded synthetic demonstration")
    demo.add_argument("--output-dir", required=True, type=Path)
    demo.add_argument("--days", type=int, default=900)
    demo.add_argument("--seed", type=int, default=20260818)

    analyze = commands.add_parser(
        "analyze", help="analyze a caller-supplied date/close CSV"
    )
    analyze.add_argument("--prices", required=True, type=Path)
    analyze.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        prices = generate_synthetic_prices(
            SyntheticSpec(days=args.days, seed=args.seed)
        )
    else:
        if not args.prices.is_file():
            raise SystemExit(f"price file does not exist: {args.prices}")
        prices = pd.read_csv(args.prices, usecols=["date", "close"])
    paths = run_study(prices, args.output_dir)
    print(f"wrote {len(paths)} aggregate artifacts")
    return 0
