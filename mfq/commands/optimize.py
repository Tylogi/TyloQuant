"""Unified CLI entry for storage-layout optimization."""

from __future__ import annotations

import argparse


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "optimize-layout",
        help="rewrite packed weights into native execution layouts",
    )
    parser.add_argument("--input", required=True, help="source MFQ model")
    parser.add_argument("--output", required=True, help="optimized MFQ model")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(_impl=_run)
    return parser


def _run(args: argparse.Namespace) -> int:
    from mfq.tools.optimize_layout import run

    return run(args)
