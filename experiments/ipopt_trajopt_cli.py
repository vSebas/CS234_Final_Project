#!/usr/bin/env python3
"""
Unified IPOPT trajectory-optimization CLI.

Subcommands:
- single: one scenario with plots/animation/log output
- batch: randomized scenario benchmark sweep
- tune: grid search over solver settings
"""

import argparse
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from experiments import _ipopt_batch_eval_impl
from experiments import run_ipopt_trajopt_demo
from experiments import _ipopt_tuning_grid_impl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified IPOPT trajectory optimization CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_single = subparsers.add_parser(
        "single",
        help="Run one IPOPT trajectory optimization scenario and save plots.",
    )
    p_single.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to experiments/run_ipopt_trajopt_demo.py",
    )

    p_batch = subparsers.add_parser(
        "batch",
        help="Run randomized batch evaluation over multiple scenarios.",
    )
    p_batch.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to IPOPT batch-eval implementation.",
    )

    p_tune = subparsers.add_parser(
        "tune",
        help="Run trajectory optimization tuning grid.",
    )
    p_tune.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to IPOPT tuning-grid implementation.",
    )
    return parser.parse_args(argv)


def _strip_double_dash(forwarded):
    if forwarded and forwarded[0] == "--":
        return forwarded[1:]
    return forwarded


def main(argv=None):
    ns = parse_args(argv)
    forwarded = _strip_double_dash(ns.args)

    if ns.command == "single":
        cfg = run_ipopt_trajopt_demo.parse_args(forwarded)
        run_ipopt_trajopt_demo.run_demo(cfg)
        return
    if ns.command == "batch":
        _ipopt_batch_eval_impl.main(forwarded)
        return
    if ns.command == "tune":
        _ipopt_tuning_grid_impl.main(forwarded)
        return
    raise ValueError(f"Unsupported command: {ns.command}")


if __name__ == "__main__":
    main()
