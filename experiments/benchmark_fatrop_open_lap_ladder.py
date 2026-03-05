#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


IPOPT_LINE = re.compile(
    r"\[ipopt\]\s+success=(\w+)\s+iterations=([\-0-9]+)\s+cost=([0-9eE+\-.]+)\s+solve_time=([0-9eE+\-.]+)s"
)
FATROP_LINE = re.compile(
    r"\[fatrop-native\]\s+success=(\w+)\s+iterations=([\-0-9]+)\s+cost=([0-9eE+\-.]+)\s+solve_time=([0-9eE+\-.]+)s"
)
BUILD_LINE = re.compile(r"build_time=([0-9eE+\-.]+)s")
TOTAL_LINE = re.compile(r"total_time=([0-9eE+\-.]+)s")


def _parse_line(text: str, rgx: re.Pattern[str]) -> Dict[str, object]:
    m = rgx.search(text)
    if not m:
        return {
            "success": False,
            "iterations": -1,
            "cost": float("inf"),
            "solve_time_s": float("inf"),
        }
    return {
        "success": (m.group(1) == "True"),
        "iterations": int(m.group(2)),
        "cost": float(m.group(3)),
        "solve_time_s": float(m.group(4)),
    }


def _parse_scalar(text: str, rgx: re.Pattern[str], default: float = float("inf")) -> float:
    m = rgx.search(text)
    if not m:
        return float(default)
    return float(m.group(1))


def _run_once(
    *,
    py_bin: str,
    map_file: str,
    N: int,
    timeout_s: float,
    fatrop_env: Dict[str, str],
) -> Dict[str, object]:
    cmd = [
        py_bin,
        "experiments/run_fatrop_native_trajopt.py",
        "--map-file",
        map_file,
        "--N",
        str(N),
        "--compare-ipopt",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env.update(fatrop_env)

    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            env=env,
            cwd=".",
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        wall = time.time() - t0
        text = (p.stdout or "") + "\n" + (p.stderr or "")
        ip = _parse_line(text, IPOPT_LINE)
        fa = _parse_line(text, FATROP_LINE)
        fa_build = _parse_scalar(text, BUILD_LINE)
        fa_total = _parse_scalar(text, TOTAL_LINE)
        return {
            "ipopt_timeout": False,
            "fatrop_timeout": False,
            "ipopt_returncode": p.returncode,
            "fatrop_returncode": p.returncode,
            "ipopt_wall_time_s": wall,
            "fatrop_wall_time_s": wall,
            "wall_time_s": wall,
            "ipopt_success": bool(ip["success"]),
            "ipopt_iterations": int(ip["iterations"]),
            "ipopt_cost": float(ip["cost"]),
            "ipopt_solve_time_s": float(ip["solve_time_s"]),
            "fatrop_success": bool(fa["success"]),
            "fatrop_iterations": int(fa["iterations"]),
            "fatrop_cost": float(fa["cost"]),
            "fatrop_solve_time_s": float(fa["solve_time_s"]),
            "fatrop_build_time_s": float(fa_build),
            "fatrop_total_time_s": float(fa_total),
            "timeout": False,
            "returncode": p.returncode,
            "error_snippet": "" if p.returncode == 0 else text[-1200:],
        }
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        return {
            "ipopt_timeout": True,
            "fatrop_timeout": True,
            "ipopt_returncode": -999,
            "fatrop_returncode": -999,
            "ipopt_wall_time_s": wall,
            "fatrop_wall_time_s": wall,
            "wall_time_s": wall,
            "ipopt_success": False,
            "ipopt_iterations": -1,
            "ipopt_cost": float("inf"),
            "ipopt_solve_time_s": float("inf"),
            "fatrop_success": False,
            "fatrop_iterations": -1,
            "fatrop_cost": float("inf"),
            "fatrop_solve_time_s": float("inf"),
            "fatrop_build_time_s": float("inf"),
            "fatrop_total_time_s": float("inf"),
            "timeout": True,
            "returncode": -999,
            "error_snippet": f"timeout>{timeout_s:.1f}s",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    ap.add_argument("--Ns", type=str, default="40,60,80,100,120")
    ap.add_argument("--python-bin", type=str, default="/home/saveas/.conda/envs/DT_trajopt/bin/python")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--out-dir", type=str, default="results/solver_benchmarks")
    ap.add_argument("--preset", type=str, default="obstacle_fast")
    ap.add_argument("--structure", type=str, default="none")
    ap.add_argument("--expand", type=int, default=0, choices=[0, 1])
    ap.add_argument("--max-iter", type=int, default=800)
    ap.add_argument("--stage-local-cost", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dynamics-scheme", type=str, default="euler", choices=["trapezoidal", "euler"])
    ap.add_argument("--closure-mode", type=str, default="open", choices=["open", "soft", "hard"])
    ap.add_argument("--closure-soft-weight", type=float, default=100.0)
    args = ap.parse_args()

    Ns: List[int] = [int(x.strip()) for x in args.Ns.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_csv = out_dir / f"fatrop_openlap_ladder_{Path(args.map_file).stem}_{ts}.csv"

    fatrop_env = {
        "FATROP_PRESET": str(args.preset),
        "FATROP_STRUCTURE_DETECTION": str(args.structure),
        "FATROP_EXPAND": str(args.expand),
        "FATROP_MAX_ITER": str(args.max_iter),
        "FATROP_STAGE_LOCAL_COST": str(args.stage_local_cost),
        "FATROP_DYNAMICS_SCHEME": str(args.dynamics_scheme),
        "FATROP_CLOSURE_MODE": str(args.closure_mode),
        "FATROP_CLOSURE_SOFT_WEIGHT": str(args.closure_soft_weight),
        "FATROP_PRINT_LEVEL": "0",
    }

    rows: List[Dict[str, object]] = []
    for n in Ns:
        print(f"[openlap_ladder] N={n} timeout={args.timeout_s:.1f}s", flush=True)
        r = _run_once(
            py_bin=str(args.python_bin),
            map_file=str(args.map_file),
            N=int(n),
            timeout_s=float(args.timeout_s),
            fatrop_env=fatrop_env,
        )
        row = {
            "map": Path(args.map_file).name,
            "N": int(n),
            "timeout_s": float(args.timeout_s),
            "preset": fatrop_env["FATROP_PRESET"],
            "structure": fatrop_env["FATROP_STRUCTURE_DETECTION"],
            "expand": int(fatrop_env["FATROP_EXPAND"]),
            "fatrop_max_iter": int(fatrop_env["FATROP_MAX_ITER"]),
            "fatrop_stage_local_cost": int(fatrop_env["FATROP_STAGE_LOCAL_COST"]),
            "fatrop_dynamics_scheme": str(fatrop_env["FATROP_DYNAMICS_SCHEME"]),
            "fatrop_closure_mode": str(fatrop_env["FATROP_CLOSURE_MODE"]),
            "fatrop_closure_soft_weight": float(fatrop_env["FATROP_CLOSURE_SOFT_WEIGHT"]),
            **r,
        }
        rows.append(row)
        print(
            f"  -> ipopt_ok={row['ipopt_success']} ipopt_t={row['ipopt_solve_time_s']}s "
            f"fatrop_ok={row['fatrop_success']} fatrop_t={row['fatrop_solve_time_s']}s timeout={row['timeout']}",
            flush=True,
        )

    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "map",
            "N",
            "timeout_s",
            "preset",
            "structure",
            "expand",
            "fatrop_max_iter",
            "fatrop_stage_local_cost",
            "fatrop_dynamics_scheme",
            "fatrop_closure_mode",
            "fatrop_closure_soft_weight",
            "timeout",
            "returncode",
            "wall_time_s",
            "ipopt_timeout",
            "fatrop_timeout",
            "ipopt_returncode",
            "fatrop_returncode",
            "ipopt_wall_time_s",
            "fatrop_wall_time_s",
            "ipopt_success",
            "ipopt_iterations",
            "ipopt_cost",
            "ipopt_solve_time_s",
            "fatrop_success",
            "fatrop_iterations",
            "fatrop_cost",
            "fatrop_solve_time_s",
            "fatrop_build_time_s",
            "fatrop_total_time_s",
            "error_snippet",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={out_csv}", flush=True)


if __name__ == "__main__":
    main()
