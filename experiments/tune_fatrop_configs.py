#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


FATROP_LINE = re.compile(
    r"\[fatrop-native\]\s+success=(\w+)\s+iterations=([\-0-9]+)\s+cost=([0-9eE+\-.]+)\s+solve_time=([0-9eE+\-.]+)s"
)


@dataclass
class Config:
    name: str
    env: Dict[str, str]


DEFAULT_CONFIGS: List[Config] = [
    Config("obstacle_fast_none", {"FATROP_PRESET": "obstacle_fast", "FATROP_STRUCTURE_DETECTION": "none"}),
    Config("fast_none", {"FATROP_PRESET": "fast", "FATROP_STRUCTURE_DETECTION": "none"}),
    Config("balanced_none", {"FATROP_PRESET": "balanced", "FATROP_STRUCTURE_DETECTION": "none"}),
    Config("obstacle_fast_auto", {"FATROP_PRESET": "obstacle_fast", "FATROP_STRUCTURE_DETECTION": "auto"}),
    Config("fast_auto", {"FATROP_PRESET": "fast", "FATROP_STRUCTURE_DETECTION": "auto"}),
    Config("obstacle_fast_manual", {"FATROP_PRESET": "obstacle_fast", "FATROP_STRUCTURE_DETECTION": "manual"}),
]


def _run_once(
    py_bin: str,
    map_file: str,
    N: int,
    timeout_s: float,
    env_extra: Dict[str, str],
) -> Dict[str, object]:
    cmd = [
        py_bin,
        "experiments/run_fatrop_native_trajopt.py",
        "--map-file",
        map_file,
        "--N",
        str(N),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["FATROP_PRINT_LEVEL"] = "0"
    env.setdefault("FATROP_DYNAMICS_SCHEME", "euler")
    env.setdefault("FATROP_CLOSURE_MODE", "soft")
    env.setdefault("FATROP_CLOSURE_SOFT_WEIGHT", "30")
    env.update(env_extra)

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
        m = FATROP_LINE.search(text)
        if not m:
            return {
                "success": False,
                "iterations": -1,
                "cost": float("inf"),
                "solve_time_s": float("inf"),
                "wall_time_s": wall,
                "timeout": False,
                "returncode": p.returncode,
                "error_snippet": text[-1200:],
            }
        return {
            "success": (m.group(1) == "True"),
            "iterations": int(m.group(2)),
            "cost": float(m.group(3)),
            "solve_time_s": float(m.group(4)),
            "wall_time_s": wall,
            "timeout": False,
            "returncode": p.returncode,
            "error_snippet": "" if p.returncode == 0 else text[-1200:],
        }
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        return {
            "success": False,
            "iterations": -1,
            "cost": float("inf"),
            "solve_time_s": float("inf"),
            "wall_time_s": wall,
            "timeout": True,
            "returncode": -999,
            "error_snippet": f"timeout>{timeout_s:.1f}s",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--python-bin", type=str, default="/home/saveas/.conda/envs/DT_trajopt/bin/python")
    parser.add_argument("--base-timeout-s", type=float, default=120.0)
    parser.add_argument("--timeout-mult", type=float, default=1.8)
    parser.add_argument("--timeout-min-s", type=float, default=60.0)
    parser.add_argument("--timeout-max-s", type=float, default=360.0)
    parser.add_argument("--out-dir", type=str, default="results/solver_benchmarks")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"fatrop_tune_N{args.N}_{Path(args.map_file).stem}_{ts}.csv"

    rows: List[Dict[str, object]] = []
    best_wall: float | None = None
    for i, cfg in enumerate(DEFAULT_CONFIGS, start=1):
        if best_wall is None:
            timeout_s = float(args.base_timeout_s)
        else:
            timeout_s = max(
                float(args.timeout_min_s),
                min(float(args.timeout_max_s), float(args.timeout_mult) * best_wall),
            )

        print(
            f"[{i}/{len(DEFAULT_CONFIGS)}] {cfg.name} timeout={timeout_s:.1f}s best_wall={best_wall}",
            flush=True,
        )
        res = _run_once(
            py_bin=str(args.python_bin),
            map_file=str(args.map_file),
            N=int(args.N),
            timeout_s=float(timeout_s),
            env_extra=cfg.env,
        )
        row = {
            "name": cfg.name,
            "preset": cfg.env.get("FATROP_PRESET", ""),
            "structure": cfg.env.get("FATROP_STRUCTURE_DETECTION", ""),
            "timeout_s": timeout_s,
            **res,
        }
        rows.append(row)
        print(
            f"  -> success={row['success']} timeout={row['timeout']} "
            f"iter={row['iterations']} solve={row['solve_time_s']} wall={row['wall_time_s']:.3f}s",
            flush=True,
        )
        if bool(row["success"]):
            bw = float(row["wall_time_s"])
            best_wall = bw if best_wall is None else min(best_wall, bw)

    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "name",
            "preset",
            "structure",
            "timeout_s",
            "success",
            "timeout",
            "iterations",
            "solve_time_s",
            "wall_time_s",
            "cost",
            "returncode",
            "error_snippet",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out_csv}", flush=True)


if __name__ == "__main__":
    main()
