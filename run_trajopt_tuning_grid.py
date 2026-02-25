#!/usr/bin/env python3
"""
Grid tuning for IPOPT trajectory optimization on a fixed map.

Primary goal: find faster feasible settings (lower lap-time cost) for Medium Oval.
Outputs:
- CSV with one row per config
- JSON with ranked summary + best config
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

from models import load_vehicle_from_yaml
from planning import ObstacleCircle, TrajectoryOptimizer
from world.world import World


@dataclass
class GridRecord:
    N: int
    obs_subsamples: int
    smoothness_w: float
    ux_min: float
    obstacle_clearance_m: float
    success: bool
    accepted: bool
    cost_s: float
    solve_time_s: float
    iterations: int
    max_obstacle_slack: float
    min_obstacle_clearance_m: float


def parse_grid(s: str, cast):
    vals = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(cast(token))
    if not vals:
        raise ValueError("Empty grid specification.")
    return vals


def create_vehicle(project_root: Path):
    config_file = project_root / "models" / "config" / "vehicle_params_gti.yaml"
    return load_vehicle_from_yaml(config_file)


def load_obstacles_from_world(world: World):
    data = world.data
    required = {
        "obstacles_s_m",
        "obstacles_e_m",
        "obstacles_radius_m",
        "obstacles_margin_m",
    }
    if not required.issubset(set(data.keys())):
        return []

    s_vals = np.atleast_1d(data["obstacles_s_m"]).astype(float)
    e_vals = np.atleast_1d(data["obstacles_e_m"]).astype(float)
    r_vals = np.atleast_1d(data["obstacles_radius_m"]).astype(float)
    m_vals = np.atleast_1d(data["obstacles_margin_m"]).astype(float)
    if not (len(s_vals) == len(e_vals) == len(r_vals) == len(m_vals)):
        raise ValueError("Obstacle arrays in map have inconsistent lengths.")

    obs: list[ObstacleCircle] = []
    for s_m, e_m, r_m, m_m in zip(s_vals, e_vals, r_vals, m_vals):
        obs.append(
            ObstacleCircle(
                s_m=float(s_m),
                e_m=float(e_m),
                radius_m=float(r_m),
                margin_m=float(m_m),
            )
        )
    return obs


def accepted(result, max_slack: float, min_clearance: float) -> bool:
    return (
        bool(result.success)
        and float(result.max_obstacle_slack) <= max_slack
        and float(result.min_obstacle_clearance) >= min_clearance
    )


def rank_key(r: GridRecord):
    # Rank successful+accepted first, then faster lap time, then faster solve.
    return (
        0 if (r.success and r.accepted) else 1,
        float(r.cost_s),
        float(r.solve_time_s),
    )


def main():
    parser = argparse.ArgumentParser(description="Grid tuning for trajectory optimizer")
    parser.add_argument("--map-file", type=str, default="maps/Medium_Oval_Map_260m.mat")
    parser.add_argument("--N-grid", type=str, default="100,120,140,160")
    parser.add_argument("--obs-subsamples-grid", type=str, default="5,7,9")
    parser.add_argument("--smoothness-grid", type=str, default="0.3,0.6,1.0")
    parser.add_argument("--ux-min-grid", type=str, default="2.0,2.5,3.0")
    parser.add_argument("--obstacle-clearance-grid", type=str, default="0.0")
    parser.add_argument("--obs-window-m", type=float, default=30.0)
    parser.add_argument("--track-buffer-m", type=float, default=0.0)
    parser.add_argument("--accept-min-clearance-m", type=float, default=-0.001)
    parser.add_argument("--accept-max-slack", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/trajectory_optimization")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    world = World(str(project_root / args.map_file), "TuneGrid", diagnostic_plotting=False)
    vehicle = create_vehicle(project_root)
    optimizer = TrajectoryOptimizer(vehicle, world)
    obstacles = load_obstacles_from_world(world)

    N_grid = parse_grid(args.N_grid, int)
    subs_grid = parse_grid(args.obs_subsamples_grid, int)
    smooth_grid = parse_grid(args.smoothness_grid, float)
    uxmin_grid = parse_grid(args.ux_min_grid, float)
    clear_grid = parse_grid(args.obstacle_clearance_grid, float)

    combos = list(product(N_grid, subs_grid, smooth_grid, uxmin_grid, clear_grid))
    print(f"Running {len(combos)} configurations on map: {args.map_file}")
    print(f"Obstacles loaded from map: {len(obstacles)}")

    records: list[GridRecord] = []
    for i, (N, subs, sw, ux_min, clear) in enumerate(combos, start=1):
        ds_m = float(world.length_m) / float(N)
        res = optimizer.solve(
            N=N,
            ds_m=ds_m,
            track_buffer_m=args.track_buffer_m,
            obstacles=obstacles,
            obstacle_window_m=args.obs_window_m,
            obstacle_clearance_m=clear,
            obstacle_subsamples_per_segment=subs,
            smoothness_weight=sw,
            obstacle_aware_init=True,
            obstacle_init_sigma_m=8.0,
            obstacle_init_margin_m=0.3,
            ux_min=ux_min,
            convergent_lap=True,
            verbose=False,
        )
        rec = GridRecord(
            N=N,
            obs_subsamples=subs,
            smoothness_w=float(sw),
            ux_min=float(ux_min),
            obstacle_clearance_m=float(clear),
            success=bool(res.success),
            accepted=accepted(res, args.accept_max_slack, args.accept_min_clearance_m),
            cost_s=float(res.cost),
            solve_time_s=float(res.solve_time),
            iterations=int(res.iterations),
            max_obstacle_slack=float(res.max_obstacle_slack),
            min_obstacle_clearance_m=float(res.min_obstacle_clearance),
        )
        records.append(rec)
        print(
            f"[{i:03d}/{len(combos):03d}] "
            f"ok={rec.success and rec.accepted} "
            f"N={N:<3} sub={subs:<2} sw={sw:<4.2f} ux_min={ux_min:<3.1f} "
            f"cost={rec.cost_s:7.3f}s t={rec.solve_time_s:6.2f}s"
        )

    ranked = sorted(records, key=rank_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"trajopt_tuning_grid_medium_oval_{timestamp}.csv"
    json_path = output_dir / f"trajopt_tuning_grid_medium_oval_{timestamp}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else ["N"])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    n_ok = sum(1 for r in records if r.success and r.accepted)
    best = ranked[0] if ranked else None
    summary = {
        "timestamp": timestamp,
        "map_file": args.map_file,
        "num_configs": len(records),
        "num_success_and_accepted": n_ok,
        "accept_min_clearance_m": args.accept_min_clearance_m,
        "accept_max_slack": args.accept_max_slack,
        "grid": {
            "N": N_grid,
            "obs_subsamples": subs_grid,
            "smoothness_w": smooth_grid,
            "ux_min": uxmin_grid,
            "obstacle_clearance_m": clear_grid,
        },
        "best": asdict(best) if best else None,
        "top_k": [asdict(r) for r in ranked[: max(1, args.top_k)]],
        "csv_path": str(csv_path),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nGrid tuning complete.")
    print(f"  Configs evaluated: {len(records)}")
    print(f"  Success+accepted: {n_ok}")
    if best is not None:
        print(
            "  Best: "
            f"N={best.N}, sub={best.obs_subsamples}, sw={best.smoothness_w}, ux_min={best.ux_min}, "
            f"clear={best.obstacle_clearance_m}, cost={best.cost_s:.3f}s"
        )
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
