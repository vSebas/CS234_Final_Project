#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import List

from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer
from world.world import World

from experiments.run_fatrop_trajopt import _load_obstacles_from_world, solve_fatrop_trajopt


def _resolve_maps(mode: str, explicit_maps: List[str]) -> List[Path]:
    if explicit_maps:
        return [Path(m) for m in explicit_maps]
    maps_dir = Path("maps")
    if mode == "obstacles":
        return sorted(maps_dir.glob("*_Obstacles.mat"))
    if mode == "no_obstacles":
        all_maps = sorted(maps_dir.glob("*.mat"))
        return [m for m in all_maps if "_Obstacles" not in m.stem]
    return sorted(maps_dir.glob("*.mat"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mode", choices=("obstacles", "no_obstacles", "all"), default="obstacles")
    parser.add_argument("--map", action="append", dest="maps", default=[])
    parser.add_argument("--out-dir", type=str, default="results/solver_benchmarks")
    args = parser.parse_args()

    maps = _resolve_maps(args.mode, args.maps)
    if not maps:
        raise RuntimeError("No maps found for selected mode.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"ipopt_vs_fatrop_N{args.N}_{args.mode}_{ts}.csv"

    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    rows = []

    # Use tuned FATROP defaults unless user overrides from shell.
    os.environ.setdefault("FATROP_PRESET", "obstacle_fast")
    os.environ.setdefault("FATROP_STRUCTURE_DETECTION", "none")
    os.environ.setdefault("FATROP_PRINT_LEVEL", "0")

    for map_path in maps:
        if not map_path.exists():
            continue
        ipopt_times = []
        fatrop_times = []
        ipopt_costs = []
        fatrop_costs = []
        ipopt_successes = 0
        fatrop_successes = 0
        ipopt_iters = []
        fatrop_iters = []
        fatrop_min_clearances = []

        for _ in range(args.repeats):
            world = World(str(map_path), map_path.stem, diagnostic_plotting=False)
            obstacles = _load_obstacles_from_world(world)
            ds_m = float(world.length_m / int(args.N))

            ipopt_res = TrajectoryOptimizer(vehicle, world).solve(
                N=int(args.N),
                ds_m=ds_m,
                lambda_u=0.005,
                ux_min=0.5,
                track_buffer_m=0.0,
                eps_s=0.1,
                eps_kappa=0.05,
                obstacles=obstacles,
                obstacle_clearance_m=0.0,
                obstacle_use_slack=False,
                obstacle_enforce_midpoints=False,
                obstacle_subsamples_per_segment=5,
                convergent_lap=True,
                verbose=False,
            )
            fatrop_res = solve_fatrop_trajopt(
                vehicle=vehicle,
                world=world,
                N=int(args.N),
                ds_m=ds_m,
                obstacles=obstacles,
                lambda_u=0.005,
                ux_min=0.5,
                track_buffer_m=0.0,
                obstacle_clearance_m=0.0,
                eps_s=0.1,
                eps_kappa=0.05,
                convergent_lap=True,
                verbose=False,
            )

            ipopt_times.append(float(ipopt_res.solve_time))
            fatrop_times.append(float(fatrop_res.solve_time))
            ipopt_costs.append(float(ipopt_res.cost))
            fatrop_costs.append(float(fatrop_res.cost))
            ipopt_successes += int(bool(ipopt_res.success))
            fatrop_successes += int(bool(fatrop_res.success))
            ipopt_iters.append(int(ipopt_res.iterations))
            fatrop_iters.append(int(fatrop_res.iterations))
            fatrop_min_clearances.append(float(fatrop_res.min_obstacle_clearance))

        row = {
            "map": map_path.name,
            "N": int(args.N),
            "repeats": int(args.repeats),
            "ipopt_success_rate": ipopt_successes / args.repeats,
            "fatrop_success_rate": fatrop_successes / args.repeats,
            "ipopt_time_med_s": median(ipopt_times),
            "fatrop_time_med_s": median(fatrop_times),
            "speedup_ipopt_over_fatrop": median(ipopt_times) / max(median(fatrop_times), 1e-9),
            "ipopt_iter_med": median(ipopt_iters),
            "fatrop_iter_med": median(fatrop_iters),
            "ipopt_cost_med": median(ipopt_costs),
            "fatrop_cost_med": median(fatrop_costs),
            "fatrop_cost_delta_pct_vs_ipopt": (median(fatrop_costs) - median(ipopt_costs))
            / max(abs(median(ipopt_costs)), 1e-9)
            * 100.0,
            "fatrop_min_clearance_med": median(fatrop_min_clearances),
        }
        rows.append(row)
        print(
            f"{map_path.name}: ipopt={row['ipopt_time_med_s']:.3f}s "
            f"fatrop={row['fatrop_time_med_s']:.3f}s "
            f"speedup={row['speedup_ipopt_over_fatrop']:.3f}x "
            f"cost_delta={row['fatrop_cost_delta_pct_vs_ipopt']:.2f}%"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark: {out_csv}")


if __name__ == "__main__":
    main()
