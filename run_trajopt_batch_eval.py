#!/usr/bin/env python3
"""
Batch evaluation for IPOPT trajectory optimization under randomized obstacle scenarios.

Outputs:
- JSON summary
- CSV per-scenario details
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from models import load_vehicle_from_yaml
from planning import ObstacleCircle, TrajectoryOptimizer
from world.world import World


@dataclass
class AttemptRecord:
    name: str
    N: int
    subsamples: int
    clearance_m: float
    success: bool
    accepted: bool
    cost_s: float
    solve_time_s: float
    iterations: int
    max_obstacle_slack: float
    min_obstacle_clearance_m: float


def create_vehicle(project_root: Path):
    config_file = project_root / "models" / "config" / "vehicle_params_gti.yaml"
    return load_vehicle_from_yaml(config_file)


def wrap_s_dist(s_a: float, s_b: float, length_m: float) -> float:
    d = (s_a - s_b) % length_m
    if d > 0.5 * length_m:
        d -= length_m
    return d


def sample_obstacles(
    world: World,
    rng: np.random.Generator,
    n_obs: int,
    vehicle_radius_m: float,
    track_buffer_m: float,
) -> List[ObstacleCircle]:
    length_m = float(world.length_m)
    min_ds = 0.12 * length_m / max(1, n_obs)
    obs: List[ObstacleCircle] = []
    s_vals = []

    attempts = 0
    while len(obs) < n_obs and attempts < 2000:
        attempts += 1
        s = float(rng.uniform(0.0, length_m))
        if any(abs(wrap_s_dist(s, s_prev, length_m)) < min_ds for s_prev in s_vals):
            continue

        radius = float(rng.uniform(1.2, 1.8))
        margin = float(rng.uniform(0.6, 0.9))
        hw = float(world.track_width_m_LUT(s % length_m)) / 2.0
        # Keep center away from edges by required obstacle radius + small margin.
        e_limit = hw - track_buffer_m - (radius + margin + vehicle_radius_m) - 0.4
        if e_limit <= 0.4:
            continue
        e = float(rng.uniform(-e_limit, e_limit))

        obs.append(
            ObstacleCircle(
                s_m=s,
                e_m=e,
                radius_m=radius,
                margin_m=margin,
            )
        )
        s_vals.append(s)

    return obs


def accepted(result, max_slack: float, min_clearance: float) -> bool:
    return (
        bool(result.success)
        and float(result.max_obstacle_slack) <= max_slack
        and float(result.min_obstacle_clearance) >= min_clearance
    )


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for IPOPT trajectory optimizer")
    parser.add_argument("--num-scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-file", type=str, default="maps/Medium_Oval_Map_260m.mat")
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--obs-subsamples", type=int, default=7)
    parser.add_argument("--obs-window-m", type=float, default=30.0)
    parser.add_argument("--track-buffer-m", type=float, default=0.0)
    parser.add_argument("--lambda-u", type=float, default=1e-3)
    parser.add_argument("--vehicle-radius-m", type=float, default=0.0)
    parser.add_argument("--eps-s", type=float, default=0.1)
    parser.add_argument("--eps-kappa", type=float, default=0.05)
    parser.add_argument("--accept-min-clearance-m", type=float, default=-0.001)
    parser.add_argument("--accept-max-slack", type=float, default=0.0)
    parser.add_argument("--min-obstacles", type=int, default=3)
    parser.add_argument("--max-obstacles", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="results/trajectory_optimization")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    world = World(str(project_root / args.map_file), "BatchEval", diagnostic_plotting=False)
    vehicle = create_vehicle(project_root)
    optimizer = TrajectoryOptimizer(vehicle, world)
    rng = np.random.default_rng(args.seed)
    lambda_u = args.lambda_u

    retry_N_1 = max(args.N, 160)
    retry_subsamples_2 = max(args.obs_subsamples, 11)
    retry_clearance_3 = 0.10
    retry_N_3 = max(retry_N_1, 180)
    retry_subsamples_3 = max(retry_subsamples_2, 13)

    attempt_cfgs = [
        ("baseline", args.N, args.obs_subsamples, 0.0),
        ("retry_higher_N", retry_N_1, args.obs_subsamples, 0.0),
        ("retry_higher_subsamples", retry_N_1, retry_subsamples_2, 0.0),
        ("retry_more_conservative", retry_N_3, retry_subsamples_3, retry_clearance_3),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"trajopt_batch_eval_{timestamp}.csv"
    json_path = output_dir / f"trajopt_batch_eval_{timestamp}.json"

    rows = []
    accepted_count = 0
    success_count = 0

    for i in range(args.num_scenarios):
        n_obs = int(rng.integers(args.min_obstacles, args.max_obstacles + 1))
        obstacles = sample_obstacles(world, rng, n_obs, args.vehicle_radius_m, args.track_buffer_m)
        if len(obstacles) < n_obs:
            rows.append(
                {
                    "scenario_id": i,
                    "n_obstacles": len(obstacles),
                    "status": "sampling_failed",
                }
            )
            continue

        attempt_records: List[AttemptRecord] = []
        chosen = None

        for name, N_i, subs_i, clear_i in attempt_cfgs:
            ds_i = world.length_m / N_i
            res = optimizer.solve(
                N=N_i,
                ds_m=ds_i,
                track_buffer_m=args.track_buffer_m,
                obstacles=obstacles,
                obstacle_window_m=args.obs_window_m,
                obstacle_clearance_m=clear_i,
                obstacle_subsamples_per_segment=subs_i,
                lambda_u=lambda_u,
                vehicle_radius_m=args.vehicle_radius_m,
                eps_s=args.eps_s,
                eps_kappa=args.eps_kappa,
                obstacle_aware_init=True,
                obstacle_init_sigma_m=8.0,
                obstacle_init_margin_m=0.3,
                ux_min=3.0,
                convergent_lap=True,
                verbose=False,
            )
            rec = AttemptRecord(
                name=name,
                N=N_i,
                subsamples=subs_i,
                clearance_m=clear_i,
                success=bool(res.success),
                accepted=accepted(res, args.accept_max_slack, args.accept_min_clearance_m),
                cost_s=float(res.cost),
                solve_time_s=float(res.solve_time),
                iterations=int(res.iterations),
                max_obstacle_slack=float(res.max_obstacle_slack),
                min_obstacle_clearance_m=float(res.min_obstacle_clearance),
            )
            attempt_records.append(rec)
            if rec.accepted:
                chosen = rec
                break

        if chosen is None:
            chosen = attempt_records[-1]

        success_any = any(r.success for r in attempt_records)
        accepted_any = any(r.accepted for r in attempt_records)
        success_count += int(success_any)
        accepted_count += int(accepted_any)

        rows.append(
            {
                "scenario_id": i,
                "n_obstacles": n_obs,
                "status": "ok",
                "accepted": accepted_any,
                "success_any": success_any,
                "selected_attempt": chosen.name,
                "selected_N": chosen.N,
                "selected_subsamples": chosen.subsamples,
                "selected_clearance_m": chosen.clearance_m,
                "selected_cost_s": chosen.cost_s,
                "selected_solve_time_s": chosen.solve_time_s,
                "selected_iterations": chosen.iterations,
                "selected_max_obstacle_slack": chosen.max_obstacle_slack,
                "selected_min_obstacle_clearance_m": chosen.min_obstacle_clearance_m,
                "attempts": json.dumps([asdict(a) for a in attempt_records]),
            }
        )

        print(
            f"[{i+1:03d}/{args.num_scenarios:03d}] "
            f"accepted={accepted_any} success_any={success_any} "
            f"attempt={chosen.name} clear={chosen.min_obstacle_clearance_m:.4f}m "
            f"time={chosen.solve_time_s:.2f}s"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scenario_id", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    summary = {
        "timestamp": timestamp,
        "num_scenarios": args.num_scenarios,
        "num_valid_sampled": n_ok,
        "solver_success_rate_over_valid": (success_count / n_ok) if n_ok > 0 else 0.0,
        "accepted_rate_over_valid": (accepted_count / n_ok) if n_ok > 0 else 0.0,
        "accept_min_clearance_m": args.accept_min_clearance_m,
        "accept_max_slack": args.accept_max_slack,
        "map_file": args.map_file,
        "base_N": args.N,
        "base_obs_subsamples": args.obs_subsamples,
        "obs_window_m": args.obs_window_m,
        "track_buffer_m": args.track_buffer_m,
        "lambda_u": lambda_u,
        "vehicle_radius_m": args.vehicle_radius_m,
        "eps_s": args.eps_s,
        "eps_kappa": args.eps_kappa,
        "seed": args.seed,
        "csv_path": str(csv_path),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBatch evaluation complete.")
    print(f"  Valid scenarios: {n_ok}/{args.num_scenarios}")
    print(f"  Solver success rate: {summary['solver_success_rate_over_valid']:.3f}")
    print(f"  Acceptance rate: {summary['accepted_rate_over_valid']:.3f}")
    print(f"  Summary JSON: {json_path}")
    print(f"  Details CSV:  {csv_path}")


if __name__ == "__main__":
    main()
