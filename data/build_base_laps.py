#!/usr/bin/env python3
"""
Build periodic base laps (no-obstacle and obstacle scenarios).

Outputs:
- data/base_laps/<map_id>/<base_id>.npz
- data/base_laps/<map_id>/manifest.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml
from planning import ObstacleCircle, TrajectoryOptimizer
from utils.world import World

from data.schema import sha256_file, sha256_json


def build_world(map_file: Path) -> World:
    return World(str(map_file), map_file.stem, diagnostic_plotting=False)


def build_vehicle() -> object:
    return load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")


def wrap_s_dist(s_a: float, s_b: float, length_m: float) -> float:
    d = (s_a - s_b) % length_m
    if d > 0.5 * length_m:
        d -= length_m
    return d


def sample_obstacles(
    world: World,
    rng: np.random.Generator,
    n_obs: int,
    radius_min: float,
    radius_max: float,
    margin_m: float,
    clearance_m: float,
    vehicle_radius_m: float,
    track_buffer_m: float,
) -> List[ObstacleCircle]:
    length_m = float(world.length_m)
    min_ds = 0.12 * length_m / max(1, n_obs)
    obs: List[ObstacleCircle] = []
    s_vals: List[float] = []

    attempts = 0
    while len(obs) < n_obs and attempts < 4000:
        attempts += 1
        s = float(rng.uniform(0.0, length_m))
        if any(abs(wrap_s_dist(s, s_prev, length_m)) < min_ds for s_prev in s_vals):
            continue

        radius = float(rng.uniform(radius_min, radius_max))
        hw = float(world.track_width_m_LUT(s % length_m)) / 2.0
        required = radius + margin_m + clearance_m + vehicle_radius_m
        e_limit = hw - track_buffer_m - required - 0.1
        if e_limit <= 0.3:
            continue
        e = float(rng.uniform(-e_limit, e_limit))

        obs.append(
            ObstacleCircle(
                s_m=s,
                e_m=e,
                radius_m=radius,
                margin_m=margin_m,
            )
        )
        s_vals.append(s)

    return obs


def write_base_lap(
    out_dir: Path,
    base_id: str,
    map_id: str,
    map_hash: str,
    solver_config: Dict[str, float],
    obstacles: List[ObstacleCircle],
    result,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{base_id}.npz"

    t_arr = result.X[TrajectoryOptimizer.IDX_T, :]
    dt = np.diff(t_arr)

    np.savez_compressed(
        npz_path,
        s_m=result.s_m,
        X_full=result.X.T,
        U=result.U.T,
        dt=dt,
        obstacles=[obs.__dict__ for obs in obstacles],
        solver_config=solver_config,
        map_id=map_id,
        map_hash=map_hash,
    )

    manifest_path = out_dir / "manifest.jsonl"
    entry = {
        "base_id": base_id,
        "map_id": map_id,
        "map_hash": map_hash,
        "npz_path": str(npz_path),
        "solver_config": solver_config,
        "solver_config_hash": sha256_json(solver_config),
        "obstacles": [obs.__dict__ for obs in obstacles],
    }
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_manifest_ids(manifest_path: Path) -> set[str]:
    ids: set[str] = set()
    if not manifest_path.exists():
        return ids
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["base_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build periodic base laps.")
    parser.add_argument("--map-files", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--output-dir", type=str, default="data/base_laps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--lambda-u", type=float, default=0.005)
    parser.add_argument("--ux-min", type=float, default=0.5)
    parser.add_argument("--track-buffer-m", type=float, default=0.0)
    parser.add_argument("--eps-s", type=float, default=0.1)
    parser.add_argument("--eps-kappa", type=float, default=0.05)
    parser.add_argument("--base-laps", type=int, default=6)
    parser.add_argument("--obstacle-laps", type=int, default=8)
    parser.add_argument("--min-obstacles", type=int, default=1)
    parser.add_argument("--max-obstacles", type=int, default=4)
    parser.add_argument("--radius-min", type=float, default=0.8)
    parser.add_argument("--radius-max", type=float, default=1.5)
    parser.add_argument("--clearance-m", type=float, default=0.3)
    parser.add_argument("--margin-m", type=float, default=0.3)
    parser.add_argument("--vehicle-radius-m", type=float, default=0.0)
    parser.add_argument("--obs-window-m", type=float, default=30.0)
    parser.add_argument("--obs-subsamples", type=int, default=11)
    parser.add_argument(
        "--obs-enforce-midpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--obs-init-sigma-m", type=float, default=8.0)
    parser.add_argument("--obs-init-margin-m", type=float, default=0.5)
    parser.add_argument("--accept-min-clearance-m", type=float, default=-0.005)
    parser.add_argument("--ipopt-tol", type=float, default=1e-6)
    parser.add_argument("--ipopt-acceptable-tol", type=float, default=1e-4)
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by keeping existing accepted base laps and appending missing ones.",
    )
    args = parser.parse_args()

    map_files = [Path(p.strip()) for p in args.map_files.split(",") if p.strip()]
    rng = np.random.default_rng(args.seed)

    os.environ["IPOPT_TOL"] = str(args.ipopt_tol)
    os.environ["IPOPT_ACCEPTABLE_TOL"] = str(args.ipopt_acceptable_tol)
    os.environ["IPOPT_MAX_ITER"] = str(args.ipopt_max_iter)

    vehicle = build_vehicle()

    for map_file in map_files:
        if not map_file.exists():
            raise FileNotFoundError(f"Map file not found: {map_file}")

        world = build_world(map_file)
        optimizer = TrajectoryOptimizer(vehicle, world)
        map_hash = sha256_file(str(map_file))
        ds_m = float(world.length_m) / float(args.N)
        map_id = map_file.stem
        base_dir = Path(args.output_dir) / map_id
        base_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = base_dir / "manifest.jsonl"
        existing_ids = load_manifest_ids(manifest_path) if args.resume else set()

        solver_base = {
            "N": int(args.N),
            "ds_m": float(ds_m),
            "lambda_u": float(args.lambda_u),
            "ux_min": float(args.ux_min),
            "track_buffer_m": float(args.track_buffer_m),
            "eps_s": float(args.eps_s),
            "eps_kappa": float(args.eps_kappa),
            "convergent_lap": True,
            "obstacles": [],
            "obstacle_clearance_m": 0.0,
        }

        for idx in range(args.base_laps):
            base_id = f"noobs_{idx:02d}"
            if args.resume and base_id in existing_ids and (base_dir / f"{base_id}.npz").exists():
                print(f"[{map_id}] base lap {base_id} already present; skipping.", flush=True)
                continue
            print(f"[{map_id}] solving base lap {base_id} ({idx + 1}/{args.base_laps})", flush=True)
            result = optimizer.solve(
                N=int(args.N),
                ds_m=ds_m,
                lambda_u=float(args.lambda_u),
                ux_min=float(args.ux_min),
                track_buffer_m=float(args.track_buffer_m),
                obstacles=None,
                obstacle_enforce_midpoints=False,
                obstacle_subsamples_per_segment=1,
                obstacle_use_slack=False,
                obstacle_clearance_m=0.0,
                vehicle_radius_m=float(args.vehicle_radius_m),
                eps_s=float(args.eps_s),
                eps_kappa=float(args.eps_kappa),
                convergent_lap=True,
                verbose=False,
            )
            if not result.success:
                print(f"[{map_id}] base lap {base_id} failed; skipping.")
                continue
            write_base_lap(base_dir, base_id, map_id, map_hash, solver_base, [], result)
            existing_ids.add(base_id)
            print(f"[{map_id}] base lap {base_id} accepted.", flush=True)

        idx = 0
        attempts = 0
        max_attempts = max(args.obstacle_laps * 20, 100)
        while idx < args.obstacle_laps and attempts < max_attempts:
            base_id = f"obs_{idx:02d}"
            if args.resume and base_id in existing_ids and (base_dir / f"{base_id}.npz").exists():
                print(f"[{map_id}] obstacle lap {base_id} already present; skipping.", flush=True)
                idx += 1
                continue
            attempts += 1
            print(
                f"[{map_id}] solving obstacle lap {base_id} ({idx + 1}/{args.obstacle_laps})",
                flush=True,
            )
            n_obs = int(rng.integers(args.min_obstacles, args.max_obstacles + 1))
            obs = sample_obstacles(
                world=world,
                rng=rng,
                n_obs=n_obs,
                radius_min=float(args.radius_min),
                radius_max=float(args.radius_max),
                margin_m=float(args.margin_m),
                clearance_m=float(args.clearance_m),
                vehicle_radius_m=float(args.vehicle_radius_m),
                track_buffer_m=float(args.track_buffer_m),
            )
            if len(obs) < n_obs:
                print(f"[{map_id}] obstacle lap {base_id} only placed {len(obs)}/{n_obs} obstacles; skipping.")
                continue

            solver_config = dict(solver_base)
            solver_config.update(
                {
                    "obstacles": [o.__dict__ for o in obs],
                    "obstacle_clearance_m": float(args.clearance_m),
                    "obstacle_use_slack": False,
                }
            )
            result = optimizer.solve(
                N=int(args.N),
                ds_m=ds_m,
                lambda_u=float(args.lambda_u),
                ux_min=float(args.ux_min),
                track_buffer_m=float(args.track_buffer_m),
                obstacles=obs,
                obstacle_window_m=float(args.obs_window_m),
                obstacle_clearance_m=float(args.clearance_m),
                obstacle_use_slack=False,
                obstacle_enforce_midpoints=bool(args.obs_enforce_midpoints),
                obstacle_subsamples_per_segment=int(args.obs_subsamples),
                obstacle_slack_weight=1e4,
                obstacle_aware_init=True,
                obstacle_init_sigma_m=float(args.obs_init_sigma_m),
                obstacle_init_margin_m=float(args.obs_init_margin_m),
                vehicle_radius_m=float(args.vehicle_radius_m),
                eps_s=float(args.eps_s),
                eps_kappa=float(args.eps_kappa),
                convergent_lap=True,
                verbose=False,
            )
            if not result.success or result.min_obstacle_clearance < float(args.accept_min_clearance_m):
                print(f"[{map_id}] obstacle lap {base_id} failed or collided; skipping.")
                continue
            write_base_lap(base_dir, base_id, map_id, map_hash, solver_config, obs, result)
            existing_ids.add(base_id)
            print(f"[{map_id}] obstacle lap {base_id} accepted.", flush=True)
            idx += 1

        if idx < args.obstacle_laps:
            print(
                f"[{map_id}] warning: only generated {idx}/{args.obstacle_laps} obstacle base laps after {attempts} attempts.",
                flush=True,
            )


if __name__ == "__main__":
    main()
