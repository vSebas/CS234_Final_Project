#!/usr/bin/env python3
"""
Generate trajectory optimization datasets per PLAN.md schema.

Stage A: no obstacles, non-periodic runs with randomized lateral offset (e0).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer
from world.world import World

from data.schema import EpisodeHeader, compute_rtg, sha256_file, sha256_json


def _yaw_wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def build_world(map_file: Path) -> World:
    return World(str(map_file), map_file.stem, diagnostic_plotting=False)


def build_vehicle() -> object:
    return load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")


def sample_initial_conditions(
    rng: np.random.Generator,
    world: World,
    track_buffer_m: float,
    ux_min: float,
) -> Dict[str, float]:
    s0 = rng.uniform(0.0, float(world.length_m))
    hw = 0.5 * float(np.array(world.track_width_m_LUT(np.array([s0]))).squeeze())
    e_max = max(0.05, hw - track_buffer_m)
    e0 = rng.uniform(-e_max, e_max)
    return {
        "s_offset_m": float(s0),
        "e0": float(e0),
        "ux0": float(ux_min),
    }


def compute_global_pose(world: World, s_m: np.ndarray, e_m: np.ndarray) -> Dict[str, np.ndarray]:
    s_mod = np.mod(s_m, world.length_m)
    posE_cl = np.array(world.posE_m_interp_fcn(s_mod)).astype(float).squeeze()
    posN_cl = np.array(world.posN_m_interp_fcn(s_mod)).astype(float).squeeze()
    psi_cl = np.array(world.psi_rad_interp_fcn(s_mod)).astype(float).squeeze()
    posE = posE_cl - e_m * np.sin(psi_cl)
    posN = posN_cl + e_m * np.cos(psi_cl)
    return {"pos_E": posE, "pos_N": posN, "psi_cl": psi_cl}


def compute_track_features(world: World, s_m: np.ndarray) -> Dict[str, np.ndarray]:
    s_mod = np.mod(s_m, world.length_m)
    kappa = np.array(world.psi_s_radpm_LUT(s_mod)).astype(float).squeeze()
    half_width = 0.5 * np.array(world.track_width_m_LUT(s_mod)).astype(float).squeeze()
    if hasattr(world, "grade_rad_LUT"):
        grade = np.array(world.grade_rad_LUT(s_mod)).astype(float).squeeze()
    else:
        grade = np.zeros_like(kappa)
    if hasattr(world, "bank_rad_LUT"):
        bank = np.array(world.bank_rad_LUT(s_mod)).astype(float).squeeze()
    else:
        bank = np.zeros_like(kappa)
    return {
        "kappa": kappa,
        "half_width": half_width,
        "grade": grade,
        "bank": bank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate no-obstacle dataset (Stage A).")
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--output-dir", type=str, default="data/datasets/oval_no_obstacles_1k")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--lambda-u", type=float, default=0.005)
    parser.add_argument("--ux-min", type=float, default=0.5)
    parser.add_argument("--track-buffer-m", type=float, default=0.0)
    parser.add_argument("--eps-s", type=float, default=0.1)
    parser.add_argument("--eps-kappa", type=float, default=0.05)
    parser.add_argument("--ipopt-tol", type=float, default=1e-6)
    parser.add_argument("--ipopt-acceptable-tol", type=float, default=1e-4)
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument(
        "--mode",
        type=str,
        default="periodic_shift",
        choices=["periodic_shift"],
        help="Stage A mode: periodic lap with circular shifts.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    map_file = Path(args.map_file)
    if not map_file.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    rng = np.random.default_rng(args.seed)
    os.environ["IPOPT_TOL"] = str(args.ipopt_tol)
    os.environ["IPOPT_ACCEPTABLE_TOL"] = str(args.ipopt_acceptable_tol)
    os.environ["IPOPT_MAX_ITER"] = str(args.ipopt_max_iter)

    world = build_world(map_file)
    vehicle = build_vehicle()
    optimizer = TrajectoryOptimizer(vehicle, world)

    ds_m = world.length_m / float(args.N)
    map_hash = sha256_file(str(map_file))

    solver_config = {
        "N": int(args.N),
        "ds_m": float(ds_m),
        "lambda_u": float(args.lambda_u),
        "ux_min": float(args.ux_min),
        "track_buffer_m": float(args.track_buffer_m),
        "eps_s": float(args.eps_s),
        "eps_kappa": float(args.eps_kappa),
        "convergent_lap": True,
        "obstacles": "none",
    }
    solver_config_hash = sha256_json(solver_config)

    manifest_f = open(manifest_path, "w", encoding="utf-8")
    t_start = time.time()
    successes = 0

    # Solve once with periodic closure, then generate episodes by circular shift.
    result = optimizer.solve(
        N=int(args.N),
        ds_m=float(ds_m),
        lambda_u=float(args.lambda_u),
        ux_min=float(args.ux_min),
        ux_max=None,
        track_buffer_m=float(args.track_buffer_m),
        obstacles=None,
        obstacle_window_m=30.0,
        obstacle_clearance_m=0.0,
        obstacle_use_slack=False,
        obstacle_enforce_midpoints=False,
        obstacle_subsamples_per_segment=1,
        obstacle_slack_weight=1e4,
        vehicle_radius_m=0.0,
        eps_s=float(args.eps_s),
        eps_kappa=float(args.eps_kappa),
        convergent_lap=True,
        verbose=False,
    )
    if not result.success:
        raise RuntimeError("Periodic solve failed; cannot generate dataset.")

    s_m = result.s_m.copy()
    X = result.X.copy()
    U = result.U.copy()
    t_arr = X[TrajectoryOptimizer.IDX_T, :].copy()
    dt_base = np.diff(t_arr)

    for episode_idx in range(args.num_episodes):
        k0 = int(rng.integers(0, len(s_m)))
        s_roll = np.roll(s_m, -k0)
        s_offset = float(s_roll[0])
        s_shift = np.mod(s_roll - s_roll[0], world.length_m)

        X_roll = np.roll(X, -k0, axis=1)
        U_roll = np.roll(U, -k0, axis=1)
        dt_roll = np.roll(dt_base, -k0)
        t_new = np.concatenate([[0.0], np.cumsum(dt_roll)])
        X_roll[TrajectoryOptimizer.IDX_T, :] = t_new

        e_arr = X_roll[TrajectoryOptimizer.IDX_E, :].copy()
        dpsi_arr = X_roll[TrajectoryOptimizer.IDX_DPSI, :].copy()

        pose = compute_global_pose(world, s_shift, e_arr)
        yaw_world = _yaw_wrap(pose["psi_cl"] + dpsi_arr)
        track_feat = compute_track_features(world, s_shift)

        reward = -dt_roll
        rtg = compute_rtg(reward)

        episode_id = f"oval_no_obs_{episode_idx:06d}"
        npz_path = episodes_dir / f"{episode_id}.npz"

        np.savez_compressed(
            npz_path,
            s_m=s_shift,
            X_full=X_roll.T,
            U=U_roll.T,
            dt=dt_roll,
            reward=reward,
            rtg=rtg,
            pos_E=pose["pos_E"],
            pos_N=pose["pos_N"],
            yaw_world=yaw_world,
            kappa=track_feat["kappa"],
            half_width=track_feat["half_width"],
            grade=track_feat["grade"],
            bank=track_feat["bank"],
            s_offset_m=s_offset,
        )

        header = EpisodeHeader(
            episode_id=episode_id,
            map_id=map_file.stem,
            map_hash=map_hash,
            solver_config=solver_config,
            solver_config_hash=solver_config_hash,
            discretization={"N": int(args.N), "ds_m": float(ds_m)},
            obstacles=[],
            s_offset_m=float(s_offset),
            npz_path=str(npz_path),
        )
        manifest_f.write(json.dumps(header.to_dict()) + "\n")
        successes += 1

        if (episode_idx + 1) % args.save_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[{episode_idx + 1}/{args.num_episodes}] "
                f"accepted={successes} elapsed={elapsed:.1f}s"
            )

    manifest_f.close()
    elapsed = time.time() - t_start
    print(f"Done. accepted={successes}/{args.num_episodes} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
