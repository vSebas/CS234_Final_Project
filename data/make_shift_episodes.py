#!/usr/bin/env python3
"""
Generate shift episodes from periodic base laps.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.world import World
from planning import TrajectoryOptimizer
from data.schema import EpisodeHeader, compute_rtg, sha256_file, sha256_json


def _yaw_wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


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
    parser = argparse.ArgumentParser(description="Generate shift episodes from base laps.")
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--base-laps-dir", type=str, default="data/base_laps")
    parser.add_argument("--output-dir", type=str, default="data/datasets/shift_episodes")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument(
        "--all-shifts",
        action="store_true",
        help="Generate all unique shifts for every base lap (k0=0..N).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by appending only missing shift episodes.",
    )
    args = parser.parse_args()

    map_file = Path(args.map_file)
    base_dir = Path(args.base_laps_dir) / map_file.stem
    if not base_dir.exists():
        raise FileNotFoundError(f"Base laps dir not found: {base_dir}")

    world = World(str(map_file), map_file.stem, diagnostic_plotting=False)
    map_hash = sha256_file(str(map_file))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    existing_count = 0
    if args.resume and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())

    rng = np.random.default_rng(args.seed)
    base_files = sorted(base_dir.glob("*.npz"))
    if not base_files:
        raise FileNotFoundError(f"No base laps found in {base_dir}")

    manifest_f = open(manifest_path, "a", encoding="utf-8")
    t_start = time.time()
    episode_idx = existing_count
    successes = existing_count

    def _emit_shift(base_id: str, solver_config: dict, solver_config_hash: str, obstacles, k0: int, s_m, X, U, dt_base):
        nonlocal episode_idx, successes
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

        episode_id = f"{map_file.stem}_{episode_idx:06d}"
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
            episode_type="shift",
            map_id=map_file.stem,
            map_hash=map_hash,
            base_id=base_id,
            solver_config=solver_config,
            solver_config_hash=solver_config_hash,
            discretization={"N": int(len(s_shift) - 1), "ds_m": float(s_m[1] - s_m[0])},
            obstacles=list(obstacles) if isinstance(obstacles, (list, tuple)) else [],
            s_offset_m=float(s_offset),
            npz_path=str(npz_path),
        )
        manifest_f.write(json.dumps(header.to_dict()) + "\n")
        successes += 1
        episode_idx += 1

        if episode_idx % args.save_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[{episode_idx}{'' if args.all_shifts else f'/{args.num_episodes}'}] "
                f"accepted={successes} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if args.all_shifts:
        slot_idx = 0
        for base_path in base_files:
            base_id = base_path.stem
            data = np.load(base_path, allow_pickle=True)
            s_m = data["s_m"].astype(float)
            X = data["X_full"].astype(float).T
            U = data["U"].astype(float).T
            dt_base = data["dt"].astype(float)
            obstacles = data.get("obstacles", [])
            if isinstance(obstacles, np.ndarray):
                obstacles = obstacles.tolist()
            solver_config = data.get("solver_config", {})
            solver_config = solver_config.item() if isinstance(solver_config, np.ndarray) else solver_config
            solver_config_hash = sha256_json(solver_config) if solver_config else ""

            for k0 in range(len(s_m)):
                if slot_idx < existing_count:
                    slot_idx += 1
                    continue
                _emit_shift(base_id, solver_config, solver_config_hash, obstacles, k0, s_m, X, U, dt_base)
                slot_idx += 1
    else:
        while episode_idx < args.num_episodes:
            base_path = rng.choice(base_files)
            base_id = base_path.stem
            data = np.load(base_path, allow_pickle=True)
            s_m = data["s_m"].astype(float)
            X = data["X_full"].astype(float).T
            U = data["U"].astype(float).T
            dt_base = data["dt"].astype(float)
            obstacles = data.get("obstacles", [])
            if isinstance(obstacles, np.ndarray):
                obstacles = obstacles.tolist()
            solver_config = data.get("solver_config", {})
            solver_config = solver_config.item() if isinstance(solver_config, np.ndarray) else solver_config
            solver_config_hash = sha256_json(solver_config) if solver_config else ""

            max_shifts = min(len(s_m), args.num_episodes - episode_idx)
            k0_list = rng.choice(len(s_m), size=max_shifts, replace=False)

            for k0 in k0_list:
                if episode_idx >= args.num_episodes:
                    break
                _emit_shift(base_id, solver_config, solver_config_hash, obstacles, k0, s_m, X, U, dt_base)

    manifest_f.close()
    elapsed = time.time() - t_start
    if args.all_shifts:
        print(f"Done. accepted={successes} elapsed={elapsed:.1f}s", flush=True)
    else:
        print(f"Done. accepted={successes}/{args.num_episodes} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
