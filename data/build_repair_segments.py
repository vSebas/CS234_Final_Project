#!/usr/bin/env python3
"""
Build short-horizon repair segments (Fix B).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

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


def build_terminal_mask() -> List[bool]:
    mask = [False] * 8
    # ux, uy, r, e, dpsi
    mask[TrajectoryOptimizer.IDX_UX] = True
    mask[TrajectoryOptimizer.IDX_UY] = True
    mask[TrajectoryOptimizer.IDX_R] = True
    mask[TrajectoryOptimizer.IDX_E] = True
    mask[TrajectoryOptimizer.IDX_DPSI] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repair segments (Fix B).")
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--base-laps-dir", type=str, default="data/base_laps")
    parser.add_argument("--output-dir", type=str, default="data/datasets/repair_segments")
    parser.add_argument("--num-segments", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--H", type=int, default=50)
    parser.add_argument("--lambda-u", type=float, default=0.005)
    parser.add_argument("--ux-min", type=float, default=0.5)
    parser.add_argument("--track-buffer-m", type=float, default=0.0)
    parser.add_argument("--eps-s", type=float, default=0.1)
    parser.add_argument("--eps-kappa", type=float, default=0.05)
    parser.add_argument("--e-perturb-m", type=float, default=0.5)
    parser.add_argument("--dpsi-perturb-rad", type=float, default=0.10)
    parser.add_argument("--terminal-weight", type=float, default=5.0)
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
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by appending only missing repair segments.",
    )
    args = parser.parse_args()

    map_file = Path(args.map_file)
    base_dir = Path(args.base_laps_dir) / map_file.stem
    if not base_dir.exists():
        raise FileNotFoundError(f"Base laps dir not found: {base_dir}")

    os.environ["IPOPT_TOL"] = str(args.ipopt_tol)
    os.environ["IPOPT_ACCEPTABLE_TOL"] = str(args.ipopt_acceptable_tol)
    os.environ["IPOPT_MAX_ITER"] = str(args.ipopt_max_iter)

    world = build_world(map_file)
    vehicle = build_vehicle()
    optimizer = TrajectoryOptimizer(vehicle, world)
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

    term_mask = build_terminal_mask()
    manifest_f = open(manifest_path, "a", encoding="utf-8")
    t_start = time.time()
    successes = existing_count

    for seg_idx in range(args.num_segments):
        if seg_idx < existing_count:
            continue
        if seg_idx > 0 and seg_idx % max(1, args.save_every) == 0:
            elapsed = time.time() - t_start
            print(
                f"[{seg_idx}/{args.num_segments}] repairs accepted={successes} elapsed={elapsed:.1f}s",
                flush=True,
            )
        base_path = rng.choice(base_files)
        base_id = base_path.stem
        data = np.load(base_path, allow_pickle=True)
        s_m = data["s_m"].astype(float)
        X_full = data["X_full"].astype(float)
        U_full = data["U"].astype(float)
        obstacles = data.get("obstacles", [])
        if isinstance(obstacles, np.ndarray):
            obstacles = obstacles.tolist()
        solver_config = data.get("solver_config", {})
        solver_config = solver_config.item() if isinstance(solver_config, np.ndarray) else solver_config
        solver_config_seg = dict(solver_config) if isinstance(solver_config, dict) else {}
        solver_config_seg.update(
            {
                "H": int(args.H),
                "lambda_u": float(args.lambda_u),
                "ux_min": float(args.ux_min),
                "track_buffer_m": float(args.track_buffer_m),
                "eps_s": float(args.eps_s),
                "eps_kappa": float(args.eps_kappa),
                "terminal_weight": float(args.terminal_weight),
                "obstacle_window_m": float(args.obs_window_m),
                "obstacle_subsamples_per_segment": int(args.obs_subsamples),
                "obstacle_enforce_midpoints": bool(args.obs_enforce_midpoints),
                "obstacle_init_sigma_m": float(args.obs_init_sigma_m),
                "obstacle_init_margin_m": float(args.obs_init_margin_m),
                "accept_min_clearance_m": float(args.accept_min_clearance_m),
            }
        )
        solver_config_hash = sha256_json(solver_config_seg) if solver_config_seg else ""
        obstacles_list = list(obstacles) if isinstance(obstacles, (list, tuple)) else []

        N_base = X_full.shape[0] - 1
        k0 = int(rng.integers(0, N_base))
        idxs = (k0 + np.arange(args.H + 1)) % (N_base + 1)

        X_seg = X_full[idxs, :].T
        U_seg = U_full[idxs, :].T
        s0_abs = float(s_m[k0])

        x0 = X_seg[:, 0].copy()
        e0 = float(x0[TrajectoryOptimizer.IDX_E]) + float(rng.uniform(-args.e_perturb_m, args.e_perturb_m))
        dpsi0 = float(x0[TrajectoryOptimizer.IDX_DPSI]) + float(
            rng.uniform(-args.dpsi_perturb_rad, args.dpsi_perturb_rad)
        )

        hw = float(world.track_width_m_LUT(s0_abs % world.length_m)) / 2.0
        e0 = float(np.clip(e0, -hw + args.track_buffer_m, hw - args.track_buffer_m))
        x0[TrajectoryOptimizer.IDX_E] = e0
        x0[TrajectoryOptimizer.IDX_DPSI] = dpsi0

        X_init = X_seg.copy()
        X_init[TrajectoryOptimizer.IDX_E, 0] = e0
        X_init[TrajectoryOptimizer.IDX_DPSI, 0] = dpsi0

        term_state = X_seg[:, -1].copy()

        result = optimizer.solve(
            N=int(args.H),
            ds_m=float(s_m[1] - s_m[0]),
            x0=x0,
            X_init=X_init,
            U_init=U_seg,
            lambda_u=float(args.lambda_u),
            ux_min=float(args.ux_min),
            track_buffer_m=float(args.track_buffer_m),
            obstacles=obstacles_list if obstacles_list else None,
            obstacle_window_m=float(args.obs_window_m),
            obstacle_clearance_m=0.3 if obstacles_list else 0.0,
            obstacle_use_slack=False,
            obstacle_enforce_midpoints=bool(args.obs_enforce_midpoints),
            obstacle_subsamples_per_segment=int(args.obs_subsamples),
            obstacle_slack_weight=1e4,
            obstacle_aware_init=True,
            obstacle_init_sigma_m=float(args.obs_init_sigma_m),
            obstacle_init_margin_m=float(args.obs_init_margin_m),
            vehicle_radius_m=0.0,
            eps_s=float(args.eps_s),
            eps_kappa=float(args.eps_kappa),
            convergent_lap=False,
            s0_offset_m=s0_abs,
            terminal_state=term_state,
            terminal_mask=term_mask,
            terminal_weight=float(args.terminal_weight),
            verbose=False,
        )

        if not result.success:
            continue
        if obstacles_list and result.min_obstacle_clearance < float(args.accept_min_clearance_m):
            continue

        s_abs = result.s_m.copy()
        X_opt = result.X.copy()
        U_opt = result.U.copy()
        t_arr = X_opt[TrajectoryOptimizer.IDX_T, :]
        dt = np.diff(t_arr)

        e_arr = X_opt[TrajectoryOptimizer.IDX_E, :].copy()
        dpsi_arr = X_opt[TrajectoryOptimizer.IDX_DPSI, :].copy()
        pose = compute_global_pose(world, s_abs, e_arr)
        yaw_world = _yaw_wrap(pose["psi_cl"] + dpsi_arr)
        track_feat = compute_track_features(world, s_abs)

        reward = -dt
        rtg = compute_rtg(reward)

        episode_id = f"{map_file.stem}_repair_{seg_idx:06d}"
        npz_path = episodes_dir / f"{episode_id}.npz"

        np.savez_compressed(
            npz_path,
            s_m=s_abs,
            X_full=X_opt.T,
            U=U_opt.T,
            dt=dt,
            reward=reward,
            rtg=rtg,
            pos_E=pose["pos_E"],
            pos_N=pose["pos_N"],
            yaw_world=yaw_world,
            kappa=track_feat["kappa"],
            half_width=track_feat["half_width"],
            grade=track_feat["grade"],
            bank=track_feat["bank"],
            s_offset_m=s0_abs,
        )

        header = EpisodeHeader(
            episode_id=episode_id,
            episode_type="repair",
            map_id=map_file.stem,
            map_hash=map_hash,
            base_id=base_id,
            solver_config=solver_config_seg,
            solver_config_hash=solver_config_hash,
            discretization={"N": int(args.H), "ds_m": float(s_m[1] - s_m[0])},
            obstacles=obstacles_list,
            s_offset_m=float(s0_abs),
            npz_path=str(npz_path),
        )
        manifest_f.write(json.dumps(header.to_dict()) + "\n")
        successes += 1

        if successes % args.save_every == 0:
            elapsed = time.time() - t_start
            print(f"[{successes}/{args.num_segments}] accepted elapsed={elapsed:.1f}s", flush=True)

    manifest_f.close()
    elapsed = time.time() - t_start
    print(f"Done. accepted={successes}/{args.num_segments} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
