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
from typing import Dict, List, Tuple

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


def build_initial_masked_state(x_state: np.ndarray) -> List[float | None]:
    masked = [None] * len(x_state)
    for idx in (
        TrajectoryOptimizer.IDX_UX,
        TrajectoryOptimizer.IDX_UY,
        TrajectoryOptimizer.IDX_R,
        TrajectoryOptimizer.IDX_E,
        TrajectoryOptimizer.IDX_DPSI,
    ):
        masked[idx] = float(x_state[idx])
    return masked


def load_existing_repairs(manifest_path: Path) -> Tuple[int, int]:
    accepted = 0
    next_episode_idx = 0
    if not manifest_path.exists():
        return accepted, next_episode_idx

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            accepted += 1
            row = json.loads(line)
            episode_id = str(row.get("episode_id", ""))
            try:
                suffix = int(episode_id.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                continue
            next_episode_idx = max(next_episode_idx, suffix + 1)
    return accepted, next_episode_idx


def load_repair_state(state_path: Path) -> Tuple[int, dict | None]:
    if not state_path.exists():
        return 0, None
    with open(state_path, "r", encoding="utf-8") as f:
        row = json.load(f)
    attempts = int(row.get("attempts", 0))
    rng_state = row.get("rng_state")
    return attempts, rng_state


def save_repair_state(state_path: Path, attempts: int, rng: np.random.Generator) -> None:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "attempts": int(attempts),
                "rng_state": rng.bit_generator.state,
            },
            f,
        )


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_hotspot_positions(hotspot_json: Path | None, map_id: str) -> np.ndarray:
    if hotspot_json is None or not hotspot_json.exists():
        return np.zeros((0,), dtype=float)
    with open(hotspot_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if map_id in payload:
            payload = payload[map_id]
        elif "hotspots_s_m" in payload:
            payload = payload["hotspots_s_m"]

    if not isinstance(payload, list):
        return np.zeros((0,), dtype=float)
    return np.asarray([float(x) for x in payload], dtype=float)


def circular_distance_m(a: np.ndarray, b: np.ndarray, length_m: float) -> np.ndarray:
    diff = np.abs(a - b)
    return np.minimum(diff, length_m - diff)


def pick_repair_start_index(
    rng: np.random.Generator,
    s_m: np.ndarray,
    obstacles_list: List[Dict],
    H: int,
    length_m: float,
    hard_mode: bool,
    hotspot_s_m: np.ndarray,
    hotspot_window_m: float,
    obstacle_bias_window_m: float,
    hotspot_bias_prob: float,
    obstacle_bias_prob: float,
) -> Tuple[int, Dict[str, float | str | bool]]:
    n_base = len(s_m) - 1
    candidate_max = max(1, n_base)
    candidate_indices = np.arange(candidate_max, dtype=int)

    metadata: Dict[str, float | str | bool] = {
        "hard_mode": bool(hard_mode),
        "sampling_reason": "uniform",
        "near_obstacle_start": False,
        "near_hotspot_start": False,
        "start_min_obstacle_ds_m": float("inf"),
        "start_min_hotspot_ds_m": float("inf"),
    }

    if not hard_mode:
        return int(rng.integers(0, candidate_max)), metadata

    s_candidates = s_m[candidate_indices]
    obstacle_near = np.zeros(candidate_indices.shape, dtype=bool)
    obstacle_ds = np.full(candidate_indices.shape, np.inf, dtype=float)
    if obstacles_list:
        obs_s = np.asarray(
            [float(obs.get("s_m", obs.get("s_obs", obs.get("s", 0.0)))) for obs in obstacles_list],
            dtype=float,
        )
        dmat = np.stack([circular_distance_m(s_candidates, s_obs, length_m) for s_obs in obs_s], axis=1)
        obstacle_ds = np.min(dmat, axis=1)
        obstacle_near = obstacle_ds <= obstacle_bias_window_m

    hotspot_near = np.zeros(candidate_indices.shape, dtype=bool)
    hotspot_ds = np.full(candidate_indices.shape, np.inf, dtype=float)
    if hotspot_s_m.size > 0:
        dmat = np.stack([circular_distance_m(s_candidates, s_h, length_m) for s_h in hotspot_s_m], axis=1)
        hotspot_ds = np.min(dmat, axis=1)
        hotspot_near = hotspot_ds <= hotspot_window_m

    if hotspot_near.any() and rng.random() < hotspot_bias_prob:
        viable = candidate_indices[hotspot_near]
        k0 = int(rng.choice(viable))
        metadata["sampling_reason"] = "hotspot"
    elif obstacle_near.any() and rng.random() < obstacle_bias_prob:
        viable = candidate_indices[obstacle_near]
        k0 = int(rng.choice(viable))
        metadata["sampling_reason"] = "obstacle"
    else:
        k0 = int(rng.integers(0, candidate_max))

    metadata["near_obstacle_start"] = bool(np.isfinite(obstacle_ds[k0]) and obstacle_ds[k0] <= obstacle_bias_window_m)
    metadata["near_hotspot_start"] = bool(np.isfinite(hotspot_ds[k0]) and hotspot_ds[k0] <= hotspot_window_m)
    metadata["start_min_obstacle_ds_m"] = float(obstacle_ds[k0]) if np.isfinite(obstacle_ds[k0]) else float("inf")
    metadata["start_min_hotspot_ds_m"] = float(hotspot_ds[k0]) if np.isfinite(hotspot_ds[k0]) else float("inf")
    return k0, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repair segments (Fix B).")
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--base-laps-dir", type=str, default="data/base_laps")
    parser.add_argument("--output-dir", type=str, default="data/datasets/repair_segments")
    parser.add_argument("--num-segments", type=int, default=200, help="Target number of accepted repair segments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--lambda-u", type=float, default=0.005)
    parser.add_argument("--ux-min", type=float, default=0.5)
    parser.add_argument("--track-buffer-m", type=float, default=0.0)
    parser.add_argument("--eps-s", type=float, default=0.1)
    parser.add_argument("--eps-kappa", type=float, default=0.05)
    parser.add_argument("--e-perturb-m", type=float, default=1.0)
    parser.add_argument("--dpsi-perturb-rad", type=float, default=0.10)
    parser.add_argument("--uy-perturb-mps", type=float, default=0.0)
    parser.add_argument("--r-perturb-radps", type=float, default=0.0)
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
    parser.add_argument("--ipopt-tol", type=float, default=1e-5)
    parser.add_argument("--ipopt-acceptable-tol", type=float, default=1e-3)
    parser.add_argument("--ipopt-max-iter", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--hard-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate a hard-repair shard with biased starts, mixed horizons, and metadata.",
    )
    parser.add_argument(
        "--hard-horizons",
        type=str,
        default="20,40,60",
        help="Comma-separated horizon values used in hard mode.",
    )
    parser.add_argument(
        "--hard-horizon-probs",
        type=str,
        default="0.6,0.25,0.15",
        help="Comma-separated probabilities for hard horizons.",
    )
    parser.add_argument("--hotspot-json", type=str, default=None, help="Optional JSON file with hotspot s-positions by map.")
    parser.add_argument("--hotspot-window-m", type=float, default=20.0)
    parser.add_argument("--obstacle-bias-window-m", type=float, default=25.0)
    parser.add_argument("--hotspot-bias-prob", type=float, default=0.6)
    parser.add_argument("--obstacle-bias-prob", type=float, default=0.9)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum solve attempts. Defaults to 5x target repairs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by appending only missing accepted repair segments toward the target.",
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
    hotspot_s_m = load_hotspot_positions(Path(args.hotspot_json) if args.hotspot_json else None, map_file.stem)
    hard_horizons = parse_int_list(args.hard_horizons)
    hard_horizon_probs = parse_float_list(args.hard_horizon_probs)
    if args.hard_mode:
        if not hard_horizons:
            raise ValueError("--hard-horizons must contain at least one horizon in hard mode")
        if len(hard_horizons) != len(hard_horizon_probs):
            raise ValueError("--hard-horizons and --hard-horizon-probs must have the same length")
        probs = np.asarray(hard_horizon_probs, dtype=float)
        if np.any(probs < 0):
            raise ValueError("--hard-horizon-probs must be nonnegative")
        if probs.sum() <= 0:
            raise ValueError("--hard-horizon-probs must sum to a positive value")
        hard_horizon_probs = (probs / probs.sum()).tolist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    state_path = output_dir / "repair_state.json"
    existing_count = 0
    next_episode_idx = 0
    if args.resume:
        existing_count, next_episode_idx = load_existing_repairs(manifest_path)

    rng = np.random.default_rng(args.seed)
    attempts = 0
    if args.resume:
        attempts, rng_state = load_repair_state(state_path)
        if rng_state is not None:
            rng.bit_generator.state = rng_state
    base_files = sorted(base_dir.glob("*.npz"))
    if not base_files:
        raise FileNotFoundError(f"No base laps found in {base_dir}")

    term_mask = build_terminal_mask()
    manifest_f = open(manifest_path, "a", encoding="utf-8")
    t_start = time.time()
    successes = existing_count
    if successes >= args.num_segments:
        manifest_f.close()
        print(f"Done. accepted={successes}/{args.num_segments} attempts={attempts} elapsed=0.0s", flush=True)
        return

    max_attempts = args.max_attempts if args.max_attempts is not None else max(args.num_segments * 5, args.num_segments)

    while successes < args.num_segments and attempts < max_attempts:
        attempts += 1
        if attempts > 1 and attempts % max(1, args.save_every) == 0:
            elapsed = time.time() - t_start
            print(
                f"[attempt {attempts}/{max_attempts}] repairs accepted={successes}/{args.num_segments} elapsed={elapsed:.1f}s",
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
                "hard_mode": bool(args.hard_mode),
            }
        )
        obstacles_list = list(obstacles) if isinstance(obstacles, (list, tuple)) else []

        H_cur = int(args.H)
        if args.hard_mode:
            H_cur = int(rng.choice(np.asarray(hard_horizons, dtype=int), p=np.asarray(hard_horizon_probs, dtype=float)))
        N_base = X_full.shape[0] - 1
        if H_cur > N_base:
            H_cur = N_base
        solver_config_seg["H"] = int(H_cur)
        solver_config_hash = sha256_json(solver_config_seg) if solver_config_seg else ""
        k0, hard_meta = pick_repair_start_index(
            rng=rng,
            s_m=s_m,
            obstacles_list=obstacles_list,
            H=H_cur,
            length_m=world.length_m,
            hard_mode=bool(args.hard_mode),
            hotspot_s_m=hotspot_s_m,
            hotspot_window_m=float(args.hotspot_window_m),
            obstacle_bias_window_m=float(args.obstacle_bias_window_m),
            hotspot_bias_prob=float(args.hotspot_bias_prob),
            obstacle_bias_prob=float(args.obstacle_bias_prob),
        )
        idxs = (k0 + np.arange(H_cur + 1)) % (N_base + 1)

        X_seg = X_full[idxs, :].T
        U_seg = U_full[idxs, :].T
        s0_abs = float(s_m[k0])

        x0 = X_seg[:, 0].copy()
        e0 = float(x0[TrajectoryOptimizer.IDX_E]) + float(rng.uniform(-args.e_perturb_m, args.e_perturb_m))
        dpsi0 = float(x0[TrajectoryOptimizer.IDX_DPSI]) + float(
            rng.uniform(-args.dpsi_perturb_rad, args.dpsi_perturb_rad)
        )
        uy0 = float(x0[TrajectoryOptimizer.IDX_UY]) + float(
            rng.uniform(-args.uy_perturb_mps, args.uy_perturb_mps)
        )
        r0 = float(x0[TrajectoryOptimizer.IDX_R]) + float(
            rng.uniform(-args.r_perturb_radps, args.r_perturb_radps)
        )

        hw = float(world.track_width_m_LUT(s0_abs % world.length_m)) / 2.0
        e0 = float(np.clip(e0, -hw + args.track_buffer_m, hw - args.track_buffer_m))
        x0[TrajectoryOptimizer.IDX_E] = e0
        x0[TrajectoryOptimizer.IDX_DPSI] = dpsi0
        x0[TrajectoryOptimizer.IDX_UY] = uy0
        x0[TrajectoryOptimizer.IDX_R] = r0

        X_init = X_seg.copy()
        X_init[TrajectoryOptimizer.IDX_E, 0] = e0
        X_init[TrajectoryOptimizer.IDX_DPSI, 0] = dpsi0
        X_init[TrajectoryOptimizer.IDX_UY, 0] = uy0
        X_init[TrajectoryOptimizer.IDX_R, 0] = r0

        term_state = X_seg[:, -1].copy()

        x0_masked = build_initial_masked_state(x0)

        result = optimizer.solve(
            N=int(H_cur),
            ds_m=float(s_m[1] - s_m[0]),
            x0=x0_masked,
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
            save_repair_state(state_path, attempts, rng)
            continue
        if obstacles_list and result.min_obstacle_clearance < float(args.accept_min_clearance_m):
            save_repair_state(state_path, attempts, rng)
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

        episode_id = f"{map_file.stem}_repair_{next_episode_idx:06d}"
        npz_path = episodes_dir / f"{episode_id}.npz"

        metadata_dict = {
            "hard_mode": bool(args.hard_mode),
            "sampling_reason": hard_meta["sampling_reason"],
            "start_e_abs_m": float(abs(e0)),
            "start_dpsi_abs_rad": float(abs(dpsi0)),
            "start_uy_abs_mps": float(abs(uy0)),
            "start_r_abs_radps": float(abs(r0)),
            "start_half_width_m": float(hw),
            "start_min_obstacle_ds_m": float(hard_meta["start_min_obstacle_ds_m"]),
            "start_min_hotspot_ds_m": float(hard_meta["start_min_hotspot_ds_m"]),
            "near_obstacle_start": bool(hard_meta["near_obstacle_start"]),
            "near_hotspot_start": bool(hard_meta["near_hotspot_start"]),
            "min_clearance_result_m": float(getattr(result, "min_obstacle_clearance", float("inf"))),
            "solver_iterations": int(result.iterations),
            "solver_success": bool(result.success),
            "H": int(H_cur),
        }

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
            metadata_json=np.array(json.dumps(metadata_dict)),
        )

        header = EpisodeHeader(
            episode_id=episode_id,
            episode_type="repair",
            map_id=map_file.stem,
            map_hash=map_hash,
            base_id=base_id,
            solver_config=solver_config_seg,
            solver_config_hash=solver_config_hash,
            discretization={"N": int(H_cur), "ds_m": float(s_m[1] - s_m[0])},
            obstacles=obstacles_list,
            s_offset_m=float(s0_abs),
            npz_path=str(npz_path),
            metadata=metadata_dict,
        )
        manifest_f.write(json.dumps(header.to_dict()) + "\n")
        manifest_f.flush()
        successes += 1
        next_episode_idx += 1
        save_repair_state(state_path, attempts, rng)

        if successes % args.save_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[accepted {successes}/{args.num_segments}] attempts={attempts}/{max_attempts} elapsed={elapsed:.1f}s",
                flush=True,
            )

    manifest_f.close()
    elapsed = time.time() - t_start
    print(
        f"Done. accepted={successes}/{args.num_segments} attempts={attempts}/{max_attempts} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
