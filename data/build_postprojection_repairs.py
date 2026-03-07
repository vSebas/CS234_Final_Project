#!/usr/bin/env python3
"""
Build post-projection labeled repair segments from DT rollout traces.
"""

import argparse
import glob
import json
import os
import signal
import sys
import time
import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer
from utils.world import World
from experiments.run_fatrop_native_trajopt import solve_fatrop_native

from data.schema import EpisodeHeader, compute_rtg, sha256_file, sha256_json
from data.build_repair_segments import (
    _yaw_wrap,
    build_terminal_mask,
    compute_global_pose,
    compute_track_features,
    load_existing_repairs,
    load_repair_state,
    normalize_obstacles,
    save_repair_state,
)


class SolveTimeoutError(RuntimeError):
    """Raised when a single solver attempt exceeds the configured timeout."""


def _run_with_timeout(seconds: float, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with a wall-clock timeout on Unix platforms."""
    if seconds <= 0:
        return fn(*args, **kwargs)
    if os.name == "nt":
        # Windows does not support SIGALRM; keep behavior unchanged.
        return fn(*args, **kwargs)

    def _handler(signum, frame):  # noqa: ARG001
        raise SolveTimeoutError(f"solve attempt exceeded timeout ({seconds:.1f}s)")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


def build_world(map_file: Path) -> World:
    return World(str(map_file), map_file.stem, diagnostic_plotting=False)


def build_vehicle() -> object:
    return load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")


def parse_trace_inputs(text: str) -> List[Path]:
    paths: List[Path] = []
    for token in [x.strip() for x in text.split(",") if x.strip()]:
        expanded = sorted(glob.glob(token))
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            p = Path(token)
            if p.exists():
                paths.append(p)
    uniq: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def circular_distance_m(a: float, b: np.ndarray, length_m: float) -> np.ndarray:
    diff = np.abs(b - a)
    return np.minimum(diff, length_m - diff)


def build_postproj_masked_state(x_state: np.ndarray) -> List[float | None]:
    masked = [None] * len(x_state)
    for idx in (
        TrajectoryOptimizer.IDX_UY,
        TrajectoryOptimizer.IDX_R,
        TrajectoryOptimizer.IDX_E,
        TrajectoryOptimizer.IDX_DPSI,
    ):
        masked[idx] = float(x_state[idx])
    return masked


def load_trace_rows(
    trace_paths: List[Path],
    only_triggered: bool,
    max_rows: int,
    rng: np.random.Generator,
) -> Dict[str, List[Dict]]:
    rows_by_map: Dict[str, List[Dict]] = defaultdict(list)
    loaded = 0
    for path in trace_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if only_triggered and not bool(row.get("triggered", False)):
                    continue
                map_file = row.get("map_file")
                x_after = row.get("x_after_projection")
                if not map_file or not isinstance(x_after, list) or len(x_after) < 8:
                    continue
                map_stem = Path(str(map_file)).stem
                rows_by_map[map_stem].append(row)
                loaded += 1
                if max_rows > 0 and loaded >= max_rows:
                    break
        if max_rows > 0 and loaded >= max_rows:
            break

    for map_stem, rows in rows_by_map.items():
        rng.shuffle(rows)
        rows_by_map[map_stem] = rows
    return rows_by_map


def _split_targets(total: int, keys: List[str]) -> Dict[str, int]:
    base = total // max(1, len(keys))
    rem = total - base * len(keys)
    out: Dict[str, int] = {}
    for i, key in enumerate(keys):
        out[key] = base + (1 if i < rem else 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate post-projection repair shards.")
    parser.add_argument("--trace-jsonl", type=str, required=True, help="Comma-separated JSONL paths or globs.")
    parser.add_argument("--base-laps-dir", type=str, default="data/base_laps")
    parser.add_argument("--output-root", type=str, default="data/datasets")
    parser.add_argument("--output-suffix", type=str, default="repairs_postproj")
    parser.add_argument("--num-segments", type=int, default=600, help="Total accepted repairs across maps.")
    parser.add_argument("--per-map-target", type=int, default=0, help="Accepted repairs per map (overrides total split).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--long-horizon", type=int, default=40)
    parser.add_argument("--long-horizon-prob", type=float, default=0.2)
    parser.add_argument("--lambda-u", type=float, default=0.005)
    parser.add_argument("--ux-min", type=float, default=0.5)
    parser.add_argument("--track-buffer-m", type=float, default=0.0)
    parser.add_argument("--eps-s", type=float, default=0.1)
    parser.add_argument("--eps-kappa", type=float, default=0.05)
    parser.add_argument("--solver", type=str, choices=("ipopt", "fatrop"), default="fatrop")
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
    parser.add_argument("--max-attempts-factor", type=float, default=8.0)
    parser.add_argument(
        "--solve-timeout-s",
        type=float,
        default=0.0,
        help="Per-attempt solve timeout in seconds; 0 disables timeout.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--clear-cache-every",
        type=int,
        default=5,
        help="Clear optimizer NLP cache every N attempts to avoid memory growth.",
    )
    parser.add_argument("--max-trace-rows", type=int, default=0)
    parser.add_argument(
        "--only-triggered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only rows marked as triggered by eval_warmstart export.",
    )
    parser.add_argument("--ipopt-tol", type=float, default=1e-5)
    parser.add_argument("--ipopt-acceptable-tol", type=float, default=1e-3)
    parser.add_argument("--ipopt-max-iter", type=int, default=100)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by appending only missing accepted segments toward targets.",
    )
    args = parser.parse_args()

    if args.solver == "ipopt":
        os.environ["IPOPT_TOL"] = str(args.ipopt_tol)
        os.environ["IPOPT_ACCEPTABLE_TOL"] = str(args.ipopt_acceptable_tol)
        os.environ["IPOPT_MAX_ITER"] = str(args.ipopt_max_iter)

    rng = np.random.default_rng(args.seed)
    trace_paths = parse_trace_inputs(args.trace_jsonl)
    if not trace_paths:
        raise FileNotFoundError(f"No trace JSONL files matched: {args.trace_jsonl}")

    rows_by_map = load_trace_rows(
        trace_paths=trace_paths,
        only_triggered=bool(args.only_triggered),
        max_rows=int(args.max_trace_rows),
        rng=rng,
    )
    if not rows_by_map:
        raise ValueError("No usable trace rows were loaded.")

    map_stems = sorted(rows_by_map.keys())
    if args.per_map_target > 0:
        targets = {stem: int(args.per_map_target) for stem in map_stems}
    else:
        targets = _split_targets(int(args.num_segments), map_stems)

    vehicle = build_vehicle()
    term_mask = build_terminal_mask()
    output_root = Path(args.output_root)
    base_root = Path(args.base_laps_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    t_global_start = time.time()

    total_accepted = 0
    total_attempts = 0

    for map_stem in map_stems:
        target = int(targets.get(map_stem, 0))
        if target <= 0:
            continue

        map_file = Path(rows_by_map[map_stem][0]["map_file"])
        if not map_file.exists():
            raise FileNotFoundError(f"Map file not found for trace rows: {map_file}")

        world = build_world(map_file)
        optimizer = TrajectoryOptimizer(vehicle, world) if args.solver == "ipopt" else None
        map_hash = sha256_file(str(map_file))

        base_dir = base_root / map_stem
        if not base_dir.exists():
            raise FileNotFoundError(f"Base laps directory missing: {base_dir}")
        base_files = sorted(base_dir.glob("*.npz"))
        if not base_files:
            raise FileNotFoundError(f"No base laps in {base_dir}")
        obs_base_files = [p for p in base_files if p.stem.startswith("obs_")]
        noobs_base_files = [p for p in base_files if p.stem.startswith("noobs_")]

        output_dir = output_root / f"{map_stem}_{args.output_suffix}"
        episodes_dir = output_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.jsonl"
        state_path = output_dir / "repair_state.json"

        existing_count = 0
        next_episode_idx = 0
        if args.resume:
            existing_count, next_episode_idx = load_existing_repairs(manifest_path)

        if existing_count >= target:
            print(f"[{map_stem}] already complete: accepted={existing_count}/{target}", flush=True)
            total_accepted += existing_count
            continue

        attempts = 0
        if args.resume:
            attempts, rng_state = load_repair_state(state_path)
            if rng_state is not None:
                rng.bit_generator.state = rng_state

        remaining = max(0, int(target - existing_count))
        max_additional_attempts = max(
            int(np.ceil(float(max(1, remaining)) * float(args.max_attempts_factor))),
            remaining,
            1,
        )
        attempts_start = int(attempts)
        successes = existing_count
        trace_rows = rows_by_map[map_stem]
        manifest_f = open(manifest_path, "a", encoding="utf-8")
        t_start = time.time()

        base_cache: Dict[str, Dict] = {}
        while successes < target and (attempts - attempts_start) < max_additional_attempts:
            attempts += 1
            if (
                args.solver == "ipopt"
                and args.clear_cache_every > 0
                and attempts % int(args.clear_cache_every) == 0
            ):
                # Obstacles vary per trace row; clearing cached NLP templates prevents
                # unbounded memory growth from per-obstacle cache keys.
                assert optimizer is not None
                optimizer._nlp_cache.clear()
                gc.collect()
            if attempts > 1 and attempts % max(1, args.save_every) == 0:
                elapsed = time.time() - t_start
                print(
                    (
                        f"[{map_stem} attempt {attempts - attempts_start}/{max_additional_attempts}] "
                        f"accepted={successes}/{target} elapsed={elapsed:.1f}s"
                    ),
                    flush=True,
                )

            row = trace_rows[int(rng.integers(0, len(trace_rows)))]
            obstacles_list = row.get("obstacles", [])
            if not isinstance(obstacles_list, list):
                obstacles_list = []

            if obstacles_list and obs_base_files:
                base_path = obs_base_files[int(rng.integers(0, len(obs_base_files)))]
            elif (not obstacles_list) and noobs_base_files:
                base_path = noobs_base_files[int(rng.integers(0, len(noobs_base_files)))]
            else:
                base_path = base_files[int(rng.integers(0, len(base_files)))]

            cache_key = str(base_path)
            if cache_key not in base_cache:
                base_data = np.load(base_path, allow_pickle=True)
                solver_config = base_data.get("solver_config", {})
                solver_config = solver_config.item() if isinstance(solver_config, np.ndarray) else solver_config
                base_cache[cache_key] = {
                    "s_m": base_data["s_m"].astype(float),
                    "X_full": base_data["X_full"].astype(float),
                    "U_full": base_data["U"].astype(float),
                    "solver_config": solver_config if isinstance(solver_config, dict) else {},
                    "base_id": base_path.stem,
                }
            cached = base_cache[cache_key]
            s_m = cached["s_m"]
            X_full = cached["X_full"]
            U_full = cached["U_full"]
            N_base = X_full.shape[0] - 1
            if N_base < 2:
                save_repair_state(state_path, attempts, rng)
                continue

            p_long = float(np.clip(args.long_horizon_prob, 0.0, 1.0))
            H_cur = int(args.long_horizon if rng.random() < p_long else args.H)
            H_cur = min(H_cur, N_base)

            s_start = float(row.get("s_m", 0.0)) % world.length_m
            d = circular_distance_m(s_start, np.mod(s_m, world.length_m), world.length_m)
            k0 = int(np.argmin(d))
            idxs = (k0 + np.arange(H_cur + 1)) % (N_base + 1)

            X_seg = X_full[idxs, :].T
            U_seg = U_full[idxs, :].T
            s0_abs = float(s_m[k0])

            x_after = np.asarray(row.get("x_after_projection", []), dtype=float)
            if x_after.shape[0] < X_seg.shape[0]:
                save_repair_state(state_path, attempts, rng)
                continue

            x0 = X_seg[:, 0].copy()
            x0[TrajectoryOptimizer.IDX_UY] = float(x_after[TrajectoryOptimizer.IDX_UY])
            x0[TrajectoryOptimizer.IDX_R] = float(x_after[TrajectoryOptimizer.IDX_R])
            x0[TrajectoryOptimizer.IDX_E] = float(x_after[TrajectoryOptimizer.IDX_E])
            x0[TrajectoryOptimizer.IDX_DPSI] = float(x_after[TrajectoryOptimizer.IDX_DPSI])

            hw = float(world.track_width_m_LUT(s0_abs % world.length_m)) / 2.0
            e0 = float(np.clip(x0[TrajectoryOptimizer.IDX_E], -hw + args.track_buffer_m, hw - args.track_buffer_m))
            x0[TrajectoryOptimizer.IDX_E] = e0

            X_init = X_seg.copy()
            X_init[TrajectoryOptimizer.IDX_UY, 0] = x0[TrajectoryOptimizer.IDX_UY]
            X_init[TrajectoryOptimizer.IDX_R, 0] = x0[TrajectoryOptimizer.IDX_R]
            X_init[TrajectoryOptimizer.IDX_E, 0] = x0[TrajectoryOptimizer.IDX_E]
            X_init[TrajectoryOptimizer.IDX_DPSI, 0] = x0[TrajectoryOptimizer.IDX_DPSI]

            term_state = X_seg[:, -1].copy()
            x0_masked = build_postproj_masked_state(x0)

            solver_config_seg = dict(cached["solver_config"])
            solver_config_seg.update(
                {
                    "solver": str(args.solver),
                    "H": int(H_cur),
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
                    "postprojection_source": True,
                }
            )
            solver_config_hash = sha256_json(solver_config_seg) if solver_config_seg else ""

            try:
                if args.solver == "fatrop":
                    result = _run_with_timeout(
                        float(args.solve_timeout_s),
                        solve_fatrop_native,
                        vehicle,
                        world,
                        N=int(H_cur),
                        ds_m=float(s_m[1] - s_m[0]),
                        obstacles=normalize_obstacles(obstacles_list) if obstacles_list else None,
                        lambda_u=float(args.lambda_u),
                        ux_min=float(args.ux_min),
                        track_buffer_m=float(args.track_buffer_m),
                        obstacle_window_m=float(args.obs_window_m),
                        obstacle_clearance_m=0.0,
                        vehicle_radius_m=0.0,
                        eps_s=float(args.eps_s),
                        eps_kappa=float(args.eps_kappa),
                        X_init=X_init,
                        U_init=U_seg,
                        x0=x0_masked,
                        s0_offset_m=s0_abs,
                        terminal_state=term_state,
                        terminal_mask=term_mask,
                        terminal_weight=float(args.terminal_weight),
                        verbose=False,
                    )
                else:
                    assert optimizer is not None
                    result = _run_with_timeout(
                        float(args.solve_timeout_s),
                        optimizer.solve,
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
                        obstacle_clearance_m=0.0,
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
            except Exception:
                save_repair_state(state_path, attempts, rng)
                continue

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

            episode_id = f"{map_stem}_postproj_{next_episode_idx:06d}"
            npz_path = episodes_dir / f"{episode_id}.npz"
            metadata_dict = {
                "postprojection_source": True,
                "trace_scenario_id": row.get("scenario_id"),
                "trace_checkpoint_path": row.get("checkpoint_path"),
                "trace_k": int(row.get("k", -1)),
                "trace_projection_mag": float(row.get("projection_mag", 0.0)),
                "trace_clearance_proxy_m": float(row.get("clearance_proxy_m", np.inf)),
                "trace_fallback_used": bool(row.get("fallback_used", False)),
                "trace_triggered": bool(row.get("triggered", False)),
                "start_e_abs_m": float(abs(x0[TrajectoryOptimizer.IDX_E])),
                "start_dpsi_abs_rad": float(abs(x0[TrajectoryOptimizer.IDX_DPSI])),
                "start_uy_abs_mps": float(abs(x0[TrajectoryOptimizer.IDX_UY])),
                "start_r_abs_radps": float(abs(x0[TrajectoryOptimizer.IDX_R])),
                "start_half_width_m": float(hw),
                "solver_iterations": int(result.iterations),
                "solver_success": bool(result.success),
                "min_clearance_result_m": float(getattr(result, "min_obstacle_clearance", np.inf)),
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
                episode_type="repair_postproj",
                map_id=map_stem,
                map_hash=map_hash,
                base_id=str(cached["base_id"]),
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

            if successes % max(1, args.save_every) == 0:
                elapsed = time.time() - t_start
                print(
                    (
                        f"[{map_stem} accepted {successes}/{target}] "
                        f"attempts={attempts - attempts_start}/{max_additional_attempts} elapsed={elapsed:.1f}s"
                    ),
                    flush=True,
                )

        manifest_f.close()
        total_accepted += successes
        total_attempts += attempts
        elapsed = time.time() - t_start
        print(
            (
                f"[{map_stem}] done accepted={successes}/{target} "
                f"attempts={attempts - attempts_start}/{max_additional_attempts} elapsed={elapsed:.1f}s"
            ),
            flush=True,
        )

    elapsed_global = time.time() - t_global_start
    print(
        f"Done. total_accepted={total_accepted} total_attempts={total_attempts} elapsed={elapsed_global:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
