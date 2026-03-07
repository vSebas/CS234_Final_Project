#!/usr/bin/env python3
"""
Standalone MadNLP trajectory-optimization runner.

This script is intentionally independent from TrajectoryOptimizer.solve() backend
selection so the core IPOPT optimizer path remains unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

from models import load_vehicle_from_yaml
from planning import ObstacleCircle, OptimizationResult, TrajectoryOptimizer
from planning.madnlp_bridge import solve_with_julia_madnlp
from utils.world import World


def _load_obstacles_from_world(world: World) -> List[dict]:
    req = ("obstacles_s_m", "obstacles_e_m", "obstacles_radius_m", "obstacles_margin_m")
    data = getattr(world, "data", {})
    if any(k not in data for k in req):
        return []
    s_vals = np.atleast_1d(data["obstacles_s_m"]).astype(float)
    e_vals = np.atleast_1d(data["obstacles_e_m"]).astype(float)
    r_vals = np.atleast_1d(data["obstacles_radius_m"]).astype(float)
    m_vals = np.atleast_1d(data["obstacles_margin_m"]).astype(float)
    n = int(min(len(s_vals), len(e_vals), len(r_vals), len(m_vals)))
    out = []
    for i in range(n):
        out.append(
            {
                "s_m": float(s_vals[i]),
                "e_m": float(e_vals[i]),
                "radius_m": float(r_vals[i]),
                "margin_m": float(m_vals[i]),
            }
        )
    return out


def _normalize_obstacles(obstacles: Sequence[Union[ObstacleCircle, dict]]) -> List[ObstacleCircle]:
    out: List[ObstacleCircle] = []
    for obs in obstacles or []:
        if isinstance(obs, ObstacleCircle):
            out.append(obs)
        else:
            out.append(ObstacleCircle(**obs))
    return out


def _frenet_to_en(world: World, s_m: float, e_m: float):
    east_arr, north_arr, _ = world.map_match_vectorized(
        np.asarray([s_m], dtype=float), np.asarray([e_m], dtype=float)
    )
    return float(east_arr[0]), float(north_arr[0])


def _nearest_centerline_s(world: World, east_m: float, north_m: float) -> float:
    dx = world.data["posE_m"] - east_m
    dy = world.data["posN_m"] - north_m
    idx = int(np.argmin(dx * dx + dy * dy))
    return float(world.data["s_m"][idx])


def solve_madnlp_trajopt(
    vehicle,
    world: World,
    N: int,
    ds_m: float,
    *,
    obstacles: Sequence[Union[ObstacleCircle, dict]] | None = None,
    lambda_u: float = 0.005,
    ux_min: float = 0.5,
    ux_max: float | None = None,
    track_buffer_m: float = 0.0,
    obstacle_window_m: float = 30.0,
    obstacle_clearance_m: float = 0.0,
    vehicle_radius_m: float = 0.0,
    eps_s: float = 0.1,
    eps_kappa: float = 0.05,
    convergent_lap: bool = True,
    s0_offset_m: float = 0.0,
    verbose: bool = False,
) -> OptimizationResult:
    opt = TrajectoryOptimizer(vehicle, world)
    obs_list = _normalize_obstacles(obstacles)

    s_grid = float(s0_offset_m) + np.linspace(0, N * ds_m, N + 1)
    s_mod = np.mod(s_grid, world.length_m)
    k_psi_data = np.array(world.psi_s_radpm_LUT(s_mod)).astype(float).squeeze()
    if hasattr(world, "grade_rad_LUT"):
        theta_data = np.array(world.grade_rad_LUT(s_mod)).astype(float).squeeze()
    else:
        theta_data = np.zeros(N + 1)
    if hasattr(world, "bank_rad_LUT"):
        phi_data = np.array(world.bank_rad_LUT(s_mod)).astype(float).squeeze()
    else:
        phi_data = np.zeros(N + 1)
    track_hw = 0.5 * np.array(world.track_width_m_LUT(s_mod)).astype(float).squeeze()
    psi_cl_data = np.array(world.psi_rad_interp_fcn(s_mod)).astype(float).squeeze()
    posE_cl_data = np.array(world.posE_m_interp_fcn(s_mod)).astype(float).squeeze()
    posN_cl_data = np.array(world.posN_m_interp_fcn(s_mod)).astype(float).squeeze()

    ux_init = max(5.0, ux_min + 1.0)
    X_init = np.zeros((8, N + 1), dtype=float)
    X_init[0, :] = ux_init
    X_init[5, :] = np.cumsum(np.ones(N + 1) * ds_m / ux_init) - ds_m / ux_init
    U_init = np.zeros((2, N + 1), dtype=float)
    U_init[1, :] = 0.5

    obs_east: list[float] = []
    obs_north: list[float] = []
    obs_s_center: list[float] = []
    obs_r_tilde: list[float] = []
    obs_payload = []
    for o in obs_list:
        if o.east_m is not None and o.north_m is not None:
            east_m = float(o.east_m)
            north_m = float(o.north_m)
        elif o.s_m is not None and o.e_m is not None:
            east_m, north_m = _frenet_to_en(world, float(o.s_m), float(o.e_m))
        else:
            continue
        s_center = float(o.s_m) if o.s_m is not None else _nearest_centerline_s(world, east_m, north_m)
        obs_east.append(east_m)
        obs_north.append(north_m)
        obs_s_center.append(s_center)
        obs_r_tilde.append(float(o.radius_m + o.margin_m))
        obs_payload.append(
            {
                "radius_m": float(o.radius_m),
                "margin_m": float(o.margin_m),
                "s_m": None if o.s_m is None else float(o.s_m),
                "e_m": None if o.e_m is None else float(o.e_m),
                "east_m": east_m,
                "north_m": north_m,
            }
        )

    payload = {
        "version": 1,
        "problem": "trajectory_direct_collocation_single_track",
        "N": int(N),
        "ds_m": float(ds_m),
        "world": {
            "length_m": float(world.length_m),
            "s_grid_m": s_grid.tolist(),
            "kappa_radpm": k_psi_data.tolist(),
            "grade_rad": theta_data.tolist(),
            "bank_rad": phi_data.tolist(),
            "track_half_width_m": track_hw.tolist(),
            "psi_cl_rad": psi_cl_data.tolist(),
            "posE_cl_m": posE_cl_data.tolist(),
            "posN_cl_m": posN_cl_data.tolist(),
        },
        "vehicle": {
            "params": vehicle.params.__dict__,
            "front_tire": vehicle.f_tire.__dict__,
            "rear_tire": vehicle.r_tire.__dict__,
            "enable_weight_transfer": bool(vehicle.enable_weight_transfer),
            "state_dim": 8,
            "control_dim": 2,
        },
        "options": {
            "lambda_u": float(lambda_u),
            "ux_min": float(ux_min),
            "ux_max": None if ux_max is None else float(ux_max),
            "track_buffer_m": float(track_buffer_m),
            "obstacle_window_m": float(obstacle_window_m),
            "obstacle_clearance_m": float(obstacle_clearance_m),
            "vehicle_radius_m": float(vehicle_radius_m),
            "eps_s": float(eps_s),
            "eps_kappa": float(eps_kappa),
            "convergent_lap": bool(convergent_lap),
            "s0_offset_m": float(s0_offset_m),
            "terminal_weight": 0.0,
            "verbose": bool(verbose),
            "tol": float(os.environ.get("MADNLP_TOL", "1e-6")),
            "acceptable_tol": float(os.environ.get("MADNLP_ACCEPTABLE_TOL", "1e-4")),
            "max_iter": int(os.environ.get("MADNLP_MAX_ITER", "1000")),
            "max_cpu_time": float(os.environ.get("MADNLP_MAX_CPU_TIME", "30.0")),
            "linear_solver": os.environ.get("MADNLP_LINEAR_SOLVER", "").strip(),
            "kkt_system": os.environ.get("MADNLP_KKT_SYSTEM", "").strip(),
            "hsllib": os.environ.get("MADNLP_HSLIB", "").strip(),
            "enforce_periodic_controls": bool(int(os.environ.get("MADNLP_PERIODIC_CONTROLS", "1"))),
            "dynamics_mode": os.environ.get("MADNLP_DYNAMICS_MODE", "simple").strip().lower(),
        },
        "obstacles": obs_payload,
        "obstacle_resolved": {
            "east_m": obs_east,
            "north_m": obs_north,
            "s_center_m": obs_s_center,
            "r_tilde_m": obs_r_tilde,
        },
        "x0": None,
        "X_init": X_init.tolist(),
        "U_init": U_init.tolist(),
        "terminal_state": None,
        "terminal_mask": None,
    }

    timeout_s = int(float(os.environ.get("MADNLP_EXA_TIMEOUT_S", "1800")))
    result = solve_with_julia_madnlp(payload=payload, timeout_s=timeout_s)
    require_gpu = os.environ.get("MADNLP_REQUIRE_GPU", "0").strip() == "1"
    if require_gpu and not bool(result.get("gpu_active", False)):
        raise RuntimeError(
            "MADNLP_REQUIRE_GPU=1 was set, but MadNLP did not run on GPU. "
            "Check CUDA/MadNLPGPU/CUDSS setup."
        )
    X = np.asarray(result.get("X", []), dtype=float)
    U = np.asarray(result.get("U", []), dtype=float)
    if X.shape != (8, N + 1) or U.shape != (2, N + 1):
        raise RuntimeError(
            f"MadNLP returned invalid shape: X={X.shape}, U={U.shape}, expected (8,{N+1}) and (2,{N+1})"
        )
    return OptimizationResult(
        success=bool(result.get("success", False)),
        s_m=s_grid,
        X=X,
        U=U,
        cost=float(result.get("cost", float("inf"))),
        iterations=int(result.get("iterations", -1)),
        solve_time=float(result.get("solve_time", 0.0)),
        k_psi=k_psi_data,
        theta=theta_data,
        phi=phi_data,
        max_obstacle_slack=float(result.get("max_obstacle_slack", 0.0)),
        min_obstacle_clearance=float(result.get("min_obstacle_clearance", float("inf"))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--compare-ipopt", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    map_file = Path(args.map_file)
    if not map_file.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    world = World(str(map_file), map_file.stem, diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    obstacles = _load_obstacles_from_world(world)
    ds_m = float(world.length_m / int(args.N))

    if args.compare_ipopt:
        base = TrajectoryOptimizer(vehicle, world).solve(
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
        print(
            f"[ipopt] success={base.success} iterations={base.iterations} "
            f"cost={base.cost:.6f} solve_time={base.solve_time:.3f}s"
        )

    mad = solve_madnlp_trajopt(
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
        verbose=bool(args.verbose),
    )
    print(
        f"[madnlp] success={mad.success} iterations={mad.iterations} "
        f"cost={mad.cost:.6f} solve_time={mad.solve_time:.3f}s "
        f"min_clearance={mad.min_obstacle_clearance:.4f}"
    )


if __name__ == "__main__":
    main()
