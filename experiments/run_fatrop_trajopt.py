#!/usr/bin/env python3
"""
Standalone FATROP trajectory optimization runner.

This script keeps FATROP separate from the core IPOPT optimizer implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List
import time

import casadi as ca
import numpy as np

from models import load_vehicle_from_yaml
from planning import ObstacleCircle, OptimizationResult, TrajectoryOptimizer
from world.world import World


def _load_obstacles_from_world(world: World) -> List[ObstacleCircle]:
    req = ("obstacles_s_m", "obstacles_e_m", "obstacles_radius_m", "obstacles_margin_m")
    data = getattr(world, "data", {})
    if any(k not in data for k in req):
        return []
    s_vals = np.atleast_1d(data["obstacles_s_m"]).astype(float)
    e_vals = np.atleast_1d(data["obstacles_e_m"]).astype(float)
    r_vals = np.atleast_1d(data["obstacles_radius_m"]).astype(float)
    m_vals = np.atleast_1d(data["obstacles_margin_m"]).astype(float)
    n = int(min(len(s_vals), len(e_vals), len(r_vals), len(m_vals)))
    out: List[ObstacleCircle] = []
    for i in range(n):
        out.append(
            ObstacleCircle(
                s_m=float(s_vals[i]),
                e_m=float(e_vals[i]),
                radius_m=float(r_vals[i]),
                margin_m=float(m_vals[i]),
            )
        )
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


def solve_fatrop_trajopt(
    vehicle,
    world: World,
    N: int,
    ds_m: float,
    *,
    obstacles: List[ObstacleCircle] | None = None,
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
    verbose: bool = False,
) -> OptimizationResult:
    p = vehicle.params
    nx = 8
    nu = 2
    obs_list = obstacles or []

    s_grid = np.linspace(0.0, N * ds_m, N + 1)
    s_mod = np.mod(s_grid, world.length_m)
    kappa = np.array(world.psi_s_radpm_LUT(s_mod)).astype(float).squeeze()
    theta = (
        np.array(world.grade_rad_LUT(s_mod)).astype(float).squeeze()
        if hasattr(world, "grade_rad_LUT")
        else np.zeros(N + 1)
    )
    phi = (
        np.array(world.bank_rad_LUT(s_mod)).astype(float).squeeze()
        if hasattr(world, "bank_rad_LUT")
        else np.zeros(N + 1)
    )
    track_hw = 0.5 * np.array(world.track_width_m_LUT(s_mod)).astype(float).squeeze()
    psi_cl = np.array(world.psi_rad_interp_fcn(s_mod)).astype(float).squeeze()
    posE_cl = np.array(world.posE_m_interp_fcn(s_mod)).astype(float).squeeze()
    posN_cl = np.array(world.posN_m_interp_fcn(s_mod)).astype(float).squeeze()

    obs_east = np.zeros(len(obs_list))
    obs_north = np.zeros(len(obs_list))
    obs_r_tilde = np.zeros(len(obs_list))
    obs_s_center = np.zeros(len(obs_list))
    for j, obs in enumerate(obs_list):
        if obs.east_m is not None and obs.north_m is not None:
            east_m = float(obs.east_m)
            north_m = float(obs.north_m)
        elif obs.s_m is not None and obs.e_m is not None:
            east_m, north_m = _frenet_to_en(world, float(obs.s_m), float(obs.e_m))
        else:
            continue
        obs_east[j] = east_m
        obs_north[j] = north_m
        obs_r_tilde[j] = float(obs.radius_m + obs.margin_m)
        obs_s_center[j] = (
            float(obs.s_m) if obs.s_m is not None else _nearest_centerline_s(world, east_m, north_m)
        )

    opti = ca.Opti()
    x_vars = [opti.variable(nx) for _ in range(N + 1)]
    u_vars = [opti.variable(nu) for _ in range(N + 1)]
    nx_list = [nx for _ in range(N + 1)]
    nu_list = [nu for _ in range(N + 1)]
    ng_list: List[int] = [0 for _ in range(N + 1)]

    # Objective
    reg_terms = []
    for k in range(N):
        reg_terms.append(ca.sumsqr(u_vars[k + 1] - u_vars[k]))
    reg_cost = lambda_u * ca.sum1(ca.vertcat(*reg_terms)) if reg_terms else 0.0
    objective = x_vars[N][5] + reg_cost
    opti.minimize(objective)

    # Dynamics
    for k in range(N):
        xk = x_vars[k]
        xkp1 = x_vars[k + 1]
        uk = u_vars[k]
        ukp1 = u_vars[k + 1]
        dx_dt_k, sdot_k = vehicle.dynamics_dt_path_vec(xk, uk, kappa[k], theta[k], phi[k])
        dx_dt_kp1, sdot_kp1 = vehicle.dynamics_dt_path_vec(xkp1, ukp1, kappa[k + 1], theta[k + 1], phi[k + 1])
        opti.subject_to(xkp1 == xk + 0.5 * ds_m * (dx_dt_k / sdot_k + dx_dt_kp1 / sdot_kp1))

    # Stage constraints and ng bookkeeping for FATROP manual structure mode
    for k in range(N + 1):
        xk = x_vars[k]
        uk = u_vars[k]
        ux_k = xk[0]
        uy_k = xk[1]
        e_k = xk[6]
        dpsi_k = xk[7]
        delta_k = uk[0]
        fx_k = uk[1]

        cons = []
        cons.append(ux_k >= ux_min)
        if ux_max is not None:
            cons.append(ux_k <= ux_max)
        one_minus = 1 - kappa[k] * e_k
        if abs(float(kappa[k])) > 1e-12:
            cons.append(one_minus >= eps_kappa)
        sdot_k = (ux_k * ca.cos(dpsi_k) - uy_k * ca.sin(dpsi_k)) / one_minus
        cons.append(sdot_k >= eps_s)
        cons.append(e_k >= -track_hw[k] + track_buffer_m)
        cons.append(e_k <= track_hw[k] - track_buffer_m)
        cons.append(delta_k >= -p.max_delta_rad)
        cons.append(delta_k <= p.max_delta_rad)
        cons.append(fx_k >= p.min_fx_kn)
        cons.append(fx_k <= p.max_fx_kn)

        if len(obs_list) > 0:
            posE_k = posE_cl[k] - e_k * np.sin(psi_cl[k])
            posN_k = posN_cl[k] + e_k * np.cos(psi_cl[k])
            for j in range(len(obs_list)):
                ds_wrap = (s_grid[k] - obs_s_center[j]) % world.length_m
                if ds_wrap > 0.5 * world.length_m:
                    ds_wrap -= world.length_m
                if abs(ds_wrap) <= obstacle_window_m:
                    req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
                    cons.append((posE_k - obs_east[j]) ** 2 + (posN_k - obs_north[j]) ** 2 >= req_r**2)

        if k == 0:
            cons.append(xk[5] == 0.0)

        if convergent_lap and k == N:
            for idx_state in (0, 1, 2, 3, 4, 6, 7):
                cons.append(xk[idx_state] == x_vars[0][idx_state])
            cons.append(uk[0] == u_vars[0][0])
            cons.append(uk[1] == u_vars[0][1])

        for c in cons:
            opti.subject_to(c)
            ng_list[k] += int(c.nnz())

    # Initial guess
    ux_init = max(5.0, ux_min + 1.0)
    t_init = np.cumsum(np.ones(N + 1) * ds_m / ux_init) - ds_m / ux_init
    for k in range(N + 1):
        opti.set_initial(x_vars[k][0], ux_init)
        opti.set_initial(x_vars[k][1], 0.0)
        opti.set_initial(x_vars[k][2], 0.0)
        opti.set_initial(x_vars[k][3], 0.0)
        opti.set_initial(x_vars[k][4], 0.0)
        opti.set_initial(x_vars[k][5], float(t_init[k]))
        opti.set_initial(x_vars[k][6], 0.0)
        opti.set_initial(x_vars[k][7], 0.0)
        opti.set_initial(u_vars[k][0], 0.0)
        opti.set_initial(u_vars[k][1], 0.5)

    # Solve with FATROP
    preset = os.environ.get("FATROP_PRESET", "fast").strip().lower()
    preset_cfg = {
        "fast": {"mu_init": 0.2, "tol": 1e-4, "acceptable_tol": 1e-3},
        "obstacle_fast": {"mu_init": 0.3, "tol": 1e-4, "acceptable_tol": 1e-3},
        "balanced": {"mu_init": 0.1},
        "accurate": {"mu_init": 0.1, "tol": 1e-6, "acceptable_tol": 1e-6},
    }.get(preset, {"mu_init": 0.2, "tol": 1e-4, "acceptable_tol": 1e-3})

    fatrop_opts = {
        "mu_init": float(os.environ.get("FATROP_MU_INIT", str(preset_cfg["mu_init"]))),
        "print_level": int(os.environ.get("FATROP_PRINT_LEVEL", "0")),
    }
    if "tol" in preset_cfg:
        fatrop_opts["tol"] = float(preset_cfg["tol"])
    if "acceptable_tol" in preset_cfg:
        fatrop_opts["acceptable_tol"] = float(preset_cfg["acceptable_tol"])
    if "FATROP_TOL" in os.environ:
        try:
            fatrop_opts["tol"] = float(os.environ["FATROP_TOL"])
        except ValueError:
            pass
    if "FATROP_ACCEPTABLE_TOL" in os.environ:
        try:
            fatrop_opts["acceptable_tol"] = float(os.environ["FATROP_ACCEPTABLE_TOL"])
        except ValueError:
            pass
    use_debug = os.environ.get("FATROP_DEBUG", "0") == "1"
    prefer_structure_detection = os.environ.get("FATROP_STRUCTURE_DETECTION", "none").strip().lower()
    convexify_strategy = os.environ.get("FATROP_CONVEXIFY_STRATEGY", "").strip()
    convexify_margin_raw = os.environ.get("FATROP_CONVEXIFY_MARGIN", "").strip()

    option_attempts = []
    first_opts = {
        "print_time": bool(verbose),
        "expand": os.environ.get("FATROP_EXPAND", "1") != "0",
        "debug": use_debug,
        "fatrop": dict(fatrop_opts),
    }
    if convexify_strategy:
        first_opts["convexify_strategy"] = convexify_strategy
    if convexify_margin_raw:
        try:
            first_opts["convexify_margin"] = float(convexify_margin_raw)
        except ValueError:
            pass
    if prefer_structure_detection:
        first_opts["structure_detection"] = prefer_structure_detection
    if prefer_structure_detection == "manual":
        first_opts["nx"] = nx_list
        first_opts["nu"] = nu_list
        first_opts["ng"] = ng_list
        first_opts["N"] = N
    option_attempts.append(first_opts)
    option_attempts.append(
        {
            "print_time": bool(verbose),
            "expand": os.environ.get("FATROP_EXPAND", "1") != "0",
            "debug": use_debug,
            "structure_detection": "auto",
            "fatrop": dict(fatrop_opts),
        }
    )
    option_attempts.append(
        {
            "print_time": bool(verbose),
            "expand": False,
            "debug": use_debug,
            "structure_detection": "none",
            "fatrop": dict(fatrop_opts),
        }
    )

    solve_time = 0.0
    success = False
    X_opt = None
    U_opt = None
    cost_opt = float("inf")
    iterations = -1
    errors = []
    for idx, opts in enumerate(option_attempts):
        t0 = time.time()
        try:
            opti.solver("fatrop", opts)
            sol = opti.solve()
            solve_time = time.time() - t0
            success = True
            X_opt = np.column_stack([np.asarray(sol.value(xk), dtype=float) for xk in x_vars])
            U_opt = np.column_stack([np.asarray(sol.value(uk), dtype=float) for uk in u_vars])
            cost_opt = float(sol.value(objective))
            stats = sol.stats()
            iterations = int(stats.get("iter_count", -1))
            break
        except RuntimeError as err:
            solve_time = time.time() - t0
            errors.append(f"attempt {idx+1}: {err}")
            continue

    if X_opt is None or U_opt is None:
        raise RuntimeError(
            "FATROP failed for all option attempts.\n" + "\n\n".join(errors)
        )

    min_clearance = float("inf")
    if len(obs_list) > 0:
        e_opt = X_opt[6, :]
        for k in range(N + 1):
            posE_k = posE_cl[k] - e_opt[k] * np.sin(psi_cl[k])
            posN_k = posN_cl[k] + e_opt[k] * np.cos(psi_cl[k])
            for j in range(len(obs_list)):
                ds_wrap = (s_grid[k] - obs_s_center[j]) % world.length_m
                if ds_wrap > 0.5 * world.length_m:
                    ds_wrap -= world.length_m
                if abs(ds_wrap) <= obstacle_window_m:
                    req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
                    d = np.sqrt((posE_k - obs_east[j]) ** 2 + (posN_k - obs_north[j]) ** 2)
                    min_clearance = min(min_clearance, float(d - req_r))

    return OptimizationResult(
        success=success,
        s_m=s_grid,
        X=X_opt,
        U=U_opt,
        cost=cost_opt,
        iterations=iterations,
        solve_time=solve_time,
        k_psi=kappa,
        theta=theta,
        phi=phi,
        max_obstacle_slack=0.0,
        min_obstacle_clearance=min_clearance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--compare-ipopt", action="store_true")
    parser.add_argument("--no-convergent-lap", action="store_true")
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
            convergent_lap=not bool(args.no_convergent_lap),
            verbose=False,
        )
        print(
            f"[ipopt] success={base.success} iterations={base.iterations} "
            f"cost={base.cost:.6f} solve_time={base.solve_time:.3f}s"
        )

    fat = solve_fatrop_trajopt(
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
        convergent_lap=not bool(args.no_convergent_lap),
        verbose=bool(args.verbose),
    )
    print(
        f"[fatrop] success={fat.success} iterations={fat.iterations} "
        f"cost={fat.cost:.6f} solve_time={fat.solve_time:.3f}s "
        f"min_clearance={fat.min_obstacle_clearance:.4f}"
    )


if __name__ == "__main__":
    main()
