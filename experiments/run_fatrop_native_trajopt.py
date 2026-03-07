#!/usr/bin/env python3
"""
Dedicated FATROP-native trajectory optimization runner.

This formulation is intentionally stage-local:
- multiple shooting style dynamics: x_{k+1} = F(x_k, u_k)
- stage-local path constraints
- stage-sum control regularization
- optional soft terminal closure penalty (instead of hard periodic closure)

Integration schemes (FATROP_DYNAMICS_SCHEME):
  euler        - 1st-order Euler, O(ds²) local error (default, backward-compatible)
  rk4          - 4th-order Runge-Kutta, O(ds⁵) local error, still stage-local
  trapezoidal  - 2nd-order trapezoidal (cross-stage, breaks FATROP structure)

Smooth controls (FATROP_SMOOTH_CONTROLS=1):
  Lifts (δ, Fx) into the state vector. Optimization controls become arc-length
  derivatives (dδ/ds, dFx/ds), making the rate regularizer ||u_k||^2 stage-local.
  Actuator bounds on (δ, Fx) are enforced as state constraints at all stages.
  When combined with rk4, the full augmented 10-state system is integrated with RK4.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import casadi as ca
import numpy as np

from models import load_vehicle_from_yaml
from planning import ObstacleCircle, OptimizationResult
from utils.visualization import TrajectoryVisualizer, create_animation
from utils.world import World


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


def _build_obstacle_aware_e_init(
    s_grid: np.ndarray,
    track_hw: np.ndarray,
    obs_s_center: np.ndarray,
    obs_e_center: np.ndarray,
    obs_r_tilde: np.ndarray,
    length_m: float,
    obstacle_clearance_m: float = 0.0,
    vehicle_radius_m: float = 0.0,
    track_buffer_m: float = 0.0,
    init_sigma_m: float = 8.0,
    init_margin_m: float = 0.3,
) -> np.ndarray:
    """Build a smooth lateral-offset init that biases e(s) away from each obstacle.

    Ported from planning/optimizer.py:_build_obstacle_aware_e_init.
    Each obstacle adds a Gaussian bump in e(s) centered at obs_s_center, pushing
    the vehicle to the opposite side from the obstacle's Frenet lateral offset.
    """
    def _wrap_s_dist(s_a: float, s_b: float) -> float:
        d = (s_a - s_b) % length_m
        return d - length_m if d > 0.5 * length_m else d

    e_init = np.zeros_like(s_grid, dtype=float)
    sigma = max(2.0, float(init_sigma_m))
    for j in range(len(obs_s_center)):
        target_abs = float(obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m + init_margin_m)
        idx_c = int(np.argmin(np.abs(np.array([_wrap_s_dist(float(s), float(obs_s_center[j])) for s in s_grid]))))
        e_limit = max(0.0, float(track_hw[idx_c] - track_buffer_m - 0.05))
        target_abs = min(target_abs, 0.9 * e_limit)
        e_obs = float(obs_e_center[j])
        target = target_abs if abs(e_obs) < 1e-2 else -np.sign(e_obs) * target_abs
        for k in range(len(s_grid)):
            bump = np.exp(-0.5 * (_wrap_s_dist(float(s_grid[k]), float(obs_s_center[j])) / sigma) ** 2)
            e_init[k] += target * bump
    e_max = np.maximum(0.0, track_hw - track_buffer_m - 0.05)
    return np.clip(e_init, -e_max, e_max)


def solve_fatrop_native(
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
    verbose: bool = False,
    X_init: np.ndarray | None = None,
    U_init: np.ndarray | None = None,
    x0: list[float | None] | None = None,
    s0_offset_m: float = 0.0,
    terminal_state: np.ndarray | None = None,
    terminal_mask: list[bool] | tuple[bool, ...] | np.ndarray | None = None,
    terminal_weight: float = 0.0,
) -> OptimizationResult:
    t_build0 = time.time()
    p = vehicle.params
    nx = 8  # base vehicle path state dimension
    nu = 2  # control dimension (always 2: [δ, Fx] or [dδ/ds, dFx/ds])
    obs_list = obstacles or []

    # ---------------------------------------------------------------------------
    # Road geometry at grid nodes
    # ---------------------------------------------------------------------------
    s_grid = float(s0_offset_m) + np.linspace(0.0, N * ds_m, N + 1)
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

    # Road geometry at segment midpoints (for RK4 k2/k3 evaluations).
    s_mid = 0.5 * (s_grid[:-1] + s_grid[1:])  # shape (N,)
    s_mid_mod = np.mod(s_mid, world.length_m)
    kappa_mid = np.array(world.psi_s_radpm_LUT(s_mid_mod)).astype(float).squeeze()
    theta_mid = (
        np.array(world.grade_rad_LUT(s_mid_mod)).astype(float).squeeze()
        if hasattr(world, "grade_rad_LUT")
        else np.zeros(N)
    )
    phi_mid = (
        np.array(world.bank_rad_LUT(s_mid_mod)).astype(float).squeeze()
        if hasattr(world, "bank_rad_LUT")
        else np.zeros(N)
    )

    # ---------------------------------------------------------------------------
    # Obstacle data
    # ---------------------------------------------------------------------------
    obs_east = np.zeros(len(obs_list))
    obs_north = np.zeros(len(obs_list))
    obs_r_tilde = np.zeros(len(obs_list))
    obs_s_center = np.zeros(len(obs_list))
    obs_e_center = np.zeros(len(obs_list))
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
        obs_e_center[j] = float(obs.e_m) if obs.e_m is not None else 0.0

    # ---------------------------------------------------------------------------
    # Feature flags from environment
    # ---------------------------------------------------------------------------
    dynamics_scheme = os.environ.get("FATROP_DYNAMICS_SCHEME", "euler").strip().lower()
    if dynamics_scheme not in {"euler", "trapezoidal", "rk4"}:
        raise ValueError(
            f"Unsupported FATROP_DYNAMICS_SCHEME={dynamics_scheme!r}; "
            "expected 'euler', 'rk4', or 'trapezoidal'."
        )

    # FATROP_SMOOTH_CONTROLS=1: lift (δ, Fx) into state, optimize arc-length rates.
    smooth_controls = os.environ.get("FATROP_SMOOTH_CONTROLS", "0").strip() == "1"

    # Effective state dimension: 8 base + 2 actuator states when augmented.
    nx_eff = nx + nu if smooth_controls else nx

    # Conservative arc-length rate bounds for augmented actuator states.
    # dδ/ds in arc-length ≈ δ_dot / ux; bound by min speed to stay conservative.
    ux_ref = max(float(ux_min), 1.0)
    v_delta_max = float(p.max_delta_dot_radps) / ux_ref   # rad/m
    v_fx_max    = float(p.max_fx_dot_knps)     / ux_ref   # kN/m
    v_fx_min    = float(p.min_fx_dot_knps)     / ux_ref   # kN/m (negative)

    # ---------------------------------------------------------------------------
    # Decision variables — stage-interleaved for FATROP auto structure detection.
    # Order: x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N, u_N(dummy)
    # ---------------------------------------------------------------------------
    opti = ca.Opti()
    x_vars = []
    u_vars = []
    for k in range(N):
        x_vars.append(opti.variable(nx_eff))
        u_vars.append(opti.variable(nu))
    x_vars.append(opti.variable(nx_eff))  # terminal state x_N
    u_vars.append(opti.variable(0))       # dummy zero-dim terminal control

    nx_list = [nx_eff for _ in range(N + 1)]
    nu_list = [nu for _ in range(N)] + [0]
    ng_list: List[int] = [0 for _ in range(N + 1)]

    # ---------------------------------------------------------------------------
    # Helper: augmented spatial dynamics dz/ds at a given state z and rate v.
    # z = [x(8), δ, Fx] when smooth_controls else z = x(8).
    # v = [dδ/ds, dFx/ds] when smooth_controls else v = [δ, Fx].
    # Returns dz/ds as a CasADi expression.
    # ---------------------------------------------------------------------------
    def _aug_dz_ds(z, v, kap, the, ph):
        if smooth_controls:
            x_dyn = z[:nx]
            u_eff = z[nx:]          # (δ, Fx) from state
            dx_dt, sdot = vehicle.dynamics_dt_path_vec(x_dyn, u_eff, kap, the, ph)
            return ca.vertcat(dx_dt / sdot, v[0], v[1])
        else:
            dx_dt, sdot = vehicle.dynamics_dt_path_vec(z, v, kap, the, ph)
            return dx_dt / sdot

    # ---------------------------------------------------------------------------
    # Pre-compile dynamics and full RK4 step as CasADi Functions.
    #
    # For Euler/trapezoidal: F_stage(x, u, kap, the, phi) -> dz/ds (one call/stage).
    # For RK4: F_rk4(x, u, kap_k, the_k, phi_k, kap_mid, the_mid, phi_mid,
    #                kap_kp1, the_kp1, phi_kp1, ds) -> x_{k+1}
    #   Compiling the whole RK4 step as ONE function means each stage adds only one
    #   call node to the NLP graph, so CasADi's AD cost is O(N) not O(N·k_nest).
    # ---------------------------------------------------------------------------
    _x_sym   = ca.MX.sym('x',   nx_eff)
    _u_sym   = ca.MX.sym('u',   nu)
    _kap_sym = ca.MX.sym('kap')
    _the_sym = ca.MX.sym('the')
    _phi_sym = ca.MX.sym('phi')
    _dz_expr = _aug_dz_ds(_x_sym, _u_sym, _kap_sym, _the_sym, _phi_sym)
    F_stage  = ca.Function(
        'F_stage',
        [_x_sym, _u_sym, _kap_sym, _the_sym, _phi_sym],
        [_dz_expr],
    )

    # Full RK4 step compiled as a single function (k1..k4 are internal to the graph).
    _kapm_sym = ca.MX.sym('kapm')
    _them_sym = ca.MX.sym('them')
    _phim_sym = ca.MX.sym('phim')
    _kapn_sym = ca.MX.sym('kapn')
    _then_sym = ca.MX.sym('then')
    _phin_sym = ca.MX.sym('phin')
    _ds_sym   = ca.MX.sym('ds')
    _k1 = _aug_dz_ds(_x_sym,                     _u_sym, _kap_sym,  _the_sym,  _phi_sym)
    _k2 = _aug_dz_ds(_x_sym + 0.5*_ds_sym*_k1,  _u_sym, _kapm_sym, _them_sym, _phim_sym)
    _k3 = _aug_dz_ds(_x_sym + 0.5*_ds_sym*_k2,  _u_sym, _kapm_sym, _them_sym, _phim_sym)
    _k4 = _aug_dz_ds(_x_sym +     _ds_sym*_k3,  _u_sym, _kapn_sym, _then_sym, _phin_sym)
    _xnext = _x_sym + (_ds_sym / 6.0) * (_k1 + 2*_k2 + 2*_k3 + _k4)
    F_rk4 = ca.Function(
        'F_rk4',
        [_x_sym, _u_sym,
         _kap_sym,  _the_sym,  _phi_sym,
         _kapm_sym, _them_sym, _phim_sym,
         _kapn_sym, _then_sym, _phin_sym,
         _ds_sym],
        [_xnext],
    )

    # ---------------------------------------------------------------------------
    # Stage loop: dynamics + path constraints
    # Constraint ordering for FATROP manual structure: dynamics_k, path_k.
    # ---------------------------------------------------------------------------
    if x0 is not None and len(x0) < nx:
        raise ValueError(f"x0 must have at least {nx} entries, got {len(x0)}")

    for k in range(N):
        xk   = x_vars[k]
        xkp1 = x_vars[k + 1]
        uk   = u_vars[k]

        # --- Discrete dynamics ---
        if dynamics_scheme == "rk4":
            # 4th-order Runge-Kutta in arc-length, stage-local (same u_k throughout).
            # Uses pre-compiled F_rk4 (entire step as one Function) so each stage
            # adds exactly one call node to the NLP — same as Euler's cost per stage.
            xkp1_pred = F_rk4(
                xk, uk,
                kappa[k],     theta[k],     phi[k],
                kappa_mid[k], theta_mid[k], phi_mid[k],
                kappa[k + 1], theta[k + 1], phi[k + 1],
                ds_m,
            )
            opti.subject_to(xkp1 == xkp1_pred)
        elif dynamics_scheme == "trapezoidal":
            # Trapezoidal — cross-stage (references u_{k+1}); breaks FATROP structure.
            # Kept for reference/fallback with structure_detection=none.
            ukp1 = u_vars[k + 1] if (k + 1) < N else uk
            dz_k   = F_stage(xk,   uk,   kappa[k],     theta[k],     phi[k])
            dz_kp1 = F_stage(xkp1, ukp1, kappa[k + 1], theta[k + 1], phi[k + 1])
            opti.subject_to(xkp1 == xk + 0.5 * ds_m * (dz_k + dz_kp1))
        else:
            # Euler (default) — O(ds²) local error, fully stage-local.
            opti.subject_to(xkp1 == xk + ds_m * F_stage(xk, uk, kappa[k], theta[k], phi[k]))

        # --- Path constraints at stage k ---
        ux_k   = xk[0]
        uy_k   = xk[1]
        e_k    = xk[6]
        dpsi_k = xk[7]

        if smooth_controls:
            # Actuator values are state components [8] and [9].
            delta_k = xk[nx]
            fx_k    = xk[nx + 1]
            # Rate controls (the optimization variable): [dδ/ds, dFx/ds].
            v_delta_k = uk[0]
            v_fx_k    = uk[1]
        else:
            delta_k = uk[0]
            fx_k    = uk[1]

        cons = []
        # Initial-time boundary condition inside stage 0: treats it as a stage-0
        # path constraint so FATROP auto structure detection does not see it as a
        # "pre-stage" constraint (which would make all downstream dynamics cross-stage).
        # x0 masked initial-state constraints are also placed here for the same reason:
        # adding them after the stage loop causes the structure detector to see x_vars[0]
        # referenced at stage N, which triggers "depending on a state of the previous
        # interval" errors and forces O(N³) generic NLP mode.
        if k == 0:
            cons.append(xk[5] == 0.0)
            if x0 is not None:
                for i in range(nx):
                    if x0[i] is not None:
                        cons.append(xk[i] == float(x0[i]))

        # Speed and forward-progress constraints.
        cons.append(ux_k >= ux_min)
        if ux_max is not None:
            cons.append(ux_k <= ux_max)
        one_minus = 1 - kappa[k] * e_k
        if abs(float(kappa[k])) > 1e-12:
            cons.append(one_minus >= eps_kappa)
        sdot_k = (ux_k * ca.cos(dpsi_k) - uy_k * ca.sin(dpsi_k)) / one_minus
        cons.append(sdot_k >= eps_s)

        # Track boundary.
        cons.append(e_k >= -track_hw[k] + track_buffer_m)
        cons.append(e_k <=  track_hw[k] - track_buffer_m)

        # Actuator bounds.
        cons.append(delta_k >= -p.max_delta_rad)
        cons.append(delta_k <=  p.max_delta_rad)
        cons.append(fx_k >= p.min_fx_kn)
        cons.append(fx_k <= p.max_fx_kn)

        # Rate bounds on the arc-length-derivative controls (smooth_controls only).
        if smooth_controls:
            cons.append(v_delta_k >= -v_delta_max)
            cons.append(v_delta_k <=  v_delta_max)
            cons.append(v_fx_k    >=  v_fx_min)
            cons.append(v_fx_k    <=  v_fx_max)

        # Obstacle avoidance (within gating window).
        if len(obs_list) > 0:
            posE_k = posE_cl[k] - e_k * np.sin(psi_cl[k])
            posN_k = posN_cl[k] + e_k * np.cos(psi_cl[k])
            for j in range(len(obs_list)):
                ds_wrap = (s_grid[k] - obs_s_center[j]) % world.length_m
                if ds_wrap > 0.5 * world.length_m:
                    ds_wrap -= world.length_m
                if abs(ds_wrap) <= obstacle_window_m:
                    req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
                    cons.append(
                        (posE_k - obs_east[j]) ** 2 + (posN_k - obs_north[j]) ** 2 >= req_r ** 2
                    )

        for c in cons:
            opti.subject_to(c)
            ng_list[k] += int(c.nnz())

    # ---------------------------------------------------------------------------
    # Terminal path constraints at stage N (no dynamics).
    # ---------------------------------------------------------------------------
    xN     = x_vars[N]
    ux_N   = xN[0]
    uy_N   = xN[1]
    e_N    = xN[6]
    dpsi_N = xN[7]

    cons_N = []
    cons_N.append(ux_N >= ux_min)
    if ux_max is not None:
        cons_N.append(ux_N <= ux_max)
    one_minus_N = 1 - kappa[N] * e_N
    if abs(float(kappa[N])) > 1e-12:
        cons_N.append(one_minus_N >= eps_kappa)
    sdot_N = (ux_N * ca.cos(dpsi_N) - uy_N * ca.sin(dpsi_N)) / one_minus_N
    cons_N.append(sdot_N >= eps_s)
    cons_N.append(e_N >= -track_hw[N] + track_buffer_m)
    cons_N.append(e_N <=  track_hw[N] - track_buffer_m)

    # Actuator bounds at terminal stage (smooth_controls: (δ,Fx) are state components).
    if smooth_controls:
        delta_N = xN[nx]
        fx_N    = xN[nx + 1]
        cons_N.append(delta_N >= -p.max_delta_rad)
        cons_N.append(delta_N <=  p.max_delta_rad)
        cons_N.append(fx_N >= p.min_fx_kn)
        cons_N.append(fx_N <= p.max_fx_kn)

    if len(obs_list) > 0:
        posE_N = posE_cl[N] - e_N * np.sin(psi_cl[N])
        posN_N = posN_cl[N] + e_N * np.cos(psi_cl[N])
        for j in range(len(obs_list)):
            ds_wrap = (s_grid[N] - obs_s_center[j]) % world.length_m
            if ds_wrap > 0.5 * world.length_m:
                ds_wrap -= world.length_m
            if abs(ds_wrap) <= obstacle_window_m:
                req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
                cons_N.append(
                    (posE_N - obs_east[j]) ** 2 + (posN_N - obs_north[j]) ** 2 >= req_r ** 2
                )
    for c in cons_N:
        opti.subject_to(c)
        ng_list[N] += int(c.nnz())

    # ---------------------------------------------------------------------------
    # Objective: terminal time + stage-local control regularization + optional
    # terminal-mask penalty + optional closure term.
    # When smooth_controls=1, u_k = [dδ/ds, dFx/ds], so ||u_k||^2 penalizes rates.
    # ---------------------------------------------------------------------------
    stage_local_cost = os.environ.get("FATROP_STAGE_LOCAL_COST", "1").strip() == "1"
    reg_cost = 0.0
    if stage_local_cost:
        reg_terms = [ca.sumsqr(u_vars[k]) for k in range(N)]
        if reg_terms:
            reg_cost = lambda_u * ca.sum1(ca.vertcat(*reg_terms))

    closure_mode = os.environ.get("FATROP_CLOSURE_MODE", "soft").strip().lower()
    closure_soft_weight = float(os.environ.get("FATROP_CLOSURE_SOFT_WEIGHT", "100.0"))
    closure_cost = 0.0
    if closure_mode == "soft":
        xdiff_terms = [x_vars[N][i] - x_vars[0][i] for i in (0, 1, 2, 3, 4, 6, 7)]
        closure_cost = closure_soft_weight * ca.sumsqr(ca.vertcat(*xdiff_terms))
    elif closure_mode == "hard":
        for idx_state in (0, 1, 2, 3, 4, 6, 7):
            opti.subject_to(x_vars[N][idx_state] == x_vars[0][idx_state])
        opti.subject_to(u_vars[N - 1][0] == u_vars[0][0])
        opti.subject_to(u_vars[N - 1][1] == u_vars[0][1])

    terminal_cost = 0.0
    if terminal_state is not None and terminal_mask is not None and float(terminal_weight) > 0.0:
        term_arr = np.asarray(terminal_state, dtype=float).reshape(-1)
        mask_arr = np.asarray(terminal_mask, dtype=bool).reshape(-1)
        if term_arr.size < nx:
            raise ValueError(f"terminal_state must have at least {nx} entries, got {term_arr.size}")
        if mask_arr.size < nx:
            raise ValueError(f"terminal_mask must have at least {nx} entries, got {mask_arr.size}")
        term_terms = []
        for i in range(nx):
            if bool(mask_arr[i]):
                term_terms.append((x_vars[N][i] - float(term_arr[i])) ** 2)
        if term_terms:
            terminal_cost = float(terminal_weight) * ca.sum1(ca.vertcat(*term_terms))

    objective = x_vars[N][5] + reg_cost + terminal_cost + closure_cost
    opti.minimize(objective)

    # ---------------------------------------------------------------------------
    # Initial guess.
    # X_init shape: (8, N+1) — always the 8-state vehicle trajectory.
    # U_init shape: (2, N+1) — (δ, Fx) at each node.
    #   Non-augmented: U_init feeds u_vars directly.
    #   Augmented: U_init feeds the actuator state x_vars[k][8:10];
    #              rate controls u_vars[k] are initialized to zero.
    # ---------------------------------------------------------------------------
    if X_init is not None and U_init is not None:
        if X_init.shape != (nx, N + 1):
            raise ValueError(f"X_init shape must be {(nx, N + 1)}, got {X_init.shape}")
        if U_init.shape != (nu, N + 1):
            raise ValueError(f"U_init shape must be {(nu, N + 1)}, got {U_init.shape}")
        for k in range(N + 1):
            opti.set_initial(x_vars[k][:nx], X_init[:, k])
            if smooth_controls:
                opti.set_initial(x_vars[k][nx],     U_init[0, k])  # δ
                opti.set_initial(x_vars[k][nx + 1], U_init[1, k])  # Fx
                if k < N:
                    opti.set_initial(u_vars[k][0], 0.0)
                    opti.set_initial(u_vars[k][1], 0.0)
            else:
                if k < N:
                    opti.set_initial(u_vars[k], U_init[:, k])
    else:
        # Initial guess: straight-line cruise matching IPOPT's convention.
        # ux_seed=10 m/s is sub-optimal (true optimal ~18 m/s) so the solver must
        # actively find the optimal speed (large enough residual to avoid premature
        # KKT convergence, but close enough to give a sensible t_init).
        # Fx_seed=0.5 kN: slight over-thrust at 10 m/s (drag is ~0.26 kN), creating
        # a positive acceleration signal that biases the solver toward higher speeds.
        # e_init: obstacle-aware lateral offset, pre-biased to opposite side of
        # each obstacle so the hard obstacle constraints start near-feasible.
        # ux_seed: initial guess for longitudinal velocity at all nodes.
        # Must be strictly > ux_min so the interior-point barrier is well-defined.
        # With ux_min=0.1, ux_seed=0.5 gives slack=0.4 — safely interior.
        # Fx_seed=0.5 kN > drag(0.5 m/s)≈0.22 kN: intentional over-thrust creates
        # a gradient pointing toward faster speeds (avoids premature KKT convergence).
        # t_init uses a separate cruise estimate so the time state is sensible
        # regardless of ux_seed (ux_seed cannot drive t_init in multiple shooting
        # since x_{k+1} is an independent variable — but the dynamics residual
        # |ds/sdot - dt_per_step| should stay small for fast convergence).
        # ux_seed must be well above ux_min (interior-point needs slack) AND large
        # enough that t_init = N*ds/ux_seed is within ~4× of the optimal lap time.
        # At ux_seed=0.5 m/s, dynamics residuals at all 150 nodes are ~0.5 m/s in ux
        # (Euler step: ux_{k+1}≈ux_k+0.52) — too many large simultaneous corrections.
        # Minimum practical: ux_seed≈5 m/s. Best result: ux_seed=10 m/s.
        ux_seed = max(10.0, ux_min + 1.0)
        Fx_seed = 0.5  # kN — over-thrust vs drag(10 m/s)≈0.26 kN
        t_init = np.linspace(0.0, N * ds_m / ux_seed, N + 1)

        # Lateral offset initial guess: uniform 0 (no obstacles) or obstacle-aware.
        if len(obs_list) > 0:
            e_init = _build_obstacle_aware_e_init(
                s_grid=s_grid,
                track_hw=track_hw,
                obs_s_center=obs_s_center,
                obs_e_center=obs_e_center,
                obs_r_tilde=obs_r_tilde,
                length_m=float(world.length_m),
                obstacle_clearance_m=obstacle_clearance_m,
                vehicle_radius_m=vehicle_radius_m,
                track_buffer_m=track_buffer_m,
            )
        else:
            e_init = np.zeros(N + 1)

        for k in range(N + 1):
            opti.set_initial(x_vars[k][0], ux_seed)
            opti.set_initial(x_vars[k][1], 0.0)
            opti.set_initial(x_vars[k][2], 0.0)
            opti.set_initial(x_vars[k][3], 0.0)
            opti.set_initial(x_vars[k][4], 0.0)
            opti.set_initial(x_vars[k][5], float(t_init[k]))
            opti.set_initial(x_vars[k][6], float(e_init[k]))
            opti.set_initial(x_vars[k][7], 0.0)
            if smooth_controls:
                opti.set_initial(x_vars[k][nx],     0.0)      # δ = 0
                opti.set_initial(x_vars[k][nx + 1], Fx_seed)  # drag-balanced Fx ≈ 0
                if k < N:
                    opti.set_initial(u_vars[k][0], 0.0)       # dδ/ds = 0
                    opti.set_initial(u_vars[k][1], 0.0)       # dFx/ds = 0
            else:
                if k < N:
                    opti.set_initial(u_vars[k][0], 0.0)       # δ = 0
                    opti.set_initial(u_vars[k][1], Fx_seed)   # drag-balanced Fx ≈ 0

    # ---------------------------------------------------------------------------
    # Solver options.
    # ---------------------------------------------------------------------------
    preset = os.environ.get("FATROP_PRESET", "obstacle_fast").strip().lower()
    preset_cfg = {
        "fast":          {"mu_init": 0.2, "tol": 1e-4, "acceptable_tol": 1e-3},
        "obstacle_fast": {"mu_init": 0.3, "tol": 1e-4, "acceptable_tol": 1e-3},
        "balanced":      {"mu_init": 0.1},
        "accurate":      {"mu_init": 0.1, "tol": 1e-6, "acceptable_tol": 1e-6},
    }.get(preset, {"mu_init": 0.3, "tol": 1e-4, "acceptable_tol": 1e-3})

    fatrop_opts = {
        "mu_init":     float(os.environ.get("FATROP_MU_INIT", str(preset_cfg["mu_init"]))),
        "print_level": int(os.environ.get("FATROP_PRINT_LEVEL", "0")),
    }
    if "tol" in preset_cfg:
        fatrop_opts["tol"] = float(preset_cfg["tol"])
    if "acceptable_tol" in preset_cfg:
        fatrop_opts["acceptable_tol"] = float(preset_cfg["acceptable_tol"])
    if "FATROP_TOL" in os.environ:
        fatrop_opts["tol"] = float(os.environ["FATROP_TOL"])
    if "FATROP_ACCEPTABLE_TOL" in os.environ:
        fatrop_opts["acceptable_tol"] = float(os.environ["FATROP_ACCEPTABLE_TOL"])
    if "FATROP_MAX_ITER" in os.environ:
        try:
            fatrop_opts["max_iter"] = int(float(os.environ["FATROP_MAX_ITER"]))
        except ValueError:
            pass

    structure_mode = os.environ.get("FATROP_STRUCTURE_DETECTION", "none").strip().lower()
    opts = {
        "print_time":        bool(verbose),
        "expand":            os.environ.get("FATROP_EXPAND", "0") != "0",
        "structure_detection": structure_mode,
        "fatrop":            dict(fatrop_opts),
    }
    if structure_mode == "manual":
        opts["nx"] = nx_list
        opts["nu"] = nu_list
        opts["ng"] = ng_list
        opts["N"]  = N

    t1 = time.time()
    build_time_s = t1 - t_build0
    t0 = time.time()
    opti.solver("fatrop", opts)
    sol = opti.solve()
    solve_time = time.time() - t0
    total_time_s = build_time_s + solve_time

    # ---------------------------------------------------------------------------
    # Extract solution.
    # X_opt: (8, N+1) vehicle states (first 8 components of augmented state).
    # U_opt: (2, N+1) actuator signals (δ, Fx).
    #   Non-augmented: u_vars[k] = [δ, Fx].
    #   Augmented:     x_vars[k][8:10] = [δ, Fx]; u_vars are rates (not returned).
    # ---------------------------------------------------------------------------
    X_opt = np.column_stack([
        np.asarray(sol.value(xk[:nx]), dtype=float) for xk in x_vars
    ])
    U_opt = np.zeros((nu, N + 1), dtype=float)
    if smooth_controls:
        for k in range(N + 1):
            U_opt[:, k] = np.asarray(sol.value(x_vars[k][nx:]), dtype=float).reshape(-1)
    else:
        for k in range(N):
            U_opt[:, k] = np.asarray(sol.value(u_vars[k]), dtype=float).reshape(-1)
        U_opt[:, N] = U_opt[:, N - 1]

    cost_opt = float(sol.value(objective))
    stats = sol.stats()
    iterations = int(stats.get("iter_count", -1))

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

    result = OptimizationResult(
        success=True,
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
    setattr(result, "build_time_s", float(build_time_s))
    setattr(result, "total_time_s", float(total_time_s))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--output-dir", type=str, default="results/trajectory_optimization/fatrop")
    ap.add_argument("--animate", action="store_true")
    args = ap.parse_args()

    map_file = Path(args.map_file)
    if not map_file.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    world = World(str(map_file), map_file.stem, diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    obstacles = _load_obstacles_from_world(world)
    ds_m = float(world.length_m / int(args.N))

    fat = solve_fatrop_native(
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
        verbose=bool(args.verbose),
    )
    closure_mode = os.environ.get("FATROP_CLOSURE_MODE", "soft").strip().lower()
    smooth_controls = os.environ.get("FATROP_SMOOTH_CONTROLS", "0").strip() == "1"
    dynamics_scheme = os.environ.get("FATROP_DYNAMICS_SCHEME", "euler").strip().lower()
    print(
        f"[fatrop-native] success={fat.success} iterations={fat.iterations} "
        f"cost={fat.cost:.6f} solve_time={fat.solve_time:.3f}s "
        f"build_time={getattr(fat, 'build_time_s', float('nan')):.3f}s "
        f"total_time={getattr(fat, 'total_time_s', float('nan')):.3f}s "
        f"min_clearance={fat.min_obstacle_clearance:.4f} "
        f"scheme={dynamics_scheme} smooth={smooth_controls} closure={closure_mode}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"fatrop_N{args.N}"
    visualizer = TrajectoryVisualizer(world, output_dir=str(output_dir))
    plots = visualizer.generate_full_report(fat, prefix=prefix)
    for kind, path in plots.items():
        print(f"  [{kind}] {path}")

    if args.animate:
        try:
            anim_path = create_animation(
                visualizer, fat,
                filename=f"{prefix}_animation.gif",
                fps=15,
            )
            print(f"  [animation] {anim_path}")
        except Exception as e:
            print(f"  [animation] skipped: {e}")


if __name__ == "__main__":
    main()
