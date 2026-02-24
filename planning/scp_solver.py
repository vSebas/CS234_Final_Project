"""
Sequential Convex Programming (SCP) solver for trajectory optimization.

This is a clean implementation focused on robustness:
- Affine linearization of spatial dynamics around a reference trajectory
- Convex subproblem with virtual control slack
- Scaled trust region on states/controls
- Nonlinear-merit acceptance/rejection (true defect + lap time)
- Two-phase behavior: prioritize feasibility first, then lap-time refinement
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import casadi as ca
import numpy as np


@dataclass
class SCPParams:
    # Trust-region settings
    tr_radius_init: float = 0.75
    tr_radius_min: float = 0.01
    tr_radius_max: float = 10.0
    tr_shrink_factor: float = 0.7
    tr_expand_factor: float = 1.2

    # Iteration / convergence settings
    max_iterations: int = 80
    convergence_tol: float = 1e-4
    constraint_tol: float = 1e-3
    virtual_control_tol: float = 1e-4

    # Subproblem / merit weights
    virtual_control_weight: float = 1e4
    defect_penalty_weight: float = 1e4
    lap_time_weight: float = 1.0
    regularization_weight: float = 1e-6

    # Feasibility-first threshold
    defect_switch_tol: float = 0.05

    # Optional early exit for already feasible warm starts
    early_exit_on_feasible: bool = True
    max_solve_time_s: float = 120.0

    # Solver backend for subproblems
    qp_solver: str = "ipopt"  # "ipopt", "osqp", or "qrqp"
    verbose: bool = True


@dataclass
class SCPResult:
    success: bool
    s_m: np.ndarray
    X: np.ndarray
    U: np.ndarray
    cost: float

    iterations: int
    iteration_history: list = field(default_factory=list)
    tr_radius_history: list = field(default_factory=list)
    constraint_violation_history: list = field(default_factory=list)
    virtual_control_history: list = field(default_factory=list)

    solve_time: float = 0.0
    subproblem_times: list = field(default_factory=list)

    k_psi: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))

    converged: bool = False
    feasible: bool = False
    termination_reason: str = ""


class SCPSolver:
    IDX_UX = 0
    IDX_UY = 1
    IDX_R = 2
    IDX_DFZ_LONG = 3
    IDX_DFZ_LAT = 4
    IDX_T = 5
    IDX_E = 6
    IDX_DPSI = 7

    STATE_SCALES = np.array([10.0, 2.0, 0.5, 1.0, 1.0, 1.0, 2.0, 0.3])
    CONTROL_SCALES = np.array([0.3, 5.0])

    def __init__(self, vehicle, world, params: Optional[SCPParams] = None):
        self.vehicle = vehicle
        self.world = world
        self.params = params or SCPParams()

        self.nx = 8
        self.nu = 2

        self._build_linearization_functions()

    def _build_linearization_functions(self) -> None:
        x_sym = ca.SX.sym("x", self.nx)
        u_sym = ca.SX.sym("u", self.nu)
        k_psi_sym = ca.SX.sym("k_psi")
        theta_sym = ca.SX.sym("theta")
        phi_sym = ca.SX.sym("phi")

        dx_dt, s_dot = self.vehicle.dynamics_dt_path_vec(
            x_sym, u_sym, k_psi_sym, theta_sym, phi_sym
        )
        dx_ds = dx_dt / s_dot

        A = ca.jacobian(dx_ds, x_sym)
        B = ca.jacobian(dx_ds, u_sym)

        self.f_dynamics = ca.Function(
            "f_dynamics",
            [x_sym, u_sym, k_psi_sym, theta_sym, phi_sym],
            [dx_ds],
            ["x", "u", "k_psi", "theta", "phi"],
            ["dx_ds"],
        )

        self.f_jacobians = ca.Function(
            "f_jacobians",
            [x_sym, u_sym, k_psi_sym, theta_sym, phi_sym],
            [A, B, dx_ds],
            ["x", "u", "k_psi", "theta", "phi"],
            ["A", "B", "f"],
        )

    def _get_road_geometry(self, s: float) -> Tuple[float, float, float]:
        s_mod = s % self.world.length_m
        k_psi = float(self.world.psi_s_radpm_LUT(s_mod))
        theta = float(self.world.grade_rad_LUT(s_mod)) if hasattr(self.world, "grade_rad_LUT") else 0.0
        phi = float(self.world.bank_rad_LUT(s_mod)) if hasattr(self.world, "bank_rad_LUT") else 0.0
        return k_psi, theta, phi

    def _get_track_half_width(self, s: float) -> float:
        s_mod = s % self.world.length_m
        return float(self.world.track_width_m_LUT(s_mod)) / 2.0

    def _sanitize_linearization_point(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keep linearization points away from singular/unphysical regions.

        This prevents CasADi Jacobian evaluation from producing inf/nan values
        when cold-start guesses are poor.
        """
        p = self.vehicle.params
        xs = np.array(x, dtype=float).copy()
        us = np.array(u, dtype=float).copy()

        # Keep forward speed positive for well-defined spatial dynamics.
        xs[self.IDX_UX] = max(xs[self.IDX_UX], 3.0)

        # Keep path denominator 1 - k*e away from zero by clamping lateral error.
        xs[self.IDX_E] = float(np.clip(xs[self.IDX_E], -2.0, 2.0))
        xs[self.IDX_DPSI] = float(np.clip(xs[self.IDX_DPSI], -0.6, 0.6))

        # Clip controls to physical bounds.
        us[0] = float(np.clip(us[0], -p.max_delta_rad, p.max_delta_rad))
        us[1] = float(np.clip(us[1], p.min_fx_kn, p.max_fx_kn))
        return xs, us

    def _linearize_around_trajectory(
        self,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray,
    ) -> Tuple[list, list, list]:
        n_nodes = X_ref.shape[1]
        A_list, B_list, c_list = [], [], []

        for k in range(n_nodes):
            x_k, u_k = self._sanitize_linearization_point(X_ref[:, k], U_ref[:, k])

            A_k_out, B_k_out, f_k_out = self.f_jacobians(
                x_k, u_k, k_psi_data[k], theta_data[k], phi_data[k]
            )
            A_k = np.array(A_k_out)
            B_k = np.array(B_k_out)
            f_k = np.array(f_k_out).flatten()

            if not (np.all(np.isfinite(A_k)) and np.all(np.isfinite(B_k)) and np.all(np.isfinite(f_k))):
                A_k = np.zeros((self.nx, self.nx))
                B_k = np.zeros((self.nx, self.nu))
                f_k = np.zeros(self.nx)

            c_k = f_k - A_k @ x_k - B_k @ u_k
            A_list.append(A_k)
            B_list.append(B_k)
            c_list.append(c_k)

        return A_list, B_list, c_list

    def _compute_nonlinear_defects(
        self,
        X: np.ndarray,
        U: np.ndarray,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray,
        ds_m: float,
    ) -> np.ndarray:
        N = X.shape[1] - 1
        defects = np.zeros((self.nx, N))

        for k in range(N):
            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            u_kp1 = U[:, k + 1]

            f_k = np.array(self.f_dynamics(x_k, u_k, k_psi_data[k], theta_data[k], phi_data[k])).flatten()
            f_kp1 = np.array(
                self.f_dynamics(x_kp1, u_kp1, k_psi_data[k + 1], theta_data[k + 1], phi_data[k + 1])
            ).flatten()

            x_kp1_pred = x_k + 0.5 * ds_m * (f_k + f_kp1)
            defects[:, k] = x_kp1 - x_kp1_pred

        return defects

    def _compute_nonlinear_merit(
        self,
        X: np.ndarray,
        U: np.ndarray,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray,
        ds_m: float,
    ) -> Tuple[float, float, float, float]:
        lap_time = float(X[self.IDX_T, -1])
        defects = self._compute_nonlinear_defects(X, U, k_psi_data, theta_data, phi_data, ds_m)
        defect_norm_sq = float(np.sum(defects**2))
        defect_norm = float(np.sqrt(defect_norm_sq))

        merit = self.params.defect_penalty_weight * defect_norm_sq + self.params.lap_time_weight * lap_time
        return merit, lap_time, defect_norm, defect_norm_sq

    def _build_cold_start_guess(self, N: int, ds_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a conservative constant cold-start.

        This avoids singular linearization points that can appear in nonlinear
        rollout-based initialization.
        """
        X = np.zeros((self.nx, N + 1))
        U = np.zeros((self.nu, N + 1))

        ux_init = 10.0
        X[self.IDX_UX, :] = ux_init
        X[self.IDX_T, :] = np.linspace(0.0, N * ds_m / ux_init, N + 1)
        return X, U

    def _perturb_control_seed(self, U_ref: np.ndarray) -> np.ndarray:
        """
        Apply a small deterministic perturbation to controls to escape fixed points
        in QP-debug runs.
        """
        p = self.vehicle.params
        U_pert = U_ref.copy()
        n_nodes = U_ref.shape[1]
        phase = np.linspace(0.0, 2.0 * np.pi, n_nodes)

        U_pert[0, :] += 0.01 * np.sin(phase)  # steering perturbation [rad]
        U_pert[1, :] += 0.05 * np.cos(phase)  # force perturbation [kN]

        U_pert[0, :] = np.clip(U_pert[0, :], -p.max_delta_rad, p.max_delta_rad)
        U_pert[1, :] = np.clip(U_pert[1, :], p.min_fx_kn, p.max_fx_kn)
        return U_pert

    def _solve_convex_subproblem(
        self,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        ds_m: float,
        A_list: list,
        B_list: list,
        c_list: list,
        track_hw: np.ndarray,
        tr_multiplier: float,
        ux_min: float,
        ux_max: Optional[float],
        convergent_lap: bool,
        phase_feasibility: bool,
        iteration: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        N = X_ref.shape[1] - 1
        p = self.vehicle.params
        opti = ca.Opti()

        X = opti.variable(self.nx, N + 1)
        U = opti.variable(self.nu, N + 1)
        V = opti.variable(self.nx, N)

        ux = X[self.IDX_UX, :]
        dfz_long = X[self.IDX_DFZ_LONG, :]
        dfz_lat = X[self.IDX_DFZ_LAT, :]
        t = X[self.IDX_T, :]
        e = X[self.IDX_E, :]
        dpsi = X[self.IDX_DPSI, :]
        delta = U[0, :]
        fx = U[1, :]

        # QP debug backends (osqp/qrqp) share relaxed constraints and soft slacks.
        use_qp_debug = self.params.qp_solver in ("osqp", "qrqp")
        use_soft_constraints = use_qp_debug
        slack_penalty_track = 1e3
        slack_penalty_periodic = 1e3
        slack_penalty_heading = 5e2
        slack_penalty_control = 5e2

        if use_soft_constraints:
            # Slack for track bounds (2 per node: lower and upper)
            s_track_lo = opti.variable(N + 1)  # e >= -hw + buffer - s_track_lo
            s_track_hi = opti.variable(N + 1)  # e <= hw - buffer + s_track_hi
            opti.subject_to(s_track_lo >= 0)
            opti.subject_to(s_track_hi >= 0)

            # Slack for heading bounds (2 per node)
            s_dpsi_lo = opti.variable(N + 1)
            s_dpsi_hi = opti.variable(N + 1)
            opti.subject_to(s_dpsi_lo >= 0)
            opti.subject_to(s_dpsi_hi >= 0)

            # Slack for control bounds (4 per node: delta lower/upper, fx lower/upper)
            s_delta_lo = opti.variable(N + 1)
            s_delta_hi = opti.variable(N + 1)
            s_fx_lo = opti.variable(N + 1)
            s_fx_hi = opti.variable(N + 1)
            opti.subject_to(s_delta_lo >= 0)
            opti.subject_to(s_delta_hi >= 0)
            opti.subject_to(s_fx_lo >= 0)
            opti.subject_to(s_fx_hi >= 0)

            # Slack for periodicity constraints (one per periodic state/control)
            n_periodic = 9  # ux, uy, r, dfz_long, dfz_lat, e, dpsi, delta, fx
            s_periodic = opti.variable(n_periodic)
            opti.subject_to(s_periodic >= 0)

        # Feasibility-first behavior: initially reduce V and stay near reference.
        lap_w = 0.1 if phase_feasibility else self.params.lap_time_weight
        reg_w = 1e-4 if use_soft_constraints else self.params.regularization_weight

        cost = (
            lap_w * t[N]
            + self.params.virtual_control_weight * ca.sumsqr(V)
            + reg_w * ca.sumsqr(X - X_ref)
            + reg_w * ca.sumsqr(U - U_ref)
        )

        if use_soft_constraints:
            cost += slack_penalty_track * (ca.sumsqr(s_track_lo) + ca.sumsqr(s_track_hi))
            cost += slack_penalty_heading * (ca.sumsqr(s_dpsi_lo) + ca.sumsqr(s_dpsi_hi))
            cost += slack_penalty_control * (
                ca.sumsqr(s_delta_lo) + ca.sumsqr(s_delta_hi) + ca.sumsqr(s_fx_lo) + ca.sumsqr(s_fx_hi)
            )
            cost += slack_penalty_periodic * ca.sumsqr(s_periodic)

        opti.minimize(cost)

        # Linearized trapezoidal dynamics
        for k in range(N):
            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            u_kp1 = U[:, k + 1]

            f_k = A_list[k] @ x_k + B_list[k] @ u_k + c_list[k]
            f_kp1 = A_list[k + 1] @ x_kp1 + B_list[k + 1] @ u_kp1 + c_list[k + 1]

            opti.subject_to(x_kp1 == x_k + 0.5 * ds_m * (f_k + f_kp1) + V[:, k])

        # Trust region is disabled in QP-debug mode to isolate feasibility issues.
        if not use_qp_debug:
            for k in range(N + 1):
                for i in range(self.nx):
                    tr_i = tr_multiplier * self.STATE_SCALES[i]
                    opti.subject_to(X[i, k] >= X_ref[i, k] - tr_i)
                    opti.subject_to(X[i, k] <= X_ref[i, k] + tr_i)
                for j in range(self.nu):
                    tr_j = tr_multiplier * self.CONTROL_SCALES[j]
                    opti.subject_to(U[j, k] >= U_ref[j, k] - tr_j)
                    opti.subject_to(U[j, k] <= U_ref[j, k] + tr_j)

        # Path/state constraints
        for k in range(N + 1):
            opti.subject_to(ux[k] >= ux_min)
            if ux_max is not None:
                opti.subject_to(ux[k] <= ux_max)

            buffer = 0.5
            if use_soft_constraints:
                # Soft track bounds for OSQP
                opti.subject_to(e[k] >= -track_hw[k] + buffer - s_track_lo[k])
                opti.subject_to(e[k] <= track_hw[k] - buffer + s_track_hi[k])
            else:
                opti.subject_to(e[k] >= -track_hw[k] + buffer)
                opti.subject_to(e[k] <= track_hw[k] - buffer)
            if use_soft_constraints:
                opti.subject_to(dpsi[k] >= -np.pi / 3 - s_dpsi_lo[k])
                opti.subject_to(dpsi[k] <= np.pi / 3 + s_dpsi_hi[k])
            else:
                opti.subject_to(dpsi[k] >= -np.pi / 3)
                opti.subject_to(dpsi[k] <= np.pi / 3)

            if k > 0:
                opti.subject_to(t[k] >= t[k - 1])

        # Control constraints
        for k in range(N + 1):
            if use_soft_constraints:
                opti.subject_to(delta[k] >= -p.max_delta_rad - s_delta_lo[k])
                opti.subject_to(delta[k] <= p.max_delta_rad + s_delta_hi[k])
                opti.subject_to(fx[k] >= p.min_fx_kn - s_fx_lo[k])
                opti.subject_to(fx[k] <= p.max_fx_kn + s_fx_hi[k])
            else:
                opti.subject_to(delta[k] >= -p.max_delta_rad)
                opti.subject_to(delta[k] <= p.max_delta_rad)
                opti.subject_to(fx[k] >= p.min_fx_kn)
                opti.subject_to(fx[k] <= p.max_fx_kn)

        # Boundary conditions
        opti.subject_to(t[0] == 0)
        if convergent_lap:
            if use_soft_constraints:
                # Soft periodicity for OSQP (allow small violations)
                opti.subject_to(ux[0] - ux[N] <= s_periodic[0])
                opti.subject_to(ux[N] - ux[0] <= s_periodic[0])
                opti.subject_to(X[self.IDX_UY, 0] - X[self.IDX_UY, N] <= s_periodic[1])
                opti.subject_to(X[self.IDX_UY, N] - X[self.IDX_UY, 0] <= s_periodic[1])
                opti.subject_to(X[self.IDX_R, 0] - X[self.IDX_R, N] <= s_periodic[2])
                opti.subject_to(X[self.IDX_R, N] - X[self.IDX_R, 0] <= s_periodic[2])
                opti.subject_to(dfz_long[0] - dfz_long[N] <= s_periodic[3])
                opti.subject_to(dfz_long[N] - dfz_long[0] <= s_periodic[3])
                opti.subject_to(dfz_lat[0] - dfz_lat[N] <= s_periodic[4])
                opti.subject_to(dfz_lat[N] - dfz_lat[0] <= s_periodic[4])
                opti.subject_to(e[0] - e[N] <= s_periodic[5])
                opti.subject_to(e[N] - e[0] <= s_periodic[5])
                opti.subject_to(dpsi[0] - dpsi[N] <= s_periodic[6])
                opti.subject_to(dpsi[N] - dpsi[0] <= s_periodic[6])
                opti.subject_to(delta[0] - delta[N] <= s_periodic[7])
                opti.subject_to(delta[N] - delta[0] <= s_periodic[7])
                opti.subject_to(fx[0] - fx[N] <= s_periodic[8])
                opti.subject_to(fx[N] - fx[0] <= s_periodic[8])
            else:
                opti.subject_to(ux[0] == ux[N])
                opti.subject_to(X[self.IDX_UY, 0] == X[self.IDX_UY, N])
                opti.subject_to(X[self.IDX_R, 0] == X[self.IDX_R, N])
                opti.subject_to(dfz_long[0] == dfz_long[N])
                opti.subject_to(dfz_lat[0] == dfz_lat[N])
                opti.subject_to(e[0] == e[N])
                opti.subject_to(dpsi[0] == dpsi[N])
                opti.subject_to(delta[0] == delta[N])
                opti.subject_to(fx[0] == fx[N])

        opti.set_initial(X, X_ref)
        opti.set_initial(U, U_ref)
        opti.set_initial(V, 0)

        if use_soft_constraints:
            opti.set_initial(s_track_lo, 0)
            opti.set_initial(s_track_hi, 0)
            opti.set_initial(s_dpsi_lo, 0)
            opti.set_initial(s_dpsi_hi, 0)
            opti.set_initial(s_delta_lo, 0)
            opti.set_initial(s_delta_hi, 0)
            opti.set_initial(s_fx_lo, 0)
            opti.set_initial(s_fx_hi, 0)
            opti.set_initial(s_periodic, 0)

        if self.params.qp_solver == "osqp":
            opts = {
                "print_time": False,
                "qpsol": "osqp",
                "qpsol_options": {
                    "error_on_fail": False,
                    "osqp": {
                        "verbose": False,
                        "eps_abs": 1e-3,
                        "eps_rel": 1e-3,
                        "max_iter": 10000,
                        "polish": False,
                    }
                },
                "print_iteration": False,
                "print_header": False,
                "max_iter": 20,
                "convexify_strategy": "regularize",
            }
            opti.solver("sqpmethod", opts)
        elif self.params.qp_solver == "qrqp":
            opts = {
                "print_time": False,
                "qpsol": "qrqp",
                "qpsol_options": {
                    "error_on_fail": False,
                    "print_iter": False,
                    "print_header": False,
                },
                "print_iteration": False,
                "print_header": False,
                "max_iter": 20,
            }
            opti.solver("sqpmethod", opts)
        else:
            opts = {
                "ipopt.print_level": 0,
                "print_time": False,
                "ipopt.max_iter": 300,
                "ipopt.tol": 1e-6,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.max_cpu_time": 5.0,
            }
            opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            X_new = sol.value(X)
            U_new = sol.value(U)
            V_sol = sol.value(V)
            V_norm_sq = float(np.sum(V_sol**2))
            cost_qp = float(sol.value(cost))

            # Log slack usage for OSQP debug
            if use_soft_constraints and self.params.verbose:
                s_track_lo_val = np.array(sol.value(s_track_lo)).flatten()
                s_track_hi_val = np.array(sol.value(s_track_hi)).flatten()
                s_dpsi_lo_val = np.array(sol.value(s_dpsi_lo)).flatten()
                s_dpsi_hi_val = np.array(sol.value(s_dpsi_hi)).flatten()
                s_delta_lo_val = np.array(sol.value(s_delta_lo)).flatten()
                s_delta_hi_val = np.array(sol.value(s_delta_hi)).flatten()
                s_fx_lo_val = np.array(sol.value(s_fx_lo)).flatten()
                s_fx_hi_val = np.array(sol.value(s_fx_hi)).flatten()
                s_periodic_val = np.array(sol.value(s_periodic)).flatten()
                track_slack_norm = float(np.sqrt(np.sum(s_track_lo_val**2) + np.sum(s_track_hi_val**2)))
                dpsi_slack_norm = float(np.sqrt(np.sum(s_dpsi_lo_val**2) + np.sum(s_dpsi_hi_val**2)))
                control_slack_norm = float(
                    np.sqrt(
                        np.sum(s_delta_lo_val**2) + np.sum(s_delta_hi_val**2)
                        + np.sum(s_fx_lo_val**2) + np.sum(s_fx_hi_val**2)
                    )
                )
                periodic_slack_norm = float(np.sqrt(np.sum(s_periodic_val**2)))
                if (
                    track_slack_norm > 1e-4
                    or dpsi_slack_norm > 1e-4
                    or control_slack_norm > 1e-4
                    or periodic_slack_norm > 1e-4
                ):
                    print(
                        "    [OSQP SLACKS] "
                        f"track={track_slack_norm:.4f}, "
                        f"dpsi={dpsi_slack_norm:.4f}, "
                        f"control={control_slack_norm:.4f}, "
                        f"periodic={periodic_slack_norm:.4f}"
                    )

            return X_new, U_new, cost_qp, V_norm_sq, True
        except RuntimeError as e:
            if self.params.verbose:
                print(f"    [SUBPROBLEM FAILURE] iter={iteration}, tr={tr_multiplier:.4f}, "
                      f"solver={self.params.qp_solver}")
                print(f"    [SUBPROBLEM FAILURE] exception:\n{str(e)}")
                try:
                    stats = opti.stats()
                    print("    [SUBPROBLEM FAILURE] solver stats:")
                    for k in ("return_status", "success", "iter_count", "unified_return_status"):
                        if k in stats:
                            print(f"      - {k}: {stats[k]}")
                except Exception as stats_e:
                    print(f"    [SUBPROBLEM FAILURE] solver stats unavailable: {stats_e}")
                # Summary of reference trajectory bounds
                ux_range = (float(X_ref[self.IDX_UX, :].min()), float(X_ref[self.IDX_UX, :].max()))
                e_range = (float(X_ref[self.IDX_E, :].min()), float(X_ref[self.IDX_E, :].max()))
                print(f"    [SUBPROBLEM FAILURE] X_ref: ux=[{ux_range[0]:.2f}, {ux_range[1]:.2f}], "
                      f"e=[{e_range[0]:.2f}, {e_range[1]:.2f}]")
                try:
                    print("    [SUBPROBLEM FAILURE] infeasibility report:")
                    opti.debug.show_infeasibilities()
                except Exception as infeas_e:
                    print(f"    [SUBPROBLEM FAILURE] infeasibility report unavailable: {infeas_e}")
            return X_ref.copy(), U_ref.copy(), float("inf"), float("inf"), False

    def solve(
        self,
        N: int,
        ds_m: float,
        X_init: Optional[np.ndarray] = None,
        U_init: Optional[np.ndarray] = None,
        ux_min: float = 1.0,
        ux_max: Optional[float] = None,
        convergent_lap: bool = True,
        verbose: Optional[bool] = None,
    ) -> SCPResult:
        t_start = time.time()
        verbose = self.params.verbose if verbose is None else verbose
        use_qp_debug = self.params.qp_solver in ("osqp", "qrqp")

        s_grid = np.linspace(0, N * ds_m, N + 1)
        k_psi_data = np.zeros(N + 1)
        theta_data = np.zeros(N + 1)
        phi_data = np.zeros(N + 1)
        track_hw = np.zeros(N + 1)

        for k in range(N + 1):
            k_psi_data[k], theta_data[k], phi_data[k] = self._get_road_geometry(s_grid[k])
            track_hw[k] = self._get_track_half_width(s_grid[k])

        if X_init is None or U_init is None:
            X_guess, U_guess = self._build_cold_start_guess(N, ds_m)
            X_ref = X_guess if X_init is None else X_init.copy()
            U_ref = U_guess if U_init is None else U_init.copy()
        else:
            X_ref = X_init.copy()
            U_ref = U_init.copy()

        tr_multiplier = self.params.tr_radius_init
        iteration_history = []
        tr_radius_history = []
        constraint_violation_history = []
        virtual_control_history = []
        subproblem_times = []

        merit_prev, lap_prev, defect_prev, _ = self._compute_nonlinear_merit(
            X_ref, U_ref, k_psi_data, theta_data, phi_data, ds_m
        )

        if verbose:
            print(f"SCP Solver: N={N}, ds={ds_m}m, track length={self.world.length_m:.1f}m")
            print(f"Initial: lap_time={lap_prev:.4f}s, defect={defect_prev:.6f}, merit={merit_prev:.4f}")
            print("-" * 70)

        if self.params.early_exit_on_feasible and defect_prev < self.params.constraint_tol:
            if verbose:
                print(f"Early exit: initial trajectory already feasible (defect={defect_prev:.2e})")
            return SCPResult(
                success=True,
                s_m=s_grid,
                X=X_ref,
                U=U_ref,
                cost=lap_prev,
                iterations=0,
                iteration_history=[lap_prev],
                tr_radius_history=[tr_multiplier],
                constraint_violation_history=[defect_prev],
                virtual_control_history=[0.0],
                solve_time=time.time() - t_start,
                subproblem_times=[],
                k_psi=k_psi_data,
                theta=theta_data,
                phi=phi_data,
                converged=True,
                feasible=True,
                termination_reason="Early exit: initial trajectory already feasible",
            )

        converged = False
        termination_reason = ""
        min_tr_reject_streak = 0
        tiny_step_streak = 0

        for it in range(self.params.max_iterations):
            if (time.time() - t_start) >= self.params.max_solve_time_s:
                termination_reason = f"Solve time limit reached ({self.params.max_solve_time_s:.1f}s)"
                break

            t_iter = time.time()

            phase_feas = defect_prev > self.params.defect_switch_tol

            A_list, B_list, c_list = self._linearize_around_trajectory(
                X_ref, U_ref, k_psi_data, theta_data, phi_data
            )

            X_new, U_new, _, V_norm_sq, qp_ok = self._solve_convex_subproblem(
                X_ref,
                U_ref,
                ds_m,
                A_list,
                B_list,
                c_list,
                track_hw,
                tr_multiplier,
                ux_min,
                ux_max,
                convergent_lap,
                phase_feas,
                iteration=it,
            )

            subproblem_times.append(time.time() - t_iter)

            if not qp_ok:
                if not use_qp_debug:
                    tr_multiplier = max(self.params.tr_radius_min, tr_multiplier * self.params.tr_shrink_factor)
                iteration_history.append(lap_prev)
                tr_radius_history.append(tr_multiplier)
                constraint_violation_history.append(defect_prev)
                virtual_control_history.append(float("inf"))
                if verbose:
                    print(f"  Iter {it}: subproblem failed, shrinking trust region")
                continue

            step_x_norm = float(np.linalg.norm(X_new - X_ref))
            step_u_norm = float(np.linalg.norm(U_new - U_ref))
            if verbose:
                print(f"    [STEP DIAG] ||dX||={step_x_norm:.3e}, ||dU||={step_u_norm:.3e}")

            if use_qp_debug and step_x_norm < 1e-7 and step_u_norm < 1e-7:
                tiny_step_streak += 1
            else:
                tiny_step_streak = 0

            if use_qp_debug and tiny_step_streak >= 3:
                U_ref = self._perturb_control_seed(U_ref)
                tiny_step_streak = 0
                if verbose:
                    print("    [DEBUG RESEED] tiny-step fixed point detected, perturbing control seed")
                iteration_history.append(lap_prev)
                tr_radius_history.append(tr_multiplier)
                constraint_violation_history.append(defect_prev)
                virtual_control_history.append(float(np.sqrt(V_norm_sq)))
                continue

            merit_new, lap_new, defect_new, _ = self._compute_nonlinear_merit(
                X_new, U_new, k_psi_data, theta_data, phi_data, ds_m
            )

            actual_decrease = merit_prev - merit_new
            if phase_feas:
                # In feasibility phase, prioritize true nonlinear defect reduction.
                accept = (defect_new < defect_prev - 1e-4) or (
                    defect_prev > 0 and defect_new / defect_prev < 0.999
                )
            else:
                accept = actual_decrease > 1e-8

            V_norm = float(np.sqrt(V_norm_sq))

            if accept:
                min_tr_reject_streak = 0
                X_ref = X_new
                U_ref = U_new
                merit_prev = merit_new
                lap_prev = lap_new
                defect_prev = defect_new

                if actual_decrease > 1e-4:
                    tr_multiplier = min(self.params.tr_radius_max, tr_multiplier * self.params.tr_expand_factor)
                accepted_txt = "accepted"
            else:
                if not use_qp_debug:
                    tr_multiplier = max(self.params.tr_radius_min, tr_multiplier * self.params.tr_shrink_factor)
                    if tr_multiplier <= self.params.tr_radius_min + 1e-12:
                        min_tr_reject_streak += 1
                accepted_txt = "rejected"

            iteration_history.append(lap_prev)
            tr_radius_history.append(tr_multiplier)
            constraint_violation_history.append(defect_prev)
            virtual_control_history.append(V_norm)

            if verbose:
                print(
                    f"  Iter {it}: t={lap_prev:.4f}s, defect={defect_prev:.6f}, "
                    f"||V||={V_norm:.2e}, tr={tr_multiplier:.3f} ({accepted_txt})"
                )

            if accept and len(iteration_history) >= 2:
                dcost = abs(iteration_history[-1] - iteration_history[-2])
                if defect_prev < self.params.constraint_tol:
                    converged = True
                    termination_reason = f"Feasibility reached: defect={defect_prev:.2e}"
                    break
                if (
                    dcost < self.params.convergence_tol
                    and defect_prev < self.params.constraint_tol
                    and V_norm < self.params.virtual_control_tol
                ):
                    converged = True
                    termination_reason = (
                        f"Converged: Δcost={dcost:.2e}, defect={defect_prev:.2e}, ||V||={V_norm:.2e}"
                    )
                    break

            if (not use_qp_debug) and tr_multiplier <= self.params.tr_radius_min and min_tr_reject_streak >= 6:
                termination_reason = "Trust region stuck at minimum with repeated rejections"
                break
        else:
            termination_reason = f"Max iterations ({self.params.max_iterations}) reached"

        solve_time = time.time() - t_start

        if verbose:
            print("-" * 70)
            print(f"Termination: {termination_reason}")
            print(
                f"Final: lap_time={lap_prev:.4f}s, defect={defect_prev:.6f}, "
                f"iterations={len(iteration_history)}, time={solve_time:.2f}s"
            )

        feasible = defect_prev < self.params.constraint_tol

        return SCPResult(
            success=converged,
            s_m=s_grid,
            X=X_ref,
            U=U_ref,
            cost=lap_prev,
            iterations=len(iteration_history),
            iteration_history=iteration_history,
            tr_radius_history=tr_radius_history,
            constraint_violation_history=constraint_violation_history,
            virtual_control_history=virtual_control_history,
            solve_time=solve_time,
            subproblem_times=subproblem_times,
            k_psi=k_psi_data,
            theta=theta_data,
            phi=phi_data,
            converged=converged,
            feasible=feasible,
            termination_reason=termination_reason,
        )

    def solve_with_warm_start(self, N: int, ds_m: float, warm_start_result, **kwargs) -> SCPResult:
        return self.solve(N=N, ds_m=ds_m, X_init=warm_start_result.X, U_init=warm_start_result.U, **kwargs)


def compare_warm_starts(
    solver: SCPSolver,
    N: int,
    ds_m: float,
    warm_starts: Dict[str, Tuple[np.ndarray, np.ndarray]],
    **solve_kwargs,
) -> Dict[str, SCPResult]:
    results = {}
    for name, (X_init, U_init) in warm_starts.items():
        print(f"\n=== Warm-start: {name} ===")
        results[name] = solver.solve(N, ds_m, X_init=X_init, U_init=U_init, **solve_kwargs)

    print("\n" + "=" * 70)
    print("WARM-START COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Name':<20} {'Iters':<8} {'Cost':<10} {'Defect':<12} {'Time':<10} {'Conv'}")
    print("-" * 70)
    for name, result in results.items():
        defect = result.constraint_violation_history[-1] if result.constraint_violation_history else 0.0
        print(f"{name:<20} {result.iterations:<8} {result.cost:<10.4f} {defect:<12.2e} {result.solve_time:<10.2f} {result.converged}")

    return results
