"""
Trajectory Optimizer using Direct Collocation

Solves minimum-time trajectory optimization for the unified single-track vehicle model.
Uses CasADi's Opti interface for clean problem formulation.

State: [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi] (8 states)
Control: [delta, fx_kn] (2 inputs)
"""

import numpy as np
import casadi as ca
from typing import Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import time


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    success: bool
    s_m: np.ndarray           # Arc length coordinates [N+1]
    X: np.ndarray             # State trajectory [nx, N+1]
    U: np.ndarray             # Control trajectory [nu, N+1]
    cost: float               # Optimal cost (lap time)
    iterations: int           # Solver iterations
    solve_time: float         # Wall clock time [s]

    # Road geometry at each point
    k_psi: np.ndarray         # Curvature [N+1]
    theta: np.ndarray         # Grade [N+1]
    phi: np.ndarray           # Bank [N+1]
    max_obstacle_slack: float = 0.0
    min_obstacle_clearance: float = float("inf")


@dataclass
class ObstacleCircle:
    """Static circular obstacle for planning constraints."""
    radius_m: float
    margin_m: float = 0.0
    s_m: Optional[float] = None
    e_m: Optional[float] = None
    east_m: Optional[float] = None
    north_m: Optional[float] = None


class TrajectoryOptimizer:
    """
    Minimum-time trajectory optimizer for vehicle dynamics.

    Uses direct collocation with trapezoidal integration and
    IPOPT as the NLP solver.

    Compatible with unified SingleTrackModel.
    """

    # State indices (matching unified model)
    IDX_UX = 0
    IDX_UY = 1
    IDX_R = 2
    IDX_DFZ_LONG = 3
    IDX_DFZ_LAT = 4
    IDX_T = 5
    IDX_E = 6
    IDX_DPSI = 7

    def __init__(self, vehicle, world):
        """
        Initialize the optimizer.

        Args:
            vehicle: SingleTrackModel instance
            world: World/track instance
        """
        self.vehicle = vehicle
        self.world = world

        # Problem dimensions (unified model)
        self.nx = 8  # State dimension
        self.nu = 2  # Control dimension

    def _get_road_geometry(self, s):
        """
        Get road geometry at arc length s.

        Returns: (k_psi, theta, phi)
            k_psi: path curvature [1/m]
            theta: road grade [rad]
            phi: road bank [rad]
        """
        s_mod = s % self.world.length_m

        k_psi = float(self.world.psi_s_radpm_LUT(s_mod))

        # Check if world has grade/bank (some tracks are flat)
        if hasattr(self.world, 'grade_rad_LUT'):
            theta = float(self.world.grade_rad_LUT(s_mod))
        else:
            theta = 0.0

        if hasattr(self.world, 'bank_rad_LUT'):
            phi = float(self.world.bank_rad_LUT(s_mod))
        else:
            phi = 0.0

        return k_psi, theta, phi

    def _get_track_half_width(self, s):
        """Get track half-width at arc length s."""
        s_mod = s % self.world.length_m
        return float(self.world.track_width_m_LUT(s_mod)) / 2.0

    def _frenet_to_en(self, s_m: float, e_m: float) -> Tuple[float, float]:
        """Convert one (s, e) pair to EN coordinates using local centerline heading."""
        s_mod = s_m % self.world.length_m
        e_cl = float(self.world.posE_m_interp_fcn(s_mod))
        n_cl = float(self.world.posN_m_interp_fcn(s_mod))
        psi = float(self.world.psi_rad_interp_fcn(s_mod))
        east_m = e_cl - e_m * np.sin(psi)
        north_m = n_cl + e_m * np.cos(psi)
        return east_m, north_m

    def _nearest_centerline_s(self, east_m: float, north_m: float) -> float:
        """Approximate along-track coordinate of a world-frame position."""
        dx = self.world.data["posE_m"] - east_m
        dy = self.world.data["posN_m"] - north_m
        idx = int(np.argmin(dx * dx + dy * dy))
        return float(self.world.data["s_m"][idx])

    @staticmethod
    def _wrap_s_dist(s_a: float, s_b: float, length_m: float) -> float:
        """Shortest signed distance along a circular track."""
        d = (s_a - s_b) % length_m
        if d > 0.5 * length_m:
            d -= length_m
        return d

    def solve(
        self,
        N: int,
        ds_m: float,
        x0: Optional[np.ndarray] = None,
        X_init: Optional[np.ndarray] = None,
        U_init: Optional[np.ndarray] = None,
        weight_delta_dot: float = 5.0,
        weight_fx_dot: float = 5.0,
        ux_min: float = 1.0,
        ux_max: Optional[float] = None,
        stage: str = "time",
        obstacles: Optional[Sequence[Union[ObstacleCircle, Dict]]] = None,
        obstacle_window_m: float = 30.0,
        obstacle_slack_weight: float = 1e4,
        smoothness_weight: float = 1.0,
        time_weight_feas: float = 1e-2,
        convergent_lap: bool = True,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Solve the trajectory optimization problem.

        Args:
            N: Number of discretization steps
            ds_m: Step size in arc length [m]
            x0: Initial state (optional, for non-convergent lap)
            X_init: Initial state trajectory guess [nx, N+1] (optional)
            U_init: Initial control trajectory guess [nu, N+1] (optional)
            weight_delta_dot: Penalty on steering rate
            weight_fx_dot: Penalty on force rate
            ux_min: Minimum forward speed [m/s]
            ux_max: Maximum speed limit [m/s] (optional)
            stage: Optimization stage, one of {"feas", "time"}
            obstacles: Optional static circular obstacles
            obstacle_window_m: Along-track gating window for obstacle constraints [m]
            obstacle_slack_weight: Penalty on obstacle slack sum
            smoothness_weight: Weight for state/control smoothness terms
            time_weight_feas: Time objective weight during feasibility stage
            convergent_lap: Whether start and end should match (periodic)
            verbose: Print solver output

        Returns:
            OptimizationResult with optimal trajectory
        """
        t_start = time.time()
        p = self.vehicle.params
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.nx, N + 1)  # States at each node
        U = opti.variable(self.nu, N + 1)  # Controls at each node

        # State components
        ux = X[self.IDX_UX, :]
        uy = X[self.IDX_UY, :]
        r = X[self.IDX_R, :]
        dfz_long = X[self.IDX_DFZ_LONG, :]
        dfz_lat = X[self.IDX_DFZ_LAT, :]
        t = X[self.IDX_T, :]
        e = X[self.IDX_E, :]
        dpsi = X[self.IDX_DPSI, :]

        # Control components
        delta = U[0, :]
        fx = U[1, :]

        # Arc length grid
        s_grid = np.linspace(0, N * ds_m, N + 1)

        # Precompute road geometry at each node
        k_psi_data = np.zeros(N + 1)
        theta_data = np.zeros(N + 1)
        phi_data = np.zeros(N + 1)
        track_hw = np.zeros(N + 1)

        for k in range(N + 1):
            k_psi_data[k], theta_data[k], phi_data[k] = self._get_road_geometry(s_grid[k])
            track_hw[k] = self._get_track_half_width(s_grid[k])

        if stage not in {"feas", "time"}:
            raise ValueError(f"Unsupported stage={stage}. Use 'feas' or 'time'.")

        # Precompute centerline EN/heading at nodes for obstacle constraints.
        posE_cl_data = np.zeros(N + 1)
        posN_cl_data = np.zeros(N + 1)
        psi_cl_data = np.zeros(N + 1)
        for k in range(N + 1):
            s_mod = s_grid[k] % self.world.length_m
            posE_cl_data[k] = float(self.world.posE_m_interp_fcn(s_mod))
            posN_cl_data[k] = float(self.world.posN_m_interp_fcn(s_mod))
            psi_cl_data[k] = float(self.world.psi_rad_interp_fcn(s_mod))

        # Normalize obstacle input.
        obs_list: List[ObstacleCircle] = []
        for obs in obstacles or []:
            if isinstance(obs, ObstacleCircle):
                obs_list.append(obs)
            else:
                obs_list.append(ObstacleCircle(**obs))

        obs_east = np.zeros(len(obs_list))
        obs_north = np.zeros(len(obs_list))
        obs_r_tilde = np.zeros(len(obs_list))
        obs_s_center = np.zeros(len(obs_list))

        for j, obs in enumerate(obs_list):
            if obs.east_m is not None and obs.north_m is not None:
                east_m = float(obs.east_m)
                north_m = float(obs.north_m)
            elif obs.s_m is not None and obs.e_m is not None:
                east_m, north_m = self._frenet_to_en(float(obs.s_m), float(obs.e_m))
            else:
                raise ValueError(
                    "Each obstacle needs either (east_m, north_m) or (s_m, e_m)."
                )

            obs_east[j] = east_m
            obs_north[j] = north_m
            obs_r_tilde[j] = float(obs.radius_m + obs.margin_m)
            obs_s_center[j] = float(obs.s_m) if obs.s_m is not None else self._nearest_centerline_s(east_m, north_m)

        sigma_obs = None
        if len(obs_list) > 0:
            sigma_obs = opti.variable(N + 1, len(obs_list))
            opti.subject_to(ca.vec(sigma_obs) >= 0)

        # === Objective ===
        time_cost = 0
        reg_cost = 0
        for k in range(N):
            # s_dot at node k
            s_dot_k = (ux[k] * ca.cos(dpsi[k]) - uy[k] * ca.sin(dpsi[k])) / (1 - k_psi_data[k] * e[k])

            # Time = integral of ds / s_dot
            time_cost += ds_m / s_dot_k

            # Control rate penalties (regularization)
            if k < N:
                delta_dot = (U[0, k+1] - U[0, k]) / ds_m * s_dot_k
                fx_dot = (U[1, k+1] - U[1, k]) / ds_m * s_dot_k
                reg_cost += weight_delta_dot * delta_dot**2 * ds_m / s_dot_k
                reg_cost += weight_fx_dot * fx_dot**2 * ds_m / s_dot_k
                reg_cost += smoothness_weight * (e[k+1] - e[k])**2
                reg_cost += smoothness_weight * (dpsi[k+1] - dpsi[k])**2

        slack_cost = 0
        if sigma_obs is not None:
            slack_cost = obstacle_slack_weight * ca.sum1(ca.sum2(sigma_obs))

        if stage == "feas":
            cost = time_weight_feas * time_cost + reg_cost + slack_cost
        else:
            cost = time_cost + reg_cost + slack_cost

        opti.minimize(cost)

        # === Dynamics constraints (trapezoidal collocation) ===
        for k in range(N):
            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            u_kp1 = U[:, k + 1]

            # Road geometry at this segment
            k_psi_k = k_psi_data[k]
            k_psi_kp1 = k_psi_data[k + 1]
            theta_k = theta_data[k]
            theta_kp1 = theta_data[k + 1]
            phi_k = phi_data[k]
            phi_kp1 = phi_data[k + 1]

            # State derivatives using unified model
            dx_dt_k, s_dot_k = self.vehicle.dynamics_dt_path_vec(
                x_k, u_k, k_psi_k, theta_k, phi_k
            )
            dx_dt_kp1, s_dot_kp1 = self.vehicle.dynamics_dt_path_vec(
                x_kp1, u_kp1, k_psi_kp1, theta_kp1, phi_kp1
            )

            # Convert to spatial derivatives: dx/ds = (dx/dt) / (ds/dt)
            dx_ds_k = dx_dt_k / s_dot_k
            dx_ds_kp1 = dx_dt_kp1 / s_dot_kp1

            # Trapezoidal integration: x_{k+1} = x_k + ds/2 * (f_k + f_{k+1})
            opti.subject_to(x_kp1 == x_k + ds_m / 2 * (dx_ds_k + dx_ds_kp1))

        # === State constraints ===
        for k in range(N + 1):
            # Minimum forward speed (avoid singularity)
            opti.subject_to(ux[k] >= ux_min)

            # Maximum speed (if specified)
            if ux_max is not None:
                opti.subject_to(ux[k] <= ux_max)

            # Track bounds (with buffer)
            buffer = 0.5  # [m]
            opti.subject_to(e[k] >= -track_hw[k] + buffer)
            opti.subject_to(e[k] <= track_hw[k] - buffer)

            # Heading error bounds
            opti.subject_to(dpsi[k] >= -np.pi/3)
            opti.subject_to(dpsi[k] <= np.pi/3)

        # === Control constraints ===
        for k in range(N + 1):
            opti.subject_to(delta[k] >= -p.max_delta_rad)
            opti.subject_to(delta[k] <= p.max_delta_rad)
            opti.subject_to(fx[k] >= p.min_fx_kn)
            opti.subject_to(fx[k] <= p.max_fx_kn)

        # === Obstacle constraints (with slack + s-window gating) ===
        if sigma_obs is not None:
            for k in range(N + 1):
                posE_k = posE_cl_data[k] - e[k] * np.sin(psi_cl_data[k])
                posN_k = posN_cl_data[k] + e[k] * np.cos(psi_cl_data[k])
                for j in range(len(obs_list)):
                    if abs(self._wrap_s_dist(float(s_grid[k]), float(obs_s_center[j]), float(self.world.length_m))) <= obstacle_window_m:
                        dx = posE_k - obs_east[j]
                        dy = posN_k - obs_north[j]
                        g_kj = dx * dx + dy * dy - obs_r_tilde[j] ** 2
                        opti.subject_to(g_kj + sigma_obs[k, j] >= 0)

        # === Boundary conditions ===
        # Time starts at zero
        opti.subject_to(t[0] == 0)

        if convergent_lap:
            # Periodic boundary conditions (start = end for closed track)
            opti.subject_to(ux[0] == ux[N])
            opti.subject_to(uy[0] == uy[N])
            opti.subject_to(r[0] == r[N])
            opti.subject_to(dfz_long[0] == dfz_long[N])
            opti.subject_to(dfz_lat[0] == dfz_lat[N])
            opti.subject_to(e[0] == e[N])
            opti.subject_to(dpsi[0] == dpsi[N])
            opti.subject_to(delta[0] == delta[N])
            opti.subject_to(fx[0] == fx[N])
        else:
            # Fixed initial state
            if x0 is not None:
                for i in range(self.nx):
                    if x0[i] is not None:
                        opti.subject_to(X[i, 0] == x0[i])

        # === Initial guess ===
        if X_init is not None:
            if X_init.shape != (self.nx, N + 1):
                raise ValueError(f"X_init must have shape {(self.nx, N + 1)}, got {X_init.shape}")
            opti.set_initial(X, X_init)
        else:
            ux_init = 10.0  # [m/s]
            opti.set_initial(ux, ux_init)
            opti.set_initial(uy, 0)
            opti.set_initial(r, 0)
            opti.set_initial(dfz_long, 0)
            opti.set_initial(dfz_lat, 0)
            opti.set_initial(t, np.cumsum(np.ones(N + 1) * ds_m / ux_init) - ds_m / ux_init)
            opti.set_initial(e, 0)
            opti.set_initial(dpsi, 0)

        if U_init is not None:
            if U_init.shape != (self.nu, N + 1):
                raise ValueError(f"U_init must have shape {(self.nu, N + 1)}, got {U_init.shape}")
            opti.set_initial(U, U_init)
        else:
            opti.set_initial(delta, 0)
            opti.set_initial(fx, 0.5)

        if sigma_obs is not None:
            opti.set_initial(sigma_obs, 0.0)

        # === Solver options ===
        opts = {
            'ipopt.print_level': 5 if verbose else 0,
            'print_time': verbose,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
        }
        opti.solver('ipopt', opts)

        # === Solve ===
        try:
            sol = opti.solve()
            success = True
            X_opt = sol.value(X)
            U_opt = sol.value(U)
            cost_opt = sol.value(cost)
            iterations = sol.stats()['iter_count']
            sigma_opt = sol.value(sigma_obs) if sigma_obs is not None else None
        except RuntimeError as err:
            if verbose:
                print(f"Solver failed: {err}")
            success = False
            X_opt = opti.debug.value(X)
            U_opt = opti.debug.value(U)
            cost_opt = float('inf')
            iterations = -1
            sigma_opt = opti.debug.value(sigma_obs) if sigma_obs is not None else None

        solve_time = time.time() - t_start

        max_obstacle_slack = 0.0
        min_obstacle_clearance = float("inf")
        if len(obs_list) > 0:
            if sigma_opt is not None:
                max_obstacle_slack = float(np.max(sigma_opt))
            e_opt = X_opt[self.IDX_E, :]
            posE_opt = posE_cl_data - e_opt * np.sin(psi_cl_data)
            posN_opt = posN_cl_data + e_opt * np.cos(psi_cl_data)
            for k in range(N + 1):
                for j in range(len(obs_list)):
                    if abs(self._wrap_s_dist(float(s_grid[k]), float(obs_s_center[j]), float(self.world.length_m))) <= obstacle_window_m:
                        d = np.sqrt((posE_opt[k] - obs_east[j]) ** 2 + (posN_opt[k] - obs_north[j]) ** 2)
                        clearance = float(d - obs_r_tilde[j])
                        if clearance < min_obstacle_clearance:
                            min_obstacle_clearance = clearance

        return OptimizationResult(
            success=success,
            s_m=s_grid,
            X=X_opt,
            U=U_opt,
            cost=cost_opt,
            iterations=iterations,
            solve_time=solve_time,
            k_psi=k_psi_data,
            theta=theta_data,
            phi=phi_data,
            max_obstacle_slack=max_obstacle_slack,
            min_obstacle_clearance=min_obstacle_clearance
        )


def plan_trajectory(vehicle, world, params: Dict) -> Dict:
    """
    High-level interface for trajectory planning.

    Args:
        vehicle: SingleTrackModel instance
        world: Track/world model
        params: Parameters dict with keys:
            - N: discretization steps
            - DS_M: step size [m]
            - WEIGHT_DELTA_DOT, WEIGHT_FX_DOT: regularization weights
            - UX_MAX: speed limit [m/s] (optional)
            - CONVERGENT_LAP: periodic BC flag

    Returns:
        Dictionary with optimization results
    """
    optimizer = TrajectoryOptimizer(vehicle, world)

    print(f"Solving trajectory optimization (N={params['N']}, ds={params['DS_M']}m)...")

    result = optimizer.solve(
        N=params['N'],
        ds_m=params['DS_M'],
        stage=params.get('STAGE', 'time'),
        obstacles=params.get('OBSTACLES'),
        obstacle_window_m=params.get('OBSTACLE_WINDOW_M', 30.0),
        obstacle_slack_weight=params.get('OBSTACLE_SLACK_WEIGHT', 1e4),
        smoothness_weight=params.get('SMOOTHNESS_WEIGHT', 1.0),
        time_weight_feas=params.get('TIME_WEIGHT_FEAS', 1e-2),
        weight_delta_dot=params.get('WEIGHT_DELTA_DOT', 5.0),
        weight_fx_dot=params.get('WEIGHT_FX_DOT', 5.0),
        ux_max=params.get('UX_MAX'),
        convergent_lap=params.get('CONVERGENT_LAP', True),
    )

    print(f"  Lap time: {result.cost:.2f}s")
    print(f"  Success: {result.success}")
    print(f"  Solve time: {result.solve_time:.1f}s")
    print(f"  Iterations: {result.iterations}")

    return {
        'result': result,
        'params': params,
    }
