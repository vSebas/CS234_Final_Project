"""
Sequential Convex Programming (SCP) Solver for Trajectory Optimization

Implements an SCP algorithm that:
1. Linearizes nonlinear dynamics around a reference trajectory
2. Solves convex QP subproblems iteratively
3. Uses trust regions for convergence
4. Tracks iteration count (key metric for warm-start evaluation)

Reference: Mao et al., "Successive Convexification of Non-Convex Optimal Control Problems", 2016.
"""

import numpy as np
import casadi as ca
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time


@dataclass
class SCPParams:
    """SCP algorithm parameters."""
    # Trust region
    tr_radius_init: float = 2.0          # Initial trust region radius
    tr_radius_min: float = 0.01          # Minimum trust region radius
    tr_radius_max: float = 10.0          # Maximum trust region radius
    tr_shrink_factor: float = 0.5        # Factor to shrink trust region on rejection
    tr_expand_factor: float = 1.5        # Factor to expand trust region on good progress

    # Convergence criteria
    max_iterations: int = 50             # Maximum SCP iterations
    convergence_tol: float = 1e-4        # Convergence tolerance (change in cost)
    constraint_tol: float = 1e-3         # Constraint violation tolerance

    # Step acceptance
    rho_min: float = 0.1                 # Minimum ratio for step acceptance
    rho_good: float = 0.7                # Ratio threshold for trust region expansion

    # Penalty weights
    virtual_control_weight: float = 1e4  # Weight for virtual control (slack)
    soft_constraint_weight: float = 1e3  # Weight for soft constraint violations

    # QP solver options
    qp_solver: str = 'ipopt'             # 'ipopt' or 'osqp'
    verbose: bool = True


@dataclass
class SCPResult:
    """Container for SCP optimization results."""
    success: bool
    s_m: np.ndarray                # Arc length coordinates [N+1]
    X: np.ndarray                  # State trajectory [nx, N+1]
    U: np.ndarray                  # Control trajectory [nu, N+1]
    cost: float                    # Final cost (lap time)

    # SCP-specific metrics
    iterations: int                # Number of SCP iterations
    iteration_history: list = field(default_factory=list)  # Cost at each iteration
    tr_radius_history: list = field(default_factory=list)  # Trust region at each iteration
    constraint_violation_history: list = field(default_factory=list)

    solve_time: float = 0.0        # Total solve time [s]
    subproblem_times: list = field(default_factory=list)  # Time per subproblem

    # Road geometry at each point
    k_psi: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))

    converged: bool = False
    termination_reason: str = ""


class SCPSolver:
    """
    Sequential Convex Programming solver for vehicle trajectory optimization.

    Uses successive linearization and trust region methods to solve
    the nonlinear minimum-time trajectory optimization problem.
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

    def __init__(self, vehicle, world, params: Optional[SCPParams] = None):
        """
        Initialize SCP solver.

        Args:
            vehicle: SingleTrackModel instance
            world: World/track instance
            params: SCP algorithm parameters
        """
        self.vehicle = vehicle
        self.world = world
        self.params = params or SCPParams()

        # Problem dimensions
        self.nx = 8  # State dimension
        self.nu = 2  # Control dimension

        # Pre-build CasADi functions for linearization
        self._build_linearization_functions()

    def _build_linearization_functions(self):
        """Build CasADi functions for dynamics linearization."""
        # Symbolic variables
        x_sym = ca.SX.sym('x', self.nx)
        u_sym = ca.SX.sym('u', self.nu)
        k_psi_sym = ca.SX.sym('k_psi')
        theta_sym = ca.SX.sym('theta')
        phi_sym = ca.SX.sym('phi')

        # Get dynamics derivatives
        dx_dt, s_dot = self.vehicle.dynamics_dt_path_vec(
            x_sym, u_sym, k_psi_sym, theta_sym, phi_sym
        )

        # Spatial dynamics: dx/ds = (dx/dt) / (ds/dt)
        dx_ds = dx_dt / s_dot

        # Jacobians
        A = ca.jacobian(dx_ds, x_sym)  # d(dx_ds)/dx
        B = ca.jacobian(dx_ds, u_sym)  # d(dx_ds)/du

        # Build CasADi functions
        self.f_dynamics = ca.Function('f_dynamics',
            [x_sym, u_sym, k_psi_sym, theta_sym, phi_sym],
            [dx_ds],
            ['x', 'u', 'k_psi', 'theta', 'phi'],
            ['dx_ds'])

        self.f_s_dot = ca.Function('f_s_dot',
            [x_sym, u_sym, k_psi_sym, theta_sym, phi_sym],
            [s_dot],
            ['x', 'u', 'k_psi', 'theta', 'phi'],
            ['s_dot'])

        self.f_jacobians = ca.Function('f_jacobians',
            [x_sym, u_sym, k_psi_sym, theta_sym, phi_sym],
            [A, B, dx_ds],
            ['x', 'u', 'k_psi', 'theta', 'phi'],
            ['A', 'B', 'f'])

    def _get_road_geometry(self, s):
        """Get road geometry at arc length s."""
        s_mod = s % self.world.length_m

        k_psi = float(self.world.psi_s_radpm_LUT(s_mod))

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

    def _linearize_around_trajectory(
        self,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        s_grid: np.ndarray,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray
    ) -> Tuple[list, list, list]:
        """
        Linearize dynamics around reference trajectory.

        Returns: (A_list, B_list, c_list) where dynamics is:
            x_{k+1} = A_k @ (x_k - x_ref_k) + B_k @ (u_k - u_ref_k) + f_k + x_ref_k

        Or equivalently:
            x_{k+1} = A_k @ x_k + B_k @ u_k + c_k
            where c_k = f_k + x_ref_k - A_k @ x_ref_k - B_k @ u_ref_k
        """
        N = len(s_grid) - 1
        A_list = []
        B_list = []
        c_list = []

        for k in range(N):
            x_k = X_ref[:, k]
            u_k = U_ref[:, k]

            # Get Jacobians at this point
            A_k_out, B_k_out, f_k_out = self.f_jacobians(
                x_k, u_k, k_psi_data[k], theta_data[k], phi_data[k]
            )

            A_k = np.array(A_k_out)
            B_k = np.array(B_k_out)
            f_k = np.array(f_k_out).flatten()

            # Affine constant: c = f(x_ref, u_ref) - A @ x_ref - B @ u_ref
            # But for integration: x_{k+1} = x_k + ds * (A*x + B*u + c)
            # So c_k = f_k - A_k @ x_k - B_k @ u_k
            c_k = f_k - A_k @ x_k - B_k @ u_k

            A_list.append(A_k)
            B_list.append(B_k)
            c_list.append(c_k)

        return A_list, B_list, c_list

    def _compute_nonlinear_cost_and_dynamics(
        self,
        X: np.ndarray,
        U: np.ndarray,
        s_grid: np.ndarray,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray,
        ds_m: float
    ) -> Tuple[float, np.ndarray]:
        """
        Compute actual nonlinear cost and dynamics defects.

        Returns:
            cost: Nonlinear objective value
            defects: Dynamics defects [nx, N]
        """
        N = len(s_grid) - 1
        cost = 0.0
        defects = np.zeros((self.nx, N))

        for k in range(N):
            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            u_kp1 = U[:, k + 1]

            # s_dot at k
            ux_k = x_k[self.IDX_UX]
            uy_k = x_k[self.IDX_UY]
            dpsi_k = x_k[self.IDX_DPSI]
            e_k = x_k[self.IDX_E]

            s_dot_k = (ux_k * np.cos(dpsi_k) - uy_k * np.sin(dpsi_k)) / (1 - k_psi_data[k] * e_k)

            # Nonlinear cost: integral of dt = ds / s_dot
            if s_dot_k > 0.1:
                cost += ds_m / s_dot_k
            else:
                cost += ds_m / 0.1  # Penalty for very slow progress

            # Dynamics defects (trapezoidal)
            f_k = np.array(self.f_dynamics(x_k, u_k, k_psi_data[k], theta_data[k], phi_data[k])).flatten()
            f_kp1 = np.array(self.f_dynamics(x_kp1, u_kp1, k_psi_data[k+1], theta_data[k+1], phi_data[k+1])).flatten()

            x_kp1_predicted = x_k + ds_m / 2 * (f_k + f_kp1)
            defects[:, k] = x_kp1 - x_kp1_predicted

        return cost, defects

    def _solve_qp_subproblem(
        self,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        s_grid: np.ndarray,
        ds_m: float,
        A_list: list,
        B_list: list,
        c_list: list,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray,
        track_hw: np.ndarray,
        tr_radius: float,
        ux_min: float = 1.0,
        ux_max: Optional[float] = None,
        convergent_lap: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Solve convex QP subproblem.

        Minimizes linearized cost subject to:
        - Linearized dynamics
        - Trust region constraints
        - State/control bounds
        - Track boundaries

        Returns:
            X_new: New state trajectory
            U_new: New control trajectory
            cost_qp: QP objective value
            success: Whether QP solved successfully
        """
        N = len(s_grid) - 1
        p = self.vehicle.params

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.nx, N + 1)
        U = opti.variable(self.nu, N + 1)

        # Virtual control for constraint relaxation
        V = opti.variable(self.nx, N)  # Virtual control (slack)

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

        # === Linearized objective ===
        # Approximate minimum time: minimize sum of 1/s_dot * ds
        # Linearize around reference: 1/s_dot ≈ 1/s_dot_ref + d(1/s_dot)/dx * (x - x_ref)
        cost = 0
        for k in range(N):
            # s_dot at reference point
            ux_ref = X_ref[self.IDX_UX, k]
            uy_ref = X_ref[self.IDX_UY, k]
            dpsi_ref = X_ref[self.IDX_DPSI, k]
            e_ref = X_ref[self.IDX_E, k]

            denom_ref = 1 - k_psi_data[k] * e_ref
            s_dot_ref = (ux_ref * np.cos(dpsi_ref) - uy_ref * np.sin(dpsi_ref)) / denom_ref
            s_dot_ref = max(s_dot_ref, 0.5)  # Prevent division issues

            # Linearized time: just use current s_dot approximation
            # For stability, keep it simple: minimize actual s_dot inverse
            denom_k = 1 - k_psi_data[k] * e[k]
            s_dot_k = (ux[k] * ca.cos(dpsi[k]) - uy[k] * ca.sin(dpsi[k])) / denom_k

            cost += ds_m / s_dot_k

        # Virtual control penalty
        cost += self.params.virtual_control_weight * ca.sumsqr(V)

        opti.minimize(cost)

        # === Linearized dynamics constraints (trapezoidal) ===
        for k in range(N):
            A_k = A_list[k]
            B_k = B_list[k]
            c_k = c_list[k]

            # Linearized dynamics: dx_ds = A*x + B*u + c
            # Trapezoidal: x_{k+1} = x_k + ds/2 * (f_k + f_{k+1})
            # With linearization: f_k ≈ A_k*x_k + B_k*u_k + c_k

            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            u_kp1 = U[:, k + 1]

            # Get linearization at k+1 (using reference)
            if k + 1 < N:
                A_kp1 = A_list[k + 1]
                B_kp1 = B_list[k + 1]
                c_kp1 = c_list[k + 1]
            else:
                # Last point - use same linearization
                A_kp1 = A_k
                B_kp1 = B_k
                c_kp1 = c_k

            f_k = A_k @ x_k + B_k @ u_k + c_k
            f_kp1 = A_kp1 @ x_kp1 + B_kp1 @ u_kp1 + c_kp1

            # Trapezoidal with virtual control
            opti.subject_to(x_kp1 == x_k + ds_m / 2 * (f_k + f_kp1) + V[:, k])

        # === Trust region constraints ===
        for k in range(N + 1):
            for i in range(self.nx):
                opti.subject_to(X[i, k] >= X_ref[i, k] - tr_radius)
                opti.subject_to(X[i, k] <= X_ref[i, k] + tr_radius)
            for j in range(self.nu):
                opti.subject_to(U[j, k] >= U_ref[j, k] - tr_radius)
                opti.subject_to(U[j, k] <= U_ref[j, k] + tr_radius)

        # === State constraints ===
        for k in range(N + 1):
            opti.subject_to(ux[k] >= ux_min)
            if ux_max is not None:
                opti.subject_to(ux[k] <= ux_max)

            # Track bounds
            buffer = 0.5
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

        # === Boundary conditions ===
        opti.subject_to(t[0] == 0)

        if convergent_lap:
            opti.subject_to(ux[0] == ux[N])
            opti.subject_to(uy[0] == uy[N])
            opti.subject_to(r[0] == r[N])
            opti.subject_to(dfz_long[0] == dfz_long[N])
            opti.subject_to(dfz_lat[0] == dfz_lat[N])
            opti.subject_to(e[0] == e[N])
            opti.subject_to(dpsi[0] == dpsi[N])
            opti.subject_to(delta[0] == delta[N])
            opti.subject_to(fx[0] == fx[N])

        # === Initial guess (warm-start with reference) ===
        opti.set_initial(X, X_ref)
        opti.set_initial(U, U_ref)
        opti.set_initial(V, 0)

        # === Solver options ===
        if self.params.qp_solver == 'osqp':
            opts = {'qpsol': 'osqp', 'print_time': False}
            opti.solver('sqpmethod', opts)
        else:
            opts = {
                'ipopt.print_level': 0,
                'print_time': False,
                'ipopt.max_iter': 200,
                'ipopt.tol': 1e-6,
            }
            opti.solver('ipopt', opts)

        # === Solve ===
        try:
            sol = opti.solve()
            X_new = sol.value(X)
            U_new = sol.value(U)
            cost_qp = sol.value(cost)
            success = True
        except RuntimeError:
            X_new = opti.debug.value(X)
            U_new = opti.debug.value(U)
            cost_qp = float('inf')
            success = False

        return X_new, U_new, cost_qp, success

    def solve(
        self,
        N: int,
        ds_m: float,
        X_init: Optional[np.ndarray] = None,
        U_init: Optional[np.ndarray] = None,
        ux_min: float = 1.0,
        ux_max: Optional[float] = None,
        convergent_lap: bool = True,
        verbose: Optional[bool] = None
    ) -> SCPResult:
        """
        Solve trajectory optimization using SCP.

        Args:
            N: Number of discretization steps
            ds_m: Step size in arc length [m]
            X_init: Initial state trajectory [nx, N+1] (warm-start)
            U_init: Initial control trajectory [nu, N+1] (warm-start)
            ux_min: Minimum forward speed [m/s]
            ux_max: Maximum speed limit [m/s]
            convergent_lap: Whether to enforce periodic boundary conditions
            verbose: Print progress (overrides params.verbose)

        Returns:
            SCPResult with optimized trajectory and metrics
        """
        t_start = time.time()
        verbose = verbose if verbose is not None else self.params.verbose

        # Arc length grid
        s_grid = np.linspace(0, N * ds_m, N + 1)

        # Precompute road geometry
        k_psi_data = np.zeros(N + 1)
        theta_data = np.zeros(N + 1)
        phi_data = np.zeros(N + 1)
        track_hw = np.zeros(N + 1)

        for k in range(N + 1):
            k_psi_data[k], theta_data[k], phi_data[k] = self._get_road_geometry(s_grid[k])
            track_hw[k] = self._get_track_half_width(s_grid[k])

        # === Initialize reference trajectory ===
        if X_init is None:
            # Simple straight-line initialization
            ux_init = 10.0
            X_ref = np.zeros((self.nx, N + 1))
            X_ref[self.IDX_UX, :] = ux_init
            X_ref[self.IDX_T, :] = np.cumsum(np.ones(N + 1) * ds_m / ux_init) - ds_m / ux_init
        else:
            X_ref = X_init.copy()

        if U_init is None:
            U_ref = np.zeros((self.nu, N + 1))
            U_ref[1, :] = 0.5  # Small positive force
        else:
            U_ref = U_init.copy()

        # === SCP iteration ===
        tr_radius = self.params.tr_radius_init

        iteration_history = []
        tr_radius_history = []
        constraint_violation_history = []
        subproblem_times = []

        cost_prev, defects_prev = self._compute_nonlinear_cost_and_dynamics(
            X_ref, U_ref, s_grid, k_psi_data, theta_data, phi_data, ds_m
        )
        max_defect_prev = np.max(np.abs(defects_prev))

        if verbose:
            print(f"SCP Solver: N={N}, ds={ds_m}m, track length={self.world.length_m:.1f}m")
            print(f"Initial cost: {cost_prev:.4f}, max defect: {max_defect_prev:.6f}")
            print("-" * 60)

        converged = False
        termination_reason = ""

        for iteration in range(self.params.max_iterations):
            t_iter_start = time.time()

            # Linearize around current reference
            A_list, B_list, c_list = self._linearize_around_trajectory(
                X_ref, U_ref, s_grid, k_psi_data, theta_data, phi_data
            )

            # Solve QP subproblem
            X_new, U_new, cost_qp, qp_success = self._solve_qp_subproblem(
                X_ref, U_ref, s_grid, ds_m,
                A_list, B_list, c_list,
                k_psi_data, theta_data, phi_data, track_hw,
                tr_radius, ux_min, ux_max, convergent_lap
            )

            t_iter = time.time() - t_iter_start
            subproblem_times.append(t_iter)

            if not qp_success:
                if verbose:
                    print(f"  Iter {iteration}: QP failed, shrinking trust region")
                tr_radius *= self.params.tr_shrink_factor
                tr_radius = max(tr_radius, self.params.tr_radius_min)
                tr_radius_history.append(tr_radius)
                iteration_history.append(cost_prev)
                continue

            # Evaluate nonlinear cost at new point
            cost_new, defects_new = self._compute_nonlinear_cost_and_dynamics(
                X_new, U_new, s_grid, k_psi_data, theta_data, phi_data, ds_m
            )
            max_defect_new = np.max(np.abs(defects_new))

            # Compute improvement ratio
            predicted_decrease = cost_prev - cost_qp
            actual_decrease = cost_prev - cost_new

            if abs(predicted_decrease) < 1e-10:
                rho = 1.0
            else:
                rho = actual_decrease / predicted_decrease

            # Accept or reject step
            if rho >= self.params.rho_min and max_defect_new < max_defect_prev * 2:
                # Accept step
                X_ref = X_new
                U_ref = U_new
                cost_prev = cost_new
                max_defect_prev = max_defect_new

                # Adjust trust region
                if rho >= self.params.rho_good:
                    tr_radius *= self.params.tr_expand_factor
                    tr_radius = min(tr_radius, self.params.tr_radius_max)

                accepted = True
            else:
                # Reject step, shrink trust region
                tr_radius *= self.params.tr_shrink_factor
                tr_radius = max(tr_radius, self.params.tr_radius_min)
                accepted = False

            # Record history
            iteration_history.append(cost_prev)
            tr_radius_history.append(tr_radius)
            constraint_violation_history.append(max_defect_prev)

            if verbose:
                status = "accepted" if accepted else "rejected"
                print(f"  Iter {iteration}: cost={cost_prev:.4f}, defect={max_defect_prev:.6f}, "
                      f"tr={tr_radius:.3f}, rho={rho:.3f} ({status}), time={t_iter:.2f}s")

            # Check convergence
            if len(iteration_history) >= 2:
                cost_change = abs(iteration_history[-1] - iteration_history[-2])
                if cost_change < self.params.convergence_tol and max_defect_prev < self.params.constraint_tol:
                    converged = True
                    termination_reason = f"Converged: cost change {cost_change:.2e} < tol, defect {max_defect_prev:.2e}"
                    break

            # Check trust region collapse
            if tr_radius <= self.params.tr_radius_min:
                termination_reason = "Trust region collapsed to minimum"
                break
        else:
            termination_reason = f"Max iterations ({self.params.max_iterations}) reached"

        solve_time = time.time() - t_start

        if verbose:
            print("-" * 60)
            print(f"Termination: {termination_reason}")
            print(f"Final cost: {cost_prev:.4f}s, Iterations: {len(iteration_history)}, Time: {solve_time:.2f}s")

        return SCPResult(
            success=converged or (max_defect_prev < self.params.constraint_tol),
            s_m=s_grid,
            X=X_ref,
            U=U_ref,
            cost=cost_prev,
            iterations=len(iteration_history),
            iteration_history=iteration_history,
            tr_radius_history=tr_radius_history,
            constraint_violation_history=constraint_violation_history,
            solve_time=solve_time,
            subproblem_times=subproblem_times,
            k_psi=k_psi_data,
            theta=theta_data,
            phi=phi_data,
            converged=converged,
            termination_reason=termination_reason
        )

    def solve_with_warm_start(
        self,
        N: int,
        ds_m: float,
        warm_start_result,  # Result from another solver (e.g., direct collocation)
        **kwargs
    ) -> SCPResult:
        """
        Solve using warm-start from another solver's result.

        This is the key method for evaluating Decision Transformer warm-starts.

        Args:
            N: Number of discretization steps
            ds_m: Step size
            warm_start_result: Result object with X and U attributes
            **kwargs: Additional arguments passed to solve()

        Returns:
            SCPResult with warm-started optimization
        """
        return self.solve(
            N=N,
            ds_m=ds_m,
            X_init=warm_start_result.X,
            U_init=warm_start_result.U,
            **kwargs
        )


def compare_warm_starts(
    solver: SCPSolver,
    N: int,
    ds_m: float,
    warm_starts: Dict[str, Tuple[np.ndarray, np.ndarray]],
    **solve_kwargs
) -> Dict[str, SCPResult]:
    """
    Compare different warm-start strategies.

    Args:
        solver: SCPSolver instance
        N: Discretization steps
        ds_m: Step size
        warm_starts: Dict mapping name -> (X_init, U_init)
        **solve_kwargs: Additional arguments for solve()

    Returns:
        Dict mapping name -> SCPResult
    """
    results = {}

    for name, (X_init, U_init) in warm_starts.items():
        print(f"\n=== Warm-start: {name} ===")
        result = solver.solve(N, ds_m, X_init=X_init, U_init=U_init, **solve_kwargs)
        results[name] = result

    # Summary
    print("\n" + "=" * 60)
    print("WARM-START COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Name':<20} {'Iterations':<12} {'Cost':<12} {'Time (s)':<12} {'Converged'}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<20} {result.iterations:<12} {result.cost:<12.4f} {result.solve_time:<12.2f} {result.converged}")

    return results
