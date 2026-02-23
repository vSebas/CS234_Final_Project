"""
Sequential Convex Programming (SCP) Solver for Trajectory Optimization

Implements a proper SCP algorithm with:
1. Convex QP subproblems (linear objective + quadratic virtual control penalty)
2. Consistent merit function for predicted/actual decrease comparison
3. Scaled trust regions for mixed-unit state vectors
4. Virtual control for constraint relaxation
5. Fast convex QP solvers (OSQP, qrqp) for subproblems

QP Solver options:
- 'osqp': OSQP via CasADi (recommended, requires: pip install osqp)
- 'qrqp': CasADi's built-in QP solver (no extra dependencies)
- 'ipopt': General NLP solver (works but overkill for QPs)

Reference: Mao et al., "Successive Convexification of Non-Convex Optimal Control Problems", 2016.
"""

import numpy as np
import casadi as ca
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import time


def _check_qp_solver_available(solver_name: str) -> bool:
    """Check if a QP solver is available in CasADi."""
    try:
        if solver_name == 'osqp':
            # Try to create a simple QP with OSQP
            x = ca.SX.sym('x')
            qp = {'x': x, 'f': x**2}
            ca.qpsol('test', 'osqp', qp, {'osqp': {'verbose': False}})
            return True
        elif solver_name == 'qrqp':
            x = ca.SX.sym('x')
            qp = {'x': x, 'f': x**2}
            ca.qpsol('test', 'qrqp', qp, {'print_iter': False})
            return True
        else:
            return True  # Assume IPOPT is always available
    except Exception:
        return False


def _get_best_qp_solver() -> str:
    """Get the best available QP solver."""
    for solver in ['osqp', 'qrqp', 'ipopt']:
        if _check_qp_solver_available(solver):
            return solver
    return 'ipopt'


@dataclass
class SCPParams:
    """SCP algorithm parameters."""
    # Trust region (now per-dimension scales)
    tr_radius_init: float = 1.0           # Initial trust region multiplier
    tr_radius_min: float = 0.01           # Minimum trust region multiplier
    tr_radius_max: float = 10.0           # Maximum trust region multiplier
    tr_shrink_factor: float = 0.5         # Factor to shrink trust region on rejection
    tr_expand_factor: float = 1.5         # Factor to expand trust region on good progress

    # Convergence criteria
    max_iterations: int = 50              # Maximum SCP iterations
    convergence_tol: float = 1e-4         # Convergence tolerance (change in merit)
    constraint_tol: float = 1e-3          # Constraint violation tolerance
    virtual_control_tol: float = 1e-4     # Virtual control magnitude tolerance

    # Step acceptance
    rho_min: float = 0.1                  # Minimum ratio for step acceptance
    rho_good: float = 0.7                 # Ratio threshold for trust region expansion

    # Penalty weights (MUST be equal for consistent ρ computation)
    virtual_control_weight: float = 1e4   # Weight for virtual control (slack)
    defect_penalty_weight: float = 1e4    # Weight for defects in merit function (must match virtual_control_weight)

    # QP solver options
    # Note: IPOPT is most robust. OSQP/qrqp are faster but may fail on ill-conditioned problems.
    qp_solver: str = 'ipopt'              # 'ipopt' (robust), 'osqp', or 'qrqp'
    verbose: bool = True

    # Early exit behavior
    # If True: return immediately when warm-start is already feasible (iterations=0)
    # If False: always run at least one SCP iteration (for algorithm validation)
    early_exit_on_feasible: bool = True


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
    iteration_history: list = field(default_factory=list)  # Merit at each iteration
    tr_radius_history: list = field(default_factory=list)  # Trust region at each iteration
    constraint_violation_history: list = field(default_factory=list)
    virtual_control_history: list = field(default_factory=list)  # ||V|| at each iteration

    solve_time: float = 0.0        # Total solve time [s]
    subproblem_times: list = field(default_factory=list)  # Time per subproblem

    # Road geometry at each point
    k_psi: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))

    converged: bool = False
    feasible: bool = False             # Whether defect < constraint_tol (separate from converged)
    termination_reason: str = ""


class SCPSolver:
    """
    Sequential Convex Programming solver for vehicle trajectory optimization.

    Key correctness features:
    1. Convex subproblem: minimize t[N] + w*||V||^2 (linear + quadratic)
    2. Consistent merit function Φ = t[N] + λ*||defects||^2 for ρ computation
    3. Scaled trust regions: per-dimension Δ based on typical magnitudes
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

    # Typical scales for trust region normalization
    # These represent expected magnitudes/variations of each variable
    STATE_SCALES = np.array([
        10.0,    # ux [m/s] - typical speed ~10-30 m/s
        2.0,     # uy [m/s] - lateral velocity ~0-5 m/s
        0.5,     # r [rad/s] - yaw rate ~0-1 rad/s
        1.0,     # dfz_long [kN] - weight transfer ~0-2 kN
        1.0,     # dfz_lat [kN] - weight transfer ~0-2 kN
        1.0,     # t [s] - time scale (per segment)
        2.0,     # e [m] - lateral deviation ~0-5 m
        0.3,     # dpsi [rad] - heading error ~0-0.5 rad
    ])

    CONTROL_SCALES = np.array([
        0.3,     # delta [rad] - steering ~0-0.5 rad
        5.0,     # fx [kN] - force ~-10 to 10 kN
    ])

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

        # Check QP solver availability
        requested_solver = self.params.qp_solver
        if not _check_qp_solver_available(requested_solver):
            fallback = _get_best_qp_solver()
            print(f"Warning: QP solver '{requested_solver}' not available, using '{fallback}'")
            self.params.qp_solver = fallback

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

        Returns: (A_list, B_list, c_list) where linearized dynamics is:
            dx/ds ≈ A_k @ x + B_k @ u + c_k
        """
        N = len(s_grid) - 1
        A_list = []
        B_list = []
        c_list = []

        for k in range(N + 1):  # Linearize at all points including last
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
            c_k = f_k - A_k @ x_k - B_k @ u_k

            A_list.append(A_k)
            B_list.append(B_k)
            c_list.append(c_k)

        return A_list, B_list, c_list

    def _compute_nonlinear_defects(
        self,
        X: np.ndarray,
        U: np.ndarray,
        s_grid: np.ndarray,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray,
        ds_m: float
    ) -> np.ndarray:
        """
        Compute nonlinear dynamics defects using trapezoidal integration.

        Returns:
            defects: Dynamics defects [nx, N]
        """
        N = len(s_grid) - 1
        defects = np.zeros((self.nx, N))

        for k in range(N):
            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            u_kp1 = U[:, k + 1]

            # Nonlinear dynamics at k and k+1
            f_k = np.array(self.f_dynamics(
                x_k, u_k, k_psi_data[k], theta_data[k], phi_data[k]
            )).flatten()
            f_kp1 = np.array(self.f_dynamics(
                x_kp1, u_kp1, k_psi_data[k+1], theta_data[k+1], phi_data[k+1]
            )).flatten()

            # Trapezoidal integration
            x_kp1_predicted = x_k + ds_m / 2 * (f_k + f_kp1)
            defects[:, k] = x_kp1 - x_kp1_predicted

        return defects

    def _compute_nonlinear_merit(
        self,
        X: np.ndarray,
        U: np.ndarray,
        s_grid: np.ndarray,
        k_psi_data: np.ndarray,
        theta_data: np.ndarray,
        phi_data: np.ndarray,
        ds_m: float
    ) -> Tuple[float, float, float]:
        """
        Compute NONLINEAR merit function: Φ_nl = t[N] + λ * ||defects_nl||^2

        This is evaluated on the TRUE nonlinear dynamics and contains NO virtual
        control V. V is an artifact of the convex relaxation and does not exist
        in the original problem.

        Used for computing ACTUAL decrease in trust-region ρ.

        Args:
            X: State trajectory
            U: Control trajectory
            s_grid: Arc length grid
            k_psi_data, theta_data, phi_data: Road geometry
            ds_m: Step size

        Returns:
            merit: Nonlinear merit function value (no V!)
            lap_time: t[N] component
            defect_norm: ||defects_nl||_2
        """
        # Lap time (primary objective)
        lap_time = X[self.IDX_T, -1]

        # Compute nonlinear defects
        defects = self._compute_nonlinear_defects(
            X, U, s_grid, k_psi_data, theta_data, phi_data, ds_m
        )
        defect_norm_sq = np.sum(defects**2)
        defect_norm = np.sqrt(defect_norm_sq)

        # Nonlinear merit function (NO V term!)
        merit = lap_time + self.params.defect_penalty_weight * defect_norm_sq

        return merit, lap_time, defect_norm

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
        tr_multiplier: float,
        ux_min: float = 1.0,
        ux_max: Optional[float] = None,
        convergent_lap: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """
        Solve CONVEX QP subproblem.

        Objective: minimize t[N] + w * ||V||^2
        This is convex (linear + quadratic).

        Returns:
            X_new: New state trajectory
            U_new: New control trajectory
            cost_qp: QP objective value (for merit comparison)
            V_norm_sq: ||V||^2 from solution
            success: Whether QP solved successfully
        """
        N = len(s_grid) - 1
        p = self.vehicle.params

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.nx, N + 1)
        U = opti.variable(self.nu, N + 1)

        # Virtual control for constraint relaxation
        V = opti.variable(self.nx, N)

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

        # === CONVEX OBJECTIVE ===
        # minimize t[N] + w * ||V||^2
        # This is linear in t[N] plus quadratic in V - genuinely convex!
        cost = t[N] + self.params.virtual_control_weight * ca.sumsqr(V)

        opti.minimize(cost)

        # === Linearized dynamics constraints (trapezoidal) ===
        for k in range(N):
            A_k = A_list[k]
            B_k = B_list[k]
            c_k = c_list[k]

            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            u_kp1 = U[:, k + 1]

            # Linearization at k+1
            A_kp1 = A_list[k + 1]
            B_kp1 = B_list[k + 1]
            c_kp1 = c_list[k + 1]

            # Linearized dx/ds at k and k+1
            f_k = A_k @ x_k + B_k @ u_k + c_k
            f_kp1 = A_kp1 @ x_kp1 + B_kp1 @ u_kp1 + c_kp1

            # Trapezoidal collocation with virtual control
            opti.subject_to(x_kp1 == x_k + ds_m / 2 * (f_k + f_kp1) + V[:, k])

        # === SCALED Trust region constraints ===
        # Per-dimension trust region: |x_i - x_ref_i| <= Δ * scale_i
        for k in range(N + 1):
            for i in range(self.nx):
                tr_i = tr_multiplier * self.STATE_SCALES[i]
                opti.subject_to(X[i, k] >= X_ref[i, k] - tr_i)
                opti.subject_to(X[i, k] <= X_ref[i, k] + tr_i)
            for j in range(self.nu):
                tr_j = tr_multiplier * self.CONTROL_SCALES[j]
                opti.subject_to(U[j, k] >= U_ref[j, k] - tr_j)
                opti.subject_to(U[j, k] <= U_ref[j, k] + tr_j)

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

            # Time must be non-decreasing (physical constraint)
            if k > 0:
                opti.subject_to(t[k] >= t[k-1])

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

        # === Select QP solver ===
        if self.params.qp_solver == 'osqp':
            # Use OSQP via CasADi's sqpmethod
            # OSQP options must be nested inside 'osqp' dict
            opts = {
                'print_time': False,
                'qpsol': 'osqp',
                'qpsol_options': {
                    'osqp': {
                        'verbose': False,
                        'eps_abs': 1e-5,
                        'eps_rel': 1e-5,
                        'max_iter': 4000,
                        'polish': True,
                    },
                },
                'print_iteration': False,
                'print_header': False,
                'max_iter': 20,  # Allow multiple SQP iterations for robustness
                'convexify_strategy': 'regularize',
            }
            opti.solver('sqpmethod', opts)
        elif self.params.qp_solver == 'qrqp':
            # Use CasADi's built-in QP solver
            opts = {
                'print_time': False,
                'qpsol': 'qrqp',
                'qpsol_options': {'print_iter': False, 'print_header': False},
                'print_iteration': False,
                'print_header': False,
                'max_iter': 20,  # Allow multiple SQP iterations for robustness
            }
            opti.solver('sqpmethod', opts)
        else:
            # Fallback to IPOPT (works but overkill for QP)
            opts = {
                'ipopt.print_level': 0,
                'print_time': False,
                'ipopt.max_iter': 300,
                'ipopt.tol': 1e-6,
                'ipopt.warm_start_init_point': 'yes',
            }
            opti.solver('ipopt', opts)

        # === Solve ===
        try:
            sol = opti.solve()
            X_new = sol.value(X)
            U_new = sol.value(U)
            V_sol = sol.value(V)
            V_norm_sq = float(np.sum(V_sol**2))
            cost_qp = sol.value(cost)
            success = True
        except RuntimeError:
            # Solver failed - return reference trajectory
            X_new = X_ref.copy()
            U_new = U_ref.copy()
            V_norm_sq = float('inf')
            cost_qp = float('inf')
            success = False

        return X_new, U_new, cost_qp, V_norm_sq, success

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
        tr_multiplier = self.params.tr_radius_init

        iteration_history = []
        tr_radius_history = []
        constraint_violation_history = []
        virtual_control_history = []
        subproblem_times = []

        # Initial nonlinear merit (no V - V doesn't exist in the original problem)
        merit_prev, lap_time_prev, defect_norm_prev = self._compute_nonlinear_merit(
            X_ref, U_ref, s_grid, k_psi_data, theta_data, phi_data, ds_m
        )

        if verbose:
            print(f"SCP Solver: N={N}, ds={ds_m}m, track length={self.world.length_m:.1f}m")
            print(f"Initial: lap_time={lap_time_prev:.4f}s, defect={defect_norm_prev:.6f}, merit={merit_prev:.4f}")
            print("-" * 70)

        # === Early convergence check ===
        # If warm-start already satisfies feasibility and early_exit is enabled, return immediately
        # Set early_exit_on_feasible=False for algorithm validation (forces at least one SCP iteration)
        if self.params.early_exit_on_feasible and defect_norm_prev < self.params.constraint_tol:
            if verbose:
                print(f"Early exit: initial trajectory already feasible (defect={defect_norm_prev:.2e})")
            return SCPResult(
                success=True,
                s_m=s_grid,
                X=X_ref,
                U=U_ref,
                cost=lap_time_prev,
                iterations=0,
                iteration_history=[lap_time_prev],
                tr_radius_history=[self.params.tr_radius_init],
                constraint_violation_history=[defect_norm_prev],
                virtual_control_history=[0.0],
                solve_time=time.time() - t_start,
                subproblem_times=[],
                k_psi=k_psi_data,
                theta=theta_data,
                phi=phi_data,
                converged=True,
                feasible=True,
                termination_reason="Early exit: initial trajectory already feasible"
            )

        converged = False
        termination_reason = ""

        for iteration in range(self.params.max_iterations):
            t_iter_start = time.time()

            # Linearize around current reference
            A_list, B_list, c_list = self._linearize_around_trajectory(
                X_ref, U_ref, s_grid, k_psi_data, theta_data, phi_data
            )

            # Solve QP subproblem
            X_new, U_new, cost_qp, V_norm_sq, qp_success = self._solve_qp_subproblem(
                X_ref, U_ref, s_grid, ds_m,
                A_list, B_list, c_list,
                k_psi_data, theta_data, phi_data, track_hw,
                tr_multiplier, ux_min, ux_max, convergent_lap
            )

            t_iter = time.time() - t_iter_start
            subproblem_times.append(t_iter)

            if not qp_success:
                if verbose:
                    print(f"  Iter {iteration}: QP failed, shrinking trust region")
                tr_multiplier *= self.params.tr_shrink_factor
                tr_multiplier = max(tr_multiplier, self.params.tr_radius_min)
                tr_radius_history.append(tr_multiplier)
                iteration_history.append(merit_prev)
                virtual_control_history.append(float('inf'))
                constraint_violation_history.append(defect_norm_prev)
                continue

            # === CORRECT ρ COMPUTATION ===
            # Two separate merit functions (as per SCP theory):
            #
            # 1) Model merit (QP objective): Φ_model = t[N] + w_v * ||V||^2
            #    - Used for PREDICTED decrease
            #    - V is an artifact of the convex relaxation
            #
            # 2) Nonlinear merit: Φ_nl = t[N] + w_def * ||defects_nl||^2
            #    - Used for ACTUAL decrease
            #    - No V term! V doesn't exist in the original problem.
            #
            # This makes ρ meaningful: it measures how well the linearization
            # predicts the true nonlinear behavior.

            # Predicted decrease (from convex model)
            # Φ_model(X_ref, V=0) - Φ_model(X_new, V_new)
            # At reference: V=0, so Φ_model(ref) = t_ref[N]
            model_merit_ref = X_ref[self.IDX_T, -1]
            model_merit_new = cost_qp  # = t_new[N] + w_v * ||V||^2
            predicted_decrease = model_merit_ref - model_merit_new

            # Actual decrease (from true nonlinear dynamics, NO V!)
            # Φ_nl(X_ref) - Φ_nl(X_new)
            merit_new, lap_time_new, defect_norm_new = self._compute_nonlinear_merit(
                X_new, U_new, s_grid, k_psi_data, theta_data, phi_data, ds_m
            )
            actual_decrease = merit_prev - merit_new

            # Compute ρ = actual / predicted
            if abs(predicted_decrease) < 1e-10:
                rho = 1.0 if actual_decrease >= 0 else 0.0
            else:
                rho = actual_decrease / predicted_decrease

            V_norm = np.sqrt(V_norm_sq)

            # === Step acceptance based on merit function ===
            # Require strict non-increase in merit (no arbitrary tolerance)
            accept_step = (rho >= self.params.rho_min) and (merit_new <= merit_prev)

            if accept_step:
                # Accept step
                X_ref = X_new
                U_ref = U_new
                merit_prev = merit_new
                lap_time_prev = lap_time_new
                defect_norm_prev = defect_norm_new

                # Adjust trust region
                if rho >= self.params.rho_good:
                    tr_multiplier *= self.params.tr_expand_factor
                    tr_multiplier = min(tr_multiplier, self.params.tr_radius_max)

                accepted = True
            else:
                # Reject step, shrink trust region
                tr_multiplier *= self.params.tr_shrink_factor
                tr_multiplier = max(tr_multiplier, self.params.tr_radius_min)
                accepted = False

            # Record history
            iteration_history.append(lap_time_prev)  # Track lap time for user
            tr_radius_history.append(tr_multiplier)
            constraint_violation_history.append(defect_norm_prev)
            virtual_control_history.append(V_norm)

            if verbose:
                status = "accepted" if accepted else "rejected"
                print(f"  Iter {iteration}: t={lap_time_prev:.4f}s, defect={defect_norm_prev:.6f}, "
                      f"||V||={V_norm:.2e}, tr={tr_multiplier:.3f}, rho={rho:.3f} ({status})")

            # Check convergence
            if len(iteration_history) >= 2 and accepted:
                cost_change = abs(iteration_history[-1] - iteration_history[-2])
                if (cost_change < self.params.convergence_tol and
                    defect_norm_prev < self.params.constraint_tol and
                    V_norm < self.params.virtual_control_tol):
                    converged = True
                    termination_reason = (f"Converged: Δcost={cost_change:.2e}, "
                                        f"defect={defect_norm_prev:.2e}, ||V||={V_norm:.2e}")
                    break

            # Check trust region collapse
            if tr_multiplier <= self.params.tr_radius_min:
                termination_reason = "Trust region collapsed to minimum"
                break
        else:
            termination_reason = f"Max iterations ({self.params.max_iterations}) reached"

        solve_time = time.time() - t_start

        if verbose:
            print("-" * 70)
            print(f"Termination: {termination_reason}")
            print(f"Final: lap_time={lap_time_prev:.4f}s, defect={defect_norm_prev:.6f}, "
                  f"iterations={len(iteration_history)}, time={solve_time:.2f}s")

        # Separate feasibility from convergence for clear benchmarking
        feasible = defect_norm_prev < self.params.constraint_tol

        return SCPResult(
            success=converged,  # For SCP benchmarking, success = converged
            s_m=s_grid,
            X=X_ref,
            U=U_ref,
            cost=lap_time_prev,
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
            termination_reason=termination_reason
        )

    def solve_with_warm_start(
        self,
        N: int,
        ds_m: float,
        warm_start_result,
        **kwargs
    ) -> SCPResult:
        """
        Solve using warm-start from another solver's result.

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
    print("\n" + "=" * 70)
    print("WARM-START COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Name':<20} {'Iters':<8} {'Cost':<10} {'Defect':<12} {'Time':<10} {'Conv'}")
    print("-" * 70)
    for name, result in results.items():
        print(f"{name:<20} {result.iterations:<8} {result.cost:<10.4f} "
              f"{result.constraint_violation_history[-1] if result.constraint_violation_history else 0:<12.2e} "
              f"{result.solve_time:<10.2f} {result.converged}")

    return results
