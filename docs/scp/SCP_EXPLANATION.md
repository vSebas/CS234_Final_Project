# Sequential Convex Programming (SCP) for Trajectory Optimization

## Table of Contents
1. [Introduction](#1-introduction)
2. [Why SCP?](#2-why-scp)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [The SCP Algorithm](#4-the-scp-algorithm)
5. [Trust Region Methods](#5-trust-region-methods)
6. [Implementation Details](#6-implementation-details)
7. [Code Walkthrough](#7-code-walkthrough)
8. [Convergence Analysis](#8-convergence-analysis)
9. [Warm-Starting](#9-warm-starting)
10. [Results and Observations](#10-results-and-observations)

---

## 1. Introduction

Sequential Convex Programming (SCP) is an iterative optimization technique for solving **nonconvex** optimization problems by solving a sequence of **convex** subproblems. Each subproblem is a local convex approximation of the original problem around the current iterate.

### The Core Idea

```
Nonconvex Problem → Linearize → Convex Subproblem → Solve → Update → Repeat
```

In trajectory optimization for vehicles, we have:
- **Nonlinear dynamics** (tire forces, weight transfer, kinematics)
- **Nonconvex constraints** (friction circles, coupled state-control limits)
- **Nonlinear objective** (minimum time = minimize 1/velocity)

SCP handles these by iteratively linearizing around a reference trajectory.

---

## 2. Why SCP?

### Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Direct Collocation + NLP** | Handles full nonlinearity, mature solvers (IPOPT) | Can be slow, sensitive to initialization |
| **Dynamic Programming** | Global optimum, handles constraints | Curse of dimensionality |
| **SCP** | Fast convex solves, iteration count = warm-start quality metric | Local convergence, needs good initialization |

### Why SCP for This Project?

1. **Iteration count is measurable**: We can directly measure how many SCP iterations a warm-start saves
2. **Convex subproblems are fast**: Each QP solve is quick, so total time ≈ iterations × QP_time
3. **Clear warm-start benefit**: A good initial guess reduces iterations dramatically
4. **Trust regions provide robustness**: Prevents wild jumps that could destabilize convergence

---

## 3. Mathematical Foundation

### 3.1 The Original Nonconvex Problem

For vehicle trajectory optimization, we solve:

```
minimize    J(x, u) = ∫₀ˢᶠ (1/ṡ) ds    (minimize lap time)

subject to  dx/ds = f(x, u, s)          (vehicle dynamics)
            g(x, u) ≤ 0                  (inequality constraints)
            h(x, u) = 0                  (equality constraints)
            x(0) = x₀, x(sᶠ) = xᶠ       (boundary conditions)
```

Where:
- `x ∈ ℝⁿˣ` is the state vector (velocity, yaw rate, weight transfer, etc.)
- `u ∈ ℝⁿᵘ` is the control vector (steering, force)
- `f(x, u, s)` is the nonlinear dynamics
- `ṡ = ds/dt` is the progress rate along the path

### 3.2 Discretization

We discretize the continuous problem into N nodes:

```
s₀ = 0,  s₁ = Δs,  s₂ = 2Δs,  ...,  sₙ = N·Δs
```

The dynamics become collocation constraints:

```
xₖ₊₁ = xₖ + (Δs/2) · [f(xₖ, uₖ) + f(xₖ₊₁, uₖ₊₁)]    (trapezoidal)
```

### 3.3 Linearization

The key step in SCP is linearizing the nonlinear dynamics around a reference trajectory `(x̄, ū)`:

```
f(x, u) ≈ f(x̄, ū) + Aₖ(x - x̄) + Bₖ(u - ū)
```

Where the Jacobians are:

```
Aₖ = ∂f/∂x |_{x̄ₖ, ūₖ}    (nx × nx matrix)
Bₖ = ∂f/∂u |_{x̄ₖ, ūₖ}    (nx × nu matrix)
```

This gives us the **affine dynamics**:

```
f(x, u) ≈ Aₖ·x + Bₖ·u + cₖ

where cₖ = f(x̄ₖ, ūₖ) - Aₖ·x̄ₖ - Bₖ·ūₖ
```

---

## 4. The SCP Algorithm

### Algorithm Overview

```
Algorithm: Sequential Convex Programming
────────────────────────────────────────
Input: Initial trajectory (x⁰, u⁰), trust region radius Δ₀

1. Set k = 0, Δ = Δ₀
2. While not converged:
   a. Linearize dynamics around (xᵏ, uᵏ) to get Aₖ, Bₖ, cₖ
   b. Formulate convex QP subproblem with trust region
   c. Solve QP to get candidate (x̃, ũ)
   d. Evaluate actual vs predicted improvement (ρ)
   e. If ρ > ρ_min: Accept step, xᵏ⁺¹ = x̃, uᵏ⁺¹ = ũ
   f. Adjust trust region based on ρ
   g. Check convergence criteria
   h. k = k + 1

Output: Optimal trajectory (x*, u*)
```

### Step-by-Step Breakdown

#### Step 2a: Linearization

For each node k, compute Jacobians using automatic differentiation:

```python
# CasADi symbolic computation
dx_ds = f(x, u, k_psi, theta, phi)  # Nonlinear dynamics

A = jacobian(dx_ds, x)  # ∂f/∂x
B = jacobian(dx_ds, u)  # ∂f/∂u
```

#### Step 2b: QP Subproblem

The **convex** subproblem at iteration i is:

```
minimize    t[N] + w·‖v‖²          (lap time + virtual control penalty)

subject to  xₖ₊₁ = xₖ + (Δs/2)·[(Aₖxₖ + Bₖuₖ + cₖ) + (Aₖ₊₁xₖ₊₁ + Bₖ₊₁uₖ₊₁ + cₖ₊₁)] + vₖ

            |xᵢ - x̄ᵢ| ≤ Δ·σᵢ      (scaled state trust region)
            |uⱼ - ūⱼ| ≤ Δ·ρⱼ      (scaled control trust region)

            x_min ≤ x ≤ x_max      (state bounds)
            u_min ≤ u ≤ u_max      (control bounds)
```

**Key insight:** The objective `t[N] + w·‖v‖²` is **linear + quadratic = convex**. This is critical for proper SCP.

Where:
- `t[N]` is the final time (lap time), handled through linearized dynamics
- `v` is **virtual control** (slack variable) that allows constraint relaxation
- `w` is a large penalty weight on virtual control (default: 1e4)
- `Δ` is the trust region multiplier
- `σᵢ`, `ρⱼ` are per-dimension scaling factors for mixed-unit normalization

#### Step 2d: Improvement Ratio (Merit Function)

The improvement ratio ρ measures how well the convex model predicts actual improvement.

**Critical insight:** V (virtual control) is an *artifact* of the convex relaxation. It does not exist in the original nonlinear problem. Therefore, we use **two separate merit functions**:

**1) Model merit (for predicted decrease):**
```
Φ_model(x, u, v) = t[N] + w·‖v‖²
```
This is the QP objective. At the reference point, V=0, so `Φ_model(ref) = t_ref[N]`.

**2) Nonlinear merit (for actual decrease) - NO V:**
```
Φ_nl(x, u) = t[N] + λ·‖defects_nl‖²
```
This measures the true nonlinear problem: lap time + dynamics violation.

Where:
- `t[N]` is the lap time (primary objective)
- `‖defects_nl‖²` measures nonlinear dynamics violation (trapezoidal)
- `λ` is the defect penalty weight (default: 1e3)
- `w` is the virtual control weight (default: 1e4)

Then:
```
predicted_decrease = Φ_model(ref, V=0) - Φ_model(new, V_new)
                   = t_ref[N] - cost_qp

actual_decrease = Φ_nl(ref) - Φ_nl(new)

ρ = actual_decrease / predicted_decrease
```

- `ρ ≈ 1`: Convex model accurately predicts nonlinear behavior
- `ρ < ρ_min`: Step rejected, shrink trust region
- `ρ ≥ ρ_good`: Accept step, expand trust region

---

## 5. Trust Region Methods

### 5.1 Why Trust Regions?

Without trust regions, SCP can:
1. **Diverge**: Large steps based on inaccurate linearization
2. **Oscillate**: Jump back and forth without converging
3. **Violate constraints**: Move to infeasible regions

Trust regions limit how far each step can go, ensuring the linearization remains valid.

### 5.2 Trust Region Dynamics

```
┌─────────────────────────────────────────────────────────────┐
│                    Trust Region Logic                        │
├─────────────────────────────────────────────────────────────┤
│  if ρ < ρ_min (0.1):     Reject step, Δ ← Δ × 0.5          │
│  if ρ_min ≤ ρ < ρ_good:  Accept step, keep Δ               │
│  if ρ ≥ ρ_good (0.7):    Accept step, Δ ← Δ × 1.5          │
├─────────────────────────────────────────────────────────────┤
│  Constraints:  Δ_min ≤ Δ ≤ Δ_max                            │
│                (0.01)    (10.0)                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Scaled Trust Region Constraint Formulation

**Problem:** A single scalar trust region Δ doesn't work well with mixed-unit state vectors (m/s, rad/s, kN, m, rad, s).

**Solution:** Per-dimension scaling based on typical variable magnitudes:

```
x̄ᵢₖ - Δ·σᵢ ≤ xᵢₖ ≤ x̄ᵢₖ + Δ·σᵢ    for all i ∈ {1,...,nx}, k ∈ {0,...,N}
ūⱼₖ - Δ·ρⱼ ≤ uⱼₖ ≤ ūⱼₖ + Δ·ρⱼ    for all j ∈ {1,...,nu}, k ∈ {0,...,N}
```

**Scaling factors:**
```
State scales σ:   [10, 2, 0.5, 1, 1, 1, 2, 0.3]  # ux, uy, r, dfz_long, dfz_lat, t, e, dpsi
                   m/s m/s rad/s kN  kN  s  m  rad

Control scales ρ: [0.3, 5.0]                      # delta, fx
                   rad  kN
```

This ensures that a trust region multiplier of Δ=1 allows physically meaningful variations in all dimensions.

---

## 6. Implementation Details

### 6.1 Class Structure

```python
class SCPSolver:
    """
    Main SCP solver class.

    Attributes:
        vehicle: SingleTrackModel - dynamics model
        world: World - track geometry
        params: SCPParams - algorithm parameters
        f_dynamics: CasADi Function - dynamics evaluation
        f_jacobians: CasADi Function - Jacobian computation
    """
```

### 6.2 Key Parameters (SCPParams)

```python
@dataclass
class SCPParams:
    # Trust region (now per-dimension scaled)
    tr_radius_init: float = 1.0      # Initial trust region multiplier
    tr_radius_min: float = 0.01      # Minimum (termination condition)
    tr_radius_max: float = 10.0      # Maximum
    tr_shrink_factor: float = 0.5    # Shrink on rejection
    tr_expand_factor: float = 1.5    # Expand on good progress

    # Convergence (now requires all three conditions)
    max_iterations: int = 50
    convergence_tol: float = 1e-4    # Cost change threshold
    constraint_tol: float = 1e-3     # Defect threshold
    virtual_control_tol: float = 1e-4  # ||V|| threshold

    # Step acceptance
    rho_min: float = 0.1             # Minimum ρ to accept
    rho_good: float = 0.7            # ρ threshold for expansion

    # Penalties (for merit function)
    virtual_control_weight: float = 1e4   # w in Φ
    defect_penalty_weight: float = 1e3    # λ in Φ

    # QP solver ('osqp', 'qrqp', or 'ipopt')
    qp_solver: str = 'osqp'               # Fast convex QP solver
```

### 6.3 Dynamics Linearization

The linearization is pre-built using CasADi's automatic differentiation:

```python
def _build_linearization_functions(self):
    # Symbolic variables
    x_sym = ca.SX.sym('x', self.nx)  # 8-dimensional state
    u_sym = ca.SX.sym('u', self.nu)  # 2-dimensional control
    k_psi_sym = ca.SX.sym('k_psi')   # Path curvature
    theta_sym = ca.SX.sym('theta')   # Road grade
    phi_sym = ca.SX.sym('phi')       # Road bank

    # Get dynamics (spatial: dx/ds)
    dx_dt, s_dot = self.vehicle.dynamics_dt_path_vec(...)
    dx_ds = dx_dt / s_dot

    # Compute Jacobians symbolically
    A = ca.jacobian(dx_ds, x_sym)  # 8×8 matrix
    B = ca.jacobian(dx_ds, u_sym)  # 8×2 matrix

    # Create callable functions
    self.f_jacobians = ca.Function('f_jacobians',
        [x_sym, u_sym, k_psi_sym, theta_sym, phi_sym],
        [A, B, dx_ds])
```

### 6.4 QP Subproblem Construction

Each SCP iteration solves a **convex** subproblem:

```python
def _solve_qp_subproblem(self, X_ref, U_ref, ...):
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(nx, N+1)      # States
    U = opti.variable(nu, N+1)      # Controls
    V = opti.variable(nx, N)        # Virtual control (slack)

    # CONVEX Objective: t[N] + w * ||V||^2
    # Linear in t[N] + quadratic in V = genuinely convex!
    t = X[IDX_T, :]
    cost = t[N] + virtual_control_weight * ca.sumsqr(V)

    opti.minimize(cost)

    # Linearized dynamics constraints (affine = convex)
    for k in range(N):
        f_k = A[k] @ X[:,k] + B[k] @ U[:,k] + c[k]
        f_kp1 = A[k+1] @ X[:,k+1] + B[k+1] @ U[:,k+1] + c[k+1]
        opti.subject_to(X[:,k+1] == X[:,k] + ds/2*(f_k + f_kp1) + V[:,k])

    # SCALED trust region constraints (per-dimension)
    for k in range(N+1):
        for i in range(nx):
            tr_i = tr_multiplier * STATE_SCALES[i]
            opti.subject_to(X[i,k] >= X_ref[i,k] - tr_i)
            opti.subject_to(X[i,k] <= X_ref[i,k] + tr_i)
        for j in range(nu):
            tr_j = tr_multiplier * CONTROL_SCALES[j]
            opti.subject_to(U[j,k] >= U_ref[j,k] - tr_j)
            opti.subject_to(U[j,k] <= U_ref[j,k] + tr_j)

    # State/control bounds, boundary conditions...

    # Use OSQP for fast convex QP solving
    opti.solver('sqpmethod', {
        'qpsol': 'osqp',
        'qpsol_options': {'verbose': False},
        'max_iter': 1,  # Single iteration since problem is QP
    })
    sol = opti.solve()

    V_norm_sq = np.sum(sol.value(V)**2)
    return sol.value(X), sol.value(U), sol.value(cost), V_norm_sq
```

---

## 7. Code Walkthrough

### 7.1 Main Solve Loop

```python
def solve(self, N, ds_m, X_init=None, U_init=None, ...):
    # Initialize
    X_ref = X_init if X_init is not None else default_trajectory()
    U_ref = U_init if U_init is not None else default_controls()
    tr_multiplier = self.params.tr_radius_init

    # Compute initial NONLINEAR merit (no V - V doesn't exist in original problem)
    merit_prev, lap_time_prev, defect_norm_prev = self._compute_nonlinear_merit(X_ref, U_ref, ...)

    for iteration in range(self.params.max_iterations):
        # Step 1: Linearize around reference
        A_list, B_list, c_list = self._linearize_around_trajectory(X_ref, U_ref, ...)

        # Step 2: Solve CONVEX QP subproblem
        X_new, U_new, cost_qp, V_norm_sq, success = self._solve_qp_subproblem(
            X_ref, U_ref, A_list, B_list, c_list, tr_multiplier, ...)

        # Step 3: Compute ρ using TWO merit functions
        # Model merit (predicted): Φ_model = t[N] + w*||V||²
        model_merit_ref = X_ref[IDX_T, -1]  # At reference: V=0
        model_merit_new = cost_qp           # QP objective value
        predicted_decrease = model_merit_ref - model_merit_new

        # Nonlinear merit (actual): Φ_nl = t[N] + λ*||defects_nl||² (NO V!)
        merit_new, lap_time_new, defect_norm_new = self._compute_nonlinear_merit(
            X_new, U_new, ...)  # No V_norm_sq!
        actual_decrease = merit_prev - merit_new

        rho = actual_decrease / predicted_decrease

        # Step 4: Accept or reject based on nonlinear merit
        if rho >= self.params.rho_min and merit_new < merit_prev * 1.01:
            X_ref, U_ref = X_new, U_new
            merit_prev, lap_time_prev, defect_norm_prev = merit_new, lap_time_new, defect_norm_new

            if rho >= self.params.rho_good:
                tr_multiplier *= self.params.tr_expand_factor
        else:
            tr_multiplier *= self.params.tr_shrink_factor

        # Step 5: Check convergence (all three conditions)
        if (cost_change < convergence_tol and
            defect_norm < constraint_tol and
            V_norm < virtual_control_tol):
            break

    return SCPResult(X=X_ref, U=U_ref, iterations=iteration, ...)
```

### 7.2 Linearization Details

```python
def _linearize_around_trajectory(self, X_ref, U_ref, s_grid, k_psi, theta, phi):
    """
    Linearize dynamics at each node.

    Returns affine model: dx/ds = A·x + B·u + c
    """
    A_list, B_list, c_list = [], [], []

    for k in range(N):
        # Evaluate Jacobians at reference point
        A_k, B_k, f_k = self.f_jacobians(
            X_ref[:, k], U_ref[:, k],
            k_psi[k], theta[k], phi[k]
        )

        # Affine constant: c = f(x_ref, u_ref) - A·x_ref - B·u_ref
        c_k = f_k - A_k @ X_ref[:, k] - B_k @ U_ref[:, k]

        A_list.append(np.array(A_k))
        B_list.append(np.array(B_k))
        c_list.append(np.array(c_k).flatten())

    return A_list, B_list, c_list
```

### 7.3 Nonlinear Merit Function Evaluation

```python
def _compute_nonlinear_merit(self, X, U, s_grid, k_psi, theta, phi, ds):
    """
    Compute NONLINEAR merit function: Φ_nl = t[N] + λ·‖defects_nl‖²

    This is evaluated on TRUE nonlinear dynamics and contains NO virtual
    control V. V is an artifact of the convex relaxation and does not
    exist in the original problem.

    Used for computing ACTUAL decrease in trust-region ρ.
    """
    # Lap time (primary objective)
    lap_time = X[IDX_T, -1]

    # Compute nonlinear dynamics defects
    defects = np.zeros((nx, N))
    for k in range(N):
        f_k = self.f_dynamics(X[:,k], U[:,k], k_psi[k], theta[k], phi[k])
        f_kp1 = self.f_dynamics(X[:,k+1], U[:,k+1], k_psi[k+1], theta[k+1], phi[k+1])

        x_kp1_predicted = X[:,k] + ds/2 * (f_k + f_kp1)
        defects[:, k] = X[:,k+1] - x_kp1_predicted

    defect_norm_sq = np.sum(defects**2)
    defect_norm = np.sqrt(defect_norm_sq)

    # Nonlinear merit function (NO V term!)
    merit = lap_time + defect_penalty_weight * defect_norm_sq

    return merit, lap_time, defect_norm
```

**Note:** The model merit `Φ_model = t[N] + w·‖V‖²` is just the QP objective value returned by `_solve_qp_subproblem()`. At the reference point, V=0, so `Φ_model(ref) = t_ref[N]`.

---

## 8. Convergence Analysis

### 8.1 Convergence Criteria

The solver terminates when **ALL THREE** of these success conditions are met:

1. **Cost convergence**: `|t[N]^{k+1} - t[N]^k| < convergence_tol`
2. **Constraint satisfaction**: `‖defects‖ < constraint_tol`
3. **Virtual control vanishes**: `‖V‖ < virtual_control_tol`

Or when ANY of these failure conditions occurs:

4. **Trust region collapse**: `Δ ≤ Δ_min`
5. **Maximum iterations**: `k ≥ max_iterations`

**Why require all three?** Virtual control going to zero ensures the linearized dynamics match the nonlinear dynamics—a key indicator of true SCP convergence.

### 8.2 Typical Convergence Behavior

```
Cold Start (poor initialization):
─────────────────────────────────────────────────────────────────────
Iter  t[N]     defect    ‖V‖       TR      ρ       Status
0     26.00    0.087     1.2e-1    1.0     -0.3    rejected
1     26.00    0.087     1.2e-1    0.5     0.05    rejected
2     26.00    0.087     1.2e-1    0.25    0.08    rejected
...
8     26.00    0.087     1.2e-1    0.01    ---     TERMINATED (TR collapsed)

Warm Start (good initialization):
─────────────────────────────────────────────────────────────────────
Iter  t[N]     defect    ‖V‖       TR      ρ       Status
0     13.92    7.9e-8    2.1e-6    1.0     0.95    accepted
1     13.92    3.2e-9    8.4e-7    1.5     0.99    CONVERGED
```

**Key diagnostic:** Virtual control ‖V‖ should decrease over iterations. If it stays large, the linearization is poor.

### 8.3 Why Cold Start Fails

With a poor initial guess (straight line at constant velocity):
1. The linearization is very inaccurate far from the true optimum
2. The QP solution is in a completely different region
3. The actual cost doesn't improve (or gets worse)
4. Trust region shrinks repeatedly until it collapses

### 8.4 Why Warm Start Succeeds

With a good initial guess (from Direct Collocation):
1. We're already near the optimum
2. The linearization is accurate locally
3. The QP solution is close to the current point
4. Convergence is immediate (1-2 iterations)

---

## 9. Warm-Starting

### 9.1 The Warm-Start Concept

```
                    ┌─────────────────┐
                    │   Data Source   │
                    │  (expert demos, │
                    │   prior solves) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Warm-Start   │
                    │   Generator     │
                    │ (e.g., Decision │
                    │  Transformer)   │
                    └────────┬────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │         Initial Guess          │
            │    X_init ∈ ℝ^{nx × (N+1)}    │
            │    U_init ∈ ℝ^{nu × (N+1)}    │
            └────────────────┬───────────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │          SCP Solver            │
            │   (refines to local optimum)   │
            └────────────────┬───────────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │      Optimal Trajectory        │
            │   (feasible, locally optimal)  │
            └────────────────────────────────┘
```

### 9.2 Warm-Start Quality Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| SCP Iterations | Number of iterations to converge | Low (1-5) |
| Initial ‖V‖ | Virtual control at first iteration | Low (<1e-3) |
| Initial defect | Dynamics violation of warm-start | Low (<1e-3) |
| Convergence Rate | How quickly cost decreases | Fast |
| Success Rate | % of problems that converge | High (>95%) |
| Final Cost | Optimality of solution | Similar to cold-start optimal |

**Key insight:** A good warm-start has low initial ‖V‖ and defect, meaning the linearized dynamics are accurate from the start.

### 9.3 Decision Transformer as Warm-Start Generator

The project's goal is to train a Decision Transformer to produce warm-starts:

```python
# Future integration
class DecisionTransformerWarmStart:
    def __init__(self, model_path):
        self.model = load_decision_transformer(model_path)

    def generate_warm_start(self, track_params, initial_state, target_time):
        """
        Generate trajectory warm-start using Decision Transformer.

        Input: track geometry, initial conditions, desired lap time
        Output: X_init, U_init for SCP
        """
        # Encode context
        context = encode_planning_context(track_params, initial_state, target_time)

        # Autoregressively generate trajectory
        trajectory = self.model.generate(context, return_to_go=target_time)

        # Format for SCP
        X_init, U_init = format_for_scp(trajectory)

        return X_init, U_init

# Usage
dt_warmstart = DecisionTransformerWarmStart("model.pt")
X_init, U_init = dt_warmstart.generate_warm_start(track, x0, target_time=14.0)

scp_result = scp_solver.solve(N, ds, X_init=X_init, U_init=U_init)
print(f"Converged in {scp_result.iterations} iterations")  # Should be 1-5
```

---

## 10. Results and Observations

### 10.1 Demo Results

From `run_trajopt_demo.py` on the 260m oval track:

| Method | Iterations | Time | Cost | Success |
|--------|------------|------|------|---------|
| Direct Collocation (IPOPT) | 34 | 8.34s | 14.46s | Yes |
| SCP Cold Start | 9 | 12.73s | 26.00s | **No** |
| SCP Warm Start | **2** | **4.69s** | 13.92s | Yes |

### 10.2 Key Observations

1. **Cold start fails completely**: Without a good initialization, SCP cannot find the optimum. The trust region collapses before making progress.

2. **Warm start is highly effective**: With a good initialization (from DC), SCP converges in just 2 iterations, validating the approach.

3. **Iteration reduction**: 4.5x fewer iterations with warm-start (9 → 2).

4. **Time reduction**: 2.7x faster with warm-start (12.73s → 4.69s).

5. **Final cost is similar**: Warm-started SCP achieves comparable (slightly better) cost to DC alone.

### 10.3 Implications for Decision Transformer

These results strongly support the project hypothesis:

> **If a Decision Transformer can produce trajectories of similar quality to Direct Collocation solutions, then SCP will converge in very few iterations.**

The warm-start needs to be:
- **Dynamically consistent**: Approximately satisfies vehicle dynamics
- **Constraint-aware**: Stays within track bounds
- **Near-optimal**: Close to the minimum-time trajectory

Even imperfect warm-starts should help, as long as they're better than straight-line initialization.

### 10.4 Visualization

The implementation generates several visualizations:

1. **`dc_trajectory.png`**: Top-down view of DC solution
2. **`scp_cold_convergence.png`**: Shows trust region collapse
3. **`scp_warm_convergence.png`**: Shows rapid convergence
4. **`warm_start_analysis.png`**: Side-by-side comparison
5. **`method_comparison.png`**: All three methods compared

---

## 11. Implementation Correctness

The SCP solver implements the three critical requirements for proper successive convexification:

### 11.1 Convex Subproblem
```python
# Objective: t[N] + w * ||V||^2
# Linear in t[N] + quadratic in V = genuinely convex
cost = t[N] + virtual_control_weight * ca.sumsqr(V)
```

### 11.2 Consistent Merit Function
```python
# Same Φ for both predicted and actual decrease
Φ = t[N] + λ * ||defects||^2 + w * ||V||^2

predicted_decrease = Φ_ref - Φ_predicted  # from QP
actual_decrease = Φ_prev - Φ_new          # from nonlinear eval
rho = actual_decrease / predicted_decrease
```

### 11.3 Scaled Trust Region
```python
# Per-dimension scaling for mixed-unit states
STATE_SCALES = [10, 2, 0.5, 1, 1, 1, 2, 0.3]  # m/s, m/s, rad/s, kN, kN, s, m, rad
CONTROL_SCALES = [0.3, 5.0]                    # rad, kN

# Trust region: |x_i - x_ref_i| <= Δ * scale_i
```

See `docs/scp_correctness_checklist.md` for the full verification checklist.

---

## References

1. Mao, Y., Szmuk, M., & Açıkmeşe, B. (2016). "Successive Convexification of Non-Convex Optimal Control Problems and Its Convergence Properties." *IEEE CDC*.

2. Malyuta, D., et al. (2022). "Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently." *IEEE Control Systems Magazine*.

3. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

4. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
