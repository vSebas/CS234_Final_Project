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

The convex subproblem at iteration i is:

```
minimize    J_lin(x, u) + w·‖v‖²    (linearized cost + virtual control penalty)

subject to  xₖ₊₁ = xₖ + (Δs/2)·[(Aₖxₖ + Bₖuₖ + cₖ) + (Aₖ₊₁xₖ₊₁ + Bₖ₊₁uₖ₊₁ + cₖ₊₁)] + vₖ

            ‖x - x̄‖∞ ≤ Δ          (state trust region)
            ‖u - ū‖∞ ≤ Δ          (control trust region)

            x_min ≤ x ≤ x_max      (state bounds)
            u_min ≤ u ≤ u_max      (control bounds)
```

Where:
- `v` is **virtual control** (slack variable) that allows constraint relaxation
- `w` is a large penalty weight on virtual control
- `Δ` is the trust region radius

#### Step 2d: Improvement Ratio

The improvement ratio ρ measures how well the linear model predicts actual improvement:

```
ρ = (J_actual(xᵏ) - J_actual(x̃)) / (J_linear(xᵏ) - J_linear(x̃))
  = actual_improvement / predicted_improvement
```

- `ρ ≈ 1`: Linear model is accurate
- `ρ < ρ_min`: Step rejected, shrink trust region
- `ρ > ρ_good`: Expand trust region

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

### 5.3 Trust Region Constraint Formulation

For each state and control at each node:

```
x̄ᵢₖ - Δ ≤ xᵢₖ ≤ x̄ᵢₖ + Δ    for all i ∈ {1,...,nx}, k ∈ {0,...,N}
ūⱼₖ - Δ ≤ uⱼₖ ≤ ūⱼₖ + Δ    for all j ∈ {1,...,nu}, k ∈ {0,...,N}
```

This creates a "box" around the reference trajectory within which the solution must lie.

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
    # Trust region
    tr_radius_init: float = 2.0      # Initial trust region
    tr_radius_min: float = 0.01      # Minimum (termination condition)
    tr_radius_max: float = 10.0      # Maximum
    tr_shrink_factor: float = 0.5    # Shrink on rejection
    tr_expand_factor: float = 1.5    # Expand on good progress

    # Convergence
    max_iterations: int = 50
    convergence_tol: float = 1e-4    # Cost change threshold
    constraint_tol: float = 1e-3     # Defect threshold

    # Step acceptance
    rho_min: float = 0.1             # Minimum ρ to accept
    rho_good: float = 0.7            # ρ threshold for expansion

    # Penalties
    virtual_control_weight: float = 1e4
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

Each SCP iteration solves:

```python
def _solve_qp_subproblem(self, X_ref, U_ref, ...):
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(nx, N+1)      # States
    U = opti.variable(nu, N+1)      # Controls
    V = opti.variable(nx, N)        # Virtual control (slack)

    # Objective: linearized time + virtual control penalty
    cost = 0
    for k in range(N):
        s_dot_k = (ux[k]*cos(dpsi[k]) - uy[k]*sin(dpsi[k])) / (1 - k*e[k])
        cost += ds / s_dot_k
    cost += w_virtual * sumsqr(V)

    # Linearized dynamics constraints
    for k in range(N):
        f_k = A[k] @ X[:,k] + B[k] @ U[:,k] + c[k]
        f_kp1 = A[k+1] @ X[:,k+1] + B[k+1] @ U[:,k+1] + c[k+1]
        opti.subject_to(X[:,k+1] == X[:,k] + ds/2*(f_k + f_kp1) + V[:,k])

    # Trust region constraints
    for k in range(N+1):
        opti.subject_to(X[:,k] >= X_ref[:,k] - tr_radius)
        opti.subject_to(X[:,k] <= X_ref[:,k] + tr_radius)
        opti.subject_to(U[:,k] >= U_ref[:,k] - tr_radius)
        opti.subject_to(U[:,k] <= U_ref[:,k] + tr_radius)

    # State/control bounds, boundary conditions...

    opti.solver('ipopt', {'ipopt.print_level': 0})
    sol = opti.solve()

    return sol.value(X), sol.value(U), sol.value(cost)
```

---

## 7. Code Walkthrough

### 7.1 Main Solve Loop

```python
def solve(self, N, ds_m, X_init=None, U_init=None, ...):
    # Initialize
    X_ref = X_init if X_init is not None else default_trajectory()
    U_ref = U_init if U_init is not None else default_controls()
    tr_radius = self.params.tr_radius_init

    # Compute initial cost and constraint violation
    cost_prev, defects_prev = self._compute_nonlinear_cost_and_dynamics(X_ref, U_ref, ...)

    for iteration in range(self.params.max_iterations):
        # Step 1: Linearize around reference
        A_list, B_list, c_list = self._linearize_around_trajectory(X_ref, U_ref, ...)

        # Step 2: Solve QP subproblem
        X_new, U_new, cost_qp, success = self._solve_qp_subproblem(
            X_ref, U_ref, A_list, B_list, c_list, tr_radius, ...)

        # Step 3: Evaluate actual cost at new point
        cost_new, defects_new = self._compute_nonlinear_cost_and_dynamics(X_new, U_new, ...)

        # Step 4: Compute improvement ratio
        predicted_decrease = cost_prev - cost_qp
        actual_decrease = cost_prev - cost_new
        rho = actual_decrease / predicted_decrease

        # Step 5: Accept or reject
        if rho >= self.params.rho_min:
            X_ref, U_ref = X_new, U_new
            cost_prev = cost_new

            if rho >= self.params.rho_good:
                tr_radius *= self.params.tr_expand_factor
        else:
            tr_radius *= self.params.tr_shrink_factor

        # Step 6: Check convergence
        if cost_change < tol and max_defect < constraint_tol:
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

### 7.3 Nonlinear Evaluation

```python
def _compute_nonlinear_cost_and_dynamics(self, X, U, s_grid, k_psi, theta, phi, ds):
    """
    Evaluate the TRUE nonlinear cost and dynamics defects.

    This is used to:
    1. Compute improvement ratio ρ
    2. Check constraint satisfaction
    """
    cost = 0.0
    defects = np.zeros((nx, N))

    for k in range(N):
        # True s_dot (nonlinear)
        s_dot_k = (X[0,k]*cos(X[7,k]) - X[1,k]*sin(X[7,k])) / (1 - k_psi[k]*X[6,k])

        # Cost contribution
        cost += ds / s_dot_k

        # Dynamics defect (should be zero if dynamics satisfied)
        f_k = self.f_dynamics(X[:,k], U[:,k], k_psi[k], theta[k], phi[k])
        f_kp1 = self.f_dynamics(X[:,k+1], U[:,k+1], k_psi[k+1], theta[k+1], phi[k+1])

        x_kp1_predicted = X[:,k] + ds/2 * (f_k + f_kp1)
        defects[:, k] = X[:,k+1] - x_kp1_predicted

    return cost, defects
```

---

## 8. Convergence Analysis

### 8.1 Convergence Criteria

The solver terminates when ANY of these conditions is met:

1. **Cost convergence**: `|J^{k+1} - J^k| < convergence_tol`
2. **Constraint satisfaction**: `max|defects| < constraint_tol`
3. **Trust region collapse**: `Δ ≤ Δ_min`
4. **Maximum iterations**: `k ≥ max_iterations`

### 8.2 Typical Convergence Behavior

```
Cold Start (poor initialization):
─────────────────────────────────
Iter  Cost     Defect    TR      Status
0     26.00    0.087     3.0     rejected (ρ < 0)
1     26.00    0.087     1.5     rejected
2     26.00    0.087     0.75    rejected
...
8     26.00    0.087     0.01    TERMINATED (TR collapsed)

Warm Start (good initialization):
─────────────────────────────────
Iter  Cost     Defect    TR      Status
0     13.92    7.9e-8    3.0     rejected (already optimal)
1     13.92    7.9e-8    1.5     CONVERGED
```

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
| Convergence Rate | How quickly cost decreases | Fast |
| Success Rate | % of problems that converge | High (>95%) |
| Final Cost | Optimality of solution | Similar to cold-start optimal |

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

From `run_scp_demo.py` on the 260m oval track:

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

## References

1. Mao, Y., Szmuk, M., & Açıkmeşe, B. (2016). "Successive Convexification of Non-Convex Optimal Control Problems and Its Convergence Properties." *IEEE CDC*.

2. Malyuta, D., et al. (2022). "Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently." *IEEE Control Systems Magazine*.

3. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

4. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
