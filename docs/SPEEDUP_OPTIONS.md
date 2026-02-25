# Trajectory Optimization Speedup Options

## Current Performance Baseline

| Metric | Value |
|--------|-------|
| Solver | IPOPT via CasADi (Python) |
| Typical solve (N=120) | 2.26s |
| Bottleneck | Hessian computation (71% of time) |
| Root cause | CasADi symbolic interpretation overhead |

---

## Option 1: CasADi C Code Generation (Recommended First Step)

**Speedup: 2-3x**

CasADi can generate standalone C code from symbolic expressions. Currently unused.

### Implementation

```python
# In optimizer.py or vehicle.py
f_dynamics = ca.Function("f_dynamics", [x, u, params], [dx_dt])
f_dynamics.generate("generated/f_dynamics.c", {"with_header": True})

# Compile to shared library
# gcc -shared -fPIC -O3 f_dynamics.c -o libf_dynamics.so

# Load back into CasADi
f_dynamics_compiled = ca.external("f_dynamics", "./libf_dynamics.so")
```

### What to Generate

1. **Vehicle dynamics** (`dynamics_dt_path_vec`)
2. **Tire force computation** (`calc_fy_kn`, `calc_fx_kn`)
3. **Jacobians** (A, B matrices for linearization)
4. **Full NLP function/gradient/Hessian**

### Pros/Cons

| Pros | Cons |
|------|------|
| Minimal code changes | Build system complexity |
| Drop-in replacement | Regenerate on model changes |
| Works with existing IPOPT flow | |

---

## Option 2: Hand-Coded C++ Dynamics with Analytical Jacobians

**Speedup: 5-8x**

Replace CasADi symbolic dynamics with hand-written C++ and explicit Jacobian formulas.

### Architecture

```
cpp/
├── dynamics/
│   ├── single_track_model.hpp    # State derivatives
│   ├── single_track_model.cpp
│   ├── fiala_tire.hpp            # Tire model
│   ├── fiala_tire.cpp
│   └── jacobians.cpp             # Hand-coded ∂f/∂x, ∂f/∂u
├── bindings/
│   └── pybind_dynamics.cpp       # Python bindings
└── CMakeLists.txt
```

### Key Functions to Implement

```cpp
// State derivative (8 states, 2 controls)
void dynamics_dt(
    const double* x,      // [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi]
    const double* u,      // [delta, fx_kn]
    double k_psi,         // curvature
    double theta,         // grade
    double phi,           // bank
    double* dx_dt,        // output: 8 derivatives
    double* s_dot         // output: ds/dt
);

// Analytical Jacobians (exploit sparsity)
void dynamics_jacobians(
    const double* x,
    const double* u,
    double k_psi, double theta, double phi,
    double* A,            // 8x8 ∂f/∂x
    double* B             // 8x2 ∂f/∂u
);
```

### Jacobian Sparsity Structure

The 8x8 state Jacobian has exploitable sparsity:

```
        ux  uy  r   dfz_l dfz_t t   e   dpsi
ux    [ x   x   x   x     x     .   .   .   ]
uy    [ x   x   x   x     x     .   .   .   ]
r     [ x   x   x   x     x     .   .   .   ]
dfz_l [ x   .   .   x     .     .   .   .   ]
dfz_t [ .   x   x   .     x     .   .   .   ]
t     [ x   x   .   .     .     .   x   x   ]
e     [ x   x   .   .     .     .   .   x   ]
dpsi  [ x   x   x   .     .     .   x   x   ]
```

~40% zeros exploitable.

### Python Binding

```cpp
// pybind_dynamics.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "single_track_model.hpp"

PYBIND11_MODULE(cpp_dynamics, m) {
    m.def("dynamics_dt", &dynamics_dt_py);
    m.def("dynamics_jacobians", &dynamics_jacobians_py);
}
```

---

## Option 3: CasADi C++ API (No Python in Solve Loop)

**Speedup: 5-10x**

Use CasADi's native C++ API to build and solve the NLP entirely in C++.

### Architecture

```
cpp/
├── trajectory_optimizer.hpp
├── trajectory_optimizer.cpp      # NLP construction in C++
├── ipopt_interface.cpp           # Direct IPOPT callback
└── python_wrapper.cpp            # Thin Python entry point
```

### Key Change

Move from:
```python
# Python (slow)
opti = ca.Opti()
X = opti.variable(8, N+1)
for k in range(N):
    opti.subject_to(...)  # Python loop overhead
sol = opti.solve()
```

To:
```cpp
// C++ (fast)
casadi::Opti opti;
casadi::MX X = opti.variable(8, N+1);
for (int k = 0; k < N; k++) {
    opti.subject_to(...);  // Native C++ loop
}
casadi::OptiSol sol = opti.solve();
```

### Build System

```cmake
find_package(casadi REQUIRED)
find_package(pybind11 REQUIRED)

add_library(cpp_trajopt MODULE
    trajectory_optimizer.cpp
    python_wrapper.cpp
)
target_link_libraries(cpp_trajopt casadi pybind11::module)
```

---

## Option 4: Direct IPOPT C++ Interface (Maximum Performance)

**Speedup: 10-15x**

Bypass CasADi entirely. Implement IPOPT's `TNLP` interface directly in C++.

### Architecture

```
cpp/
├── nlp/
│   ├── trajectory_nlp.hpp        # IPOPT TNLP implementation
│   ├── trajectory_nlp.cpp
│   ├── dynamics.hpp              # Vehicle model
│   ├── constraints.hpp           # Track bounds, obstacles
│   └── sparse_hessian.cpp        # Hand-coded sparse Hessian
├── solvers/
│   └── ipopt_solver.cpp          # IPOPT setup and solve
└── bindings/
    └── python_interface.cpp
```

### IPOPT TNLP Interface

```cpp
class TrajectoryNLP : public Ipopt::TNLP {
public:
    // Problem dimensions
    bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                      Index& nnz_h_lag, IndexStyleEnum& index_style) override;

    // Variable bounds
    bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                         Index m, Number* g_l, Number* g_u) override;

    // Objective + gradient
    bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) override;
    bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) override;

    // Constraints + Jacobian (sparse)
    bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) override;
    bool eval_jac_g(Index n, const Number* x, bool new_x, Index m,
                    Index nele_jac, Index* iRow, Index* jCol, Number* values) override;

    // Hessian of Lagrangian (sparse)
    bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor,
                Index m, const Number* lambda, bool new_lambda,
                Index nele_hess, Index* iRow, Index* jCol, Number* values) override;

private:
    int N_;                        // Discretization nodes
    double ds_;                    // Step size
    VehicleParams params_;
    std::vector<Obstacle> obstacles_;

    // Precomputed sparsity patterns
    std::vector<int> jac_rows_, jac_cols_;
    std::vector<int> hess_rows_, hess_cols_;
};
```

### Sparse Hessian Structure

For N=120 nodes, 8 states, 2 controls:
- Variables: 1210
- Dense Hessian: 1.46M entries
- **Sparse Hessian: ~15K entries** (block-tridiagonal structure from collocation)

```cpp
// Hessian sparsity: block-tridiagonal from dynamics coupling
// Only adjacent nodes interact via collocation constraints
//
// [ H_0   C_0                           ]
// [ C_0^T H_1   C_1                     ]
// [       C_1^T H_2   C_2               ]
// [             ...   ...   ...         ]
// [                   C_{N-1}^T  H_N    ]
//
// H_k: 10x10 (states + controls at node k)
// C_k: 10x10 (coupling between nodes k and k+1)
```

---

## Option 5: GPU-Accelerated Batch Solves (For Dataset Generation)

**Speedup: 50-100x throughput (batch)**

For generating training data, solve many trajectories in parallel on GPU.

### Approaches

1. **cuOpt / NVIDIA Optimization**: Commercial GPU optimizer
2. **Custom CUDA kernels**: Parallelize across scenarios
3. **JAX + jaxopt**: GPU-accelerated optimization in Python

### When Useful

- Dataset generation (1000s of solves)
- Real-time MPC with multiple horizon samples
- Monte Carlo scenario evaluation

### JAX Example

```python
import jax
import jax.numpy as jnp
from jaxopt import ProjectedGradient

@jax.jit
def dynamics_residual(X_flat, U_flat, params):
    # Vectorized dynamics for GPU
    ...

# Batch solve across 100 scenarios
scenarios = jax.vmap(solve_single_trajectory)
results = scenarios(batch_params)  # Runs on GPU
```

---

## Option 6: Algorithmic Improvements (Complementary)

**Speedup: 2-5x (orthogonal to C++)**

### 6.1 Warm-Starting

```python
# Use previous solution as initial guess
result = optimizer.solve(N, ds, X_init=prev_X, U_init=prev_U)
```

Current: Cold start every solve
Improvement: 2-3x fewer iterations

### 6.2 Adaptive Mesh Refinement

```python
# Start coarse, refine where needed
result_coarse = solve(N=30)
result_fine = solve(N=120, X_init=interpolate(result_coarse))
```

### 6.3 Multiple Shooting vs Collocation

Current: Direct collocation (dense coupling)
Alternative: Multiple shooting (parallelizable segments)

### 6.4 Inexact Newton Steps

```python
opts = {
    'ipopt.linear_solver': 'ma57',      # Faster than MUMPS
    'ipopt.mehrotra_algorithm': 'yes',  # Predictor-corrector
    'ipopt.mu_strategy': 'adaptive',
}
```

---

## Comparison Summary

| Option | Speedup | Effort | Maintainability | Best For |
|--------|---------|--------|-----------------|----------|
| 1. CasADi codegen | 2-3x | Low | High | Quick win |
| 2. C++ dynamics | 5-8x | Medium | Medium | Production |
| 3. CasADi C++ API | 5-10x | Medium | Medium | Clean rewrite |
| 4. Direct IPOPT | 10-15x | High | Low | Maximum speed |
| 5. GPU batch | 50-100x | High | Low | Dataset gen |
| 6. Algorithmic | 2-5x | Low | High | Complement any |

---

## Recommended Path

### Phase 1: Quick Wins (Week 1)
1. Enable CasADi code generation for dynamics
2. Add warm-starting to optimizer
3. Switch to MA57 linear solver (if available)

### Phase 2: C++ Core (Weeks 2-4)
1. Implement vehicle dynamics in C++ with pybind11
2. Hand-code analytical Jacobians with sparsity
3. Replace CasADi dynamics calls with C++ module

### Phase 3: Full C++ Solver (Weeks 5-8)
1. Implement IPOPT TNLP interface
2. Exploit block-tridiagonal Hessian structure
3. Thin Python wrapper for orchestration

### Expected Final Performance

| Metric | Current | After Phase 3 |
|--------|---------|---------------|
| Single solve (N=120) | 2.26s | 0.15-0.25s |
| Dataset gen (1000 solves) | 37 min | 3-4 min |
| Real-time capable | No | Yes (10-20 Hz) |

---

## Files to Modify/Create

```
CS234_Final_Project/
├── cpp/                              # NEW: C++ source
│   ├── CMakeLists.txt
│   ├── dynamics/
│   │   ├── single_track_model.cpp
│   │   ├── fiala_tire.cpp
│   │   └── jacobians.cpp
│   ├── nlp/
│   │   ├── trajectory_nlp.cpp        # IPOPT interface
│   │   └── sparse_structure.cpp
│   └── bindings/
│       └── pybind_trajopt.cpp
├── planning/
│   ├── optimizer.py                  # MODIFY: use compiled dynamics
│   └── optimizer_cpp.py              # NEW: thin wrapper for C++ solver
├── models/
│   └── vehicle.py                    # MODIFY: optional CasADi codegen
└── scripts/
    └── build_cpp.sh                  # NEW: build script
```
