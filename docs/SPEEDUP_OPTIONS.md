# Trajectory Optimization Speedup Options

This note is written against the **current repo behavior**: the simplified **Tier 1** NLP (min-time + Δu regularizer, node-only obstacles + dense post-check) is now the intended trajectory optimizer for dataset generation and DT warm-start refinement.

---

## Current Performance Baseline (measure the right thing)

There are **two** relevant timings:

| Metric | What it is | Typical value | Why it matters |
|---|---|---:|---|
| **IPOPT core time** | time spent inside IPOPT once the NLP is built | ~2–3 s | reflects objective/constraint eval + derivatives (Hessian/Jacobian) |
| **End-to-end solve time** | Python + CasADi graph building + precomputations + IPOPT | often **much larger** (e.g., 10–15 s) | **this is usually the real bottleneck** in the current code path |

Key observation from recent logs: IPOPT may report a few seconds, while the wrapper reports ~5–10× more. That means **we’re spending most time outside IPOPT** (problem construction and Python loops), not “solver math”.

---

## Option 0: Remove Python/NLP build overhead (fastest real-world win)

**Expected speedup: 3–8× end-to-end** (often bigger than any IPOPT tweak)

### 0.1 Skip obstacle midpoint/sample-grid work when it’s not needed
In `planning/optimizer.py`, the code currently constructs obstacle sample grids and calls track interpolants thousands of times **even when there are no obstacles** or when Tier 1 is set to “node-only + dense post-check”.

**Fix:**
- Only build `sample_grids` when: `len(obstacles) > 0 and obstacle_enforce_midpoints == True`.
- In Tier 1 dataset-generation mode: default `obstacle_enforce_midpoints = False`.

This change alone can cut many seconds from “build time”.

### 0.2 Vectorize track geometry lookups (stop per-node Python loops)
You currently loop over k and call `posE_m_interp_fcn`, `posN_m_interp_fcn`, `psi_rad_interp_fcn`, etc. per node.

**Fix:**
- Evaluate interpolants on **vector inputs** once per solve (or once per build if cached).
- Convert results to NumPy arrays once (avoid repeated Python↔CasADi overhead).

### 0.3 Cache the NLP: build once, solve many (parameterize scenarios)
For dataset generation you typically keep `(map_id, N, ds)` fixed while changing only obstacles (and maybe margins).

**Refactor goal:**
- `build_problem(map_id, N, ds, Jmax)` once.
- Treat scenario-dependent items as `Opti.parameter(...)`:
  - obstacle centers/radii (Frenet or world)
  - margins / buffers
- Then for each scenario: set parameter values + set initial guess + call `opti.solve()`.

**Important:** variable obstacle count forces rebuild. Avoid that by padding:
- choose a maximum number of obstacles `Jmax` (e.g., 8),
- if fewer, set unused obstacles “inactive” (radius=0 or place far away).

This usually makes end-to-end runtime approach IPOPT time.

### 0.4 Parallelize dataset generation (throughput multiplier)
Even without algorithmic changes:
- each scenario solve is independent
- use `multiprocessing` (or joblib) to run multiple solves in parallel

This is especially effective after caching (because each process amortizes the build cost).

---

## Option 1: IPOPT configuration tweaks (cheap, after Option 0)

**Expected speedup: 1.2–2× IPOPT time** (varies)

### 1.1 Hessian approximation: L-BFGS
Hessian evaluation can dominate IPOPT time.

Try:
- `ipopt.hessian_approximation = "limited-memory"`

Often reduces wall time substantially for collocation NLPs, at the risk of slightly more iterations.

### 1.2 Relax tolerances for dataset-gen mode
For training data, you typically do not need ultra-tight KKT tolerances.

Try:
- `ipopt.tol = 1e-4`
- `ipopt.acceptable_tol = 1e-3`

Validate that trajectories still pass acceptance checks.

### 1.3 Linear solver choice
If available on your system, switching linear solvers can help:
- MA57 / MA86 / Pardiso often outperform MUMPS

(This depends on licensing/installation.)

---

## Option 2: CasADi C Code Generation (optional, after caching)

**Expected speedup: 2–3× IPOPT time** (not end-to-end unless Option 0 is done)

CasADi can generate C code for functions (and derivatives) to reduce evaluation overhead.

### What to codegen
- vehicle dynamics in spatial form (or temporal + sdot)
- tire force computations
- constraint functions for collocation residuals
- optionally full NLP callbacks (objective, constraints, gradient, Jacobian, Hessian)

### Why this matters
Once you’ve removed Python build overhead (Option 0), IPOPT’s inner loop becomes the main cost. Codegen makes those evaluations faster.

---

## Option 4: Algorithmic changes (bigger redesign)

### 4.1 Multi-resolution / coarse-to-fine
Solve a coarse problem, then refine (warm-start) at higher N.

### 4.2 Multiple shooting vs collocation
May improve parallelism and structure; depends on implementation details.

---

## SCP like ART? (when and why)

ART uses SCP when the problem is “mostly convex” except for a small set of nonconvex constraints (e.g., keep-out zones) that can be sequentially linearized with a trust region.

For the raceline NLP:
- dynamics are nonlinear (tire + load transfer),
- min-time objective introduces nonconvexity,
- obstacle avoidance is nonconvex.

So “ART-style SCP” would require linearizing **a lot**, not just one constraint. That becomes a major solver rewrite (sequential convexification / SQP-like), and **it won’t address today’s main issue** (Python/NLP build overhead).

If later you need real-time MPC-like solves, SCP/QP solvers can be worth it, but it’s a Phase-2+ project.

---

## Recommended Path (updated)

### Phase 0: Fix the real bottleneck (days)
1. Guard midpoint/sample-grid work behind obstacle presence + flag
2. Vectorize track interpolant queries (no per-node Python loops)
3. Cache the NLP (build once) and parameterize obstacles with fixed `Jmax`
4. Parallelize dataset generation

### Phase 1: IPOPT speedups (hours)
1. `hessian_approximation = limited-memory`
2. Relax tolerances for dataset mode
3. Try a faster linear solver if available

### Phase 2: Codegen / compiled eval (days)
1. CasADi codegen for dynamics + derivatives
2. (Optional) codegen for full NLP callbacks

---

## Expected outcomes (realistic)

| Step | Main benefit | Typical impact |
|---|---|---|
| Option 0 (cache + skip work) | eliminates Python build overhead | **3–8×** end-to-end speedup |
| L-BFGS Hessian | reduces derivative work | **1.2–2×** IPOPT time |
| CasADi codegen | faster inner evals | **2–3×** IPOPT time |
