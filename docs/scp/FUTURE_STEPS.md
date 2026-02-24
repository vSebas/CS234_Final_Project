# If You Keep Trying SCP: A Practical Rescue Plan (Track + Obstacles)

This document is a **do-this-next** plan for salvaging the SCP approach in your current racing TO codebase.  
It assumes you want the benefits of SCP (fast convex subproblems + warm-start friendliness) but your current SCP loop is failing.

This plan prioritizes **globalization**, **numerical scaling**, **initialization**, and **making the convex subproblem truly convex**.

---

## 0) Goal: what “success” looks like for SCP here

You should be able to run:

- **Cold start**: a dumb initializer (or cheap heuristic) → SCP converges to a feasible trajectory (maybe not best lap time)
- **Warm start**: a decent initializer (DC/IPOPT or DT) → SCP converges in fewer iterations and less time

And log meaningful:
- `t[N]`, `||defects||`, `||V||`, trust-region radius, accept/reject, ρ.

If these signals don’t behave, you’re not doing SCP; you’re doing “iterative random trouble”.

---

## 1) Make the subproblem truly convex (and keep it that way)

### 1.1 Objective should be simple and convex
Use:
- **minimize** `t[N] + w_v * ||V||^2 + w_u * ||ΔU||^2` (optional)

Do **not** put `ds/s_dot` directly in the convex objective if it contains trig/division in decision vars.

### 1.2 Constraints must be affine/convex
- Linearized dynamics in trapezoidal form: affine equality
- Track bounds: affine inequalities in `e`
- Control bounds: box constraints
- Obstacle constraints: linearized half-spaces (for SCP), not nonlinear circles (unless you leave convex world)

### 1.3 Use an actual convex/QP solver path
Recommended:
- CVXPY + OSQP (or ECOS)
- or direct OSQP matrices

Avoid routing QP through a generic SQP wrapper that can fail for tiny tolerance reasons.

---

## 2) Globalization is not optional: trust region + line search

Your SCP will fail if it can’t reduce step size when linearization is bad.  
You need at least one of:

- Trust region **and** shrinking on rejection
- Backtracking line search on the step direction (often needed even with TR)

### 2.1 Trust region should be enabled for all solver backends
Do **not** disable TR in “debug QP” modes. That guarantees rejection loops.

Use per-dimension TR:
- `|x_i - x_ref_i| ≤ Δ_i`
- `|u_j - u_ref_j| ≤ Δ_u_j`

### 2.2 Add backtracking on α along the convex step direction
After solving the convex subproblem and obtaining `(X_qp, U_qp)`:

Try:
- `α ∈ {1, 0.5, 0.25, 0.125, ...}`

Define:
- `X_trial = X_ref + α (X_qp - X_ref)`
- `U_trial = U_ref + α (U_qp - U_ref)`

Accept the first α that decreases the **nonlinear merit** (see Section 3).

This single feature fixes a large fraction of “SCP rejects forever” failures.

---

## 3) Fix the merit function (ρ must compare apples to apples)

You must not evaluate “actual” progress using variables that are not in the original problem.

### 3.1 Use two functions
**Model merit (predicted):**
- `Φ_model(X,U,V) = t[N] + w_v * ||V||^2`  
(= convex subproblem objective)

**Nonlinear merit (actual):**
- `Φ_nl(X,U) = t[N] + w_def * ||defects_nl(X,U)||^2`  
(no V term)

### 3.2 Compute ρ correctly
Let reference be `(X_ref,U_ref)` and convex solve output `(X_qp,U_qp,V_qp)`.

- `pred = Φ_model(X_ref,U_ref,V=0) - Φ_model(X_qp,U_qp,V_qp)`
- `act  = Φ_nl(X_ref,U_ref) - Φ_nl(X_qp,U_qp)`
- `ρ = act / pred` (guard against `pred ≤ 0`)

**Acceptance rule:**
- accept if `ρ ≥ ρ_min` and `Φ_nl(new) < Φ_nl(ref)`
- otherwise reject and shrink TR

Also: line search can use `Φ_nl` directly, bypassing fragile ρ when needed.

---

## 4) Virtual control / slacks: use them correctly

Virtual control should:
- make every convex subproblem feasible
- go to zero as you converge

### 4.1 Track and penalize ||V||
Log `||V||` per iteration. Expected behavior:
- large early
- decreasing to near zero on successful runs

### 4.2 Consider a penalty continuation schedule
Instead of one huge `w_v`, use:
- start `w_v = 1e1` (example)
- when steps are accepted consistently, multiply by 10
- continue until `||V||` is tiny

This improves numerical conditioning.

---

## 5) Initialization matters: make cold start “track-consistent”

A totally naive cold start (straight, constant speed, zero yaw rate) on a curved track is usually outside the linearization basin.

### 5.1 Track-consistent initializer (cheap, good)
- `e = 0`
- `dpsi = 0`
- `δ_ff(s) ≈ atan(L * κ(s))`
- set `r` consistent with curvature and speed (even roughly)
- pick `u_x` modest (not extreme)

Even rough consistency can drop initial defects by orders of magnitude.

### 5.2 Obstacle-aware initializer for SCP
Before doing SCP obstacle constraints, build an `e_ref(s)` that shifts around obstacles with a smooth bump, then initialize `e` from that reference.

---

## 6) Obstacle avoidance in SCP: do it as convexified constraints

For circular obstacles, the true constraint is:
- `||p_k - p_obs||_2 ≥ R` (nonconvex)

In SCP, you enforce a **linearized half-space** around the current iterate:
- compute gradient at current `p_ref`
- impose the supporting hyperplane that keeps `p` outside locally

### 6.1 Keep it local with trust region
Obstacle half-spaces are only valid locally; TR/line-search is essential.

### 6.2 Use slack on obstacle constraints at first
Add `σ_obs ≥ 0` and penalize strongly:
- keeps subproblems feasible early
- gradually pushes violations to zero

---

## 7) Use a clean convex-solver pipeline (recommended implementation path)

### 7.1 CVXPY prototype (fastest to debug)
- Implement convex subproblem in CVXPY
- Solve with OSQP
- Check:
  - primal residuals
  - feasibility
  - objective value
  - variable scaling

This gets you out of CasADi solver-wrapper edge cases.

### 7.2 Only then integrate into the full codebase
Once CVXPY version works, you can decide whether to:
- keep CVXPY (fine for research project)
- or move to direct OSQP matrices for speed

---

## 8) Logging you must add (otherwise you’re blind)

Per SCP outer iteration log:
- `t[N]` (lap time proxy)
- `Φ_nl` and `Φ_model`
- `pred`, `act`, `ρ`
- `||defects||_∞`, `||defects||_2`
- `||V||_2`
- trust-region radius (per-dimension min/max or scalar summary)
- accept/reject
- if line search: accepted α

Plot these. If they don’t make sense, the solver isn’t behaving.

---

## 9) Minimal staged debugging sequence (don’t skip steps)

1. **No obstacles, wide track bounds**, conservative speed:
   - goal: basic feasibility convergence

2. **No obstacles, realistic track bounds**, still conservative:
   - goal: track constraints behave

3. **Add obstacle slacks**, one obstacle, mild penalty:
   - goal: feasibility with obstacle “pressure”

4. **Ramp obstacle penalty**, remove slack:
   - goal: true avoidance

5. **Then** tune time-optimality (reduce regularization, push speed)

If you add everything at once, you won’t know what broke.

---

## 10) Decision point (when to stop forcing SCP)

Set a hard criterion:
- If you can’t get **accepted steps** in QP-backed SCP within the first 10–20 outer iterations on the no-obstacle case, you need to fix globalization/scaling before adding complexity.

If after implementing Sections 2–3 (TR + line search + correct merit), you still fail on the no-obstacle case, SCP is not “just tuning” — it’s a formulation mismatch and you should pivot to NLP.

---

## TL;DR

If you want SCP to work, do these in order:

1) Convex subproblem = `min t[N] + w||V||^2` with affine constraints  
2) Enable trust region for all backends + shrink on rejection  
3) Add backtracking line search on the convex step  
4) Fix merit/ρ: actual merit uses nonlinear defects only (no V)  
5) Use track-consistent init; only then add obstacle half-spaces + slack  
6) Debug with CVXPY+OSQP first, then integrate

This turns “SCP failing” into a controlled engineering problem instead of guessing.
