# SCP Trajectory Optimizer Status

## Canonical Doc

This is the main SCP status document to use going forward.

Reference/archived SCP documents are stored here:
- `docs/SCP_EXPLANATION.md` (kept for reference)
- `docs/scp_archive/scp_correctness.md`
- `docs/scp_archive/plan_keep_trying_scp.md`

Code freeze policy:
- `planning/scp_solver.py` is frozen for regular development.
- Only critical bug-fix or explicitly-approved experimental changes should modify it.

## Scope

This note summarizes the current state of the trajectory-optimization SCP implementation, key failure modes observed in practice, differences vs the ART paper codebase (`art-aeroconf24-main`), and a practical assessment for project execution.

---

## Current State (This Repo)

Primary implementation:
- `planning/scp_solver.py`
- `run_trajopt_demo.py`

Current behavior by backend:

1. IPOPT-backed SCP (production path)
- Works well enough for dataset generation.
- Cold-start runs can reach feasibility target and terminate successfully.
- This is currently the only reliable SCP path.

2. QP-backed SCP (OSQP/QRQP debug paths)
- Still not reliable for cold-start nonlinear racing dynamics.
- OSQP: repeated subproblem failure (`Maximum_Iterations_Exceeded`) with no accepted SCP updates.
- QRQP: subproblems can solve, but outer SCP repeatedly rejects steps (no nonlinear-defect improvement).

Evidence logs:
- `results/trajectory_optimization/run_trajopt_demo_output.log` (or legacy `run_scp_demo_output.log`)
- `results/trajectory_optimization/osqp_debug.log`
- `results/trajectory_optimization/qrqp_debug.log`

---

## Detailed Problems Observed

### 1) QP subproblem reliability gap

- OSQP path repeatedly fails at subproblem solve stage.
- Reported status: `Maximum_Iterations_Exceeded`, `SOLVER_RET_LIMITED`.
- No accepted SCP step is produced from cold start.

### 2) Fixed-point / rejection loop in QRQP mode

- QRQP can return a candidate step, but nonlinear acceptance rejects it repeatedly.
- Defect does not decrease from initial value.
- Trust-region logic (or debug-disabled variant) cannot escape this loop robustly.

### 3) High sensitivity to linearization quality

- Current nonlinear vehicle model and discretization are strongly nonlinear.
- Early linearization points are often outside a region where convex approximations are predictive.
- As a result, convex subproblem solutions do not consistently reduce true nonlinear defect.

### 4) Practical implication

- Method-level SCP is valid, but this formulation is not robust enough for QP-backed cold-start usage.
- IPOPT-backed subproblems remain the robust path in current code.

---

## Difference vs ART Method

Reference inspected:
- `art-aeroconf24-main/optimization/ocp.py`

### ART code characteristics

1. Dynamics in SCP subproblem are linear/LTV propagation (`stm`, `cim`), not a fresh nonlinear model linearization each iteration.
2. Nonconvex elements are handled with structured convexification (linearized keep-out constraints + SOC trust region).
3. Iteration update is simpler and more deterministic:
   - Solve convex subproblem
   - If feasible/optimal, update reference
   - Shrink trust region geometrically
4. Solver stack is pure convex optimization (CVXPY with ECOS/MOSEK).

### This repo’s SCP characteristics

1. Uses repeated linearization of nonlinear racing dynamics each iteration (`A,B,c` from CasADi Jacobians).
2. Uses linearized trapezoidal dynamics plus virtual control slack.
3. Uses nonlinear merit-based accept/reject with adaptive trust-region updates.
4. Uses CasADi `Opti` pipeline with IPOPT or SQP+QP backends.

### Practical consequence of this difference

The ART setup is substantially easier to keep numerically stable in a convex SCP loop.  
This repo’s setup is more expressive but significantly harder to make robust with pure QP subproblem solves from cold start.

---

## Assessment

### What is true

- SCP itself is not invalid for nonlinear dynamics.
- The current QP-backed SCP formulation in this repo is not robust enough for production use.
- IPOPT-backed SCP is currently the best working compromise for dataset generation.

### What this means for project execution

1. Use IPOPT-backed SCP (or direct collocation) as the dataset-generation solver now.
2. Treat OSQP/QRQP SCP as an experimental optimization track.
3. Only promote QP-backed SCP after it demonstrates:
   - accepted cold-start iterations,
   - defect reduction to target feasibility,
   - repeatable runtime benefit over IPOPT-backed path.

---

## Recommended Near-Term Direction

1. Keep production generation on IPOPT-backed path.
2. Continue collecting solver diagnostics for QP-backed debug runs.
3. Consider formulation-level redesign if QP-backed SCP remains unstable:
   - stronger globalization strategy,
   - alternative trust-region/line-search mechanisms,
   - staged/curriculum initialization,
   - or direct convex model simplification for QP mode.
