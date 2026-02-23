# SCP Correctness Status

This file tracks whether the current SCP implementation is structurally correct and whether recent demo results validate behavior.

- Solver implementation: `planning/scp_solver.py`
- Demo runner: `run_scp_demo.py`
- Latest evidence log: `results/trajectory_optimization/run_scp_demo_output.log`
- Last verified: 2026-02-23

---

## 1) Implemented fixes in code

The following requested changes are present in `planning/scp_solver.py`:

1. Merit-weight consistency for rho scaling
- `virtual_control_weight = 1e4`
- `defect_penalty_weight = 1e4`

2. Early feasible exit support
- Added `SCPParams.early_exit_on_feasible` (default `True`)
- Early return before SCP loop when initial defect is below tolerance

3. Acceptance rule tightened
- Uses strict non-increase condition:
  - `accept_step = (rho >= rho_min) and (merit_new <= merit_prev)`

4. Reporting semantics separated
- `SCPResult` now has separate `converged` and `feasible`
- Final `success` is tied to `converged` for SCP benchmarking

5. OSQP/QRQP SQP wrapper settings improved
- `sqpmethod max_iter` raised from `1` to `20`
- OSQP tolerances set to `1e-5`

---

## 2) Demo runner alignment

`run_scp_demo.py` now prints SCP-specific fields:
- `Converged`
- `Feasible`
- `Early exit`

Summary table includes these columns so `iterations=0` cases are not misinterpreted.

---

## 3) Latest results (from current log)

From `results/trajectory_optimization/run_scp_demo_output.log`:

1. Direct Collocation (IPOPT)
- Success: `True`
- Lap time: `14.9013 s`
- Iterations: `17`

2. SCP Cold Start (IPOPT)
- Success: `False`
- Converged: `False`
- Feasible: `False`
- Iterations: `9`
- Termination: `Trust region collapsed to minimum`

3. SCP Warm Start (OSQP)
- Initial defect is already feasible
- Early exit triggered
- Success: `True`
- Converged: `True`
- Feasible: `True`
- Iterations: `0`
- Termination: `Early exit: initial trajectory already feasible`

---

## 4) What is still failing

1. Cold-start SCP robustness is still poor.
- Trust region collapses before convergence.

2. Warm-start result currently validates the feasibility shortcut, not iterative SCP quality.
- Because early exit occurs, OSQP subproblem behavior is not exercised in that run.

---

## 5) What to test next (required)

1. Validate iterative SCP behavior explicitly
- Re-run warm-start with `early_exit_on_feasible=False`.
- Confirm at least one SCP iteration is executed.

2. Re-check OSQP path after that run
- Ensure QP failures are not dominant.
- Verify trust region does not immediately collapse.

3. Keep two evaluation modes in reports
- Runtime mode: early-exit enabled
- Algorithm-validation mode: early-exit disabled

Only after step (1) can warm-start SCP improvement claims be treated as validated.
