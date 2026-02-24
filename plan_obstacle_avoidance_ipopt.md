# IPOPT Obstacle-Avoidance Plan (Current Status)

This file is the source of truth for trajectory optimization work in this repo.
It reflects the current production path: direct collocation + IPOPT, full lap, no SCP development.

## Scope

- Solver: IPOPT via CasADi (`planning/optimizer.py`)
- Problem: full-lap trajectory optimization with periodic closure
- Obstacles: static circles from map metadata
- Objective: single-stage combined objective (time + smoothness, optional slack penalty)

## Implemented

1. Full-lap solve is active by default (`convergent_lap=True`, `ds = track_length / N`).
2. Obstacle constraints are implemented as hard nonlinear distance constraints.
3. Along-track obstacle gating is implemented (`obstacle_window_m`).
4. Interior obstacle checks per segment are implemented (`obstacle_subsamples_per_segment`).
5. Dense post-check metric is implemented (`min_obstacle_clearance`).
6. Obstacle slack plumbing exists for diagnosis (`obstacle_use_slack`), hard constraints are default.
7. Demo script logs to file and saves trajectory/states/controls/GIF (`run_trajopt_demo.py`).
8. Frenet-to-ENU conversion is unified across optimizer and plotting: `E = E_cl - e*sin(psi)`.
9. Frenet-to-ENU conversion is unified across optimizer and plotting: `N = N_cl + e*cos(psi)`.
10. Obstacle map generation works via unified generator (`create_tracks.py`) using Frenet source fields.
11. Deterministic acceptance-gated retry policy is implemented in `run_trajopt_demo.py`.
12. Obstacle-aware initializer for `e(s)` is implemented in `planning/optimizer.py`.

## Not Implemented Yet

1. Scenario sampler and dataset generator script (`data/generate_dataset.py`).
2. Canonical versioned scenario schema and solver-config hash storage.
3. DT dataset export pipeline (states/actions/rtg/tokens/masks).
4. Warm-start evaluation harness (`planning/refine_from_warmstart.py`).
5. Batch acceptance reporting dashboard for large runs.

## Current Acceptance Gates

1. `result.success == True`
2. `max_obstacle_slack == 0` in hard-constraint mode
3. `min_obstacle_clearance >= -0.001` from dense post-check (current practical epsilon mode)
4. No track/control/dynamics violations beyond solver tolerances

## Known Current Issue

1. Some runs still show tiny negative dense clearance (millimeter-level); strict mode (`>= 0`) may fail even when practical mode passes.

## Immediate Next Steps

1. Batch-run script is implemented (`run_trajopt_batch_eval.py`) and produces JSON/CSV pass-rate reports.
2. Numerical acceptance tolerance is set to `-0.001 m` by default in demo/batch tools.
3. Validate gate pass rate on larger batches (for example 50-200 scenarios).
4. Start dataset generation only after gate pass rate is stable.
5. Implement dataset schema/versioning and DT export format.

## Out of Scope (For Now)

1. SCP feature development.
2. Two-stage feasibility/performance pipeline.
3. QP-based convex subproblem paths.
