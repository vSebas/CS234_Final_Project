# Trajectory Optimizer Configuration

This document summarizes the configuration knobs for the IPOPT trajectory optimizer
(`planning/optimizer.py`) and the demo runner (`run_trajopt_demo.py`).

It is intended to be read alongside `docs/DYNAMIC_MODEL.md`.

---

## 1) Where Configuration Lives

There are two sources of configuration:

1. **Function arguments** in `TrajectoryOptimizer.solve(...)`
2. **Environment variables** that override IPOPT settings (and some demo defaults)

The demo script (`run_trajopt_demo.py`) sets defaults for `N`, `ds`, and other solver options,
then calls `TrajectoryOptimizer.solve(...)`.

---

## 2) Core Discretization

These are the most important parameters for any new track.

- **`N`**: number of spatial nodes (decision points)
- **`ds_m`**: spatial step size, usually `track_length / N`

**Rule of thumb:**
- mild curvature: `ds_m ≈ 2.0–3.0 m`
- tight/high curvature: `ds_m ≈ 1.0–1.5 m`

Then compute:
```
N = ceil(track_length / ds_m)
ds_m = track_length / N
```

**Paper-aligned default (Aggarwal & Gerdes 2025):**
- `N = 260`, `ds_m = 1.0 m` on the 260 m oval

---

## 3) TrajectoryOptimizer.solve(...) Parameters

These are the main knobs you will set or tune:

**Basic**
- `N`: nodes
- `ds_m`: spatial step
- `lambda_u`: control smoothness regularizer (default `1e-3`)
- `ux_min`: minimum longitudinal speed (default `3.0`)
- `ux_max`: optional max speed cap
- `track_buffer_m`: shrink track bounds (default `0.0`)

**Frenet regularization**
- `eps_s`: minimum forward progress (`sdot`) to avoid singularity
- `eps_kappa`: minimum `1 - kappa*e` margin

**Obstacles**
- `obstacles`: list of `ObstacleCircle` or dicts
- `obstacle_window_m`: along-track activation window
- `obstacle_clearance_m`: extra safety margin
- `obstacle_use_slack`: allow slack variables (default `False`)
- `obstacle_enforce_midpoints`: enforce midpoints inside NLP (default `False`)
- `obstacle_subsamples_per_segment`: midpoint samples when enabled
- `obstacle_slack_weight`: penalty when slack is enabled
- `vehicle_radius_m`: footprint inflation (default `0.0`)

**Initialization**
- `X_init`, `U_init`: optional warm-starts
- `obstacle_aware_init`: bias initial lateral reference away from obstacles

**Lap type**
- `convergent_lap`: full lap (periodic boundary conditions)

---

## 4) IPOPT Configuration (Environment Variables)

The optimizer reads these environment variables to configure IPOPT:

- `IPOPT_TOL` (default `1e-6`)
- `IPOPT_ACCEPTABLE_TOL` (default `1e-4`)
- `IPOPT_MAX_ITER` (default `1000`)
- `IPOPT_PRINT_LEVEL` (default `5` if verbose, else `0`)
- `IPOPT_LINEAR_SOLVER` (optional)

### Linear solver behavior

If `IPOPT_LINEAR_SOLVER` is **not set**, the optimizer tries **MA57** first, and
falls back to IPOPT’s default solver (typically **MUMPS**) if MA57 is unavailable.

If `IPOPT_LINEAR_SOLVER` **is set**, the optimizer will use it as-is.

---

## 5) Demo Defaults (`run_trajopt_demo.py`)

The demo uses these defaults unless overridden via CLI args or env vars:

- `N = 260` (paper-aligned discretization)
- `ds_m = track_length / N`
- `lambda_u = 1e-3`
- `ux_min = 3.0`
- `obstacle_subsamples_per_segment = 7`
- `obstacle_enforce_midpoints = False`
- `obstacle_use_slack = False`
- `track_buffer_m = 0.0`
- `eps_s = 0.1`, `eps_kappa = 0.05`

Outputs are written to:
```
results/trajectory_optimization/nlp
```

---

## 6) Track Dependence Summary

**Likely to change with track**
- `N` and `ds_m`
- `ux_min` (tight tracks may need lower min speed)
- `obstacle_subsamples_per_segment` (tight/close obstacles may need more)
- `track_buffer_m` (narrow tracks may need smaller buffer)

**Usually stable across tracks**
- `lambda_u`
- `eps_s`, `eps_kappa`

---

## 7) Reference

Paper alignment for discretization and solver choice:
Aggarwal & Gerdes, *Friction-Robust Autonomous Racing Using Trajectory Optimization Over Multiple Models*, IEEE OJCS 2025.
