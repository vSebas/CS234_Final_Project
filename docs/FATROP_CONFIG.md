# FATROP Config (Canonical)

This file documents the active FATROP runner and the current tested configuration.

## Active runner

- `experiments/run_fatrop_native_trajopt.py`

## Default output path

- `results/trajectory_optimization/fatrop`

## Environment knobs used by the runner

- `FATROP_PRESET`: `fast|obstacle_fast|balanced|accurate`
- `FATROP_STRUCTURE_DETECTION`: `none|auto|manual`
- `FATROP_EXPAND`: `0|1`
- `FATROP_STAGE_LOCAL_COST`: `0|1`
- `FATROP_DYNAMICS_SCHEME`: `euler|rk4|trapezoidal`
- `FATROP_SMOOTH_CONTROLS`: `0|1`
- `FATROP_CLOSURE_MODE`: `open|soft|hard`
- `FATROP_CLOSURE_SOFT_WEIGHT`: float
- `FATROP_MAX_ITER`: int
- `FATROP_MU_INIT`: float
- `FATROP_TOL`: float
- `FATROP_ACCEPTABLE_TOL`: float

## Canonical N=120 command (Oval)

```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_CLOSURE_MODE=open \
FATROP_MAX_ITER=800 FATROP_TOL=0.01 FATROP_ACCEPTABLE_TOL=0.01 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/Oval_Track_260m.mat --N 120
```

## Latest verification run

Date: 2026-03-06  
Command: canonical N=120 command above

Observed output:

```text
[fatrop-native] success=True iterations=73 cost=16.936748 solve_time=5.825s build_time=0.511s total_time=6.337s min_clearance=inf scheme=euler smooth=False closure=open
  [trajectory] results/trajectory_optimization/fatrop/fatrop_N120_trajectory.png
  [states] results/trajectory_optimization/fatrop/fatrop_N120_states.png
  [controls] results/trajectory_optimization/fatrop/fatrop_N120_controls.png
```

## Notes

- This is now the canonical FATROP path after removing legacy FATROP/Rockit experiment scripts.
- Historical context and solver-lineage summary are kept in `docs/TRAJOPT_HISTORY.md`.
