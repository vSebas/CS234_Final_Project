# FATROP Config (Canonical)

This file documents the active FATROP runner and the current code-level FATROP
profiles used in this repo.

## Current policy (March 2026)

In the active Oval-first data loop:
- hard repairs: FATROP (`data/run_hard_repairs_fatrop.sh`, `--solver fatrop`)
- post-projection repairs: FATROP by default (`POSTPROJ_SOLVER=fatrop` in `data/run_postprojection_repairs.sh`)

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

## Profile A: canonical N=120 trajectory run (Oval)

```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_CLOSURE_MODE=open \
FATROP_MAX_ITER=800 FATROP_TOL=0.01 FATROP_ACCEPTABLE_TOL=0.01 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/Oval_Track_260m.mat --N 120
```

Notes:
- `FATROP_SMOOTH_CONTROLS` is not set above, so runner default applies (`0`).
- This profile is used for standalone full-lap FATROP trajectory tests.

## Profile B: best-quality full-lap trajectory (all 6 tracks, N150_smooth results)

Used to generate `results/trajectory_optimization/fatrop/*/N150_smooth/` on 2026-03-05.
File used: `experiments/run_fatrop_native_trajopt.py` (was `run_fatrop_native_trajopt_v2.py` before consolidation commit b5a26cc).

```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_SMOOTH_CONTROLS=1 \
FATROP_CLOSURE_MODE=open FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/<TRACK>.mat --N 150
```

Notes:
- `FATROP_SMOOTH_CONTROLS=1`: lifts (δ, Fx) into state, uses arc-length rates as controls.
- Warm start: ux=10 m/s, Fx=0.5 kN (over-thrust); obstacle-aware lateral offset init.
- Oval no-obstacle: cost=14.40s, ~7.5s solve (10× faster than IPOPT at 14.43s/78s).
- Oval with obstacles: cost=16.94s, ~4s solve (15× faster than IPOPT at 14.70s/61s).

## Profile C: hard-repair pipeline defaults (from current script)

Source of truth: `data/run_hard_repairs_fatrop.sh`

Current defaults in that script:
- `FATROP_PRESET=obstacle_fast`
- `FATROP_STRUCTURE_DETECTION=auto`
- `FATROP_EXPAND=0`
- `FATROP_STAGE_LOCAL_COST=1`
- `FATROP_DYNAMICS_SCHEME=euler`
- `FATROP_SMOOTH_CONTROLS=1`
- `FATROP_CLOSURE_MODE=open`
- `FATROP_MAX_ITER=800`
- `FATROP_TOL=5e-3`
- `FATROP_ACCEPTABLE_TOL=5e-3`

Equivalent one-off command form for hard-repair benchmarking:

```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_SMOOTH_CONTROLS=1 \
FATROP_CLOSURE_MODE=open FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 \
/home/saveas/.conda/envs/DT_trajopt/bin/python data/build_repair_segments.py \
  --map-file maps/Oval_Track_260m.mat --base-laps-dir data/base_laps \
  --output-dir /tmp/hard_repairs_fatrop --num-segments 20 --seed 0 --H 20 \
  --hard-mode --save-every 5 --max-attempts 100 --solver fatrop --no-resume
```

## Latest verification run (Profile A — N=120)

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
- Source-of-truth defaults for post-proj solver are in:
  - `data/build_postprojection_repairs.py` (`--solver` default `fatrop`)
  - `data/run_postprojection_repairs.sh` (`POSTPROJ_SOLVER` default `fatrop`)
