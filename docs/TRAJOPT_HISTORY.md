# Trajectory Optimization History

This is the trajectory-optimization history summary for the project.
Current FATROP runtime knobs and canonical command are documented in
`docs/FATROP_CONFIG.md`.

## Current status (what to use now)

- **Primary runner:** `experiments/run_fatrop_native_trajopt.py`
- **Primary baseline/reference solver in planner path:** IPOPT via `planning/optimizer.py`
- **Recommended FATROP config (Oval, N=120 stable):**
  - `FATROP_PRESET=obstacle_fast`
  - `FATROP_STRUCTURE_DETECTION=auto`
  - `FATROP_EXPAND=0`
  - `FATROP_STAGE_LOCAL_COST=1`
  - `FATROP_DYNAMICS_SCHEME=euler`
  - `FATROP_CLOSURE_MODE=open`
  - `FATROP_MAX_ITER=800`
  - `FATROP_TOL=0.01`
  - `FATROP_ACCEPTABLE_TOL=0.01`
- **Default FATROP output dir:** `results/trajectory_optimization/fatrop`

## Historical summary

### IPOPT
- IPOPT was the original production trajectory optimizer and dataset-generation solver.
- It remains the quality reference path.
- For dataset generation, `ux_min` in saved manifests is `0.5` (overridden by dataset scripts), while generic optimizer default is higher unless overridden.

### FATROP evolution
- Early FATROP experiments used multiple scripts and benchmarking helpers.
- Main failures at high horizon were caused by formulation/structure issues:
  - non-interleaved stage variable creation,
  - boundary constraint placement interfering with stage structure detection,
  - closure couplings breaking banded assumptions.
- The consolidated runner fixed these and enabled reliable `N=120` solves.
- Current tradeoff: strong speedups in solved cases, quality depends on formulation parity and acceptance checks.

### MadNLP / Rockit tracks
- MadNLP and Rockit+FATROP were explored experimentally.
- They are not the active path now and are kept as historical context only.

## Decision policy used now

1. Use FATROP for fast candidate generation.
2. Enforce acceptance gates (feasibility, clearance, smoothness).
3. Use IPOPT fallback for rejected/hard cases when quality is critical.

## Canonical commands

### FATROP run
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_CLOSURE_MODE=open \
FATROP_MAX_ITER=800 FATROP_TOL=0.01 FATROP_ACCEPTABLE_TOL=0.01 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/Oval_Track_260m.mat --N 120
```

### IPOPT baseline run
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python run_trajopt_demo.py
```
