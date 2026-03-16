# Trajectory Optimization History

This is the trajectory-optimization history summary for the project.
Current FATROP runtime knobs and canonical command are documented in
`docs/FATROP_CONFIG.md`.

## Current status (what to use now)

- **Primary runner:** `experiments/run_fatrop_native_trajopt.py`
- **Active project solver policy:** FATROP for generation + warm-start evaluation (Oval phase)
- **IPOPT status:** retained for diagnostics/reference comparisons only
- **Recommended FATROP config for standalone Oval run (N=120 stable):**
  - `FATROP_PRESET=obstacle_fast`
  - `FATROP_STRUCTURE_DETECTION=auto`
  - `FATROP_EXPAND=0`
  - `FATROP_STAGE_LOCAL_COST=1`
  - `FATROP_DYNAMICS_SCHEME=euler`
  - `FATROP_CLOSURE_MODE=open`
  - `FATROP_MAX_ITER=800`
  - `FATROP_TOL=0.01`
  - `FATROP_ACCEPTABLE_TOL=0.01`
- **Hard-repair pipeline FATROP defaults (current code path):**
  - Source: `data/run_hard_repairs_fatrop.sh`
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
- **Post-projection repair default solver (current code path):**
  - Source: `data/build_postprojection_repairs.py`, `data/run_postprojection_repairs_loop.sh`
  - single supported wrapper entrypoint: `data/run_postprojection_repairs_loop.sh`
  - default is `fatrop` (`POSTPROJ_SOLVER=fatrop`)
  - `ipopt` remains available only as explicit diagnostic override
- **Current FATROP-clean dataset snapshot (Oval):**
  - `data/datasets/Oval_Track_260m_shifts_fatrop_clean`: `31710` episodes (`N=150`)
  - `data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean`: `400` episodes
  - `data/datasets/Oval_Track_260m_repairs_postproj_fatrop_clean`: `1000` episodes
    - completion used mostly FATROP with small IPOPT fallback (`992` FATROP, `8` IPOPT)
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

1. Standalone trajopt experiments: use FATROP profiles documented in `docs/FATROP_CONFIG.md`.
2. Hard-repair dataset generation (active phase): FATROP.
3. Post-projection repair generation (active phase): FATROP default; use IPOPT fallback only when FATROP stalls on individual attempts.
4. Use IPOPT only for explicit diagnostics/smoke checks when needed.
5. Enforce acceptance gates (feasibility, clearance, smoothness) regardless of solver choice.

## Canonical commands

### FATROP standalone run
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_CLOSURE_MODE=open \
FATROP_MAX_ITER=800 FATROP_TOL=0.01 FATROP_ACCEPTABLE_TOL=0.01 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/Oval_Track_260m.mat --N 120
```

### FATROP hard-repair benchmark run
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_SMOOTH_CONTROLS=1 \
FATROP_CLOSURE_MODE=open FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 \
/home/saveas/.conda/envs/DT_trajopt/bin/python data/build_repair_segments.py \
  --map-file maps/Oval_Track_260m.mat --base-laps-dir data/base_laps \
  --output-dir /tmp/hard_repairs_fatrop --num-segments 20 --seed 0 --H 20 \
  --hard-mode --save-every 5 --max-attempts 100 --solver fatrop --no-resume
```

### IPOPT baseline run
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_ipopt_trajopt_demo.py
```
