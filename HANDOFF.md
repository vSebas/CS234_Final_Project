# Handoff (Fresh Session)

## Repo
- Path: `/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project`

## Current canonical trajopt path
- FATROP runner: `experiments/run_fatrop_native_trajopt.py`
- FATROP config doc: `docs/FATROP_CONFIG.md`
- History summary: `docs/TRAJOPT_HISTORY.md`

## What was changed recently
- Removed old FATROP helper/benchmark scripts and kept one canonical FATROP runner.
- Added FATROP support into hard-repair generation:
  - `data/build_repair_segments.py` now has `--solver {ipopt,fatrop}`.
  - `data/build_dataset.py` now has `--hard-repair-solver {ipopt,fatrop}`.
  - Added wrapper `data/run_hard_repairs_fatrop.sh`.
- FATROP runner updated to support repair-mode hooks:
  - `x0` (masked initial-state constraints)
  - `s0_offset_m`
  - `terminal_state` + `terminal_mask` + `terminal_weight`

## Key benchmark results (hard repairs)

### Baseline IPOPT
Command:
```bash
time /home/saveas/.conda/envs/DT_trajopt/bin/python data/build_repair_segments.py \
  --map-file maps/Oval_Track_260m.mat \
  --base-laps-dir data/base_laps \
  --output-dir /tmp/hard_repairs_ipopt_v2 \
  --num-segments 20 --seed 0 --H 20 --hard-mode \
  --save-every 5 --max-attempts 100 --solver ipopt --no-resume
```
Result:
- accepted `20/20`
- attempts `35/100`
- elapsed `194.5s` (real `3m21s`)

### FATROP (script settings with auto structure + smooth controls)
Command:
```bash
time PYTHONPATH=. \
  FATROP_PRESET=obstacle_fast \
  FATROP_STRUCTURE_DETECTION=auto \
  FATROP_EXPAND=0 \
  FATROP_STAGE_LOCAL_COST=1 \
  FATROP_DYNAMICS_SCHEME=euler \
  FATROP_SMOOTH_CONTROLS=1 \
  FATROP_CLOSURE_MODE=open \
  FATROP_MAX_ITER=800 \
  FATROP_TOL=5e-3 \
  FATROP_ACCEPTABLE_TOL=5e-3 \
  /home/saveas/.conda/envs/DT_trajopt/bin/python data/build_repair_segments.py \
  --map-file maps/Oval_Track_260m.mat \
  --base-laps-dir data/base_laps \
  --output-dir /tmp/hard_repairs_fatrop_v2 \
  --num-segments 20 --seed 0 --H 20 --hard-mode \
  --save-every 5 --max-attempts 100 --solver fatrop --no-resume
```
Result:
- accepted `0/20`
- attempts `100/100`
- elapsed `31.2s` (fast failure)
- repeated FATROP structure warnings:
  - "Constraint found depending on a state of the previous interval."

### FATROP (robust fallback mode tested earlier)
- `FATROP_STRUCTURE_DETECTION=none` can run and accept repairs, but was slower than IPOPT in tested setup.

## Main issue
- Historical note: an earlier run showed FATROP `auto` structural failures in hard-repair mode.
- Latest verification on 2026-03-06 (same benchmark shape) succeeded with FATROP `auto` + smooth-controls profile.
- Keep validating across maps/seeds; do not assume one map benchmark fully generalizes.

## Recommended next steps (highest ROI)
1. Run controlled multi-map/multi-seed A/B with the current FATROP hard-repair profile.
2. Track acceptance rate, attempts/accept, and wall-clock vs IPOPT.
3. Keep IPOPT fallback path available until FATROP results are stable across tracks.

## Useful commands

Canonical FATROP oval run:
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_CLOSURE_MODE=open \
FATROP_MAX_ITER=800 FATROP_TOL=0.01 FATROP_ACCEPTABLE_TOL=0.01 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/Oval_Track_260m.mat --N 120
```

IPOPT comparison run (standalone):
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python run_trajopt_demo.py \
  --map-file maps/Oval_Track_260m.mat --n 120 \
  --output-dir results/trajectory_optimization/nlp_compare
```
