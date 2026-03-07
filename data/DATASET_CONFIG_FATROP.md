# Dataset Config (FATROP-Only)

This file is the FATROP-only dataset configuration for the current Oval-first phase.

## Scope

- map: `Oval_Track_260m` only
- solver: `fatrop` only (generation + eval alignment)
- start assumptions:
  - `s0`: sampled along centerline progress
  - `e0=0`, `dpsi0=0`, `uy0=0`, `r0=0`
  - `ux0=5.0 m/s` for rollout/eval policy
- scenario families:
  - no obstacles (fixed case)
  - obstacles (1-4 obstacles, frozen set for benchmark)

## Dataset Size Goal

Primary goal (full trajectories):
- no-obstacle trajectories: ~`900`
- obstacle trajectories: ~`2250`
- total full trajectories: ~`3150`

Concrete target used in this file:
- `N=150` with `--all-shifts` gives `151` trajectories per accepted base lap.
- no-obstacle base laps: `6`  -> `6 * 151 = 906`
- obstacle base laps: `15`    -> `15 * 151 = 2265`
- total full trajectories: `3171`

Obstacle composition target:
- 25% with 1 obstacle
- 30% with 2 obstacles
- 25% with 3 obstacles
- 20% with 4 obstacles

## Mapping to Base-Lap/Shift Pipeline (`N=150`)

With `--all-shifts`, each base lap yields `N+1 = 151` full trajectories.

- no-obstacle base laps: `6`  -> `906` trajectories
- obstacle base laps: `15`    -> `2265` trajectories
- total: `3171`

Rationale:
- keep at least 6 distinct centerline anchor-start base solves in no-obstacle regime
- allocate most coverage to obstacle regimes

## Clean Shard Naming (no mixing with legacy data)

- `data/base_laps_fatrop_clean/Oval_Track_260m/*.npz`
- `data/datasets/Oval_Track_260m_shifts_fatrop_clean`
- `data/datasets/Oval_Track_260m_repairs_fatrop_clean`
- `data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean`
- `data/datasets/Oval_Track_260m_repairs_postproj_fatrop_clean`

## Generation Commands

### 1) Base laps (FATROP, Oval only)

```bash
env FATROP_CLOSURE_MODE=open FATROP_SMOOTH_CONTROLS=1 FATROP_PRESET=obstacle_fast \
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u data/build_base_laps.py \
  --map-files maps/Oval_Track_260m.mat \
  --output-dir data/base_laps_fatrop_clean \
  --solver fatrop \
  --N 150 \
  --ux-min 5.0 \
  --base-laps 6 \
  --obstacle-laps 15 \
  --min-obstacles 1 \
  --max-obstacles 4 \
  --seed 0 \
  --resume
```

### 2) Shift episodes

```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u data/make_shift_episodes.py \
  --map-file maps/Oval_Track_260m.mat \
  --base-laps-dir data/base_laps_fatrop_clean \
  --output-dir data/datasets/Oval_Track_260m_shifts_fatrop_clean \
  --all-shifts \
  --seed 0 \
  --resume
```

### 3) Standard repairs (optional small shard first)

```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u data/build_repair_segments.py \
  --map-file maps/Oval_Track_260m.mat \
  --base-laps-dir data/base_laps_fatrop_clean \
  --output-dir data/datasets/Oval_Track_260m_repairs_fatrop_clean \
  --num-segments 300 \
  --H 20 \
  --ux-min 5.0 \
  --solver fatrop \
  --seed 0 \
  --resume
```

### 4) Hard repairs

```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u data/build_repair_segments.py \
  --map-file maps/Oval_Track_260m.mat \
  --base-laps-dir data/base_laps_fatrop_clean \
  --output-dir data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean \
  --num-segments 400 \
  --H 20 \
  --hard-mode \
  --ux-min 5.0 \
  --solver fatrop \
  --solve-timeout-s 12 \
  --seed 0 \
  --resume
```

### 5) Post-projection repairs (FATROP clean shard)

```bash
TRACE_JSONL="dt/checkpoints/*/warmstarts/eval/*/warmstart_eval_*_rollout_trace.jsonl" \
OUTPUT_SUFFIX=repairs_postproj_fatrop_clean \
TOTAL_TARGET=1000 SINGLE_MAP_CAP=0 \
./data/run_postprojection_repairs_loop.sh
```

## Notes

- Keep this file as the source of truth for FATROP-only generation.
- Keep `data/DATASET_CONFIG.md` as broader/historical context.
- Current best robust config for this generation plan:
  - `FATROP_CLOSURE_MODE=open`
  - `FATROP_SMOOTH_CONTROLS=1`
  - `FATROP_STRUCTURE_DETECTION=auto`
