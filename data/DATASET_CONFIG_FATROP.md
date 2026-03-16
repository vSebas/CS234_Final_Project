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

Current goal (expanded clean full trajectories):
- no-obstacle trajectories: ~`3000`
- obstacle trajectories: ~`28000`
- total full trajectories: ~`30000`

Concrete target used now:
- `N=150` with `--all-shifts` gives `151` trajectories per accepted base lap.
- no-obstacle base laps: `20`  -> `20 * 151 = 3020`
- obstacle base laps: `190`    -> `190 * 151 = 28690`
- total full trajectories: `31710`

Obstacle composition target:
- 25% with 1 obstacle
- 30% with 2 obstacles
- 25% with 3 obstacles
- 20% with 4 obstacles

## Mapping to Base-Lap/Shift Pipeline (`N=150`)

With `--all-shifts`, each base lap yields `N+1 = 151` full trajectories.

- no-obstacle base laps: `20`  -> `3020` trajectories
- obstacle base laps: `190`    -> `28690` trajectories
- total: `31710`

Rationale:
- keep many centerline-anchor no-obstacle starts for nominal policy quality
- allocate most coverage to obstacle regimes for robustness

## Clean Shard Naming (no mixing with legacy data)

- `data/base_laps_fatrop_clean/Oval_Track_260m/*.npz`
- `data/datasets/Oval_Track_260m_shifts_fatrop_clean`
- `data/datasets/Oval_Track_260m_repairs_fatrop_clean`
- `data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean`
- `data/datasets/Oval_Track_260m_repairs_postproj_fatrop_clean`

## Current Generated Snapshot (as of 2026-03-10)

- base laps (`data/base_laps_fatrop_clean/Oval_Track_260m`):
  - total: `210`
  - no-obstacle: `20`
  - obstacle: `190`
- shift shard (`data/datasets/Oval_Track_260m_shifts_fatrop_clean`):
  - total episodes: `31710`
  - `N` distribution: `{150: 31710}`
  - obstacle-count distribution: `{0: 906, 1: 4379, 2: 9060, 3: 10570, 4: 6795}`
- hard-repair shard (`data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean`):
  - total episodes: `400`
  - horizon distribution: `{20: 250, 40: 83, 60: 67}`
  - obstacle-count distribution: `{0: 117, 1: 13, 3: 199, 4: 71}`
  - sampling reason distribution: `obstacle=258`, `uniform=142`
  - solver iterations (min/median/p90/max): `5 / 39 / 153 / 768`
- post-projection clean shard (`data/datasets/Oval_Track_260m_repairs_postproj_fatrop_clean`):
  - total episodes: `1000`
  - obstacle-count distribution: `{1: 191, 2: 197, 3: 194, 4: 418}`
  - horizon distribution: `{20: 933, 40: 67}`
  - solver mix from manifest solver_config: `{fatrop: 992, ipopt: 8}`
  - integrity check status: no missing/corrupt files; all rows `solver_success=True`

Current usage note:
- this post-proj shard is complete for the current Oval phase target (`1000`).
- do not append new post-proj rows until after benchmark verdict on current improved checkpoints.

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
  --base-laps 20 \
  --obstacle-laps 190 \
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

### 5) Train first clean model (required before post-projection)

Use only clean FATROP shards in the training manifest mix (shifts + hard repairs first).
After training, export warmstart rollout traces from that checkpoint.

### 6) Post-projection repairs (from clean-model traces)

```bash
TRACE_JSONL="dt/checkpoints/<clean_run>/warmstarts/eval/<eval_tag>/warmstart_eval_*_rollout_trace.jsonl" \
OUTPUT_SUFFIX=repairs_postproj_fatrop_clean \
TOTAL_TARGET=1000 SINGLE_MAP_CAP=0 \
./data/run_postprojection_repairs_loop.sh
```

Important:
- Do not build post-projection repairs from legacy/non-clean checkpoints.
- For this phase, keep trace scenarios in the same Oval family (`no-obstacle` and `1-4 obstacle` cases) to avoid distribution mismatch.

## Notes

- Keep this file as the source of truth for FATROP-only generation.
- Keep `data/DATASET_CONFIG.md` as broader/historical context.
- Current best robust config for this generation plan:
  - `FATROP_CLOSURE_MODE=open`
  - `FATROP_SMOOTH_CONTROLS=1`
  - `FATROP_STRUCTURE_DETECTION=auto`
