# Dataset Generation Config

This file is the canonical status/config for dataset generation used by DT training.

## Current Focus (March 7, 2026)

Current active phase is **Oval-only training iteration**:
- maps in active training loop:
  - `Oval_Track_260m` (no obstacles)
  - `Oval_Track_260m` (with obstacles)
- keep nonlinear dynamics stack unchanged
- solver policy for this phase:
  - hard-repair generation: `FATROP`
  - post-projection generation: `IPOPT` (default)

Current Oval recovery shards:
- `data/datasets/Oval_Track_260m_repairs_hard`: `416` accepted episodes (target `400`, phase complete)
- `data/datasets/Oval_Track_260m_repairs_postproj`: `602` accepted episodes (target `1000`, in progress)

Resumable commands:
- hard repairs (FATROP):
  - `PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_SMOOTH_CONTROLS=1 FATROP_CLOSURE_MODE=open FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 /home/saveas/.conda/envs/DT_trajopt/bin/python data/build_repair_segments.py --map-file maps/Oval_Track_260m.mat --base-laps-dir data/base_laps --output-dir data/datasets/Oval_Track_260m_repairs_hard --num-segments 400 --seed 0 --H 20 --hard-mode --save-every 10 --solver fatrop --resume`
- post-projection repairs (IPOPT default):
  - `TOTAL_TARGET=1000 SINGLE_MAP_CAP=0 ./data/run_postprojection_repairs_loop.sh`

Why `SINGLE_MAP_CAP=0`:
- default cap can stop intentional single-map runs early
- setting `0` disables the cap for this Oval-only completion phase

## Why Post-Projection Repairs

Post-projection repairs are included because DT rollout states are not always on the clean expert manifold:
- wrapper projection/fallback events indicate distribution shift at inference
- post-projection shards label those difficult states with short optimizer teacher solves
- this directly targets the failure mode that pure shift/repair data does not cover well

In this phase:
- hard repairs provide broad recovery coverage
- post-projection repairs provide wrapper-triggered off-manifold coverage

## Important Scope Clarification

Multi-track data already exists in this repo and is not deleted:
- shift shards for multiple tracks under `data/datasets/*_shifts`
- standard repair shards for multiple tracks under `data/datasets/*_repairs`

Current training focus is still Oval-only by choice (fast iteration), not because other-track data is missing.

## Shard Inventory (Current)

Episode counts below are read from each shard `manifest.jsonl`.

| Shard | Episodes |
|---|---:|
| `Oval_Track_260m_shifts` | 1694 |
| `Oval_Track_260m_repairs` | 84 |
| `Oval_Track_260m_repairs_hard` | 416 |
| `Oval_Track_260m_repairs_postproj` | 602 |
| `TRACK1_280m_shifts` | 1694 |
| `TRACK1_280m_repairs` | 84 |
| `TRACK1_280m_repairs_hard` | 200 |
| `TRACK2_shifts` | 1694 |
| `TRACK2_repairs` | 83 |
| `TRACK2_repairs_hard` | 200 |
| `TRACK3_300m_shifts` | 1694 |
| `TRACK3_300m_repairs` | 83 |
| `TRACK3_300m_repairs_hard` | 200 |
| `TRACK4_315m_shifts` | 1694 |
| `TRACK4_315m_repairs` | 83 |
| `TRACK4_315m_repairs_hard` | 200 |
| `TRACK5_330m_shifts` | 1694 |
| `TRACK5_330m_repairs` | 83 |
| `TRACK5_330m_repairs_hard` | 200 |

Current totals by family:
- shifts: `10164`
- repairs: `500`
- hard repairs: `1416`
- post-projection repairs: `602`

## How These Numbers Were Produced (and Why)

This section explains the current counts from generation settings and phase history.

### 1) Shift shards (`*_shifts`): `1694` per map

How generated:
- base laps per map:
  - no-obstacle base laps: `6`
  - obstacle base laps: `8`
- total base laps per map: `14`
- shifts generated with all offsets (`--all-shifts`) at `N=120`, which yields `N+1 = 121` shifts per base lap

Count math:
- `14 * 121 = 1694` shifts per map
- across 6 maps: `1694 * 6 = 10164`

Why this size:
- gives large low-cost coverage from a small number of expensive full-lap solves
- keeps map diversity from all 6 track geometries in the base dataset

### 2) Standard repair shards (`*_repairs`): total `500`

How generated:
- target repairs were set to `500` total in the multi-track run
- split approximately evenly across 6 maps

Observed counts:
- one map has `84`, five maps have `83` (sum `500`)

Why this size:
- enough to add recovery samples without dominating shift data
- keeps the base dataset mostly trajectory-following with limited off-manifold correction examples

### 3) Hard-repair shards (`*_repairs_hard`): total `1416`

How generated:
- earlier all-track hard-repair phase targeted `1200` total:
  - `200` per map across 6 maps
- later Oval-focused phase resumed and expanded only Oval hard repairs to support faster local iteration

Observed counts:
- non-Oval maps: `200` each
- Oval: `416`
- total: `200*5 + 416 = 1416`

Why this size:
- preserve all-track hard-repair coverage from earlier runs
- add extra Oval hard-repair density because current training/eval loop is intentionally Oval-first

### 4) Post-projection shard (`*_repairs_postproj`): `602` (Oval only)

How generated:
- from DT rollout trace rows (`experiments/eval_warmstart.py --export-rollout-trace`)
- labeled by short optimizer repair solves (`data/build_postprojection_repairs.py`)
- currently only Oval traces have been used in this phase

Observed count:
- `Oval_Track_260m_repairs_postproj = 602`

Why this number:
- this is an in-progress checkpoint toward the current target (`1000`)
- current phase is prioritizing fast Oval iteration before expanding post-proj labeling to other maps

## Dataset Scripts

Core scripts:
- base laps: `data/build_base_laps.py`
- shifts: `data/make_shift_episodes.py`
- repairs: `data/build_repair_segments.py`
- post-projection repairs: `data/build_postprojection_repairs.py`
- full wrapper/orchestration: `data/build_dataset.py`

Convenience wrappers:
- hard repairs: `data/run_hard_repairs_fatrop.sh`
- post-projection: `data/run_postprojection_repairs.sh`
- post-projection loop-to-target: `data/run_postprojection_repairs_loop.sh`

## Default Solver/Trajectory Settings

Current common settings used in the active phase:
- `N = 120`
- `H = 20` (base short-horizon repair)
- `lambda_u = 0.005`
- `ux_min = 0.5`
- `track_buffer_m = 0.0`
- `eps_s = 0.1`
- `eps_kappa = 0.05`

Post-projection defaults:
- base horizon: `H=20`
- optional long horizon: `H=40` at configured probability
- trigger filter uses trace rows marked `triggered=true`

## On-Disk Outputs

Shard patterns:
- shifts: `data/datasets/<map_id>_shifts/episodes/*.npz`
- repairs: `data/datasets/<map_id>_repairs/episodes/*.npz`
- hard repairs: `data/datasets/<map_id>_repairs_hard/episodes/*.npz`
- post-proj repairs: `data/datasets/<map_id>_repairs_postproj/episodes/*.npz`

Each shard stores `manifest.jsonl` with episode metadata.

## Saved Schema (Consumed by DT Loader)

Node-aligned arrays (`N+1` for shifts, `H+1` for repairs):
- `s_m`, `X_full`, `U`, `pos_E`, `pos_N`, `yaw_world`, `kappa`, `half_width`, `grade`, `bank`

Transition-aligned arrays (`N` for shifts, `H` for repairs):
- `dt`, `reward`, `rtg`

Manifest fields include:
- `episode_id`, `episode_type`, `base_id`, `map_id`, `map_hash`, `solver_config`, `solver_config_hash`, `obstacles`, `s_offset_m`, `npz_path`

This schema is the source of truth for `dt/dataset.py`.

## Provenance Notes

Historical multi-track dataset build logs remain relevant:
- canonical stitched progress: `data/datasets/progress.log`
- resumed shell logs:
  - `results/dataset_runs/run_20260302_103944.log`
  - `results/dataset_runs/run_20260302_142801.log`
  - `results/dataset_runs/run_20260302_145210.log`

Use:
- `progress.log` for final completion/accounting checks
- `results/dataset_runs/*.log` for exact command/run history

## Next Dataset Step (Current)

1. Finish `Oval_Track_260m_repairs_postproj` from `602` to `1000`.
2. Retrain DT on Oval-only shards with updated recovery mix.
3. Re-evaluate downstream benchmark gates.

## Future / Phase 2 (After Oval Iteration)

After Oval-only gate improves sufficiently:
- resume multi-track robustness training using existing non-Oval shards
- optionally generate additional post-proj shards for other tracks
- keep checkpoint-local benchmark artifacts under:
  - `dt/checkpoints/<run>/warmstarts/eval/...`
  - `dt/checkpoints/<run>/warmstarts/viz/...`
