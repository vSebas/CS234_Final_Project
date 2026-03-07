# Dataset Generation Config (Fix A + Fix B)

This dataset is generated using the staged scripts below. `data/generate_dataset.py`
remains a quick Stage A-only helper; the full pipeline uses base laps + shifts + repairs.

## Current Focus (March 6, 2026)

To unblock DT training/evaluation quickly, current dataset work is:
- **Oval-first**
- **IPOPT-only** for repair generation/fixing
- keep nonlinear dynamics
- avoid solver-stack switching for production dataset jobs

Active post-projection target (Oval):
- target accepted episodes: `600`
- current shard: `data/datasets/Oval_Track_260m_repairs_postproj`
- recommended resumable command:
  - `TOTAL_TARGET=600 SINGLE_MAP_CAP=0 ./data/run_postprojection_repairs_loop.sh`

Why `SINGLE_MAP_CAP=0`:
- default safety cap can stop single-map runs early
- setting to `0` disables that cap for intentional Oval-only completion

### Oval-only training expansion plan (current)

For current DT training iterations, use only:
- `Oval_Track_260m` (no obstacles)
- `Oval_Track_260m` with obstacles

Expanded recovery-data targets for this Oval-only phase:
- hard repairs (`data/datasets/Oval_Track_260m_repairs_hard`): target `400`
- post-projection repairs (`data/datasets/Oval_Track_260m_repairs_postproj`): target `1000`

Resumable commands:
- hard repairs (FATROP smooth-controls profile at `N=120`):
  - `PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_SMOOTH_CONTROLS=1 FATROP_CLOSURE_MODE=open FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 /home/saveas/.conda/envs/DT_trajopt/bin/python data/build_repair_segments.py --map-file maps/Oval_Track_260m.mat --base-laps-dir data/base_laps --output-dir data/datasets/Oval_Track_260m_repairs_hard --num-segments 400 --seed 0 --H 20 --hard-mode --save-every 10 --solver fatrop --resume`
- post-projection repairs:
  - `TOTAL_TARGET=1000 SINGLE_MAP_CAP=0 ./data/run_postprojection_repairs_loop.sh`

Current solver defaults for this phase:
- hard-repair generation: FATROP (`data/run_hard_repairs_fatrop.sh` or `--solver fatrop`)
- post-projection generation: IPOPT by default (`POSTPROJ_SOLVER=ipopt`)
- optional override for post-proj:
  - `POSTPROJ_SOLVER=fatrop ./data/run_postprojection_repairs.sh`

**Scripts**
- Build base laps (no obstacles + obstacle scenarios):
  - `data/build_base_laps.py`
- Generate shift episodes from base laps:
  - `data/make_shift_episodes.py`
- Build repair segments (Fix B):
  - `data/build_repair_segments.py`
- Orchestrate all steps:
  - `data/build_dataset.py`

**Defaults (current dataset run)**
- `N = 120`, `ds_m = track_length / N`
- `lambda_u = 0.005`
- `ux_min = 0.5`
- `track_buffer_m = 0.0`
- `eps_s = 0.1`
- `eps_kappa = 0.05`

**Fix A (base laps + shifts)**
- Base laps per track:
  - no-obstacle: `B_no = 6`
  - obstacle: `B_obs = 8`
- Shift episodes per base lap: up to `N` unique shifts
- `--all-shifts` behavior: generates `N+1` shifts (k0=0..N) so the final episode ends exactly at the start node; this includes one duplicate shift due to periodic closure.
- Obstacle sampling defaults:
  - count: `min_obstacles=1`, `max_obstacles=4`
  - radius: `0.8–1.5 m`
  - margin: `0.3 m`
  - clearance: `0.3 m`

**Fix B (repair segments)**
- Horizon: `H = 20`
- Perturbations: `e0` ±1.0 m, `dpsi0` ±0.10 rad
- Terminal anchor: `terminal_weight = 5.0` on `(ux, uy, r, e, dpsi)`
- Conservative IPOPT defaults: `tol = 1e-5`, `acceptable_tol = 1e-3`, `max_iter = 100`
- Mixed with/without obstacles (50/50 recommended)
- Total repairs: `N_rep = 500` for the current run (split evenly across 6 tracks → ~83–84 each)

**Outputs**
- Base laps: `data/base_laps/<map_id>/*.npz`
- Shifts: `data/datasets/<map_id>_shifts/episodes/*.npz`
- Repairs: `data/datasets/<map_id>_repairs/episodes/*.npz`
- Hard repairs: `data/datasets/<map_id>_repairs_hard/episodes/*.npz` (optional targeted shard)
- Post-projection repairs: `data/datasets/<map_id>_repairs_postproj/episodes/*.npz` (optional DAGGER-lite shard)

**Run provenance for the current generated dataset**
- The canonical aggregate build log is:
  - `data/datasets/progress.log`
- `progress.log` is the best single place to verify the final dataset because the current dataset was assembled over multiple resumed runs.
- The detailed shell logs for those resumed runs are stored in:
  - `results/dataset_runs/run_20260302_103944.log`
  - `results/dataset_runs/run_20260302_142801.log`
  - `results/dataset_runs/run_20260302_145210.log`
- Interpretation for the current dataset:
  - `run_20260302_103944.log`: first full multi-track resumed build, including all shift generation and the earlier repair pass with `H = 50`
  - `run_20260302_142801.log`: resumed repair regeneration after switching to the final `H = 20`
  - `run_20260302_145210.log`: follow-up resumed repair pass that finished additional tracks at `H = 20`
  - `data/datasets/progress.log`: final stitched record showing the complete accepted repair counts and the final completion timestamp on March 2, 2026 at `15:18:43`
- The empty `results/dataset_runs/run_20260302_151336.log` is not useful provenance for the current dataset and can be ignored.
- Practical rule:
  - use `data/datasets/progress.log` to answer "did the current dataset finish successfully?"
  - use `results/dataset_runs/run_*.log` when you need the exact shell commands and intermediate resume history that produced the current shard contents

**Saved schema used by training code**
- Node-aligned arrays (`N+1` for shifts, `H+1` for repairs):
  - `s_m`, `X_full`, `U`, `pos_E`, `pos_N`, `yaw_world`, `kappa`, `half_width`, `grade`, `bank`
- Transition-aligned arrays (`N` for shifts, `H` for repairs):
  - `dt`, `reward`, `rtg`
- Episode metadata is stored in `manifest.jsonl` with:
  - `episode_id`, `episode_type`, `base_id`, `map_id`, `map_hash`, `solver_config`, `solver_config_hash`, `obstacles`, `s_offset_m`, `npz_path`

This is the current canonical on-disk schema for the repo. The DT loader in `dt/dataset.py` reads these field names directly.

## Hard-Repair Workflow

The repo now also supports a separate hard-repair workflow for obstacle-focused
recovery data. This does not replace the existing dataset; it adds a new shard.

**Scripts**
- Build hotspot anchors from the DT diagnostic eval CSVs:
  - `data/build_hotspot_json.py`
- Generate hard repairs:
  - `data/build_repair_segments.py --hard-mode`
- Orchestrate hard repairs:
  - `data/build_dataset.py --hard-repair-segments ...`
- Convenience wrapper:
  - `./data/run_full_dataset.sh hard_repairs`
  - `./data/run_hard_repairs_fatrop.sh` (experimental FATROP backend)

**Current hard-repair design**
- separate shard:
  - `data/datasets/<map_id>_repairs_hard`
- biased starts toward:
  - low-clearance / near-obstacle states
  - optional hotspot `s` regions from the diagnostic obstacle runs
  - current hotspot JSONs are anchor points, not a continuous heatmap
- perturb mainly:
  - `e`
  - `dpsi`
- add smaller / rarer:
  - `uy`
  - `r`
- mixed horizons in hard mode:
  - default `20,40,60`
  - default probabilities `0.6,0.25,0.15`
- per-episode hardness metadata is written into:
  - manifest `metadata`
  - `.npz` field `metadata_json`

**Current first-pass sizing**
- target about `1200` hard repairs total across all tracks
- with the current mixed-horizon default, this is about `3%` of the current dataset by transition count

**Current intended first training mix**
- `75%` shifts
- `10%` existing repairs
- `15%` hard repairs
- validation should remain unweighted / natural after the split

**Current command path**
- build symmetric hotspot anchors for all tracks:
  - `./data/build_all_hotspots.sh`
- build the all-track hard-repair shard:
  - `./data/run_hard_repairs.sh`
- build the all-track hard-repair shard with FATROP:
  - `./data/run_hard_repairs_fatrop.sh`
- or use the broader dataset wrapper:
  - `./data/run_full_dataset.sh hard_repairs`

FATROP hard-repair note:
- current repair formulation requires `FATROP_STRUCTURE_DETECTION=none` for robustness
- in current tests on Oval (`H=20`, hard mode), FATROP was slower than IPOPT overall
  for accepted hard-repair generation, so IPOPT remains the default production path

Hotspot note:
- the current hotspot files store a small set of per-track anchor `s` positions
- they are derived from the bad obstacle benchmark scenarios and weighted by DT rollout difficulty
- they are useful as a first-pass sampling bias for hard repairs
- they are not yet true per-step projection/fallback heatmaps

## Post-Projection Repair Workflow

This workflow labels DT+wrapper-induced states with short repair solves and stores
them in a dedicated shard.

**Scripts**
- Build labels directly from rollout trace JSONL:
  - `data/build_postprojection_repairs.py`
- Convenience wrapper:
  - `./data/run_postprojection_repairs.sh`

**Trace input**
- Export trace rows using:
  - `python -u experiments/eval_warmstart.py ... --export-rollout-trace`
- Input rows must contain:
  - `map_file`, `s_m`, `x_after_projection`, `obstacles`

**Current defaults**
- target size:
  - `600` accepted repairs total (or `--per-map-target`)
- horizons:
  - base `H=20`
  - optional long horizon `H=40` at `20%` probability
- trigger filter:
  - uses rows marked `triggered=true` by default

**Output shard**
- `data/datasets/<map_id>_repairs_postproj`
- manifest `episode_type`:
  - `repair_postproj`

**Current status (March 4, 2026)**
- Current generated post-projection shard is effectively single-map:
  - `data/datasets/Oval_Track_260m_repairs_postproj`
- Other track post-projection shards are not populated yet because current trace inputs are Oval-focused.
