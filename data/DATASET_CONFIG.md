# Dataset Generation Config (Fix A + Fix B)

This dataset is generated using the staged scripts below. `data/generate_dataset.py`
remains a quick Stage A-only helper; the full pipeline uses base laps + shifts + repairs.

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
- or use the broader dataset wrapper:
  - `./data/run_full_dataset.sh hard_repairs`

Hotspot note:
- the current hotspot files store a small set of per-track anchor `s` positions
- they are derived from the bad obstacle benchmark scenarios and weighted by DT rollout difficulty
- they are useful as a first-pass sampling bias for hard repairs
- they are not yet true per-step projection/fallback heatmaps
