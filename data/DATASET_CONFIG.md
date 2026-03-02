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

**Defaults (paper-aligned Tier 1)**
- `N = 200`, `ds_m = track_length / N`
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
- Horizon: `H = 50`
- Perturbations: `e0` ±0.5 m, `dpsi0` ±0.10 rad
- Terminal anchor: `terminal_weight = 5.0` on `(ux, uy, r, e, dpsi)`
- Mixed with/without obstacles (50/50 recommended)

**Outputs**
- Base laps: `data/base_laps/<map_id>/*.npz`
- Shifts: `data/datasets/<map_id>_shifts/episodes/*.npz`
- Repairs: `data/datasets/<map_id>_repairs/episodes/*.npz`
