# Dataset Generation Config (No Obstacles, Random e0)

This dataset is generated using `data/generate_dataset.py` with the settings below.

**Scenario (Stage A default)**
- Track: `maps/Oval_Track_260m.mat`
- Obstacles: none
- Lap type: **periodic** (full lap) with **circular shifts**
- Randomized: `s_offset` (random start position)

**Optimizer config**
- `N = 120`
- `ds_m = track_length / N`
- `lambda_u = 0.005`
- `ux_min = 0.5`
- `track_buffer_m = 0.0`
- `eps_s = 0.1`
- `eps_kappa = 0.05`

**Dataset size**
- Target episodes: `1000`

**Output**
- Episodes: `data/datasets/oval_no_obstacles_1k/episodes/*.npz`
- Manifest: `data/datasets/oval_no_obstacles_1k/manifest.jsonl`
