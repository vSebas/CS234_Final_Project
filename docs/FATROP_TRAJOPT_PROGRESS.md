# FATROP TrajOpt Progress (vs IPOPT)

## Current status
- FATROP is implemented as a standalone runner in [`experiments/run_fatrop_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final%20Project/CS234_Final_Project/experiments/run_fatrop_trajopt.py).
- Core planner path remains IPOPT-based.
- Script now uses stage-wise decision variables (`x[k]`, `u[k]`) and stage-wise `ng[k]` bookkeeping.
- FATROP manual structure metadata is now supported in solver options (`nx`, `nu`, `ng`, `N`).
- FATROP now defaults to a `fast` preset:
  - `fatrop.mu_init=0.2`
  - `fatrop.tol=1e-4`
  - `fatrop.acceptable_tol=1e-3`
  - `fatrop.print_level=0`
  - `structure_detection=none`
  - `expand=True`
- Presets:
  - `FATROP_PRESET=fast` (default)
  - `FATROP_PRESET=obstacle_fast`
  - `FATROP_PRESET=balanced`
  - `FATROP_PRESET=accurate`
- Structure default is currently `FATROP_STRUCTURE_DETECTION=none`.
- `manual` is still available but can trigger structure warnings with current lap-closure constraint formulation.

## Reference repos checked
- [`docs/reference_repos/fatrop_demo-master`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final%20Project/CS234_Final_Project/docs/reference_repos/fatrop_demo-master)
- [`docs/reference_repos/fatrop-main/examples`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final%20Project/CS234_Final_Project/docs/reference_repos/fatrop-main/examples)

Key pattern from both repos:
- Best-practice setup uses `structure_detection='manual'` with explicit stage metadata (`nx`, `nu`, `ng`, `N`).

## Latest Oval comparison
Command:
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python \
  experiments/run_fatrop_trajopt.py --map-file maps/Oval_Track_260m.mat --N 40 --compare-ipopt
```

Observed (latest, fast preset):
- IPOPT: `success=True`, `iterations=98`, `cost=15.420789`, `solve_time=7.732s`
- FATROP (`structure_detection=manual`): `success=True`, `iterations=67`, `cost=15.665454`, `solve_time=8.166s`

## Latest Oval+Obstacle comparison
Command:
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=manual \
  /home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_trajopt.py \
  --map-file maps/Oval_Track_260m_Obstacles.mat --N 40 --compare-ipopt
```

Observed:
- IPOPT: `success=True`, `iterations=133`, `cost=15.490398`, `solve_time=8.738s`
- FATROP (`obstacle_fast`): `success=True`, `iterations=56`, `cost=15.781586`, `solve_time=6.482s`

Re-check with current defaults:
- IPOPT: `success=True`, `solve_time=8.968s`
- FATROP (`FATROP_PRESET=obstacle_fast`, default structure mode): `success=True`, `solve_time=6.411s`

## Interpretation
- FATROP is working and returns feasible solutions.
- Runtime is now close to IPOPT on no-obstacle Oval and can be faster on obstacle Oval with tuned `mu_init`.
- Tradeoff: FATROP converges faster in iterations with slightly higher objective value.
- Main remaining gap is formulation/option parity, not missing structure metadata.

## Next step to close gap
1. Benchmark the same setup on obstacle maps and larger horizons (`N>=80`) to see if the speed gap widens or narrows.
2. Sweep a narrow set of solver params around the current fast preset for obstacle cases.
3. If needed, move to explicit stage path-function definitions (closer to FATROP examples) for additional speed.
