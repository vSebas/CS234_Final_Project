# Rockit + FATROP Progress

## Scope
- Dedicated runner: [`experiments/run_rockit_fatrop_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_rockit_fatrop_trajopt.py)
- Backend solver path: [`experiments/run_fatrop_native_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_fatrop_native_trajopt.py)

## Implementation status
- Rockit+FATROP runner is implemented as a standalone script.
- It uses the stage-structured FATROP OCP formulation (multiple-shooting style, stage-local constraints, soft-closure mode by default).
- External `rockit` package is not required by this runner.

## Current defaults used by the runner
- `FATROP_PRESET=obstacle_fast`
- `FATROP_STRUCTURE_DETECTION=none`
- `FATROP_EXPAND=0`
- `FATROP_STAGE_LOCAL_COST=1`
- `FATROP_DYNAMICS_SCHEME=trapezoidal`
- `FATROP_CLOSURE_MODE=soft`
- `FATROP_CLOSURE_SOFT_WEIGHT=100`
- `FATROP_MAX_ITER=800`

## Latest smoke results (2026-03-06)
Command:
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python \
  experiments/run_rockit_fatrop_trajopt.py --map-file maps/Oval_Track_260m.mat --N 40 --compare-ipopt
```
Observed:
- IPOPT: `success=True`, `iterations=233`, `cost=14.509318`, `solve_time=15.333s`
- Rockit+FATROP: `success=True`, `iterations=35`, `cost=17.578371`, `solve_time=4.990s`, `build_time=0.363s`, `total_time=5.353s`

Command:
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python \
  experiments/run_rockit_fatrop_trajopt.py --map-file maps/Oval_Track_260m_Obstacles.mat --N 40 --compare-ipopt
```
Observed:
- IPOPT: `success=True`, `iterations=115`, `cost=14.826745`, `solve_time=10.216s`
- Rockit+FATROP: `success=True`, `iterations=106`, `cost=17.577570`, `solve_time=20.171s`, `build_time=0.437s`, `total_time=20.608s`, `min_clearance=1.8490`

## Interpretation
- Runner is working and reproducible.
- At `N=40`, no-obstacle case is substantially faster than IPOPT.
- At `N=40`, obstacle case is slower than IPOPT and has higher objective.
- This path is currently experimental; IPOPT remains the quality baseline for production trajectory generation.

## Next checks
1. Run matched `N=60/80/120` ladder with the same runner and record solve/build split.
2. Tune obstacle-focused options (`mu_init`, closure soft weight, max_iter) specifically for obstacle maps.
3. Keep checkpointed comparison tables against IPOPT for final solver selection.
