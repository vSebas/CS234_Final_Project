# FATROP Configuration

## Scope
This file lists FATROP-specific runtime knobs used by:
- [`experiments/run_fatrop_native_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_fatrop_native_trajopt.py)
- [`experiments/benchmark_ipopt_vs_fatrop.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/benchmark_ipopt_vs_fatrop.py)
- [`experiments/tune_fatrop_configs.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/tune_fatrop_configs.py)
- [`experiments/benchmark_fatrop_open_lap_ladder.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/benchmark_fatrop_open_lap_ladder.py)

## Preset selection
- `FATROP_PRESET`:
  - `fast` (default)
  - `obstacle_fast`
  - `balanced`
  - `accurate`

Preset values in code (`run_fatrop_native_trajopt.py`):
- `fast`: `mu_init=0.2`, `tol=1e-4`, `acceptable_tol=1e-3`
- `obstacle_fast`: `mu_init=0.3`, `tol=1e-4`, `acceptable_tol=1e-3`
- `balanced`: `mu_init=0.1` (no explicit `tol`/`acceptable_tol` override)
- `accurate`: `mu_init=0.1`, `tol=1e-6`, `acceptable_tol=1e-6`

Override precedence:
- Preset sets defaults first.
- If `FATROP_TOL` / `FATROP_ACCEPTABLE_TOL` are set, they override preset values.

## Core solver environment variables
- `FATROP_MU_INIT`
- `FATROP_TOL`
- `FATROP_ACCEPTABLE_TOL`
- `FATROP_MAX_ITER` (maps to FATROP `max_iter`)
- `FATROP_STAGE_LOCAL_COST` (`1` uses stage-local `||u_k||^2` regularizer; `0` uses cross-stage `||u_{k+1}-u_k||^2`)
- `FATROP_DYNAMICS_SCHEME` (`trapezoidal` or `euler`)
- `FATROP_CLOSURE_MODE` (`open`, `soft`, `hard`)
- `FATROP_CLOSURE_SOFT_WEIGHT` (soft terminal mismatch penalty weight)
- `FATROP_PRINT_LEVEL`
- `FATROP_STRUCTURE_DETECTION` (`none`, `auto`, `manual`)
- `FATROP_EXPAND` (`1`/`0`)
- `FATROP_CONVEXIFY_STRATEGY` (optional)
- `FATROP_CONVEXIFY_MARGIN` (optional)
- `FATROP_DEBUG` (`1` enables extra debug options)

## Structure detection and fallback logic
`run_fatrop_native_trajopt.py` uses a single solver configuration per run:
- `structure_detection` from `FATROP_STRUCTURE_DETECTION`
- when `manual`: provides `nx`, `nu`, `ng`, `N`

Set `FATROP_STAGE_LOCAL_COST=1` when diagnosing FATROP structure/scaling issues.
Set `FATROP_DYNAMICS_SCHEME=euler` to remove dependence on `u_{k+1}` in dynamics (more stage-local, lower fidelity).
Set `FATROP_CLOSURE_MODE=soft` to avoid hard wrap-around equalities while still biasing lap closure.

## Programmatic warm-start support
`solve_fatrop_native(...)` supports optional array initial guesses:
- `X_init`: shape `(8, N+1)`
- `U_init`: shape `(2, N+1)`

If both are provided, they are used directly as solver initial values.
If omitted, the script falls back to the built-in nominal initialization.

## Dedicated FATROP-native formulation
Use [`experiments/run_fatrop_native_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_fatrop_native_trajopt.py) for a cleaner FATROP-first OCP formulation:
- multiple shooting stage-local dynamics
- stage-local constraints
- stage-local control regularization
- closure handling via `open|soft|hard` mode
- prints `solve_time`, `build_time`, and `total_time` in the runner output line

Example command:
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=none FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_CLOSURE_MODE=soft FATROP_CLOSURE_SOFT_WEIGHT=100 \
FATROP_MAX_ITER=800 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/Oval_Track_260m.mat --N 40 --compare-ipopt
```

## CLI benchmarking controls
`experiments/benchmark_ipopt_vs_fatrop.py` supports matched-problem controls:
- `--N`, `--repeats`
- `--ux-min`, `--lambda-u`
- `--eps-s`, `--eps-kappa`
- obstacle toggles and parameters:
  - `--obstacle-clearance-m`
  - `--obs-subsamples`
  - `--obs-enforce-midpoints / --no-obs-enforce-midpoints`
  - `--obs-use-slack / --no-obs-use-slack`
- `--obstacle-window-m`
- `--convergent-lap / --no-convergent-lap`

`experiments/benchmark_fatrop_open_lap_ladder.py` defaults are now:
- `FATROP_DYNAMICS_SCHEME=euler` (for high-`N` native sweeps)
- CSV includes `fatrop_build_time_s` and `fatrop_total_time_s` in addition to `fatrop_solve_time_s`

## Practical config profiles
Use these as reproducible starting points.

Profile A: quick `N=40` obstacle comparison
```bash
export FATROP_PRESET=obstacle_fast
export FATROP_STRUCTURE_DETECTION=none
export FATROP_PRINT_LEVEL=0
export FATROP_EXPAND=0
export FATROP_MAX_ITER=800
export FATROP_STAGE_LOCAL_COST=1
export FATROP_DYNAMICS_SCHEME=trapezoidal
export FATROP_CLOSURE_MODE=soft
export FATROP_CLOSURE_SOFT_WEIGHT=100
```

Profile B: structure probing (diagnostic only)
```bash
export FATROP_PRESET=obstacle_fast
export FATROP_STRUCTURE_DETECTION=manual
export FATROP_PRINT_LEVEL=0
export FATROP_EXPAND=0
export FATROP_DEBUG=1
```

Profile C: tighter-accuracy check
```bash
export FATROP_PRESET=accurate
export FATROP_STRUCTURE_DETECTION=none
export FATROP_PRINT_LEVEL=0
export FATROP_EXPAND=1
export FATROP_MAX_ITER=1200
export FATROP_STAGE_LOCAL_COST=0
export FATROP_DYNAMICS_SCHEME=trapezoidal
```

## What to log for every FATROP run
- map name, `N`, `ds_m`
- preset + all FATROP env variables
- success/failure, iterations, solve time, objective
- min obstacle clearance (if obstacles enabled)
- whether warnings occurred (especially structure-detection warnings)

When using closure reformulation or homotopy:
- closure mode and soft weight (`FATROP_CLOSURE_MODE`, `FATROP_CLOSURE_SOFT_WEIGHT`)
- whether warm-start arrays were used (`X_init/U_init`)

## Current practical note
- At `N=120` on Oval tuning sweep (latest run), all tested preset/structure combinations timed out at 120s:
  - file: `results/solver_benchmarks/fatrop_tune_N120_Oval_Track_260m_20260305_115251.csv`
- This means current FATROP path is not yet a drop-in replacement for production IPOPT at high horizon in this codepath.

Additional supporting run:
- `results/solver_benchmarks/fatrop_quick_N120_20260305_114637.csv`
- both tested configs timed out at 90s.

## Known pain points at high horizon (`N=120`)
- Structure warnings appear in manual mode on obstacle runs:
  - `Constraint found depending on a state of the previous interval.`
- Current formulation includes periodic lap-closure equality at `k=N` coupling back to `k=0`; this can interfere with FATROP stage-structure assumptions in manual mode.
- `FATROP_MAX_ITER` should be set for high-horizon sweeps to avoid long solves.
- There is no confirmed FATROP native max wall-time option in current exposed interface; use process-level timeout in sweep scripts when strict wall-time cutoff is required.

## Best-known stable FATROP config (current)
For reformulated open-lap diagnosis runs:
- `FATROP_PRESET=obstacle_fast`
- `FATROP_STRUCTURE_DETECTION=none`
- `FATROP_EXPAND=0`
- `FATROP_STAGE_LOCAL_COST=1`
- `FATROP_DYNAMICS_SCHEME=trapezoidal`
- `FATROP_CLOSURE_MODE=soft`
- `FATROP_CLOSURE_SOFT_WEIGHT=100`
- `FATROP_MAX_ITER=800`

## Related docs
- Progress log: [`docs/FATROP_TRAJOPT_PROGRESS.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/FATROP_TRAJOPT_PROGRESS.md)
- IPOPT config baseline: [`docs/OPTIMIZER_CONFIG.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/OPTIMIZER_CONFIG.md)
