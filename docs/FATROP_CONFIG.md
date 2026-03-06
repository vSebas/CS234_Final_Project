# FATROP Configuration

## Scope
This file lists FATROP-specific runtime knobs used by:
- [`experiments/run_fatrop_native_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_fatrop_native_trajopt.py)
- [`experiments/run_rockit_fatrop_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_rockit_fatrop_trajopt.py)
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

Rockit-style dedicated entrypoint:
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python \
  experiments/run_rockit_fatrop_trajopt.py --map-file maps/Oval_Track_260m.mat --N 40 --compare-ipopt
```
This runner uses the same stage-structured FATROP formulation but keeps a separate CLI for Rockit+FATROP experiments.

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

## Current practical note (updated 2026-03-05)
- `N=120` on Oval solves in ~8s / 93 iter using `run_fatrop_native_trajopt_v2.py`.
- **Best config after fine-tuning: `N=150, tol=5e-3`** — cost=17.14 (-5.1% vs baseline), 87 iter, 13.4s.
- `N=150, tol=0.01` is also a free win over N=120: 8.6s, 56 iter, cost=17.84.
- `N=200` is not worth it: needs tol≤1e-3 to avoid premature exit, then 54s with no gain.
- Previous timeouts at N=120 were caused by two formulation bugs — now fixed in v2.
- `run_fatrop_native_trajopt.py` (v1) retains the old formulation for reference.

## Root causes of previous N=120 timeouts (fixed in v2)

1. **Non-interleaved variable creation** — v1 created all `x_vars` first, then all `u_vars`.
   FATROP auto structure detection assigns stages by variable creation order and expects
   `x_0, u_0, x_1, u_1, ...`. The grouped ordering caused structure detection to fail on
   every stage, forcing FATROP into generic (O(N³)) NLP mode.

2. **Pre-loop boundary condition** — v1 added `opti.subject_to(x_vars[0][5] == 0.0)`
   before the stage loop. This created a "pre-stage 0" constraint that made all subsequent
   dynamics (`x_1 = F(x_0, u_0)`) appear to depend on the previous interval.

3. **Soft/hard closure** — `||x_N - x_0||²` in the objective (soft closure) or
   `x_N == x_0` as a constraint (hard closure) couples stage 0 and stage N, breaking
   FATROP's banded Hessian/Riccati structure. Since periodicity is not required for
   single-lap time minimization, `FATROP_CLOSURE_MODE=open` is both correct and necessary.

4. **Euler integration accuracy** — Demanding `tol=1e-4` fights against the O(ds²) Euler
   discretization error. `tol=0.01` matches the integration accuracy and converges reliably.

## Known pain points at high horizon (`N=120`) — historical, now resolved in v2
- ~~Structure warnings: `Constraint found depending on a state of the previous interval.`~~ fixed
- ~~Periodic lap-closure coupling breaking stage structure~~ fixed (open closure)
- `FATROP_MAX_ITER` should still be set; max allowed value is ~800 (3000 rejected as out of bounds).
- No confirmed FATROP native wall-time limit; use process-level timeout when needed.

## Best-known stable FATROP configs (v2, current)
Script: `experiments/run_fatrop_native_trajopt_v2.py`

### Fast config — dataset generation (N=150, tol=0.01)
```bash
FATROP_PRESET=obstacle_fast
FATROP_STRUCTURE_DETECTION=auto
FATROP_EXPAND=0
FATROP_STAGE_LOCAL_COST=1
FATROP_DYNAMICS_SCHEME=euler
FATROP_CLOSURE_MODE=open
FATROP_MAX_ITER=800
FATROP_TOL=0.01
FATROP_ACCEPTABLE_TOL=0.01
```
Result on Oval_Track_260m N=150: **success=True, 56 iter, cost=17.84, solve_time~8.6s**

### Balanced config — best quality/speed (N=150, tol=5e-3)
```bash
FATROP_PRESET=obstacle_fast
FATROP_STRUCTURE_DETECTION=auto
FATROP_EXPAND=0
FATROP_STAGE_LOCAL_COST=1
FATROP_DYNAMICS_SCHEME=euler
FATROP_CLOSURE_MODE=open
FATROP_MAX_ITER=800
FATROP_TOL=5e-3
FATROP_ACCEPTABLE_TOL=5e-3
```
Result on Oval_Track_260m N=150: **success=True, 87 iter, cost=17.14, solve_time~13.4s**
That is -5.1% lap time vs the N=120/tol=0.01 baseline.

Example command (balanced):
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_CLOSURE_MODE=open \
FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt_v2.py \
  --map-file maps/Oval_Track_260m.mat --N 150
```

## Related docs
- Progress log: [`docs/FATROP_TRAJOPT_PROGRESS.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/FATROP_TRAJOPT_PROGRESS.md)
- IPOPT config baseline: [`docs/OPTIMIZER_CONFIG.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/OPTIMIZER_CONFIG.md)
