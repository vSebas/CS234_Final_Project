# FATROP TrajOpt Progress (vs IPOPT)

## Current status
- FATROP is implemented as a standalone runner in [`experiments/run_fatrop_native_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final%20Project/CS234_Final_Project/experiments/run_fatrop_native_trajopt.py).
- A dedicated Rockit+FATROP entrypoint is available in [`experiments/run_rockit_fatrop_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final%20Project/CS234_Final_Project/experiments/run_rockit_fatrop_trajopt.py).
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
  experiments/run_fatrop_native_trajopt.py --map-file maps/Oval_Track_260m.mat --N 40 --compare-ipopt
```

Observed (latest rerun):
- IPOPT: `success=True`, `iterations=98`, `cost=15.420789`, `solve_time=8.824s`
- FATROP (`fast`, default structure flow): `success=True`, `iterations=67`, `cost=15.665454`, `solve_time=9.368s`

## Latest Oval+Obstacle comparison
Command:
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=manual \
  /home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt.py \
  --map-file maps/Oval_Track_260m_Obstacles.mat --N 40 --compare-ipopt
```

Observed:
- IPOPT: `success=True`, `iterations=133`, `cost=15.490398`, `solve_time=10.765s`
- FATROP (`obstacle_fast`): `success=True`, `iterations=56`, `cost=15.781586`, `solve_time=10.518s`, `min_clearance=1.9117`

## FATROP vs IPOPT summary for `N=40`
Primary source runs:
- no obstacle run command above (rerun at 2026-03-05 12:16 local)
- obstacle run command above (rerun at 2026-03-05 12:17 local)
- sweep file: `results/solver_benchmarks/fatrop_tune_N40_Oval_Track_260m_20260305_115116.csv`

Comparison snapshot (single-run rerun):
- Oval no-obstacle:
  - IPOPT faster by ~`0.54s` (`8.824s` vs `9.368s`)
  - FATROP lower iterations (`67` vs `98`) but higher objective (`15.665454` vs `15.420789`)
- Oval+obstacle:
  - FATROP slightly faster by ~`0.25s` (`10.518s` vs `10.765s`)
  - FATROP lower iterations (`56` vs `133`) but higher objective (`15.781586` vs `15.490398`)

Sweep-best FATROP times at `N=40` (no IPOPT in this sweep file):
- `fast_none`: `8.137s`, iter `67`, cost `15.665454`
- `obstacle_fast_none`: `10.737s`, iter `80`, cost `15.652857`
- `balanced_none`: `11.037s`, iter `96`, cost `15.591532`
- `auto` and `manual` modes were slower in this sweep.

## Interpretation
- FATROP is working and returns feasible solutions.
- Runtime at `N=40` is in the same order as IPOPT, with case-dependent winner.
- Tradeoff: FATROP converges faster in iterations with slightly higher objective value.
- Main remaining gap is formulation/option parity, not missing structure metadata.

## N=120 resolved (2026-03-05) — v2 formulation

**`run_fatrop_native_trajopt_v2.py` solves N=120 in ~8s / 93 iterations.**

Root causes of all previous timeouts, now fixed:

### Bug 1: non-interleaved variable creation (primary cause)
v1 created variables as `[x_0..x_N], [u_0..u_{N-1}]` (all states, then all controls).
FATROP auto structure detection expects stage-interleaved order: `x_0, u_0, x_1, u_1, ...`.
With the grouped ordering, structure detection failed on every stage, forcing FATROP into
generic O(N³) NLP mode — hence timeouts at N=120 regardless of other settings.

Fix in v2: create variables inside the stage loop, one `(x_k, u_k)` pair at a time.

### Bug 2: pre-loop boundary condition
v1 added `opti.subject_to(x_vars[0][5] == 0.0)` before the stage loop.
CasADi's structure detector treated this as a "pre-stage 0" constraint, making all
downstream dynamics appear to depend on the previous interval (cascading warnings).

Fix in v2: added as a path constraint inside the `k=0` stage iteration.

### Formulation issue: closure coupling
Soft/hard closure couples stage 0 and stage N in the Hessian or constraints, which
FATROP's Riccati factorization cannot handle. Since periodicity is not required for
single-lap time minimization, `FATROP_CLOSURE_MODE=open` is both correct and sufficient.

### Tolerance: Euler integration accuracy
Demanding `tol=1e-4` fights the O(ds²) Euler discretization error. `tol=0.01` matches
the integration accuracy and converges reliably in ~93 iterations.

## Confirmed working config (v2)

Script: `experiments/run_fatrop_native_trajopt_v2.py`

```
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

Result — Oval_Track_260m, N=120:
- `success=True`, `iterations=93`, `cost=18.051533`, `solve_time~8s`, `total_time~9s`
- Reproducible across multiple runs.

Note: `max_iter=3000` was rejected by FATROP as out-of-bounds; effective max is ~800.

## Historical tuning results (v1, structure=none)
- trapezoidal, structure=`none`: N=40 ~27s, N=60 ~80s, N=120 timeout
- euler, structure=`none`: N=40 failed, N=60 timeout (large ds instability at N=40)
- soft closure, structure=`none`: N=40 8.4s, N=60 19.6s, N=80 80.3s, N=120 timeout
- All `auto`/`manual` structure attempts on v1 failed (variable ordering bug)

## Next steps
- Run v2 ladder sweep (N=40,60,80,100,120) to confirm scaling behaviour
- Validate solution quality vs IPOPT on matched open-lap formulation
- Consider adding v2 formulation fixes back into v1 (or promoting v2 to primary)
