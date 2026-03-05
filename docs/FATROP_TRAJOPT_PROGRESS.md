# FATROP TrajOpt Progress (vs IPOPT)

## Current status
- FATROP is implemented as a standalone runner in [`experiments/run_fatrop_native_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final%20Project/CS234_Final_Project/experiments/run_fatrop_native_trajopt.py).
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

## Latest high-horizon status (N=120)
Latest tuning sweep:
- `results/solver_benchmarks/fatrop_tune_N120_Oval_Track_260m_20260305_115251.csv`
- All tested configurations timed out at `120s`:
  - `obstacle_fast_none`
  - `fast_none`
  - `balanced_none`
  - `obstacle_fast_auto`
  - `fast_auto`
  - `obstacle_fast_manual`

This currently blocks adopting FATROP as the production default for high-horizon runs in the present codepath.

Additional quick check:
- `results/solver_benchmarks/fatrop_quick_N120_20260305_114637.csv`
- both tested configs timed out at `90s`.

## Detailed `N=120` problem analysis
Observed facts:
1. `N=120` had zero completed solves across all tested preset/structure combinations under 90s and 120s process timeouts.
2. `manual` structure mode emits repeated warnings on obstacle runs:
   - `Structure detection error ... Constraint found depending on a state of the previous interval.`
3. Current FATROP wrapper can try multiple option attempts in one process (preferred mode, then `auto`, then `none` with `expand=False`), so total per-config wall clock can be dominated by failed attempts.

Most likely contributors:
1. Stage-structure mismatch with current NLP graph:
   - periodic lap closure (`x_N` and `u_N` tied to stage `0`) introduces cross-stage coupling.
   - obstacle + closure constraints can break strict stage-local structure assumptions in manual mode.
2. Scale growth from `N=40` to `N=120`:
   - 3x horizon increases variable and constraint counts substantially and increases KKT factorization burden.
3. Guardrails were missing in the earlier runs:
   - those benchmark runs depended on external process timeout only.
   - the code now supports `FATROP_MAX_ITER` for explicit iteration caps.
4. `expand=True` default in first attempts can enlarge symbolic workload before/while solving.

## Reformulation and tuning update (2026-03-05)
Implemented updates:
1. Added `FATROP_STAGE_LOCAL_COST` in `run_fatrop_native_trajopt.py`:
   - `1`: stage-local control regularization (`sum ||u_k||^2`)
   - `0`: original cross-stage control difference (`sum ||u_{k+1}-u_k||^2`)
2. Added `FATROP_DYNAMICS_SCHEME`:
   - `trapezoidal` (original)
   - `euler` (more stage-local but lower fidelity)
3. Added `FATROP_MAX_ITER` to bound solve effort.
4. Added closure reformulation in FATROP path:
   - `FATROP_CLOSURE_MODE=open|soft|hard`
   - `FATROP_CLOSURE_SOFT_WEIGHT`
   - `soft` removes hard wrap-around constraints and adds terminal mismatch penalty.
5. Added programmatic warm-start support in `solve_fatrop_native(...)`:
   - accepts `X_init/U_init` arrays for homotopy-style initialization.
6. Added dedicated FATROP-native runner:
   - `experiments/run_fatrop_native_trajopt.py`
   - purpose: avoid IPOPT-style mixed formulation and keep stage-local structure explicit.

Observed tuning results:
- Open-lap, stage-local cost, trapezoidal, structure=`none`, expand=`0`, max_attempts=`1`:
  - `N=40`: FATROP success, `~27.4s`
  - `N=60`: FATROP success, `~80-89s`
  - Source: `results/solver_benchmarks/trapezoidal_stage_local/fatrop_openlap_ladder_Oval_Track_260m_20260305_125731.csv`
- Same setup but structure=`auto` or `manual` at `N=40`:
  - FATROP failed (no accepted solve)
  - Sources:
    - `results/solver_benchmarks/structure_tune_auto/fatrop_openlap_ladder_Oval_Track_260m_20260305_130049.csv`
    - `results/solver_benchmarks/structure_tune_manual/fatrop_openlap_ladder_Oval_Track_260m_20260305_130116.csv`
- Same setup but expand=`1` at `N=40`:
  - FATROP failed (no accepted solve)
  - Source: `results/solver_benchmarks/expand_tune/fatrop_openlap_ladder_Oval_Track_260m_20260305_130141.csv`
- Euler dynamics (`FATROP_DYNAMICS_SCHEME=euler`) was unstable in this setup (`N=40` failed, `N=60` timed out):
  - Source: `results/solver_benchmarks/fatrop_openlap_ladder_Oval_Track_260m_20260305_125417.csv`
- Independent-timeout diagnostic at `N=80` (open-lap, stage-local, trapezoidal, structure=`none`, expand=`0`, `max_iter=200`, per-solver timeout `60s`):
  - IPOPT: success (`iterations=192`, `solve_time=29.635s`)
  - FATROP: timeout (`60s`, no accepted solve)
  - Source: `results/solver_benchmarks/trapezoidal_stage_local/fatrop_openlap_ladder_Oval_Track_260m_20260305_131139_087754.csv`

Closure soft-mode results (same general tuning, `FATROP_CLOSURE_MODE=soft`, `FATROP_CLOSURE_SOFT_WEIGHT=100`):
- `N=40`: IPOPT `15.255s`, FATROP `8.439s` (FATROP faster)
- `N=60`: IPOPT `20.617s`, FATROP `19.551s` (roughly parity, FATROP slightly faster)
- `N=80`: IPOPT `26.342s`, FATROP `80.268s` (FATROP slower but solved under 120s, no timeout)
- Sources:
  - `results/solver_benchmarks/closure_soft/fatrop_openlap_ladder_Oval_Track_260m_20260305_131656_754433.csv`
  - `results/solver_benchmarks/closure_soft/fatrop_openlap_ladder_Oval_Track_260m_20260305_131826_119596.csv`

Dedicated FATROP-native sanity run (`N=40`, Oval, soft closure):
- IPOPT: `success=True`, `iterations=233`, `cost=14.509318`, `solve_time=12.780s`
- FATROP-native: `success=True`, `iterations=35`, `cost=17.578371`, `solve_time=4.641s`
- Interpretation: major speedup at low `N`, with higher objective vs IPOPT (quality gap still present).

Current best-known FATROP config for this reformulated path:
- `--no-convergent-lap` (open-lap benchmark mode)
- `FATROP_PRESET=obstacle_fast`
- `FATROP_STRUCTURE_DETECTION=none`
- `FATROP_EXPAND=0`
- `FATROP_STAGE_LOCAL_COST=1`
- `FATROP_DYNAMICS_SCHEME=trapezoidal`
- `FATROP_CLOSURE_MODE=soft`
- `FATROP_CLOSURE_SOFT_WEIGHT=100`
- `FATROP_MAX_ITER=800`

Required fixes before claiming `N=120` viability:
1. Add explicit FATROP per-solve limits (iteration/time) in solver options.
2. Separate/relax periodic closure handling for FATROP benchmarking (or isolate open-lap benchmark first).
3. Rebuild constraints to preserve stage-local dependency for manual structure mode.
4. Re-run a fixed ladder (`N=40, 60, 80, 100, 120`) and report solve success/time/objective deltas vs IPOPT.

## Next step to close gap
1. Done: `FATROP_MAX_ITER` is implemented in `run_fatrop_native_trajopt.py`.
2. Run fresh `N=120` sweep with explicit caps enabled to quantify whether timeouts become deterministic failures vs occasional solves.
3. Make a stage-structure-clean benchmark variant (no periodic closure coupling) and verify manual mode behavior.
4. Only then retune presets at `N>=80` and compare against IPOPT on matched config.
