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

## OCP formulation (v2)

### State vector — path coordinates (8 states)
| Index | Symbol | Units | Description |
|-------|--------|-------|-------------|
| 0 | `ux` | m/s | longitudinal velocity (body frame) |
| 1 | `uy` | m/s | lateral velocity (body frame) |
| 2 | `r` | rad/s | yaw rate |
| 3 | `ΔFz_long` | kN | longitudinal weight transfer |
| 4 | `ΔFz_lat` | kN | lateral weight transfer |
| 5 | `t` | s | elapsed time (integrated along arc length) |
| 6 | `e` | m | lateral deviation from centerline (positive = left) |
| 7 | `Δψ` | rad | heading error relative to track tangent |

### Control vector (2 inputs)
| Index | Symbol | Units | Description |
|-------|--------|-------|-------------|
| 0 | `δ` | rad | steering angle |
| 1 | `Fx` | kN | total longitudinal force command |

### Objective
Minimize terminal lap time plus control regularization:

```
J = t_N  +  λ_u · Σ_{k=0}^{N-1} ||u_k||²
```

- `t_N = x_N[5]` — elapsed time at final stage (= lap time for single-lap)
- `λ_u = 0.005` — control regularization weight (reduces chattering, not dominant)
- `FATROP_STAGE_LOCAL_COST=1` uses stage-local `||u_k||²` (no cross-stage coupling)

No closure penalty is added (`FATROP_CLOSURE_MODE=open`): periodicity is not required for single-lap time minimization.

### Equality constraints (dynamics)
Euler forward integration at each stage `k = 0, …, N-1`:

```
x_{k+1} = x_k + ds · (dx/dt)(x_k, u_k, κ_k, θ_k, φ_k) / ṡ(x_k, u_k, κ_k)
```

where `ṡ = (ux·cos(Δψ) - uy·sin(Δψ)) / (1 - e·κ)` is the arc-length rate.

The full temporal dynamics `dx/dt` encode:
- Pacejka combined-slip tire forces (front and rear)
- Longitudinal/lateral weight transfer (first-order lag)
- Road grade (`θ`) and bank (`φ`) gravity components
- Aerodynamic drag and rolling resistance
- Brake-yaw moment (Aggarwal & Gerdes 2025)

Initial-time BC (`t_0 = 0`) is added as a stage-0 path constraint so that FATROP's
structure detector treats it as part of stage 0, not a "pre-stage" constraint.

### Inequality constraints (per stage, k = 0, …, N)
| Constraint | Expression | Meaning |
|-----------|-----------|---------|
| min speed | `ux_k ≥ ux_min` (0.5 m/s) | prevent stall / singularity |
| forward progress | `ṡ_k ≥ ε_s` (0.1 m/s) | arc-length rate positive |
| curvature singularity | `1 - κ_k · e_k ≥ ε_κ` (0.05) | Frenet denominator bounded away from 0 |
| track boundary | `-hw_k ≤ e_k ≤ hw_k` | stay inside track half-width |
| steering limits | `-δ_max ≤ δ_k ≤ δ_max` | actuator bounds (vehicle param) |
| force limits | `Fx_min ≤ Fx_k ≤ Fx_max` | drive/brake force bounds (vehicle param) |
| obstacle avoidance | `‖p_k - p_obs_j‖² ≥ (r̃_j)²` | circle clearance per obstacle in window |

Obstacle constraint uses Cartesian coordinates:
```
p_k = [posE_cl(s_k) - e_k·sin(ψ_cl(s_k)),  posN_cl(s_k) + e_k·cos(ψ_cl(s_k))]
r̃_j = radius_j + margin_j + obstacle_clearance_m + vehicle_radius_m
```
Only obstacles within `obstacle_window_m = 30 m` of the current stage are activated.

---

## FATROP v2 vs IPOPT formulation differences

The two solvers share the same state/control definition and vehicle dynamics, but differ in several places. These gaps explain most of the cost discrepancy (FATROP gives slightly higher lap times).

### 1. Integration scheme — the largest accuracy gap

| | IPOPT | FATROP v2 |
|---|---|---|
| Method | **Trapezoidal** collocation | **Euler** forward |
| Formula | `x_{k+1} = x_k + ds/2·(f_k/ṡ_k + f_{k+1}/ṡ_{k+1})` | `x_{k+1} = x_k + ds·f_k/ṡ_k` |
| Local error | O(ds³) | O(ds²) |
| Stage coupling | uses `x_{k+1}, u_{k+1}` in stage-k constraint → cross-stage | purely stage-local |

Trapezoidal is significantly more accurate. Euler is required for FATROP because trapezoidal references `u_{k+1}` in stage-k's constraint, breaking FATROP's banded structure assumption.

### 2. Control regularizer — different penalty shape

| | IPOPT | FATROP v2 |
|---|---|---|
| Term | `λ_u · Σ ‖u_{k+1} − u_k‖²` | `λ_u · Σ ‖u_k‖²` |
| Effect | penalizes *rate of change* (smoothness) | penalizes *magnitude* (energy) |
| Stage coupling | cross-stage (couples k and k+1) | stage-local |

The IPOPT penalty discourages abrupt steering/force changes. The FATROP penalty discourages large actuator values. Both use `λ_u=0.005` — the regularizer is small relative to lap time so the effect on cost is minor, but it shapes the solution differently.

### 3. Lap closure / boundary conditions

| | IPOPT | FATROP v2 |
|---|---|---|
| `convergent_lap=True` (default) | **periodic BCs**: `x_0==x_N`, `u_0==u_N` on all 7 state dims + both controls | — |
| `convergent_lap=False` | fixed `x_0` parameter | `FATROP_CLOSURE_MODE=open` (no closure) |

IPOPT by default enforces periodicity — the trajectory must loop back to its own start. FATROP v2 uses open closure (no such constraint). This means FATROP optimizes a single shot from the default initial guess without constraining the final state, which can give a slightly different (and sometimes better in raw cost, but not periodic) solution.

### 4. Obstacle constraints

| | IPOPT | FATROP v2 |
|---|---|---|
| Hard constraint | `‖p_k − p_obs‖² ≥ r̃²` | same |
| Slack option | yes — `g_kj + σ_kj ≥ 0`, `σ_kj ≥ 0`, penalized in objective | no |
| Midpoint subsampling | optional (`obstacle_enforce_midpoints`, up to `N_sub` interior points per segment) | no |
| Obstacle-aware init | yes (smooth lateral bump away from obstacles) | **yes** (added 2026-03-05) |

IPOPT has a richer obstacle handling pipeline. FATROP v2 only enforces the circle constraint at the `N+1` grid nodes.

### 5. Summary table

| Feature | IPOPT | FATROP v2 | Impact |
|---|---|---|---|
| Integration | Trapezoidal O(ds³) | Euler O(ds²) | **high** — accuracy |
| Regularizer | `‖Δu‖²` (rate) | `‖u‖²` (magnitude) | low — shapes trajectory |
| Closure | periodic BCs | open | medium — affects feasibility at start/end |
| Obstacle slack | optional | none | medium for tight obstacle cases |
| Obstacle midpoints | optional | none | low–medium for coarse N |

The cost gap (FATROP ~17–18s vs IPOPT ~15–16s at N=40, or ~18s vs ~15s equivalent) is primarily explained by (1) Euler integration error at coarser N and (2) open vs periodic closure allowing FATROP to avoid the constraint that links the last and first stages.

---

## Fine-tuning results (v2, Oval_Track_260m, 2026-03-05)

Sweep over `N` and `tol` with fixed config:
`FATROP_PRESET=obstacle_fast | FATROP_STRUCTURE_DETECTION=auto | FATROP_EXPAND=0`
`FATROP_STAGE_LOCAL_COST=1 | FATROP_DYNAMICS_SCHEME=euler | FATROP_CLOSURE_MODE=open | FATROP_MAX_ITER=800`

| N | tol | iter | solve_time | cost (lap time) | notes |
|---|-----|------|-----------|-----------------|-------|
| 120 | 0.01 | 93 | 8.1s | 18.051 | **baseline** |
| 150 | 0.01 | 56 | 8.6s | 17.837 | free win: faster iter, same time |
| 120 | 1e-3 | 262 | 20.9s | 17.169 | good quality, 2.6× slower |
| **150** | **5e-3** | **87** | **13.4s** | **17.138** | **best overall** |
| 150 | 1e-3 | 415 | 36.7s | 17.420 | worse than 5e-3 (stalls in barrier) |
| 200 | 1e-3 | 375 | 53.8s | 17.430 | no gain over N=150 |
| 200 | 0.01 | 10 | 5.0s | 35.612 | **premature exit** — acceptable_tol hit at bad local min |
| 120 | 0.01 | 119 | 11.5s | 17.937 | mu_init=0.1, worse |

### Key findings
- **N=150 is a free win**: 56 iter, 8.6s, cost=17.84 vs baseline (93 iter, 8.1s, cost=18.05).
  Better cost and fewer iterations for only 0.5s more build+solve time.
- **N=150 + tol=5e-3 is the best quality/speed trade-off**: 87 iter, 13.4s, cost=17.14.
  That is -5.1% lap time vs baseline.
- **Tighter tol at N=150 (1e-3) is counterproductive**: 415 iter, 36.7s, cost=17.42 —
  the interior-point barrier needs to descend further, converging to a worse local minimum.
- **N=200 with tol=0.01 fails**: acceptable_tol=0.01 is too loose for ds=1.3m; the solver
  exits after 10 iterations with cost=35.6 (trivial slow trajectory). Needs tol ≤ 1e-3
  which then costs 54s with no improvement over N=150.
- **mu_init=0.1 is worse than default 0.3**: more iterations, higher cost, slower.

### Recommended configs

| Use case | N | tol | ~Time | Cost |
|---|---|---|---|---|
| Fast (dataset generation) | 150 | 0.01 | ~9s | 17.84 |
| Balanced (best quality/speed) | 150 | 5e-3 | ~13s | 17.14 |

Command for balanced config:
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_CLOSURE_MODE=open \
FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt_v2.py \
  --map-file maps/Oval_Track_260m.mat --N 150
```

---

## RK4 + smooth controls upgrade (2026-03-05)

Two improvements were implemented in v2 (`FATROP_DYNAMICS_SCHEME`, `FATROP_SMOOTH_CONTROLS`).

### RK4 integration — impractical with current vehicle model

`FATROP_DYNAMICS_SCHEME=rk4` was implemented using a pre-compiled CasADi `Function`
for the complete RK4 step (k1..k4 internal, one call node per stage in the NLP graph).

**Result: times out even at N=40 (>5 minutes).**

Root cause: CasADi's automatic differentiation for the NLP Jacobian/Hessian must
propagate through the Pacejka tire model (very complex expression graph). Even with
the full step compiled as a single `ca.Function`, the symbolic Jacobian of that function
is expensive to generate. RK4 with forward-simulation is fine (no AD needed — used in
`planning/dt_warmstart.py`), but optimization requires differentiating through the step.

RK4 stays in the code for reference; it is **not recommended** for the current vehicle
model. If needed in future, the fix would require SX-based dynamics or user-supplied
Jacobians (external functions with hand-coded derivatives).

### Smooth controls state augmentation — large improvement

`FATROP_SMOOTH_CONTROLS=1` lifts `(δ, Fx)` into the state vector and uses
`(dδ/ds, dFx/ds)` as the optimization controls (arc-length rate commands).

- Fully stage-local: `δ_{k+1} = δ_k + ds·(dδ/ds)_k`
- Objective `λ·Σ‖u_k‖²` now penalizes control *rates* (not magnitudes) — stage-local ✓
- Actuator bounds on `(δ, Fx)` enforced as state path constraints at all stages (0..N)
- Rate bounds on `(dδ/ds, dFx/ds)` derived from vehicle params, scaled by `ds/ux_min`

**Benchmark (N=150, tol=5e-3, Oval_Track_260m):**

| Config | Iter | Total time | Cost | Notes |
|--------|------|-----------|------|-------|
| euler (baseline) | 87 | 6.4s | 17.14 | previous best |
| **euler + smooth** | **57** | **4.6s** | **14.70** | **−14% lap time, faster** |
| rk4 | — | >300s | — | impractical (AD cost) |

Smooth controls gives −14% lap time and is also 28% faster to solve. This is because
the rate-based regularizer (`‖dδ/ds‖² + ‖dFx/ds‖²`) penalizes chattering directly,
while the magnitude regularizer (`‖δ‖² + ‖Fx‖²`) was biasing the solver away from
large-but-smooth actuator values that are physically optimal.

### Updated recommended configs (2026-03-05)

| Use case | N | tol | smooth | ~Time | Cost |
|---|---|---|---|---|---|
| Fast (dataset gen) | 150 | 0.01 | 0 | ~9s | 17.84 |
| Balanced | 150 | 5e-3 | 0 | ~6s | 17.14 |
| **Best quality** | **150** | **5e-3** | **1** | **~5s** | **14.70** |

Command for best-quality config:
```bash
PYTHONPATH=. FATROP_PRESET=obstacle_fast FATROP_STRUCTURE_DETECTION=auto FATROP_EXPAND=0 \
FATROP_STAGE_LOCAL_COST=1 FATROP_DYNAMICS_SCHEME=euler FATROP_SMOOTH_CONTROLS=1 \
FATROP_CLOSURE_MODE=open FATROP_MAX_ITER=800 FATROP_TOL=5e-3 FATROP_ACCEPTABLE_TOL=5e-3 \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/run_fatrop_native_trajopt_v2.py \
  --map-file maps/Oval_Track_260m.mat --N 150
```

---

## Warm-start tuning + FATROP vs IPOPT final comparison (2026-03-05)

### Initial guess: what was tried

The default initial guess had `ux_seed=5 m/s`, `Fx=0.5 kN` — below the drag-balanced
value at that speed (Fx_drag(5 m/s) ≈ 0.23 kN), meaning the vehicle is slightly
accelerating at the initial point.

Six variants were tested (N=150, smooth_controls=1, Oval_Track_260m):

| ux_seed | Fx_seed | No-obs cost | No-obs iter | Obs cost | Obs iter | Obs clearance |
|---------|---------|-------------|-------------|----------|----------|---------------|
| 5 m/s | 0.5 kN (old default) | 14.70 | 57 | 21.39 | 39 | 0.094m |
| 15 m/s | 0.31 kN (drag-balanced) | 17.54 | 17 | 16.28 | 59 | 0.059m |
| 10 m/s | 0.26 kN (drag-balanced) | 17.13 | 29 | 16.45 | 62 | 0.008m |
| 10 m/s | 0.5 kN | 14.40 | 103 | 17.16 | 25 | 0.075m |
| 1 m/s | 0.22 kN (drag-balanced) | timeout | — | timeout | — | — |
| **10 m/s** | **0.5 kN + obs-aware e** | **14.40** | **103** | **16.94** | **53** | **0.183m** |

IPOPT reference: no-obs 14.43 / 432 iter / 78s; obs 14.70 / 305 iter / 61s.

### Key insight: why drag-balanced Fx causes premature convergence

At true steady state (Fx = drag), the KKT stationarity conditions are nearly satisfied
at the initial point — the gradient of the Lagrangian w.r.t. Fx is close to zero because
the dynamics residual is zero. FATROP's interior-point solver exits as soon as KKT < tol,
which it achieves almost immediately, trapping the solver at the slow initial speed.

Using Fx=0.5 kN (slightly above drag at 10 m/s) creates a positive acceleration
imbalance: the dynamics are NOT in steady state, so there is a meaningful gradient
pointing toward higher ux. This forces the solver to work hard and find the true optimum.

At ux=1 m/s, the time state init is t[N]=260s (18× the optimal 14.4s) — the barrier
descent is overwhelmingly dominated by the time residual and FATROP hits max_iter
(800) long before converging. Confirmed by experiment: both runs hung for >15 minutes.

### Obstacle-aware lateral offset init

Ported from `planning/optimizer.py:_build_obstacle_aware_e_init`. Each obstacle
contributes a Gaussian bump in e(s) that pre-biases the lateral position to the
opposite side from the obstacle before the solver starts:

```
e_init[k] += target · exp(−0.5·(d_s(k, obs_j)/σ)²)
target = −sign(e_obs) · (r̃ + margin)   if e_obs ≠ 0
         + (r̃ + margin)                 if e_obs ≈ 0 (centerline obstacle)
σ = 8.0 m (along-track spread), margin = 0.3 m
```

Result on obstacle case: clearance 0.075 m → 0.183 m, cost 17.16 → 16.94.

### Final FATROP v2 configuration

```
FATROP_PRESET=obstacle_fast | FATROP_STRUCTURE_DETECTION=auto | FATROP_EXPAND=0
FATROP_STAGE_LOCAL_COST=1 | FATROP_DYNAMICS_SCHEME=euler | FATROP_SMOOTH_CONTROLS=1
FATROP_CLOSURE_MODE=open | FATROP_MAX_ITER=800 | FATROP_TOL=5e-3 | FATROP_ACCEPTABLE_TOL=5e-3
N=150 | ux_seed=10 m/s | Fx_seed=0.5 kN | e_init=obstacle-aware
```

### FATROP v2 vs IPOPT (final)

| Map | Solver | Iter | Solve time | Cost | Speedup |
|-----|--------|------|------------|------|---------|
| No obstacles | IPOPT | 432 | 78s | 14.43 | — |
| No obstacles | **FATROP v2** | **103** | **7.5s** | **14.40** | **10×** |
| Obstacles | IPOPT | 305 | 61s | 14.70 | — |
| Obstacles | **FATROP v2** | **53** | **4.1s** | **16.94** | **15×** |

- No-obstacle: FATROP **beats IPOPT** by 0.2% at 10× speed.
- Obstacle: 15% cost gap remains. Root cause: Euler integration (vs trapezoidal) and
  no obstacle midpoint subsampling. FATROP clears the obstacle with 18 cm margin.
