# Plan of Action: Trajectory Optimization + Obstacle Avoidance (without SCP)

This plan replaces SCP with a **direct transcription / collocation NLP** solved by **IPOPT (via CasADi)**, while still supporting:

- **Dataset generation** (expert trajectories)
- **Decision Transformer (DT)** warm-starting
- **Obstacle avoidance** as a first-class constraint

The goal is a pipeline that is **reliable**, **trainable**, and **measurable** (warm-start reduces runtime/iterations).

---

## 0) Default task setup (freeze the spec)

**Recommended default:** track-based planning in Frenet-like coordinates, using the existing state that includes lateral deviation `e(s)` and a track model that provides curvature and width.

### Objective
- Primary: **minimize final time** `t[N]`  
  (Time is already in your state and `dt/ds` is enforced by dynamics.)

### Constraints
- Dynamics via collocation (trapezoidal, etc.)
- Track bounds: `e_min(s_k) ≤ e_k ≤ e_max(s_k)`
- Control bounds
- Obstacle avoidance (see below)

### Full-lap default (project decision)
- Optimize one full closed lap by default.
- Use periodic boundary conditions (`convergent_lap=True`).
- Use `ds = track_length / N` so the horizon spans exactly one lap.

**Explicit project stance:** full-lap optimization is the active default in this repository.

---

## 1) Add obstacle avoidance to the NLP (in a solver-friendly way)

### 1.1 Obstacle representation
Start with **static circular obstacles** in global coordinates (ENU/EN):

- Obstacle `j`: center `(E_j, N_j)`, radius `R_j`
- Add safety margin `m` (vehicle radius + buffer):
  - `R̃_j = R_j + m`

Store `(E_j, N_j, R̃_j)` in the scenario.

### 1.2 Map vehicle position per node
At each collocation node `k`, compute global position from track map matching:
- `(E_k, N_k) = map_match(s_k, e_k)`

### 1.3 Obstacle avoidance constraint (hard, but with slack)
Use squared distance (no sqrt):
\[
g_{k,j} = (E_k - E_j)^2 + (N_k - N_j)^2 - R̃_j^2 \ge 0
\]

To avoid infeasibility killing IPOPT, introduce **slack variables**:
- `σ_{k,j} ≥ 0`
- enforce:
\[
g_{k,j} + \sigma_{k,j} \ge 0,\quad \sigma_{k,j} \ge 0
\]

and penalize slack in the objective:
\[
J \leftarrow J + w_{\sigma}\sum_{k,j}\sigma_{k,j}
\]

**Why slacks matter:** they make the solve “always solvable” and let the optimizer drive violations toward zero.

### 1.4 Don’t enforce all obstacles at all nodes (constraint gating)
To avoid a huge NLP, only enforce obstacle `j` around where it matters.

Practical gating approach:
- Precompute each obstacle’s approximate `s_obs` (projection onto centerline / nearest along-track).
- Enforce constraints only for nodes where:
  - `|wrap(s_k - s_obs)| ≤ s_window` (e.g., 20–40 m)

This can reduce constraints by 10×–100×.

---

## 2) Current Solve Strategy (single-stage, full lap)

Current active path in code is **single-stage IPOPT** on a full lap.

Objective combines:
- time term (`t[N]`)
- smoothness regularization
- obstacle penalty term (only if obstacle slack is enabled)

Obstacle constraints are enforced as:
- hard constraints by default (no slack),
- node constraints + interior constraints within each collocation interval,
- along-track gating window around each obstacle (`obstacle_window_m`),
- required radius:
  - `R_required = R + margin + obstacle_clearance_m`

Important implementation note:
- Avoid multiple additive gap knobs.
- In the current demo, `obstacle_clearance_m` is fixed to `0.0` and effective obstacle size is
  controlled by map data (`radius + margin`).

### 2.1 Acceptance criteria (single-stage)
Treat a run as valid only if all are true:
- `success == True`
- `max_obstacle_slack <= eps_sigma` (hard mode target: `0`)
- `min_obstacle_clearance >= 0` (recommended with margin: `>= 0.05 m`)
- dynamics and path constraints within solver tolerance

---

## 3) Build an initializer that doesn’t drive through obstacles

Even with slacks, IPOPT can get stuck if the initial guess plows through obstacles.

### Minimal geometric initializer for `e(s)`
For each obstacle:
1. Compute approximate Frenet obstacle coordinates `(s_obs, e_obs)` once.
2. Choose passing side (left/right) based on available track width.
3. Add a smooth lateral offset “bump” to `e_ref(s)` around `s_obs`:
   - use Gaussian bump or paired sigmoids
   - ensure the peak shift `Δe ≥ R̃_j + margin`

Then set:
- `e_init = e_ref(s_grid)`
- `δ_init ≈ atan(L * κ(s))` (curvature feedforward) + small correction from `e_init`
- `u_x_init` constant or curvature-limited
- roll out states to form a consistent `X_init`

This is enough to make the single-stage solve more reliable.

---

## 4) Make IPOPT-collocation the single “official” optimizer

Refactor/keep the optimizer as one entry point:

`solve_trajectory(scenario, X_init=None, U_init=None) -> (X, U, stats)`

Key requirements:
- full-lap periodic solve
- obstacle constraints on/off
- warm-start support (DT will call this)
- strict post-solve validation against single-stage acceptance criteria

### 4.1 Retry policy (single-stage)
When a run fails acceptance:
1. Increase `N` and obstacle interior subsamples.
2. Increase `obstacle_clearance_m`.
3. Temporarily disable obstacle gating (`obstacle_window_m = large`) to validate formulation.
4. If still failing, enable obstacle slack only for diagnosis (not for production dataset generation).

### Also keep deterministic baselines
- Heuristic initializer (above)
- Naive initializer (for worst-case comparisons)

---

## 5) Dataset generation (solver-first, not ML-first)

Create `data/generate_dataset.py`:

For each sample:
1. Sample a scenario:
   - full-lap track id and initial speed
   - obstacles list (randomized)
2. Generate heuristic init
3. Run single-stage full-lap solve
4. If success: save `.npz` with:
   - `s_grid`
   - `X[0:N]`, `U[0:N-1]`, `t[0:N]`
   - per-node track features: `κ(s_k)`, widths
   - obstacles: `(E_j, N_j, R̃_j)` plus optional `(s_obs, e_obs, R̃_j)`
   - solver stats: success, runtime, iterations, max slack, min distance

Keep seeds for reproducibility.

### 5.1 Freeze a canonical scenario schema (avoid dataset drift)
Define one versioned scenario spec and use it everywhere (generation, training, eval):

- `scenario_id`
- `seed`
- `track_id`
- `x0`
- obstacle list: `(E_j, N_j, R_j, margin, R_tilde_j)`
- solver config hash/version

Reject samples that do not pass single-stage acceptance criteria, and store failure reason codes.

### Dataset size targets
- Start: **200–500 successful** obstacle scenarios
- Scale: **2,000–10,000** if runtime permits

---

## 6) DT dataset formatting (minimal and correct)

### Per-step conditioning
At step `k`, observation includes:
- state `x_k`
- track features at `s_k` (e.g., curvature, width)
- obstacle tokens (fixed-size)

### Obstacle tokenization (simple)
Choose `M` nearest obstacles ahead of the car in along-track distance:
- each token: `[Δs, e_obs, R̃]`
- pad to M, include mask if needed

### Supervision target
- predict action `u_k` (e.g., `[δ_k, F_x,k]`)

### RTG / cost-to-go
To condition on “go fast”:
- define per-step reward `r_k = -(t_{k+1}-t_k)`
- compute RTG by reverse cumulative sum

Store:
- `states`, `actions`, `rtg`, `timesteps`, `mask`, plus `obs_tokens`

---

## 7) Warm-start experiment (main result)

Build `planning/refine_from_warmstart.py`:

Compare:
1) IPOPT from heuristic init
2) IPOPT from DT init

Metrics:
- success rate (with obstacles)
- IPOPT iterations
- wall-clock runtime
- final `t[N]`
- max constraint violation (track + obstacles)
- min obstacle distance / clearance

This directly measures warm-start value without SCP.

---

## 8) De-risking checklist (do before training)

Before training DT, verify:

1. Single-stage full-lap solves succeed for most randomized obstacle scenarios.
2. Solver achieves nonnegative minimum obstacle clearance under dense checking.
3. Warm-starting IPOPT with a **better** initializer reduces iterations/time.
   - If this doesn’t hold, DT won’t help either.
4. Logs include solver diagnostics for every run:
   - IPOPT status + iteration count
   - objective components (time, slack penalty, smoothness)
   - max dynamics residual, max track/control violation
   - `max_sigma`, min obstacle clearance
   - retry path / parameter escalation used (if any)

---

## 9) Visual-overlap Debug Checklist

If plots appear to show obstacle overlap:
1. Confirm the run log is from latest code (timestamp + git diff).
2. Confirm plotting uses obstacle positions derived from Frenet `(s,e)` when available (same source as optimizer).
3. Confirm `world.map_match_vectorized` uses the same Frenet-to-ENU conversion as optimizer constraints:
   - `E = E_cl - e*sin(psi)`
   - `N = N_cl + e*cos(psi)`
   A mismatch here creates false visual collisions even when NLP constraints are satisfied.
4. Check `min_obstacle_clearance` from dense post-check, not only node checks.
5. Increase `N` and obstacle interior subsamples before concluding formulation failure.
6. If clearance remains negative, temporarily disable gating and diagnose with full-horizon obstacle constraints.

---

## 9) What to demote / remove

- Demote SCP to an appendix/experiment.
- Keep it only if you want “attempted SCP” discussion.
- Make the collocation NLP the production pipeline.

---

## TL;DR

**Do this:**
- Add obstacle constraints with slacks
- Use a two-stage solve (feasibility → min-time)
- Use a non-stupid initializer around obstacles
- Generate dataset from IPOPT solutions
- DT warm-starts IPOPT, measure runtime/iterations gains

This is reliable, standard, and ships.
