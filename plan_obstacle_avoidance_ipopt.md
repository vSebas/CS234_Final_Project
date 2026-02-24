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

### Why track segments (not full laps) are recommended initially
- Much easier to solve robustly (smaller NLP, fewer coupled nonconvex interactions)
- Faster iteration for debugging/tuning (shorter solve times, quicker failure diagnosis)
- Easier to randomize -> more diverse dataset quickly
- Same methods extend to full laps later

**Explicit project stance:** full-lap optimization is intentionally deferred until segment-level obstacle solves are consistently reliable under the Stage A/B acceptance criteria.

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

## 2) Two-stage solve (critical for robustness with obstacles)

Obstacle avoidance makes the problem nonconvex. Instead of one-shot min-time, solve in two passes:

### Stage A — Feasibility-first solve
**Goal:** produce a collision-free-ish, track-valid trajectory even from a weak initialization.

Objective:
- minimize:
  - obstacle slack `∑σ`
  - smoothness regularizers (e.g., `∑||Δu||^2`, `∑||Δx||^2`)
  - optionally a mild time term (small weight)

Constraints:
- full dynamics collocation
- track bounds
- obstacle constraints **with slack**

Output:
- a trajectory with low slack and reasonable dynamics consistency

### Stage B — Performance solve (min-time)
Warm-start Stage B with Stage A solution.

Objective:
- minimize `t[N]` (primary)
- keep small smoothness regularizers
- crank slack penalty high (`w_σ ↑↑`) so the optimizer removes any remaining slack

Constraints:
- same as Stage A

**Benefit:** this transforms a fragile nonconvex solve into a robust pipeline.

### 2.1 Explicit acceptance criteria (must be coded, not informal)
Define hard pass/fail checks for each stage:

- Stage A (feasibility) passes only if all are true:
  - `max_sigma <= eps_sigma_feas` (example: `1e-2`)
  - max dynamics defect/residual <= `eps_dyn_feas` (example: `1e-2`)
  - track bound violation <= `eps_track` (example: `1e-3`)
- Stage B (performance) passes only if all are true:
  - `max_sigma <= eps_sigma_time` (example: `1e-4`)
  - max dynamics defect/residual <= `eps_dyn_time` (example: `1e-3`)
  - track/control constraints satisfied within solver tolerance

If a stage does not pass these checks, treat it as failed even if IPOPT reports solve success.

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

This is enough to make Stage A reliable.

---

## 4) Make IPOPT-collocation the single “official” optimizer

Refactor the optimizer into one entry point:

`solve_trajectory(scenario, X_init=None, U_init=None, stage="feas"|"time") -> (X, U, stats)`

Key requirements:
- Stage A/B objective toggles
- obstacle constraints on/off
- warm-start support (DT will call this)
- strict post-solve validation against Stage A/B acceptance criteria

### 4.1 Fallback and retry policy (when Stage B fails)
Add deterministic retries before declaring failure:

1. Retry Stage B from Stage A solution with higher slack penalty schedule.
2. Retry Stage B with slightly perturbed controls/states around Stage A solution.
3. If still failing, rerun Stage A with a more conservative initializer and retry Stage B once.

Record which retry path succeeded for later diagnostics.

### Also keep deterministic baselines
- Heuristic initializer (above)
- Naive initializer (for worst-case comparisons)

---

## 5) Dataset generation (solver-first, not ML-first)

Create `data/generate_dataset.py`:

For each sample:
1. Sample a scenario:
   - `s0`, segment length, initial speed
   - obstacles list (randomized)
2. Generate heuristic init
3. Run **Stage A**, then **Stage B**
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
- `s0`, `segment_length`
- `x0`
- obstacle list: `(E_j, N_j, R_j, margin, R_tilde_j)`
- solver config hash/version

Reject samples that do not pass Stage A/B acceptance criteria, and store failure reason codes.

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
- min obstacle distance (or max slack)

This directly measures warm-start value without SCP.

---

## 8) De-risking checklist (do before training)

Before training DT, verify:

1. Stage A+B solves succeed for most randomized obstacle scenarios.
2. Solver produces near-zero slack and respects track bounds.
3. Warm-starting IPOPT with a **better** initializer (e.g., DC solution) reduces iterations/time.
   - If this doesn’t hold, DT won’t help either.
4. Logs include solver diagnostics for every run:
   - IPOPT status + iteration count
   - objective components (time, slack penalty, smoothness)
   - max dynamics residual, max track/control violation
   - `max_sigma`, min obstacle distance
   - retry path used (if any)

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
