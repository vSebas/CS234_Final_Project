# Project Plan: DT Warm-Start for Minimum-Time + Obstacle-Avoiding Raceline

This plan is written against the current repo state (IPOPT direct-collocation is the production optimizer; SCP is archived).

---

## 0) Reality check: what exists right now

### Repo structure (healthy)
- `models/`: unified single-track model + Fiala brush tires.
- `world/`: track geometry, Frenet/ENU utilities.
- `planning/optimizer.py`: direct collocation (trapezoidal in spatial domain) + IPOPT.
- `run_trajopt_demo.py`: single-scenario run with acceptance-gated retries.
- `run_trajopt_batch_eval.py`: randomized obstacle scenarios + JSON/CSV reports.
- `docs/`: background + papers + reference repos.

### What the optimizer currently solves
- **Decision variables:** state `X[:,0..N]` and control `U[:,0..N]`.
- **Cost:** sum of time increments `\sum ds / \dot s` + regularizers (control slew + small smoothness on `e` and `dpsi`) + optional obstacle slack penalty.
- **Constraints:** trapezoidal collocation for dynamics in **spatial form**, track bounds, control bounds, periodic closure (except time), obstacle distance constraints at nodes + interior samples.

---

## 1) Optimizer target formulation (simplified Tier 1)

The current optimizer is structurally aligned with the standard minimum-time racing TO setup (spatial parameterization, direct collocation, IPOPT). For this project, simplify the NLP to a **Tier 1** formulation: keep only the constraints that prevent pathological solutions and enforce safety, and use a single, simple regularizer so solutions are learnable by a Decision Transformer.

### 1.1 Proposed objective (minimum time + simple \(\Delta u\) regularizer)

Use a minimum-time objective plus a lightweight “don’t chatter” term:

- **Preferred (cleanest):** minimize final time state
  \[
  \min\; J \;=\; t_N \; +\; \lambda_u\sum_{k=0}^{N-1}\lVert u_{k+1}-u_k\rVert^2
  \]
- **Equivalent (if you don’t want to rely on \(t\) as a state):**
  \[
  \min\; J \;=\; \sum_{k=0}^{N-1}\frac{\Delta s}{\dot s_k} \; +\; \lambda_u\sum_{k=0}^{N-1}\lVert u_{k+1}-u_k\rVert^2
  \]

where \(u_k=[\delta_k,\;F_{x,k}]\). This regularizer is intentionally simpler than time-domain slew penalties; it suppresses high-frequency control oscillations and typically improves IPOPT robustness without complicating the formulation.

### 1.2 Proposed constraints (Tier 1)

Enforce these at collocation nodes \(k=0..N\):

1) **Dynamics (trapezoidal direct collocation, spatial form)**
\[
x_{k+1}=x_k+\frac{\Delta s}{2}\big(f_s(x_k,u_k)+f_s(x_{k+1},u_{k+1})\big)
\]

2) **Track bounds**
\[
e_{\min}(s_k)\le e_k\le e_{\max}(s_k)
\]

3) **Input bounds**
\[
|\delta_k|\le\delta_{\max},\qquad F_{x,\min}\le F_{x,k}\le F_{x,\max}
\]

4) **Forward progress (prevents negative/undefined time)**
\[
\dot s_k \ge \varepsilon_s \;>\;0
\]

5) **Frenet non-singularity (prevents mapping blow-ups)**
\[
1-\kappa(s_k)e_k \ge \varepsilon_\kappa
\]

6) **Periodic closure for full-lap solutions**
- enforce \(x_0=x_N\) and \(u_0=u_N\) for all states/controls **except time** (keep \(t_0=0\), do not constrain \(t_N\)).

7) **Obstacle clearance (node-only, simplest)**
For each obstacle \(i\) that is “active” at node \(k\):
\[
\lVert p(s_k,e_k)-p_{\text{obs},i}\rVert^2 \ge R_{\text{safe},i}^2
\]
with \(R_{\text{safe},i} = r_i + \text{margin}\).

**Pin:** include a conservative vehicle footprint radius:
\[
R_{\text{safe},i} = r_i + \text{margin} + R_{\text{vehicle}}
\]
(one-line upgrade; prevents center-point “corner clipping”).

**Keep the heavy lifting outside the NLP:** continue using dense post-solve collision checking (and retries / different initializations) rather than adding interior obstacle samples or homotopy inside the NLP for now.

### 1.3 Explicitly deferred (not in Tier 1)

These are valuable, but are not required to finish the project:
- hard slew-rate constraints on \(\dot\delta\), \(\dot F_x\)
- engine power limits and explicit friction feasibility constraints
- interior obstacle sampling constraints inside the NLP
- continuation / homotopy schedules for obstacles

If the Tier 1 optimizer struggles on hard scenes, add these later as “Tier 2” upgrades rather than upfront complexity.

### 1.4 Concrete edits to implement Tier 1

Modify `planning/optimizer.py`:

- **Objective:** replace the current slew/time-weighted regularizers with the single \(\Delta u\) regularizer:
  - keep the minimum-time term (`t_N` or \(`\sum ds/\dot s`\))
  - add `lambda_u * sum ||U[:,k+1]-U[:,k]||^2`
- **Constraints:** add
  - `sdot_k >= eps_s`
  - `1 - kappa(s_k) * e_k >= eps_kappa`
- **Obstacles:** keep **node-only** constraints in the NLP; keep dense post-check and retry schedule as the safety net.
- **Footprint:** add `vehicle_radius_m` into `R_safe`.

Expose new hyperparameters in config:
- `lambda_u`, `eps_s`, `eps_kappa`, and `vehicle_radius_m`.


## 2) Dataset generation pipeline (needed before DT)

### 2.1 Define a canonical scenario schema (versioned)

Create: `data/schema.py` and a JSON-serializable scenario dict:

- Track/map:
  - map file name + hash (or full track parameters if generated)
- Discretization:
  - `N`, `ds`, obstacle checking settings
- Obstacles (variable count): list of `{s, e, radius, margin}` (Frenet recommended)
- Solver config:
  - IPOPT options, Tier-1 weights/buffers, acceptance thresholds
- Random seed

Store each solved trajectory with a **future-proof contract**: keep one canonical “episode header” plus per-step arrays. DT can use a reduced observation, but we preserve the full internal model state for replay, labeling (e.g., constraints-to-go), and exact warm-start rollouts.

**Episode header (canonical, required):**
- `episode_id`
- `map_id` **and** a `map_version_hash` (or filename hash) so geometry is reproducible
- discretization: `N`, `ds`, and either `s_grid` (preferred) or `(s0, L)`
- full obstacle list for the episode (canonical scene description):
  - recommended Frenet form: `obstacles = [{s_obs, e_obs, r_obs, margin}, ...]`
- `solver_config_hash` (or embed the key IPOPT/optimizer settings)

**Per-step arrays (required):**
- `s_m[k]`: arc length at node (length `N+1`)
- `X_full[k]`: **full optimizer/model state** (keep internal dynamics complete)
  - include weight-transfer states even if DT does not observe them: `dfz_long`, `dfz_lat`
  - include path states `e`, `dpsi`, and vehicle states `ux, uy, r`
  - include time either as `t[k]` or (preferred) just store `dt[k]`
- `U[k]`: controls `[delta, Fx]`
- `dt[k]`: per-step time increment (e.g., `dt_k ≈ ds / sdot_k`)
- `reward[k]`: store `reward = -dt` (minimum-time)
- `rtg[k]`: return-to-go (backward cumulative sum of `reward`)

**Global pose (required; cached but treated as part of the contract):**
- `pos_E[k]`, `pos_N[k]`: world-frame position at the node (ENU)
- `yaw_world[k]`: world yaw at the node
  - compute as `yaw_world[k] = psi_centerline(s_k) + dpsi[k]` (then wrap to \((-\pi,\pi]\))

Rationale: global pose can be derived from `(s,e,dpsi)` and the map, but storing it makes training/evaluation/labeling stable even if map code evolves.

**Per-step map features (strongly recommended, cheap insurance):**
- `kappa[k] = kappa(s_k)` (track curvature at the node)
- `half_width[k]` (or equivalent bounds data)
- `grade[k]`, `bank[k]` (store zeros for flat tracks; keeps schema stable)

**Optional cached fields (derived; do not treat as canonical):**
- `s_dot[k]`, `one_minus_kappa_e[k]` (handy for fast validity/CtG labeling)
- `obs_feat[k, M, feat_dim]`: the “M nearest obstacles ahead” padded feature tensor used by DT (can be recomputed from `obstacles` + `s_grid`)

**Important distinction (implementation):**
- The **optimizer and rollout model** operate on `X_full`.
- The **DT observation** can be a subset, e.g. `[ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world] + track + obstacle features`. You do *not* need to feed `dfz_long/dfz_lat` to DT, but you should still **store them** and propagate them internally during rollout.

Write one “manifest” file per dataset shard:
- `data/datasets/<name>/manifest.jsonl`

### 2.2 Generate data (accepted solutions only) — concrete staged procedure

Create: `data/generate_dataset.py`

Use two stages:
- **Stage A:** no obstacles (pipeline sanity + “base raceline” + start-state definition)
- **Stage B:** randomized obstacles (actual training dataset)

Both stages should use the same acceptance-gated retry schedule style as `run_trajopt_demo.py` / `run_trajopt_batch_eval.py`, because it’s already proven to be reliable in `results/trajectory_optimization/`.

#### 2.2.1 Stage A — no obstacles (define start + generate base laps)

Goal: validate the Tier-1 optimizer, dataset writing, and pose computations before adding obstacles.

**Map choice**
- Use the map that is already stable in your logs: `maps/Medium_Oval_Map_260m.mat` (or the current default map used in your demos).

**Discretization**
- Start with the values that are already working:
  - `N = 120`
  - `ds = world.length_m / N` (do *not* hardcode)
- Track buffer: `track_buffer_m = 0.0` (increase later if needed).

**Start definition**
- The trajectory is parameterized by `s` and solved with periodic closure (`convergent_lap=True`).
- Define node `k=0` as **the start line** at:
  - `s_m[0] = 0.0`
  - `t[0] = 0.0` (already constrained in the optimizer)
- The solver chooses `X_full[:,0]` (subject to periodic closure), but the dataset start is always the node at `s=0`.

**Procedure**
1) Sample a seed and build a scenario with **no obstacles**.
2) Solve once with `convergent_lap=True`.
3) Compute and save:
   - `pos_E,pos_N = world.map_match_vectorized(s_m, e)` for each node
   - `yaw_world = psi_centerline(s_m) + dpsi`
4) Save the trajectory as a single episode.

**Optional (useful) augmentation without changing physics**
- Because the solution is periodic, you can create *additional episodes* by circularly shifting the node index:
  - roll `s_m, X_full, U, dt, kappa, half_width, pos_E, pos_N, yaw_world` by a random shift
  - then re-wrap `s_m` to be monotone in `[0, L]` by subtracting `s_m[0]` modulo `L` (store `s_offset` in the episode header if you do this)
- This is optional; it’s mostly a way to create more diverse *starting contexts* from one periodic lap. The real diversity comes from obstacles and/or multiple maps.

Stage A “done” when:
- you can write/read episodes reliably,
- pose/yaw fields look correct in plots,
- the optimizer solves deterministically and passes acceptance checks.

#### 2.2.2 Stage B — randomized obstacles (actual training dataset)

Goal: generate a large set of **accepted**, obstacle-avoiding minimum-time trajectories.

**Obstacle count**
- Use a curriculum (start easy, then hard):
  - B1: `n_obs ~ Uniform{1,2}`
  - B2: `n_obs ~ Uniform{3,6}`  (this matches what you already evaluate in `run_trajopt_batch_eval.py`)
  - (optional later) B3: `n_obs ~ Uniform{4,8}`

**Obstacle size + margin**
Start with the ranges that already appear stable in your batch eval sampler:
- radius `r ~ Uniform(1.2, 1.8)` meters
- margin `m ~ Uniform(0.6, 0.9)` meters
- required radius (Tier-1): `R_safe = r + m + R_vehicle` (include footprint)

**Obstacle placement (Frenet; consistent with current sampler)**
For each obstacle (repeat until you have `n_obs`, cap attempts to avoid infinite loops):
1) Sample along-track location:
   - `s_obs ~ Uniform(0, L)`
   - enforce a minimum along-track separation so obstacles aren’t stacked:
     - `min_ds = 0.12 * L / max(1, n_obs)`  
     - reject any new obstacle with `|wrap_s_dist(s_obs, s_prev)| < min_ds`
2) Sample lateral offset:
   - `hw = track_width(s_obs)/2`
   - `e_limit = hw - track_buffer_m - (r + m + R_vehicle) - edge_buffer`
   - use `edge_buffer = 0.4 m` (current batch eval uses this)
   - require `e_limit > 0.4 m`, else resample
   - `e_obs ~ Uniform(-e_limit, e_limit)`

**Scenario prefilter (cheap, avoids obvious infeasible setups)**
- Reject an obstacle if `(r + m + R_vehicle) >= (hw - track_buffer_m - 0.4)` at its `s_obs`.
- Optionally reject obstacle pairs that overlap in world-frame distance if they’re too close (not required if you enforce `min_ds`, but it’s a helpful extra check).

**Acceptance gates (keep only clean trajectories)**
- `result.success == True`
- `max_obstacle_slack == 0` (if using slack; Tier-1 default is no slack)
- dense min clearance `>= -1e-3` m (your current “practical epsilon”)

**Dense post-check (do not trust node-only constraints)**
- For each segment `[k,k+1]`, sample `S` interior points (e.g., `S=7`, which you already use in results).
- Convert each `(s,e)` sample to world position and check distance to every obstacle.
- Record `min_obstacle_clearance` and reject if below threshold.

#### 2.2.3 Retry schedule (proven by your results)

For each sampled obstacle scenario, attempt a small schedule of “increasing robustness” configs until one is accepted.

Start from what is already working in `run_trajopt_batch_eval.py`:

- Attempt 1: baseline
  - `N=120`, `subsamples=7`, `obstacle_clearance_m=0.0`
- Attempt 2: higher N
  - `N=160`, `subsamples=7`, `obstacle_clearance_m=0.0`
- Attempt 3: more subsamples
  - `N=160`, `subsamples=11`, `obstacle_clearance_m=0.0`
- Attempt 4: more conservative
  - `N=180`, `subsamples=13`, `obstacle_clearance_m=0.10`

Notes:
- In Tier-1, “subsamples” should apply to **post-check** (and any obstacle-aware initializer), not to extra constraints inside the NLP.
- Keep `obstacle_aware_init=True` (your current initializer already biases `e(s)` away from obstacles and helps convergence).

Always log:
- sampled obstacles (canonical list)
- which attempt was accepted
- solve iterations/time, final lap time
- min clearance after dense check

#### 2.2.4 Splits

Split by scenario seed (not by individual trajectories) so evaluation is honest:
- train/val/test = 80/10/10 or similar
- if multi-map later: split by `(map_id, seed)` and reserve held-out maps for test.

---

### 2.3 Add a deterministic “baseline initializer” (for comparisons)

Create a baseline warm-start generator that does **not** use DT:
- e.g., constant `ux`, `e=0`, `dpsi=0`, mild `F_x`, and steering from curvature.

You will compare DT warm-start against this baseline.

---

### 2.4 Dataset sizing targets (how many samples)

Decision Transformers train on **sequence windows** (context length \(K\)), so the relevant scale is **total timesteps** (not just number of laps).

Assume a typical lap discretization of \(N\approx 200\text{–}400\) steps (given your `ds`). Targets below refer to **accepted** trajectories only.

- **Stage A — pipeline / POC (single oval map):** produce 1–10 obstacle-free laps (plus optional circular-shift augmentation for sanity). This stage is not about scale.
- **Stage B — generalize across obstacle layouts (same map):** ~3k–10k accepted trajectories (≈1–3M timesteps)
- **Stage C — multi-map generalization:** ~20k–60k trajectories total across maps (≈5–20M timesteps), depending on map diversity

Operationally: keep generating until you hit a **timesteps budget** per split (train/val/test), not an attempted-solve count.

### 2.5 Reward + Return-to-Go (RTG): include it in the dataset

Even though obstacle avoidance is enforced by constraints + acceptance filtering, DT needs a conditioning signal. Use **time** as the reward source:

- Store per-step time increments:
  \[
  dt_k \approx \frac{\Delta s}{\dot s_k}
  \]
- Define per-step reward:
  \[
  r_k = -dt_k
  \]
- Compute return-to-go (RTG) offline (backward cumulative sum):
  \[
  RTG_k = \sum_{j=k}^{N-1} r_j = -(t_N - t_k)
  \]

Implementation detail: during dataset build, compute `dt`, `reward`, and `rtg` arrays and save them alongside `X` and `U` in each trajectory file (NPZ/HDF5). This keeps training code simple and avoids recomputation bugs.

### 2.6 Track map strategy (do we need more maps?)

- **To finish the project (DT warm-start works on the oval):** one map is enough. Randomize obstacle layouts aggressively.
- **To claim generalization:** add more maps.

Recommended progression:
- **Stage A/B:** 1 map + randomized obstacles (ship the full pipeline)
- **Stage C:** add **5–10 additional tracks** with diverse curvature/width; reserve **1–2 held-out test maps** never used in training

Splitting rule:
- split by `(map_id, scenario_seed)` so val/test are honest
- if multi-map, ensure **test maps are unseen** (strongest generalization check)


## 3) Decision Transformer design for this repo

### 3.1 What DT should predict

Goal: produce a **full warm-start** for IPOPT:
- `U_init[:,0..N]` (primary)
- `X_init[:,0..N]` via model rollout (secondary)

Inference style:
- **dynamics-in-the-loop rollout** (DT predicts `u_k`, then you propagate dynamics to get `x_{k+1}`), like ART-style trajectory generation.

### 3.2 Tokenization (what goes into the sequence)

A plain DT uses `(RTG_k, state_k, action_k)`.
For vehicle racing with obstacles, you should extend the state to include **scene + track context**.

Recommended per-step input vector (DT **observations** + scene context). The rollout still maintains the **full internal state** `X_full`.

- **Vehicle/path observation (DT input):**
  - default (recommended): `[ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world]`
  - optional: add `[dfz_long, dfz_lat]` if it empirically improves warm-start quality
  - note: even if omitted from DT input, `dfz_*` are preserved in `X_full` and propagated during rollout
- **Track features at s_k (DT input):**
  - minimum: `[kappa(s_k), half_width(s_k)]`
  - keep schema hooks for future 3D tracks: `[grade(s_k), bank(s_k)]` (store zeros for flat tracks)
- **Obstacle features (DT input; fixed-size, padded):**
  - compute from the **full episode obstacle list** (stored canonically)
  - choose `M` nearest obstacles *ahead* within a window
  - for each obstacle j: `[Δs_j, (e_obs_j - e_k), r_obs_j]`
  - sort by `Δs_j` and pad with zeros if < M

So:
- `state_aug_k = concat(vehicle_obs_k, track_feat_k, obstacle_feat_k)`

Why this representation:
- It makes DT generalize across obstacle placements (and later, across tracks) without needing images.

### 3.3 Return-to-go definition for minimum-time

Your optimizer minimizes time.
Define a per-step reward (and store it in the dataset):
- `dt_k = (t_{k+1} - t_k)` (or `dt_k ≈ ds / sdot_k`)
- `r_k = -dt_k`
Then:
- `RTG_k = \sum_{j=k}^{N-1} r_j = -(t_N - t_k)`.

At inference you can condition on a desired lap time:
- set `RTG_0 = -T_desired`.

Practical choice of `T_desired`:
- use the best known time from a simple heuristic baseline (or from a fast coarse solve) **minus a small margin**.

### 3.4 Model architecture

Start from the reference repo (`docs/reference_repos/decision-transformer-master`) but implement a clean, minimal PyTorch version for continuous actions.

- Causal GPT-style transformer
- modality embeddings:
  - RTG, state_aug, action
- learned timestep embedding (0..N)
- output heads predict action and next observation, trained with MSE

Core hyperparams (baseline first run — chosen to be close to DT defaults, but sized for track+obstacle tokens and the extra state head):

**Architecture**
- layers: **4**
- heads: **4**
- embedding dim (`d_model`): **128**
- MLP hidden dim: **512** (≈ 4×`d_model`)
- dropout: **0.1**
- context length `K`: **30**
- positional / timestep embedding: **learned** (0..N)

**Problem/obstacle tokenization (DT inputs)**
- number of obstacle slots: **M = 8**
- lookahead window in arc-length: **Δs_win = 60 m**
- per-obstacle features (padded, sorted by increasing Δs): **[Δs, (e_obs - e_k), r_obs]**
- always store the *full* obstacle list per episode; these per-step tokens are a derived view (optionally cached)

**Two-head outputs (ART-inspired, DT-faithful)**
- action head: predict **u_k = [δ_k, F_{x,k}]**
- state head: predict next-step **x_obs_{k+1} = [u_x,u_y,r,e,Δψ,pos_E,pos_N,yaw_world]**

**Loss weights**
- `lambda_x`: start at **0.5** (tune in **[0.1, 1.0]**; keep action loss primary)

**Training defaults**
- optimizer: **AdamW**
- learning rate: **1e-4**
- weight decay: **1e-4**
- betas: **(0.9, 0.95)**
- gradient clip: **1.0**
- batch size: **64** (sequence windows)
- LR warmup: **2,000** steps
- normalize: standardize `x_obs`, `u`, and RTG with dataset stats (store mean/std with the dataset)

**Small ablation sweep (keep everything else fixed)**
- `K ∈ {20, 30, 50}`
- layers `∈ {3, 4, 6}`
- `d_model ∈ {128, 256}`
- heads `∈ {1, 4}`
- `lambda_x ∈ {0.1, 0.5, 1.0}`

### 3.5 Training objective

Supervised loss:
\[
L = \|u_k-\hat u_k\|_2^2 \;+\; \lambda_x \|x^{obs}_{k+1}-\widehat{x}^{obs}_{k+1}\|_2^2
\]

Standard DT trick:
- sample random subsequences of length `K` from each trajectory during training.

Normalization:
- store dataset mean/std for states and actions, normalize both.

---

## 4) DT → IPOPT integration (the real deliverable)

Create: `planning/dt_warmstart.py`

### 4.1 Inference pipeline

Given a scenario:
1) build initial `state_aug_0` from baseline initial state + track/obstacle features
2) set `RTG_0 = -T_desired`
3) for k=0..N-1:
   - predict `a_k = [delta_k, fx_k]`
   - clip to actuator bounds
   - roll dynamics one step (spatial step ds) to get `x_{k+1}` (full internal state)
   - update `RTG_{k+1} = RTG_k - r_k` (use model-predicted dt or approximate from rollout)
   - append tokens
4) output `U_init` and `X_init`

### 4.2 Warm-start acceptance check (before calling IPOPT)

Reuse the same style of checks you already implemented for solver outputs:
- track bounds
- obstacle clearance (dense)
- `\dot s > 0`
- no NaNs

If DT fails acceptance:
- fall back to baseline initializer (or a cached previous IPOPT solution).

### 4.3 IPOPT refinement

Call `TrajectoryOptimizer.solve(..., X_init=X_init, U_init=U_init)`.

Measure:
- iterations
- solve time
- final cost
- success/acceptance

---

## 5) Evaluation + ablations (so you can claim “it works”)

Create: `experiments/eval_warmstart.py`

Metrics per scenario (report overall and per-map if multi-map):
- IPOPT success rate
- acceptance rate
- solve time and iteration count
- final lap time (cost)

Baselines:
1) baseline initializer only
2) baseline + acceptance retry schedule
3) DT warm-start + IPOPT

Ablations (quick wins):
- without obstacle features
- without track features
- different M (obstacle slots)
- different context length K

Plot:
- histograms of solve time / iterations
- scatter: cost vs min clearance

---

## 6) Stretch goals (after project is “done”)

1) **Multi-track generalization**
- generate multiple tracks (from `maps/reference/racetrack-database-master.zip`)
- include track features as above

2) **Feasibility-conditioned DT** (ART-style)
- add an extra token: constraint-to-go (remaining violations)
- set it to 0 at inference to bias feasibility

3) **Multi-objective conditioning**
- condition on both desired lap time and desired safety margin

---

## 7) Concrete checklist of code to add/modify

### Must modify
- `planning/optimizer.py` (Tier 1)
  - simplify objective to **minimum time + \(\Delta u\) regularizer**
  - add constraints:
    - `sdot_k >= eps_s`
    - `1 - kappa(s_k) * e_k >= eps_kappa`
  - keep obstacle constraints **node-only** in the NLP
  - keep dense post-solve obstacle checking + retries
  - inflate obstacle radius by a conservative vehicle footprint radius `vehicle_radius_m`

- `planning/config.py` (or wherever solver hyperparams live)
  - add `lambda_u`, `eps_s`, `eps_kappa`, and `vehicle_radius_m`

### Must add
- `data/schema.py`
- `data/generate_dataset.py`
- `data/datasets/<name>/...`
- `dt/` (or `models_dt/`): DT model + trainer
  - `dt/model.py`
  - `dt/dataset.py`
  - `dt/train.py`
  - `dt/eval.py`
- `planning/dt_warmstart.py`
- `experiments/eval_warmstart.py`

### Definition of done
- On a held-out batch of randomized obstacle scenarios (and held-out maps if enabled):
  - DT warm-start reduces median IPOPT iterations and solve time vs baseline
  - success/acceptance does not drop (or improves)
  - final lap time is within a small gap of baseline (or improves)
