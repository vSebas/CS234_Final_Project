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

**Goal:** generate a *large* and *diverse* offline dataset without paying for 1000+ expensive full-lap non‑periodic IPOPT solves. We do this with two complementary episode types:

- **Fix A — “circular shifts / start-state sampling” (cheap, high volume):**
  solve a small library of **periodic** optimal laps, then create many episodes by **rotating** the trajectory start index.
- **Fix B — short-horizon “repair” optimizations (moderate cost, off-manifold variety):**
  perturb the start state (e.g., random `e0`, `dpsi0`), then solve a **short segment** OCP that rejoins a nominal lap.

This avoids the failure mode you observed: **random `e0` + non‑periodic full lap is a much harder boundary-value problem and can be 10× slower** than periodic lap closure.

---

### 2.1 Canonical scenario + trajectory schema (versioned)

Create: `data/schema.py`

Store each dataset item as an **episode**:
\[
\tau = (P_{\text{episode}},\; \{P_k, X_k, U_k, dt_k, rtg_k\}_{k=0}^{T})
\]
where `P_episode` is the canonical scenario definition (map + obstacles + discretization), and `P_k` is any per-step derived view (e.g., “M nearest obstacles ahead”).

**Episode header (canonical, required):**
- `episode_id`
- `episode_type`: `"shift"` or `"repair"`
- `map_id` **and** `map_version_hash` (hash of the `.mat` / map params used)
- discretization:
  - `ds`
  - `T` (number of steps in this episode; `T=N` for full-lap episodes; `T=H` for repair segments)
  - `s0_abs_m` (absolute arc-length start along centerline, in `[0, L)`)
  - optional: `k0` (start index in the source periodic lap, for traceability)
- obstacles (canonical scene description):
  - `obstacles = [{s_obs_m, e_obs_m, r_obs_m, margin_m}, ...]` (Frenet form recommended)
  - for no-obstacle episodes: empty list

**Per-step arrays (required):**
- `s_abs_m[k]` (absolute arc-length at each step; wrap with `mod L`)
- `X_full[k]`: full model/optimizer state (keep internal dynamics complete)
  - include `dfz_long`, `dfz_lat` even if DT does not observe them
  - include `ux, uy, r, e, dpsi`
- `U[k]`: controls `[delta, Fx]`
- `dt[k]`: per-step time increment (store explicitly)
- `reward[k] = -dt[k]` (minimum-time reward)
- `rtg[k]`: return-to-go (backward cumulative sum of reward)

**Global pose cache (required; you explicitly asked to pin this):**
- `posE_m[k]`, `posN_m[k]` (ENU)
- `yaw_world_rad[k] = psi_centerline(s_abs_k) + dpsi_k`

These are derivable from `(map_id, s_abs, e, dpsi)` but saving them avoids future recomputation and makes debugging/plots cheap.

**Per-step map features (strongly recommended, cheap insurance):**
- `kappa[k] = kappa(s_abs_k)` (track curvature)
- `half_width[k]` (or equivalent bounds)
- `grade[k]`, `bank[k]` (store zeros for flat tracks; keeps schema stable)

**Optional cached DT features (derived; not canonical):**
- `obs_feat[k, M, feat_dim]`: padded “M nearest obstacles ahead” tensor
- `one_minus_kappa_e[k]`, `s_dot[k]` (fast validity/CtG labeling)

---

### 2.2 Stage 0 — Readiness gate (before generating lots of data)

Use your existing evaluation harness (`run_trajopt_demo.py`, `run_trajopt_batch_eval.py`) to confirm the optimizer is in a “dataset-ready” regime.

Minimum gates:
- no-obstacle periodic solves: near-100% accepted
- obstacle scenarios (curriculum count): stable accepted rate with dense post-check enabled
- median IPOPT iterations/time are not exploding (save logs + CSV/JSON like you already do in `results/trajectory_optimization/`)

If these gates fail, fix optimizer config first (init, weights, bounds, obstacle margins) before generating data.

---

### 2.3 Stage A — No-obstacle “base lap library” (few solves → many episodes) (Fix A)

**A1) Solve periodic, no-obstacle base laps (expensive step, done rarely)**

For each map:
- `convergent_lap=True`
- obstacles: none
- Tier 1 objective (min time + Δu)
- start with your proven discretization (e.g., `N=120` on the medium oval) then adjust if needed.

Generate `B` base laps per map:
- quick start: `B=5`
- stronger coverage: `B=20`

To avoid having 20 identical solutions, vary one knob slightly per solve:
- `lambda_u ∈ {0.002, 0.005, 0.01}` (or a small random jitter around your baseline)
- optionally: small changes in `ux_min`

Store each base lap as a canonical full-lap trajectory episode (with `episode_type="base"` optionally, not used for training directly unless you want).

**A2) Create many no-obstacle episodes by “circular shift / start-state sampling” (cheap step)**

Given a periodic lap with arrays `X[0..N]`, `U[0..N]`, `dt[0..N-1]`:

1) sample random `k0 ∈ [0, N-1]`
2) rotate (wrap) the sequence start:
   - `X_ep[k] = X[(k0 + k) mod N]`
   - `U_ep[k] = U[(k0 + k) mod N]`
   - `dt_ep[k] = dt[(k0 + k) mod N]`
3) set `s0_abs_m = s_grid[k0]`, and `s_abs_m[k] = (s0_abs_m + k*ds) mod L`
4) recompute `reward=-dt` and `rtg` (don’t “rotate” RTG; recompute cleanly)

This gives you a distribution of `e0` and speeds across the lap **without solving new NLPs**.

**A3) No-obstacle episode targets**
To produce ~1k no-obstacle training episodes cheaply:
- solve `B=10` base laps total
- sample `100` shifts from each → `1000` episodes

---

### 2.4 Stage A+ — No-obstacle “repair segments” (off-manifold variety) (Fix B)

Fix A yields “on-raceline” variety. Fix B teaches recovery from off-raceline initial states.

**B0) Choose a nominal segment**
- sample `(base_lap_id, k0)`
- define absolute start `s0_abs_m = s_grid[k0]`
- take nominal terminal target from the base lap at horizon `H`:
  - `kT = (k0 + H) mod N`
  - `x_target_obs = X_base_obs[kT]` (DT-observation subset)

**B1) Perturb the initial state (controlled, feasible)**
Start from the nominal state `x_nom0 = X_base[:, k0]` and perturb:
- `e0 += Uniform(-0.5, 0.5) m` (clamp within track bounds minus margin)
- `dpsi0 += Uniform(-0.10, 0.10) rad`
- optional (small): `uy0 += Uniform(-0.5, 0.5) m/s`, `r0 += Uniform(-0.2, 0.2) rad/s`
Keep `ux0` unchanged (or very lightly perturbed) to avoid infeasibility.

**B2) Solve a short-horizon segment OCP (moderate cost)**
- horizon: `H = 60` steps (>= 2× your DT context `K=30`; adjust if you change K)
- `convergent_lap=False` (segment, not periodic)
- enforce initial condition: `X[:,0] = x0_pert` (maskable constraint)
- **add a terminal “rejoin” penalty** so the segment doesn’t drift:
  \[
  J \;=\; t_H \;+\; \lambda_u\sum\|u_{k+1}-u_k\|^2 \;+\; w_T\|x_{obs,H}-x_{target\_obs}\|^2
  \]
  Choose `w_T` so the segment ends near the nominal manifold (start with `w_T` ~ 10–100 after normalization).

**B3) Implementation requirement: absolute s-offset support**
To solve a segment starting at arbitrary `s0_abs_m`, the optimizer must evaluate map quantities at:
- `s_abs_m[k] = (s0_abs_m + k*ds) mod L`

So add a `s0_offset_m` (parameter) to the optimizer’s geometry lookup path (curvature, centerline pose, bounds). This is also needed for obstacle Δs computations to be consistent.

**B4) Mix ratio**
A good default mix per split:
- **80%** shift episodes (Fix A)
- **20%** repair episodes (Fix B)

---

### 2.5 Stage B — Obstacle dataset (periodic obstacle laps + shifts + repairs)

Obstacle data is where you pay for solves again. Use caching + warm-start + curriculum to keep it sane.

**C1) Obstacle sampling (curriculum)**
Reuse the obstacle sampling distributions you already exercised in `run_trajopt_batch_eval.py`:

- obstacle count:
  - start: `n_obs ∈ {1,2}`
  - then: `n_obs ∈ {3,4,5,6}`
- radius: `r_obs_m ~ Uniform(1.2, 1.8)`
- margin: `margin_m ~ Uniform(0.6, 0.9)`
- longitudinal placement:
  - sample `s_obs_m` uniformly in `[0, L)` with a minimum spacing `Δs_min` (use your existing spacing logic)
- lateral placement:
  - sample within safe bounds: `e_obs ∈ [e_min + buffer, e_max - buffer]`
  - where `buffer = r_obs + margin + 0.4m` (conservative)

**C2) Solve periodic full-lap obstacle OCPs (expensive step)**
- `convergent_lap=True`
- warm-start from the nearest no-obstacle base lap solution on the same map
- keep node-only obstacle constraints in the NLP + dense post-check + retries

**Speed infrastructure (recommended)**
- Use a fixed maximum obstacle slots `Jmax` (e.g., 6) and pad unused obstacles so the NLP structure is constant.
- Cache NLPs by `(map_id/hash, N, ds, Jmax, convergent_lap, x0_mask)`.

**C3) Multiply each accepted obstacle lap into many episodes (Fix A)**
For each accepted obstacle lap:
- generate `Kshift=5..20` shifted episodes (cheap)
- each episode keeps the same obstacle list but starts at a different `s0_abs_m`

**C4) Obstacle repair segments (Fix B variant)**
As in Stage A+, but:
- start from a shifted obstacle lap state
- perturb `e0`, `dpsi0` modestly
- solve a short horizon segment with the same obstacle set
- add terminal rejoin penalty toward the nominal obstacle-avoiding lap segment

This teaches corrective behavior near obstacles without full-lap non-periodic costs.

---

### 2.6 Dataset sizing targets (shift + repair)

Targets below refer to **accepted episodes**. Let `N≈120` for full laps and `H≈60` for segments.

**Stage A (no obstacles, single map):**
- 10 periodic base laps
- 800 shift episodes + 200 repair episodes  → ~1k episodes total

**Stage B (obstacles, single map, curriculum):**
- 50–200 accepted periodic obstacle laps (depending on acceptance rate)
- per obstacle lap: 10 shift episodes (typical) → 500–2k obstacle episodes
- add 10–20% repair segments for recovery diversity

**Stage C (multi-map):**
- generate 5–10 additional tracks (see below)
- hold out 1–2 maps entirely for test
- scale obstacle episodes until you hit a total timesteps budget (e.g., 1–3M to start, 5–20M for strong generalization)

---

### 2.7 Track map strategy (including “new race tracks”)

To finish the project on the oval, one map is enough. To claim generalization, add more tracks.

**Sources already in the repo:**
- `maps/MAP1.mat`, `maps/Medium_Oval_Map_260m.mat`
- `maps/reference/racetrack-database-master.zip`
- `maps/reference/Procedural_race_track_Isaac_lab-main.zip`
- `maps/reference/procedural-tracks-master.zip`

**Recommended approach:**
1) Keep Stage A/B on the oval to stabilize the pipeline.
2) Add ~10 procedural tracks (different seeds) and export to your `.mat` format.
3) Split by `map_id` so test maps are unseen during training.

---

### 2.8 Concrete code to add/modify for dataset generation

**Must add (data scripts):**
- `data/build_base_laps.py` (solve and store periodic no-obstacle base laps)
- `data/make_shift_episodes.py` (turn base laps / obstacle laps into many shifted episodes)
- `data/repair_segments.py` (generate short-horizon repair episodes)
- `data/write_shards.py` (NPZ/HDF5 shards + `manifest.jsonl`)

**Must modify (optimizer support for segments):**
- `planning/optimizer.py`
  - accept a `s0_offset_m` parameter to evaluate map geometry at `s_abs_m[k]`
  - support segment mode with `H` and terminal penalty target
  - keep `x0` constraint parameterized (so caching remains valid across random `x0`)

**One manifest per dataset shard:**
- `data/datasets/<name>/manifest.jsonl` with paths + counts + hashes

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
- `data/build_base_laps.py`
- `data/make_shift_episodes.py`
- `data/repair_segments.py`
- `data/write_shards.py`
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
