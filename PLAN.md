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


## 2) Dataset generation pipeline (training-focused; Fix A + Fix B)

The immediate goal is **to generate a trainable offline dataset** for DT (not to prove generalization claims yet). The dataset should be large enough in **timesteps**, contain **obstacle context**, and include some **off-manifold recovery** behavior—without requiring thousands of full-lap non-periodic solves.

### 2.0 Targets (define first, so generation is mechanical)

**Timesteps targets (baseline, good enough to start DT training):**
- Train: **~1.0M timesteps**
- Val: **~0.1M timesteps**
- Test: **~0.1M timesteps**

These targets assume full-lap episodes have ~`N` steps (e.g., `N=120`) and repair segments have `H` steps (e.g., `H=50`).

**Composition targets (recommended):**
- **60%** timesteps from **obstacle episodes** (DT must learn obstacle conditioning)
- **35%** timesteps from **no-obstacle episodes** (clean dynamics + speed control)
- **5%** timesteps from **repair segments** (off-manifold recovery) split ~50/50 with/without obstacles

If repair segments turn out expensive, reduce them to 1–2% initially and add later.

### 2.1 Canonical schema (episode = trajectory + problem context)

Create: `data/schema.py` and a JSON-serializable episode header + per-step arrays. This is still “just trajectories” in the ART sense:
\[
\tau = (P_k, x_k, u_k)_{k=0}^{T}
\]

**Episode header (canonical, required):**
- `episode_id`
- `episode_type`: one of `{shift, repair}`
- `base_id`: identifier of the base lap used (for shift) or the nominal lap used (for repair)
- `map_id` and `map_version_hash` (or filename hash)
- discretization:
  - full laps: `N`, `ds`
  - repair segments: `H`, `ds`
- `s0_abs` (absolute start progress on map in meters)
- full obstacle list for the episode (canonical scene description):
  - recommended Frenet form: `obstacles = [{s_obs, e_obs, r_obs, margin}, ...]`
- `solver_config_hash` (or embed the key IPOPT/optimizer settings)
- `solver_params` (explicit, so runs are reproducible without guessing):
  - `ux_min` (**fixed across the entire dataset**, match the production optimizer)
  - `lambda_u` (Δu regularizer weight; allow small variation, see §2.2.1)
  - `eps_s`, `eps_kappa`, `vehicle_radius_m` (and any other Tier-1 constants)

**Per-step arrays (required):**
- `s_abs[k]`: absolute arc-length for each step (modulo track length)
- `X_full[k]`: full optimizer/model state (keep internal dynamics complete)
  - include `dfz_long`, `dfz_lat` even if DT does not observe them
- `U[k]`: controls `[delta, Fx]`
- `dt[k]`: per-step time increment (`dt_k ≈ ds / sdot_k`)
- `reward[k] = -dt[k]`
- `rtg[k]`: return-to-go (backward cumulative sum of `reward`)

**Pose (required; cached for convenience + labeling):**
- `posE[k]`, `posN[k]` (global position)
- `yaw_world[k]` (global yaw)

**Per-step map features (strongly recommended):**
- `kappa[k] = kappa(s_abs[k])`
- `half_width[k]`
- `grade[k]`, `bank[k]` (store zeros for flat tracks; keeps schema stable)

**Optional cached fields (derived; not canonical):**
- `obs_feat[k, M, feat_dim]` padded “nearest-ahead obstacles” tensor used by DT
- `s_dot[k]`, `one_minus_kappa_e[k]` (fast validity / CtG labeling)

Write one manifest file per dataset shard:
- `data/datasets/<name>/manifest.jsonl`

### 2.2 Fix A: Base-lap library + circular-shift episodes (cheap volume)

#### 2.2.1 A1 — Build a **base lap library** (periodic solves; expensive but few)

For each track/map, solve a small number of **periodic** laps (full lap, no obstacles first; then obstacles). Periodic laps are fast and stable compared to non-periodic full-lap boundary-value runs.

- Use `convergent_lap=True`
- Use Tier-1 objective (min time + \(\Delta u\))
- Use your stable discretization (start with `N=120` unless you have evidence a different `N` is better)

**Optimizer settings policy for dataset generation (important):**
- `ux_min`: **do not vary**. Fix it to the value you will use when DT warm-starts IPOPT (production config). Varying `ux_min` mostly changes feasibility and can create a distribution mismatch.
- `lambda_u`: allow **small variation** around the production value to improve robustness, without turning the dataset into a mixture of different objectives:
  - define `lambda_u_prod` as the production value (e.g., `0.005` in your current experiments)
  - generate **80–90%** of base laps with `lambda_u = lambda_u_prod`
  - generate **10–20%** of base laps with `lambda_u ~ Uniform(0.8, 1.25) * lambda_u_prod`
- Record `lambda_u` in the episode header (`solver_params`) and `solver_config_hash`. DT does **not** need to condition on `lambda_u` initially; IPOPT will re-optimize with the fixed production objective anyway.

**How many base laps do we need?**
A single periodic base lap can generate up to **N distinct shift episodes** (one per unique start index). Each shift episode has ~N steps, so one base lap provides up to:
\[
\text{timesteps per base lap} \approx N\times N = N^2.
\]
For `N=120`, that’s **14,400 timesteps per base lap** if you generate all unique shifts.

To hit the baseline train target of ~1.0M timesteps *using mostly shift episodes*, you need roughly:
\[
B_{\text{total}} \approx \lceil 1{,}000{,}000 / 14{,}400 \rceil \approx 70 \text{ base laps.}
\]
Spread across 5 tracks, that’s ~14 base laps/track (split between no-obstacle and obstacle base laps).

**Recommended starting counts (for 5 tracks):**
- Per track:
  - `B_no = 6` no-obstacle base laps
  - `B_obs = 8` obstacle base laps (curriculum: 3 easy, 5 hard)
  - total `B_track = 14`
- Total base solves: ~70 (manageable)

**Note on hitting timestep targets efficiently:** prioritize obstacle base laps once the no-obstacle pipeline is stable.
- Obstacle shift episodes are the most informative for DT (they teach obstacle conditioning) and count toward the timestep budget.
- If solve budget is tight, you can reduce `B_no` (e.g., 3–4/track) and increase `B_obs` accordingly, keeping `ux_min` fixed and `lambda_u` mostly at `lambda_u_prod`.

#### 2.2.2 A2 — Generate **shift episodes** by circular rotation (no new NLP solve)

Given a periodic base lap `(X_base[0..N], U_base[0..N])` with `X_base[0]=X_base[N]`, create a new episode by choosing a start index `k0` and rotating:
- `X_ep[k] = X_base[(k0 + k) mod N]`
- `U_ep[k] = U_base[(k0 + k) mod N]`
- `dt_ep[k] = dt_base[(k0 + k) mod N]`
- `s0_abs = s_base[k0]`
- `s_abs_ep[k] = (s0_abs + k*ds) mod L`

**Important: cap per base lap.**
- Maximum *non-duplicate* shift episodes per base lap is **N** (unique `k0`).
- If you need more timesteps, add more base laps (A1) and/or add obstacles and repair segments; do not oversample the same base lap with repeated `k0`.

**Train/val/test split to avoid leakage:**
Shift episodes from the *same base lap* are highly correlated. Split by `base_id`:
- assign base laps to `{train, val, test}` (e.g., 80/10/10)
- generate shifts only within that split

### 2.3 Fix B: Short-horizon “repair” segment optimizations (off-manifold variety)

Goal: expose DT to states **near** the raceline but not exactly on it (recovery behavior), without solving full non-periodic laps.

#### 2.3.1 Segment problem definition
For a chosen base lap and start index `k0`:
- define absolute start `s0_abs = s_base[k0]`
- define a short horizon `H` (recommended `H=40–60`)

Create a perturbed initial condition `x0_pert` by perturbing a subset of states:
- `e0 += Uniform(-0.5, 0.5) m` (clipped to bounds)
- `dpsi0 += Uniform(-0.10, 0.10) rad`
- optional small perturbations: `uy0`, `r0`

Then solve a short-horizon OCP with:
- `convergent_lap=False`
- initial condition enforced: `X[:,0] = x0_pert` (parameterized so it plays well with caching)
- same Tier-1 constraints (dynamics, bounds, progress, Frenet non-singularity)
- **track geometry evaluated at absolute s:** `s_abs[k] = (s0_abs + k*ds) mod L`

**Implementation requirement:** add `s0_offset_m` (or equivalent) to the optimizer so curvature/width/centerline queries use absolute `s_abs`, not a fixed `0..L` grid.

#### 2.3.2 Terminal “rejoin” anchor (prevents free-end drift)
Add a soft terminal penalty to pull the segment back toward the nominal base lap state at the end of the horizon:
\[
J \;+=\; w_T\,\lVert x^{obs}_H - x^{obs}_{\text{base}}[(k0+H)\bmod N] \rVert^2
\]
where `x_obs` is the DT-observation subset (including global pose/yaw if used).

#### 2.3.3 Repair segments with obstacles (optional but valuable)
Repeat the same procedure but using an **obstacle base lap** as the nominal reference and keeping the obstacle list active.

Suggested mix:
- 50% repair segments no-obstacle
- 50% repair segments with obstacles

Keep repair segments a small percentage of the dataset initially (e.g., 1–5%) to control compute.

### 2.4 Obstacle base laps (use them to hit timesteps efficiently)

Obstacle conditioning is the point of the project, so we want a lot of obstacle-context data without doing thousands of solves.

**Approach:** solve a moderate number of *distinct obstacle scenarios* per track (periodic full-lap), then multiply each accepted solution into many shift episodes (A2).

Obstacle sampling should reuse your existing `run_trajopt_batch_eval.py` distributions:
- obstacle count curriculum: `{1,2}` → `{3..6}`
- radius `r ~ Uniform(1.2, 1.8)` m
- margin `m ~ Uniform(0.6, 0.9)` m
- placement in Frenet with your `edge_buffer` and `min_ds` logic

**Acceptance:** keep only accepted trajectories with dense post-check min clearance `>= -1e-3`.

**Current dataset defaults (implemented):**
- obstacle count: `min_obstacles=1`, `max_obstacles=4`
- radius `r ~ Uniform(0.8, 1.5)` m
- margin `m = 0.3` m
- clearance `c = 0.3` m
- shift generation: `--all-shifts` produces `N+1` shifts (k0=0..N), including one duplicate due to periodic closure

### 2.5 Splits and use of 6 tracks

We currently have **6 tracks**:
- `Oval_Track_260m`
- `TRACK1_280m`
- `TRACK2`
- `TRACK3_300m`
- `TRACK4_315m`
- `TRACK5_330m`

- If you mainly want a strong *training* dataset now (no generalization claim):
  - use all 6 tracks in train
  - split val/test by `base_id` and obstacle scenario seed

- If you want honest “unseen track” testing (recommended even during development):
  - Train/Val: 5 tracks
  - Test: 1 held-out track
  - Ensure no base laps from the test track appear in train.

**Do you need more than 6 tracks?**
- If the 6 tracks are reasonably diverse (different curvature/width profiles): **no** for the first trainable baseline.
- If they’re all “oval-like”: generate ~5–10 more using `create_tracks.py` seeds. This helps DT not overfit a single geometry family, but it’s not required to start training.

### 2.6 Concrete counts to hit the trainable baseline (example)

Assume `N=120`, `H=50`.

**Train target: 1.0M steps**
- Shift episodes contribute ~`N` steps each.
- If you generate *all N unique shifts* from each base lap, each base lap yields `N^2 = 14,400` steps.

A workable plan with 5 tracks:
- Per track:
  - `B_no = 6` no-obstacle base laps → `6*N` shift episodes
  - `B_obs = 8` obstacle base laps → `8*N` shift episodes
- Total shift timesteps across 5 tracks:
  - `(6+8) * 5 * N^2 = 14 * 5 * 14,400 = 1,008,000` timesteps

Then add repair segments as a small augmentation:
- `N_rep = 1000` segments total (across all tracks; ~166–167 per track)
- each contributes `H` steps → `25k–75k` extra steps

This hits the baseline without needing thousands of full-lap non-periodic solves.

### 2.7 Implementation checklist (dataset-only)

Create these scripts/modules (dataset-only scope):
- `data/schema.py` (episode header + arrays contract)
- `data/build_base_laps.py`
  - solve periodic no-obstacle base laps per track
  - solve periodic obstacle base laps per track (curriculum)
  - store as `base_laps/<map_id>/<base_id>.npz`
- `data/make_shift_episodes.py`
  - for each base lap in split, generate up to N unique `k0` shifts
  - recompute `rtg` after shifting
- `data/build_repair_segments.py`
  - sample `(base_id, k0)`, perturb `x0`, solve `H`-step segment with terminal anchor
- `data/write_shards.py`
  - write to `data/datasets/<name>/{train,val,test}_*.npz` + `manifest.jsonl`

Required optimizer tweaks to support segments cleanly:
- add `s0_offset_m` (absolute s) support for track geometry queries
- allow parameterized initial condition constraints (`x0_param`, `x0_mask`) so caching works

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
