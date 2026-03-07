# DT Warm-Start Repo Review — Detailed Prioritized Recommendations (Time/Success)

This document is a concrete, prioritized plan to improve a trajectory-optimization + Decision Transformer (DT) warm-start pipeline with the primary objective:

- **Primary:** *time to solve* and *success rate* (never worse than baseline)
- **Secondary:** solution quality (lap time / objective), only after stability and speed are achieved

It is written to be actionable: exact experiments, where to change code, what to log, and how to evaluate without confounds.

Status convention used below:
- `Current phase`: implemented and actively used in the current loop.
- `Phase 2`: agreed next change, not yet implemented end-to-end.
- `Historical / proposed`: kept for context, not part of the active loop yet.

---

## Executive summary

**Current state (observed behavior):**
- DT warm-start is **usable** (often succeeds), but still frequently **slower** than baseline.
- Raw DT rollout with **projection OFF** goes off-track early.
- With projection ON, DT initialization is **projection-dominated** (projection happens on most steps).
- Post-projection repairs were introduced to reduce distribution shift, but projection pressure remains high.

**Narrowed scope for next phase:**
- Oval only
- FATROP only (generation + eval + warm-start refinement)
- Start assumptions for training/eval:
  - start anywhere along centerline progress (`s0`)
  - `e0=0`, `dpsi0=0`, `uy0=0`, `r0=0`
  - fixed moderate `ux0=5.0 m/s`
- Scenario families:
  - Family 1: no obstacles (single fixed case)
  - Family 2: 1–4 random obstacles, generated once and frozen

## Current phase execution order (locked)

1. Keep scope locked: Oval only, FATROP only, frozen two-family benchmark.
2. Keep rollout start assumptions locked: `s0` centerline-progress only, `e0=dpsi0=uy0=r0=0`, `ux0=5.0`.
3. Fix training correctness first:
   - masked-loss normalization in `dt/train.py`,
   - padded-window sampling bias in `dt/dataset.py`.
4. Deconfound warmstart behavior:
   - projection modes `off|soft|full`,
   - projection/fallback reason counters,
   - auto-fallback quality gate when DT init is poor.
5. Refresh dataset only after steps 1-4 are in place (FATROP + locked assumptions).
6. Resume training to max 40 epochs with fixed benchmark checks every 2-3 epochs.
7. Select checkpoints by benchmark metrics first, validation loss second.
8. Run strict ablations in order (projection mode, postproj on/off, DT vs baseline).

---

# 1) Top root causes (ranked)

## R1 — Projection is substituting for the model
**Symptom:** projection happens on ~all steps (e.g., ~116/120).  
**Interpretation:** the optimizer does not receive a true DT rollout; it receives a repeatedly clipped/shifted trajectory.

**Why this hurts time:**
- Frequent clipping introduces **non-smoothness** (kinks) and destroys the shape the NLP solver likes as an initial guess.
- Projection can “fight” the model, yielding trajectories that are feasible-ish but not basin-friendly.

**What success looks like:**
- Projection occurs on **substantially fewer** steps (target: < 60–80% initially, then lower),
- projection magnitudes drop,
- DT warm-start becomes **faster** than baseline on median seeds.

---

## R2 — Training/inference distribution gap (post-projection induced states)
**Symptom:** raw rollout drifts off-track quickly without projection.

**Mechanism:** classic compounding error:
- the DT is trained on expert states (clean IPOPT manifold),
- but at inference it conditions on states produced by **DT + safety wrapper**,
- those induced states are out-of-distribution.

**What success looks like:**
- raw/soft rollouts survive longer,
- fallback triggers less,
- projection reason breakdown shifts toward “rare safety interventions” rather than “every-step clamping.”

---

## R3 — Masked MSE is implemented incorrectly
If masked losses are computed as:
```python
loss = MSE(pred * mask, target * mask)  # reduction=mean
```
then the denominator includes padded zeros, so windows with heavy padding contribute artificially low gradients.

This is especially harmful when you add many short segments (repairs, postproj), because it biases learning away from meaningful tokens.

**What success looks like:**
- after fixing, action loss curves become more interpretable,
- rollout stability improves (often without any architecture change).

---

## R4 — Dataset window indexing oversamples heavily padded windows
If you include every `start in range(T)` for each episode:
- near-terminal windows have `actual_len << K`,
- many samples are mostly padding,
- combined with R3 this can distort training dramatically.

**What success looks like:**
- training samples are dominated by full-length windows where possible,
- short windows are either excluded or explicitly weighted.

---

## R5 — Warm-start acceptance is too permissive
If DT warm-start is always accepted even when heavily rescued:
- you pay the worst-case: extra projection-induced kinks + extra IPOPT iterations,
- DT can be worse than baseline.

**What success looks like:**
- add a reject gate so DT is never worse than baseline in deployment evaluation.

---

# 2) Top 5 next experiments (exact knobs/files/metrics)

## Exp 1 — Projection ablation (OFF / SOFT / FULL) + DT acceptance gating
**Goal:** Deconfound DT vs projection, and ensure “DT never worse than baseline.”

**Files:**
- `planning/dt_warmstart.py`
- `experiments/eval_warmstart.py`

**Knobs:**
- Add `--projection-mode {off,soft,full}`
- Add `--dt-reject-mode {never,threshold}` with thresholds below

Status:
- `Current phase` (implemented in `planning/dt_warmstart.py` and `experiments/eval_warmstart.py`).

**Metrics to log per scenario:**
- solver: `success`, `solve_time_s`, `iterations`, `objective`, `lap_time_s`
- Warmstart: `proj_fraction`, `proj_total_mag`, `proj_max_mag`, `fallback_count`
- Projection reason breakdown: counts of `e_clip`, `dpsi_clip`, `ux_clip`, `uy_clip`, `r_clip`, `obs_push`

**Acceptance gating (initial thresholds to start):**
- Reject DT and fall back to baseline init if **any**:
  - `fallback_count > 0` (deployment mode; for diagnostics, start with `>2`)
  - `proj_fraction > 0.8`
  - `proj_total_mag > 100`  (tune after first run)
  - `proj_max_mag > 2.0`    (tune after first run)

**Outcome interpretation:**
- If `off` collapses fast → confirms model instability.
- If `soft` is better than `full` → projection is over-aggressive and creates solver-hostile kinks.
- If gating improves median time vs ungated DT → projection-dominated inits were hurting time.

---

## Exp 2 — Fix masked MSE (R3) and resume training 5–10 epochs
**Goal:** Correct training gradients (high leverage).

**File:** `dt/train.py`

**Change: masked MSE should divide by mask sum**
For example (pseudocode):
```python
sq = (pred - target)**2
sq = sq * mask
loss = sq.sum() / mask.sum().clamp(min=1)
```
Repeat for action loss (and optional state loss if used).

Status:
- `Current phase` (implemented in `dt/train.py` with valid-token normalization).

**Run:**
- Resume from your current best checkpoint.
- Evaluate every 1–2 epochs using the fixed gate (see evaluation protocol section).

**Metrics:**
- Downstream median `solve_time_s` and `iterations`
- `proj_fraction` + reason breakdown
- `fallback_count`

---

## Exp 3 — Reduce padded-window bias (R4)
**Goal:** Ensure training windows represent what the model sees at inference (full contexts).

**File:** `dt/dataset.py`

**Simple change option:**
Index only starts that can produce full windows where possible:
- for each episode of length T and context K:
  - valid full starts: `0 .. T-K`
  - optionally include a small fraction of short starts for coverage

**Better change option:**
Sample windows by selecting an end index and taking the last K steps, so most samples are full length.

Status:
- `Current phase` (implemented in `dt/dataset.py` with reduced padded-tail window sampling).

**Run:** short resume training (5–10 epochs) after Exp 2 (or combined).

**Metrics:** same as Exp 2.

---

## Exp 4 — Postproj vs hard ablation (composition, not just counts)
**Goal:** Measure the marginal value of postproj repairs.

Runs (hold everything else fixed):
- A: shifts + hard (no postproj)
- B: shifts + postproj (no hard)
- C: shifts + hard + postproj (current direction)

**Metrics:**
- Obstacle benchmark median time/iters (primary)
- Projection pressure + reason breakdown (diagnostic)

Status:
- `Current phase` for dataset selection/mix; use existing shard/fraction controls in `dt/run_train.sh` and `dt/train.py`.

---

## Exp 5 — RTG disambiguation experiment (avoid scale confound across shards)
**Hypothesis:** RTG meaning differs across shards (full lap vs short repair) and confuses conditioning.

**Do NOT immediately replace RTG with RTG/H.** That changes semantics.

Instead, prefer one of:
1) Add an `episode_kind` token/feature:
   - shift / repair / hard / postproj
2) Add `horizon_remaining` as a feature token (or scalar)
3) Add RTG-per-step as an *additional* channel, keep original RTG too

**Files:**
- `dt/dataset.py` (feature construction)
- `dt/model.py` (input dimension alignment)

**Metrics:**
- Downstream time/iters
- Projection reason breakdown (especially obstacle pushes)

Status:
- `Phase 2`.

---

# 3) Specific code changes for dt_warmstart projection/fallback/rollout

## C1 — Add `projection_mode` (off/soft/full)
**File:** `planning/dt_warmstart.py`

Add to warmstarter init:
- `projection_mode: str = "soft"` (less clamp-heavy default)
- Validate allowed values

In `_project_state`:
- if mode == `"off"`: return x unchanged, mag=0
- if mode == `"soft"`:
  - only clamp when violating hard safety bounds (don’t “shrink envelope” every step)
  - do not clip uy/r unless state unreasonable or safety-critical
  - obstacle push only when clearance proxy violates threshold
- if mode == `"full"`: current behavior

Status:
- `Current phase` (projection-mode plumbing is active end-to-end).

## C2 — Add projection reason counters
For each projection operation, increment:
- `e_clip_count`
- `dpsi_clip_count`
- `ux_clip_count`
- `uy_clip_count`
- `r_clip_count`
- `obs_push_count`

Store per-trajectory totals and fractions.

This is essential to test “obstacle conditioning not learned” (it should change obs_push stats).

## C3 — Make fallback a rejection trigger
Fallback should keep the rollout numerically alive, but if it triggers:
- mark the warmstart as **low confidence**
- reject DT warm-start (use baseline init) in deployment mode if fallback_count > 0
- for diagnostics/ablation mode, start with fallback_count > 2 to avoid over-rejecting early experiments

This is the fastest way to enforce “DT never worse than baseline.”

## C4 — Export per-step projection magnitudes for debugging
In rollout traces, store:
- `proj_mag_per_step` (length N)
- optionally also store reason code per step (small int enum)

This lets you identify s-positions where the policy systematically fails.

---

# 4) Training-data mix recommendations (hard vs postproj)

These are **by sampled training windows**, not by on-disk episode counts.

## General principles
1) Do not overweight tiny shards (causes repetition/overfit).
2) Start conservative; raise special data fractions only after stability improves.
3) Prefer postproj for fixing induced-state shift; use hard repairs for diversity.

## Recommended starting mix (Oval focus)
- shifts: **0.70–0.80**
- hard repairs: **0.10–0.15**
- postproj repairs: **0.10–0.15**
- standard repairs: **0.00–0.05** (keep small unless you ablate away)

If postproj shard is still small (few hundred segments), start postproj at **0.01–0.05** until you have enough volume.

## Ablation schedule
- Run 1: shifts 0.80, hard 0.10, postproj 0.10
- Run 2: shifts 0.80, hard 0.15, postproj 0.05
- Run 3: shifts 0.80, hard 0.05, postproj 0.15

Select checkpoints by downstream benchmark, not val loss.

---

# 5) Evaluation protocol to deconfound projection effects

## Fixed scenario set (must be deterministic)
Generate once and reuse forever:
- 1 no-obstacle fixed case
- 10 obstacle cases with 1–4 random obstacles (frozen once and reused)

Store scenarios to disk (pickle/json) and always evaluate on the same set.

## Four conditions per checkpoint
1) Baseline (cold start)
2) DT + projection=full
3) DT + projection=soft
4) DT + projection=off (with rejection gate)

Interpretation rule:
- `projection=off` is the truth test for direct DT warm-start behavior and should be treated as the primary DT KPI.
- `projection=soft` + gating is the secondary safety/deployment path.

## Reported metrics (per condition)
Primary:
- `success_rate` (must be >= baseline)
- `median_solve_time_s`
- `median_iterations`

Secondary:
- `lap_time_s` or final objective
- `p75_solve_time_s`, `p75_iterations` (tail behavior matters)

Diagnostics:
- `proj_fraction`
- `proj_total_mag`, `proj_max_mag`
- `fallback_fraction`
- projection reason breakdown (counts/fractions)

**Key success target:**
- DT (soft/full with gating) should be **< 1.0× baseline** on median time and not reduce success.

---

## Implementation checklist (fastest sequence)
1) Keep projection_mode + reason counters + gating active in `dt_warmstart.py`
2) Keep eval flags + exports active in `eval_warmstart.py`
3) Keep masked MSE fix active in `dt/train.py`
4) Keep reduced padded-window indexing active in `dt/dataset.py`
5) Run Exp 1–3 on the same scenario gate
6) Expand postproj dataset only if projection pressure remains high

---

## Notes on “more complex DT architecture”
Do **not** scale architecture until:
- projection pressure drops or is clearly attributable to missing capacity,
- training correctness issues (masking/padding) are resolved,
- and you have deconfounded projection effects (off/soft/full).

Architecture changes are high-cost and confounded if the pipeline is still dominated by wrapper corrections.

---

## Appendix: Suggested CLI patterns

### Projection ablation
Note: these flags are implemented in the current repo.
```bash
PYTHONPATH=. python experiments/eval_warmstart.py \
  --checkpoint <ckpt.pt> \
  --map-file maps/Oval_Track_260m.mat \
  --num-scenarios 10 \
  --seed 42 \
  --min-obstacles 1 \
  --max-obstacles 4 \
  --projection-mode full

PYTHONPATH=. python experiments/eval_warmstart.py \
  --checkpoint <ckpt.pt> \
  --map-file maps/Oval_Track_260m.mat \
  --num-scenarios 10 \
  --seed 42 \
  --min-obstacles 1 \
  --max-obstacles 4 \
  --projection-mode soft

PYTHONPATH=. python experiments/eval_warmstart.py \
  --checkpoint <ckpt.pt> \
  --map-file maps/Oval_Track_260m.mat \
  --num-scenarios 10 \
  --seed 42 \
  --min-obstacles 1 \
  --max-obstacles 4 \
  --projection-mode off \
  --dt-reject-mode threshold
```

### Short resume train (after masking/window fixes)
```bash
PYTHONPATH=. python dt/train.py \
  --resume <ckpt.pt> \
  --epochs 10 \
  --eval-every 1 \
  --select-by downstream_benchmark
```

(Adjust to your actual CLI names.)

---

## Concrete Next-Step Runbook (Current Repo)

This is the practical sequence to run now, using current scripts.

1. Complete post-proj target (FATROP):
```bash
POSTPROJ_SOLVER=fatrop TOTAL_TARGET=1000 SINGLE_MAP_CAP=0 ./data/run_postprojection_repairs_loop.sh
```

2. Resume current best run to max 40 epochs (benchmark-gated outside trainer):
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u dt/train.py \
  --data-dir data/datasets/Oval_Track_260m_shifts,data/datasets/Oval_Track_260m_repairs,data/datasets/Oval_Track_260m_repairs_hard,data/datasets/Oval_Track_260m_repairs_postproj \
  --output-dir dt/checkpoints/oval_hard400_train20 \
  --context-length 30 \
  --batch-size 64 \
  --num-epochs 40 \
  --num-workers 4 \
  --lambda-x 0.0 \
  --resume auto
```

3. Every 2-3 epochs, run fixed warm-start benchmark (FATROP only):
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u experiments/eval_warmstart.py \
  --checkpoint dt/checkpoints/oval_hard400_train20/checkpoints/checkpoint_best.pt \
  --map-file maps/Oval_Track_260m.mat \
  --num-scenarios 10 \
  --seed 42 \
  --min-obstacles 1 \
  --max-obstacles 4 \
  --N 120 \
  --solver fatrop \
  --projection-mode soft \
  --dt-reject-mode threshold \
  --dt-reject-fallback-max 0 \
  --dt-reject-proj-fraction-max 0.8 \
  --dt-reject-proj-total-max 100 \
  --dt-reject-proj-step-max 2.0
```

For the fixed no-obstacle case:
```bash
PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u experiments/eval_warmstart.py \
  --checkpoint dt/checkpoints/oval_hard400_train20/checkpoints/checkpoint_best.pt \
  --map-file maps/Oval_Track_260m.mat \
  --num-scenarios 1 \
  --seed 42 \
  --min-obstacles 0 \
  --max-obstacles 0 \
  --N 120 \
  --solver fatrop \
  --projection-mode soft \
  --dt-reject-mode threshold \
  --dt-reject-fallback-max 0 \
  --dt-reject-proj-fraction-max 0.8 \
  --dt-reject-proj-total-max 100 \
  --dt-reject-proj-step-max 2.0
```

4. Keep checkpoint selection benchmark-first:
- primary: median solve time / iterations / success
- secondary: val action loss

5. After retrain, run one controlled with-vs-without-postproj ablation:
- same training settings and seed
- only toggle inclusion/fraction of `*_repairs_postproj`
- compare on the same fixed benchmark set.
