# Projection Ablation Results

This note documents the completed projection-mode ablations run on March 13, 2026 for the current Oval DT checkpoints.

## Scope

Compared checkpoints:
- `dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/checkpoints/checkpoint_best.pt`
- `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/checkpoints/checkpoint_last.pt`

Projection modes:
- `off`
- `soft`
- `full`

Frozen evaluation gates:
- no-obstacle gate:
  - map: `maps/Oval_Track_260m.mat`
  - obstacles: `0-0`
  - scenarios: `10`
  - seed: `52`
- obstacle gate:
  - map: `maps/Oval_Track_260m.mat`
  - obstacles: `1-4`
  - scenarios: `10`
  - seed: `44`

Solver:
- `ipopt`

Why IPOPT:
- FATROP fixed-gate evaluation was too slow / unreliable for interactive completion
- IPOPT was used here as the controlled diagnostic path

## Source Artifacts

Run logs:
- `results/dataset_runs/ctx50_full_ablation_20260313_023036.log`
- `results/dataset_runs/ft2_full_ablation_20260313_023036.log`

Canonical summary files used below:

`oval_fatrop_improved_ctx50_m6x192`
- noobs `off`:
  - `dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_023038_summary.json`
- noobs `soft`:
  - `dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_030901_summary.json`
- noobs `full`:
  - `dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_034718_summary.json`
- obs `off`:
  - `dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_025017_summary.json`
- obs `soft`:
  - `dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_032836_summary.json`
- obs `full`:
  - `dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_040644_summary.json`

`oval_fatrop_improved_postproj_ft2_rerun`
- noobs `off`:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_023038_summary.json`
- noobs `soft`:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_030852_summary.json`
- noobs `full`:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_034654_summary.json`
- obs `off`:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_025001_summary.json`
- obs `soft`:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_032817_summary.json`
- obs `full`:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_040634_summary.json`

Note:
- earlier `20260313_0115xx` / `0117xx` / `0119xx` / `0129xx` / `0131xx` files are one-scenario screening runs and are not the canonical full-gate ablation.

## Metrics Reported

For each run, track:
- success rate
- DT solve time
- baseline solve time
- DT total time (`warmstart_time + solve_time`)
- iterations
- warm-start acceptance rate
- fallback count mean
- projection fraction mean
- projection total magnitude mean

## Results

### 1) `oval_fatrop_improved_ctx50_m6x192`

#### No-obstacle gate (`10` scenarios)

| Mode | Baseline Solve | DT Solve | DT Total | Iterations | Acceptance | Fallback Mean | Projection Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off` | `57.29s` | `57.01s` | `59.96s` | `267.0` | `0%` | `10.0` | `0.000` |
| `soft` | `57.26s` | `56.92s` | `59.66s` | `267.0` | `0%` | `3.0` | `0.800` |
| `full` | `56.86s` | `56.68s` | `59.18s` | `267.0` | `0%` | `2.0` | `0.725` |

Read:
- DT solve time is marginally lower than baseline in all three modes.
- Total DT time is worse than baseline in all three modes once warm-start inference is included.
- `full` is the best of the three for this checkpoint on the no-obstacle gate.

#### Obstacle gate (`10` scenarios)

| Mode | Baseline Solve | DT Solve | DT Total | Iterations | Acceptance | Fallback Mean | Projection Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off` | `55.44s` | `52.95s` | `55.49s` | `240.7` | `0%` | `9.1` | `0.000` |
| `soft` | `55.28s` | `52.90s` | `55.47s` | `240.7` | `0%` | `3.0` | `0.744` |
| `full` | `54.86s` | `52.52s` | `55.22s` | `240.7` | `0%` | `2.0` | `0.729` |

Read:
- DT solve time is lower than baseline in all three modes.
- Total DT time is still slightly above baseline in all three modes.
- `full` is the best obstacle-gate regime for this checkpoint.

### 2) `oval_fatrop_improved_postproj_ft2_rerun`

#### No-obstacle gate (`10` scenarios)

| Mode | Baseline Solve | DT Solve | DT Total | Iterations | Acceptance | Fallback Mean | Projection Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off` | `56.45s` | `56.52s` | `59.15s` | `267.0` | `0%` | `4.0` | `0.000` |
| `soft` | `56.62s` | `56.56s` | `59.05s` | `267.0` | `0%` | `1.0` | `0.508` |
| `full` | `57.44s` | `57.35s` | `60.05s` | `267.0` | `0%` | `1.0` | `0.542` |

Read:
- `soft` is the best no-obstacle regime for this checkpoint.
- `soft` substantially reduces fallback relative to `off`.
- DT total time remains worse than baseline even in the best case.

#### Obstacle gate (`10` scenarios)

| Mode | Baseline Solve | DT Solve | DT Total | Iterations | Acceptance | Fallback Mean | Projection Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off` | `55.43s` | `53.35s` | `55.83s` | `240.7` | `0%` | `4.0` | `0.000` |
| `soft` | `54.48s` | `52.58s` | `55.21s` | `240.7` | `0%` | `1.0` | `0.440` |
| `full` | `55.65s` | `53.46s` | `56.29s` | `240.7` | `0%` | `1.0` | `0.491` |

Read:
- `soft` is the best obstacle-gate regime for this checkpoint.
- `soft` also gives the lowest fallback pressure.
- DT total time is still slightly worse than baseline.

## Comparison Summary

Best regime per checkpoint:
- `oval_fatrop_improved_ctx50_m6x192`
  - noobs: `full`
  - obs: `full`
- `oval_fatrop_improved_postproj_ft2_rerun`
  - noobs: `soft`
  - obs: `soft`

Best DT variant overall:
- `oval_fatrop_improved_postproj_ft2_rerun + projection=soft`

Why:
- noobs total time: `59.05s`
- obs total time: `55.21s`
- fallback means: `1.0` on both gates
- lower projection burden than `full`

## Final Verdict

1. The full projection ablation is complete.
2. `soft` is the best projection regime to carry forward.
3. `oval_fatrop_improved_postproj_ft2_rerun + projection=soft` is the best current DT operating point.
4. No DT variant beats baseline overall once `warmstart_time + solve_time` is counted.
5. Acceptance remains `0%` for all runs.
6. Iteration reduction remains `0%` for all runs.

Important gate note:
- these ablations were run with the default eval thresholds from `experiments/eval_warmstart.py`
- that default gate includes:
  - `dt_reject_fallback_max = 0`
  - `dt_reject_projection_step_max = 2.0`
- for the best current DT regime (`postproj_ft2_rerun + soft`), the observed rollout scale is closer to:
  - fallback `= 1`
  - projection max-step `= 4.7-4.8`
- so the full-ablation `0%` acceptance result should be read as:
  - a valid result under the current default gate
  - but not a pure measure of raw DT usefulness independent of gate policy

That means:
- projection choice matters,
- post-proj fine-tuning helped relative to the parent under `soft`,
- but the project bottleneck is still rollout acceptance / gating, not offline imitation quality.

## Recommended Follow-up

- Keep `projection=soft` as the main DT evaluation regime.
- Keep `projection=off` as the truth test for raw rollout quality.
- Do not launch another blind fine-tune yet.
- Next work should target:
  - acceptance logic / rejection thresholds
  - fallback triggers
  - understanding why useful-looking DT rollouts still never pass acceptance

## Relaxed-Gate Follow-Up

A follow-up diagnostic was run on the best current DT operating point:
- checkpoint:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/checkpoints/checkpoint_epoch_0006.pt`
- mode:
  - `projection=soft`
- solver:
  - `ipopt`
- diagnostic gate:
  - `dt_reject_fallback_max = 1`
  - `dt_reject_projection_fraction_max = 0.8`
  - `dt_reject_projection_total_max = 120.0`
  - `dt_reject_projection_step_max = 5.0`

Source artifacts:
- no-obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_133513_summary.json`
- obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_133513_summary.json`

Results:
- no-obstacle gate (`10` scenarios):
  - acceptance: `0%`
  - baseline solve mean: `58.61s`
  - DT total mean: `60.64s`
  - fallback mean/max: `6 / 6`
- obstacle gate (`10` scenarios):
  - acceptance: `20%`
  - baseline solve mean: `56.45s`
  - DT total mean: `59.41s`
  - iterations mean: `251.7` vs baseline `240.7`

Read:
- the default gate was definitely suppressing acceptance
- but relaxing the gate does not make the DT competitive with baseline yet
- the next bottleneck is no longer just threshold policy; it is rollout quality itself

## Rollout-Quality Diagnosis

This section inspects the per-scenario rows from the relaxed-gate run for the current best DT regime:
- checkpoint:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/checkpoints/checkpoint_epoch_0006.pt`
- mode:
  - `projection=soft`

### Obstacle gate: accepted vs rejected

Accepted scenarios (`2 / 10`):
- scenario ids:
  - `2`, `5`
- mean metrics:
  - solve time: `57.96s`
  - total time: about `60.62s`
  - iterations: `256.0`
  - fallback count: `1.0`
  - projection fraction: `0.654`
  - projection total magnitude: `45.82`
  - projection max-step: `3.34`
  - `e`-clip count: `77.0`
  - obstacle-push count: `1.5`

Rejected scenarios (`8 / 10`):
- dominant rejection reasons:
  - `gate:fallback_count` (`5`)
  - `gate:projection_total` (`2`)
  - direct obstacle collision at node `0` (`1`)
- mean metrics:
  - solve time: `56.41s`
  - total time: about `59.11s`
  - iterations: `250.6`
  - fallback count: `3.75`
  - projection fraction: `0.463`
  - projection total magnitude: `91.85`
  - projection max-step: `4.41`
  - `e`-clip count: `52.9`
  - obstacle-push count: `2.5`

Read:
- accepted obstacle cases are not actually solver-easier than the rejected ones:
  - accepted cases still have higher iteration count on average
  - accepted cases also have higher projection fraction and more `e` clipping
- what separates the accepted cases is not “clean rollout geometry”
  - it is mainly lower total projection magnitude and lower max-step magnitude
- this means the relaxed gate is admitting some runs, but those admitted runs are still not strong solver initializations

### No-obstacle gate

Initial relaxed-gate result:
- all `10 / 10` no-obstacle scenarios were rejected

Important confounder:
- the current eval path does not vary start progress
- `experiments/eval_warmstart.py` always uses the same `x0`
- `planning/dt_warmstart.py::generate_warmstart()` always initializes rollout progress with `s = 0.0`
- so this no-obstacle `10`-scenario gate is currently `10` repeats of the same start, not `10` distinct frozen anchors

Pattern:
- rejection reason:
  - `gate:fallback_count` for all `10`
- the per-scenario DT statistics are effectively identical across all starts:
  - fallback count: `6`
  - projection fraction: `0.342`
  - projection total magnitude: `91.58`
  - projection max-step: `4.85`
  - `e`-clip count: `41`
  - `dpsi`-clip count: `13`

Read:
- this is not scenario-specific noise
- the no-obstacle rollout appears to be hitting the same deterministic failure pattern for every start
- that points to a systematic rollout issue from the current fixed start
- but the benchmark should be fixed before interpreting this as failure across multiple distinct no-obstacle starts

Corrected rerun after fixing start-progress variation:
- source artifacts:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_151949_summary.json`
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_151949.csv`
- result:
  - acceptance improved from `0%` to `20%`
  - baseline solve mean: `49.06s`
  - DT total mean: `51.98s`
  - DT iterations mean: `206.8` vs baseline `191.8`
- accepted start-progress values:
  - about `4.12 m`
  - about `103.50 m`
- rejected starts had:
  - fallback mean `7.25`
  - projection total mean `102.54`
- accepted starts had:
  - fallback mean `1.0`
  - projection total mean `61.42`

Read:
- the earlier “all no-obstacle starts fail the same way” result was partly an eval artifact
- varying start progress does produce some accepted no-obstacle rollouts
- but even with corrected starts, DT still loses overall on total time and iterations

### Diagnosis summary

The current best DT regime still has two distinct problems:

1. Obstacle gate:
   - some runs can now pass the gate
   - but the accepted runs are still not better solver initializations than baseline
2. No-obstacle gate:
   - the rollout fails in the same way across every tested start
   - this suggests a systematic rollout/fallback issue even in the easiest family

Practical implication:
- the next step should be a targeted inspection of rollout mechanics on:
  - no-obstacle starts near accepted vs rejected progress regions
  - accepted obstacle cases vs baseline trajectories (why accepted still increases iterations)
