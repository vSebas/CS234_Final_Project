# DT Training Analysis

This note summarizes the completed DT training runs and their downstream warm-start checks.

## Current Status

- Current best DT run:
  - `dt/checkpoints/full_run_lambda0`
- Current diagnosis:
  - nominal no-obstacle warm starts can be useful
  - obstacle-conditioned warm starts are still the main failure mode
  - rollout/wrapper diagnostics show heavy intervention even in nominal cases, with more pressure in obstacle cases
- Latest negative result:
  - `dt/checkpoints/full_run_lambda0_hard`
  - best validation loss improved substantially, but downstream warm-start performance on the fixed Oval benchmark regressed badly
- Current next step:
  - keep `full_run_lambda0` as the current best run
  - treat validation loss as a weak shortlist signal only
  - if continuing, move to post-projection labeled data (DAGGER-like) rather than more naive repair-mix tuning

## Next Program: Post-Projection Labeled Data (DAGGER-Lite)

This is the new primary plan after the negative `full_run_lambda0_hard` downstream result.

### Why this plan

- DT rollout is frequently corrected by projection/fallback.
- That means inference-time state distribution differs from clean expert data.
- Training only on clean expert trajectories is likely a distribution-mismatch failure mode.

### Core workflow

1. Export per-step rollout traces from DT + wrapper runs.
2. Keep trigger-selected states:
   - projection magnitude over threshold
   - fallback events
   - optional low-clearance proxy
3. Label selected states with short repair solves (optimizer teacher).
4. Build a separate post-projection shard.
5. Retrain with conservative mix.

Implementation status (now):
- `planning/dt_warmstart.py` supports optional per-step rollout trace collection via `collect_rollout_trace=True`.
- `experiments/eval_warmstart.py` supports trace export and trigger filtering with:
  - `--export-rollout-trace`
  - `--trace-projection-thresh`
  - `--trace-clearance-thresh`
  - `--trace-random-keep-prob`
  - `--trace-max-keep-per-scenario`
- exported trace rows are written as:
  - `warmstart_eval_<timestamp>_rollout_trace.jsonl`
- post-projection label builder is implemented:
  - `data/build_postprojection_repairs.py`
  - output shard pattern: `data/datasets/<map_id>_repairs_postproj`
- convenience wrappers:
  - `./data/run_postprojection_repairs.sh`
  - `./dt/run_postproj_train.sh`
- DT loader source-mix now supports a 4th source:
  - `repair_postproj` via `--postproj-repair-fraction`

### Practical guardrails

- Keep benchmark usage two-tier:
  - `3/3` seeds as smoke gate
  - `10/10` seeds as decision gate
- Add labeling budgets:
  - max solve time/iterations per candidate state
  - stop after collecting a target number of accepted labels
- Serialize solver-relevant trace context (do not rely on fragile reconstruction).
- Enforce diversity caps:
  - cap contribution per checkpoint
  - cap contribution per scenario
  - stratify sampling across `s` bins

### First retrain mix for this plan

- `85%` shifts
- `10%` standard repairs
- `0%` hard repairs
- `5%` post-projection repairs

Use this as the first conservative setting to avoid over-biasing into recovery-mode behavior.

## Brief Run History

1. `full_run1`
   - `lambda_x = 0.5`
   - stable training, but DT warm-start was worse than baseline on both no-obstacle and obstacle benchmarks
   - conclusion: auxiliary state loss was hurting the main objective
2. `full_run_lambda0`
   - `lambda_x = 0.0`
   - improved nominal no-obstacle warm-starting
   - last checkpoint beat baseline on the small no-obstacle benchmark
   - obstacle-conditioned warm starts remained worse than baseline
   - this is the current best run
3. `full_run_lambda0_repairs`
   - `lambda_x = 0.0`, `repair_weight = 4.0`
   - negative result
   - naive global repair upweighting hurt both nominal and obstacle performance
   - conclusion: do not pursue global repair weighting as-is
4. `full_run_lambda0_hard`
   - `lambda_x = 0.0`
   - explicit train-only source mix:
     - `75%` shifts
     - `10%` standard repairs
     - `15%` hard repairs
   - best validation action loss improved to `0.03182` at epoch `2`
   - downstream result on the fixed Oval benchmark was still strongly negative
   - conclusion: the hard-repair shard plus simple source mixing did not improve actual warm-start utility

## Run Bundles

Use these run-local bundles when you want one self-contained folder per training run.

### `full_run1`

Run directory:
- `dt/checkpoints/full_run1`

Local bundle note:
- `dt/checkpoints/full_run1/RUN_ANALYSIS.md`

Use this folder for:
- the original `lambda_x = 0.5` training artifacts
- local copies of the first warm-start evals
- local copies of the RTG-fix reruns

### `full_run_lambda0`

Run directory:
- `dt/checkpoints/full_run_lambda0`

Local bundle note:
- `dt/checkpoints/full_run_lambda0/RUN_ANALYSIS.md`

Use this folder for:
- the action-only `lambda_x = 0.0` training artifacts
- the checkpoint-shortlist warm-start comparisons
- the no-obstacle and obstacle comparison visualizations

### `full_run_lambda0_repairs`

Run directory:
- `dt/checkpoints/full_run_lambda0_repairs`

Local bundle note:
- `dt/checkpoints/full_run_lambda0_repairs/RUN_ANALYSIS.md`

Use this folder for:
- the weighted-repair action-only training artifacts
- the negative-result warm-start evals for the repair-weighted experiment
- the direct comparison point against `full_run_lambda0`

### `full_run_lambda0_hard`

Run directory:
- `dt/checkpoints/full_run_lambda0_hard`

Local bundle note:
- `dt/checkpoints/full_run_lambda0_hard/RUN_ANALYSIS.md`

Use this folder for:
- the hard-repair + explicit source-mix training artifacts
- the fixed Oval benchmark reruns for the early best checkpoint
- the negative downstream result showing that better validation did not translate to better warm-start quality

The rest of this document keeps the cross-run narrative in one place.

## Run

Training run directory:
- `dt/checkpoints/full_run1`

Primary artifacts:
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_last.pt`
- `metrics.jsonl`
- `loss_curves.png`
- `val_loss_curves.png`

## Dataset Coverage

The loader trained on the full dataset root:
- `data/datasets`

Important clarification:
- one epoch is one full pass over the training split
- the DT is trained on sliding context windows, not on raw transitions directly
- so one epoch covers all training windows once, not each raw timestep exactly once in isolation

For `full_run1`:
- total loaded episodes: `10664`
- train episodes: `9030`
- val episodes: `1634`

## Runtime

From `metrics.jsonl`:
- total training time: `6874.275 s`
- about `1 h 54 min 34 s`

Per-epoch time was roughly stable:
- about `11.3-11.8 min` per epoch

This was substantially faster than the initial smoke-run estimate, likely due to:
- dataset loader caching
- reduced plotting overhead during training
- general runtime cleanup after the first smoke pass

## Final Metrics

Epoch-by-epoch validation summary:

| Epoch | Train Loss | Val Loss | Val Action Loss | Val State Loss |
|------:|-----------:|---------:|----------------:|---------------:|
| 1 | 0.09240 | 0.04308 | 0.03197 | 0.02223 |
| 2 | 0.01514 | 0.04623 | 0.03781 | 0.01684 |
| 3 | 0.01013 | 0.04411 | 0.03674 | 0.01474 |
| 4 | 0.00782 | 0.04333 | 0.03627 | 0.01412 |
| 5 | 0.00649 | 0.04454 | 0.03753 | 0.01403 |
| 6 | 0.00562 | 0.04658 | 0.03976 | 0.01364 |
| 7 | 0.00501 | 0.04598 | 0.03906 | 0.01383 |
| 8 | 0.00454 | 0.04656 | 0.04002 | 0.01309 |
| 9 | 0.00415 | 0.04584 | 0.03925 | 0.01318 |
| 10 | 0.00384 | 0.04597 | 0.03956 | 0.01282 |

Best validation checkpoints:
- best `val_loss`: epoch `1`
- best `val_action_loss`: epoch `1`
- best `val_state_loss`: epoch `10`

## Interpretation

### What improved

- training loss decreased smoothly and stably
- validation state loss improved steadily through the run
- the trained checkpoint can now generate a valid DT warm start with the current robustified warm-start code

### What did not improve

- validation action loss was best at epoch `1`
- total validation loss was also best at epoch `1`
- later epochs mainly improved the auxiliary state-prediction term, not the action-prediction term

This suggests the current objective is over-optimizing the auxiliary state head relative to the action head.

## Curve Reading

From `val_loss_curves.png`:
- `val_state_loss` trends downward over the whole run
- `val_action_loss` rises after epoch `1`
- `val_total_loss` is mostly flat-to-worse after the first few epochs

Practical reading:
- the model is learning something useful about state evolution
- but the action predictions used for warm-starting are not improving after the earliest checkpoint

## Which Checkpoint To Use

Use:
- `dt/checkpoints/full_run1/checkpoints/checkpoint_best.pt`

Reason:
- it corresponds to the best validation loss
- it also matches the best validation action loss in this run

Do not assume the final checkpoint is the best one for warm-starting.

## First Warm-Start Comparison

A first optimizer-level comparison was run on one fixed scenario with one obstacle using:
- `dt/checkpoints/full_run1/checkpoints/checkpoint_best.pt`

Outputs:
- `results/warmstarts/eval/full_run1_cmp_1/warmstart_eval_20260302_205240.csv`
- `results/warmstarts/eval/full_run1_cmp_1/warmstart_eval_20260302_205240_summary.json`

Result on that scenario:

| Method | Success | Solve Time | IPOPT Iterations | Lap Time |
|-------|---------|-----------:|-----------------:|---------:|
| baseline | yes | 31.28 s | 155 | 15.75 s |
| baseline_retry | yes | 31.28 s | 155 | 15.75 s |
| dt_warmstart | yes | 51.24 s | 367 | 15.95 s |

Warm-start-specific details:
- warm-start acceptance: `100%`
- warm-start inference time: `1.12 s`
- total DT time including inference: `52.36 s`

Interpretation:
- the DT warm start is technically valid
- but on this tested scenario it is worse than baseline
- it increases IPOPT iterations and total solve time

This is only one scenario, so it is not enough for a final claim, but it is not an encouraging first comparison.

## Follow-Up Fixed Benchmark Sets

To check whether the first negative result was an outlier, two small fixed benchmark sets were run with:
- `dt/checkpoints/full_run1/checkpoints/checkpoint_best.pt`
- map: `maps/Oval_Track_260m.mat`
- seed: `42`
- `3` scenarios each

### Obstacle scenarios (`1` obstacle each)

Outputs:
- `results/warmstarts/eval/full_run1_cmp_3/warmstart_eval_20260302_232346.csv`
- `results/warmstarts/eval/full_run1_cmp_3/warmstart_eval_20260302_232346_summary.json`

Summary:

| Method | Success | Solve Time | IPOPT Iterations | Lap Time |
|-------|---------|-----------:|-----------------:|---------:|
| baseline | 3/3 | 35.58 +/- 2.92 s | 174.3 +/- 14.3 | 15.73 s |
| baseline_retry | 3/3 | 35.58 +/- 2.92 s | 174.3 +/- 14.3 | 15.73 s |
| dt_warmstart | 3/3 | 54.17 +/- 2.86 s | 372.0 +/- 27.6 | 15.86 s |

Warm-start-specific details:
- warm-start inference time: `1.392 s`
- warm-start acceptance: `100%`
- total DT time including inference: `55.56 s`

Interpretation:
- the first negative comparison was not a fluke
- on this obstacle set, DT warm-start remains consistently slower and requires far more IPOPT iterations

### No-obstacle scenarios (`0` obstacles)

Outputs:
- `results/warmstarts/eval/full_run1_cmp_3_noobs/warmstart_eval_20260302_233040.csv`
- `results/warmstarts/eval/full_run1_cmp_3_noobs/warmstart_eval_20260302_233040_summary.json`

Summary:

| Method | Success | Solve Time | IPOPT Iterations | Lap Time |
|-------|---------|-----------:|-----------------:|---------:|
| baseline | 3/3 | 41.23 +/- 0.46 s | 267.0 +/- 0.0 | 15.60 s |
| baseline_retry | 3/3 | 41.23 +/- 0.46 s | 267.0 +/- 0.0 | 15.60 s |
| dt_warmstart | 3/3 | 64.83 +/- 0.08 s | 475.0 +/- 0.0 | 15.66 s |

Warm-start-specific details:
- warm-start inference time: `1.137 s`
- warm-start acceptance: `100%`
- total DT time including inference: `65.96 s`

Interpretation:
- the DT warm-start is also worse than baseline even without obstacles
- that means the current issue is not limited to obstacle-conditioning quality
- the learned warm-start policy is currently a poor optimizer initializer in the nominal racing case as well

## RTG Conditioning Audit and Rerun

An audit of the DT inference path found that the previous evaluation runs were using an unrealistic default RTG target at warm-start inference time.

### What was wrong

The old evaluator path did not pass an explicit `target_lap_time` into the warm-starter. That meant `planning/dt_warmstart.py` fell back to a heuristic based on:
- track length
- initial speed `x0[0]`

For the Oval track evaluation setup:
- `track_length ≈ 260 m`
- `x0[0] = ux_min + 5.0 = 8.0 m/s`
- heuristic target was therefore about `-29.25 s`

That was far outside the actual training-data RTG range:
- shift episodes: roughly `-22.33` to `-15.52`
- repair episodes: roughly `-5.76` to `-2.05`

So the DT was being asked to behave more aggressively than anything it had seen during training.

### Fix

`experiments/eval_warmstart.py` was updated to infer a per-track target lap time from the generated shift dataset and pass it explicitly into `generate_warmstart()`.

For `Oval_Track_260m`, the calibrated per-track target is:
- `15.517 s`

### Rerun results with RTG fix

The same fixed `3`-scenario benchmark sets were rerun after the RTG calibration change.

#### No-obstacle scenarios (`0` obstacles) with RTG fix

Outputs:
- `results/warmstarts/eval/full_run1_cmp_3_noobs_rtgfix/warmstart_eval_20260303_002308.csv`
- `results/warmstarts/eval/full_run1_cmp_3_noobs_rtgfix/warmstart_eval_20260303_002308_summary.json`

Summary:

| Method | Success | Solve Time | IPOPT Iterations | Lap Time |
|-------|---------|-----------:|-----------------:|---------:|
| baseline | 3/3 | 47.21 +/- 5.48 s | 267.0 +/- 0.0 | 15.60 s |
| baseline_retry | 3/3 | 47.21 +/- 5.48 s | 267.0 +/- 0.0 | 15.60 s |
| dt_warmstart | 3/3 | 59.36 +/- 1.03 s | 390.0 +/- 0.0 | 16.39 s |

Compared to the pre-fix no-obstacle run:
- DT solve time improved from `64.83 s` to `59.36 s`
- DT iterations improved from `475.0` to `390.0`

Interpretation:
- the RTG fix helped the no-obstacle DT warm-start
- but DT is still clearly worse than baseline

#### Obstacle scenarios (`1` obstacle each) with RTG fix

Outputs:
- `results/warmstarts/eval/full_run1_cmp_3_rtgfix/warmstart_eval_20260303_002308.csv`
- `results/warmstarts/eval/full_run1_cmp_3_rtgfix/warmstart_eval_20260303_002308_summary.json`

Summary:

| Method | Success | Solve Time | IPOPT Iterations | Lap Time |
|-------|---------|-----------:|-----------------:|---------:|
| baseline | 3/3 | 36.85 +/- 2.75 s | 174.3 +/- 14.3 | 15.73 s |
| baseline_retry | 3/3 | 36.85 +/- 2.75 s | 174.3 +/- 14.3 | 15.73 s |
| dt_warmstart | 3/3 | 73.60 +/- 14.29 s | 502.7 +/- 116.2 | 17.40 s |

Compared to the pre-fix obstacle run:
- DT solve time worsened from `54.17 s` to `73.60 s`
- DT iterations worsened from `372.0` to `502.7`

Interpretation:
- the RTG heuristic was a real issue
- but fixing it did not solve the obstacle-conditioned warm-start problem
- the current DT policy remains a poor initializer, especially in the obstacle case

## Warm-Start Engineering Fixes Added During This Work

The warm-start path required robustness fixes before it could produce valid trajectories consistently:
- canonical obstacle key support (`s_m`, `e_m`, `radius_m`, `margin_m`)
- safer rollout fallback when dynamics propagation becomes unstable
- projection back into conservative track bounds
- simple obstacle-aware lateral projection during warm-start generation

These fixes make the DT warm-start generator usable for testing, but they also mean current warm-start performance reflects both:
- the trained checkpoint quality
- the fallback/projection logic

## Conclusions

1. The training run itself was successful and numerically stable.
2. For this configuration, `10` epochs was more than enough; the best action-related validation performance happened at epoch `1`.
3. The current baseline DT can produce valid warm starts, but it is consistently worse than baseline on both small obstacle and no-obstacle benchmark sets.
4. RTG conditioning was previously misconfigured at inference time, and fixing that helped nominal no-obstacle performance somewhat.
5. Even after the RTG fix, the DT initializer remains poor, especially in obstacle scenarios.
6. The current bottleneck is no longer loader/training stability; it is downstream warm-start usefulness.

## Recommended Next Steps

1. Keep the current fixed benchmark sets as the evaluation gate:
   - `3` no-obstacle scenarios
   - `3` one-obstacle scenarios
   - reuse the same seeded scenarios after every meaningful training change
2. In the next training run, make action the only objective:
   - set `lambda_x = 0.0`
   - keep the state head code for now, but remove its training influence completely
3. Compare a short checkpoint shortlist by downstream benchmark performance rather than by one proxy scalar:
   - `checkpoint_epoch_0000.pt`
   - best total validation checkpoint
   - best `val_action_loss` checkpoint
   - last checkpoint
   - choose by feasibility first, then median IPOPT iterations or solve time
4. Audit RTG conditioning before drawing strong conclusions from the next run:
   - compare the RTG values used at inference against the training-data distribution
   - avoid conditioning on unrealistically aggressive targets that can force the DT into infeasible or unstable controls
5. Add early stopping or checkpoint selection based on `val_action_loss` or downstream warm-start performance, not only total validation loss.
6. Re-run the same fixed obstacle and no-obstacle benchmark sets after the next training change so the comparison stays apples-to-apples.
7. If action-only training still fails, then move to robustness and data-distribution interventions:
   - oversample repair segments
   - add mild noisy-context training
   - add targeted longer-horizon or harder repair segments

## Follow-Up Run: `lambda_x = 0.0`

A second training run was completed with the auxiliary state-loss weight removed:
- run directory: `dt/checkpoints/full_run_lambda0`
- `lambda_x = 0.0`
- `40` epochs
- total training time: `27140.9 s` (`7 h 32 min 21 s`)

### Validation pattern

Best validation happened early:

| Epoch | Train Loss | Val Loss | Val Action Loss | Val State Loss |
|------:|-----------:|---------:|----------------:|---------------:|
| 1 | 0.04911 | 0.04308 | 0.04308 | 0.92961 |
| 2 | 0.00577 | 0.04271 | 0.04271 | 0.91018 |
| 3 | 0.00308 | 0.04106 | 0.04106 | 0.91060 |
| 4 | 0.00201 | 0.04249 | 0.04249 | 0.91663 |
| 5 | 0.00146 | 0.04743 | 0.04743 | 0.92435 |

After epoch `3`, validation degraded while train loss kept falling.

Best validation checkpoint:
- `dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_best.pt`
- best epoch: `3`
- best `val_loss = val_action_loss = 0.04106`

### Checkpoint shortlist comparison

The following checkpoints were evaluated on the fixed benchmark sets:
- `dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_epoch_0000.pt`
- `dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_best.pt` (epoch `3`)
- `dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_last.pt`

All comparisons below use the per-track RTG calibration.

#### No-obstacle benchmark (`3` scenarios)

Outputs:
- `results/warmstarts/eval/full_run_lambda0_epoch0_noobs/warmstart_eval_20260303_084835_summary.json`
- `results/warmstarts/eval/full_run_lambda0_best_noobs/warmstart_eval_20260303_084844_summary.json`
- `results/warmstarts/eval/full_run_lambda0_last_noobs/warmstart_eval_20260303_084850_summary.json`

Summary:

| Checkpoint | DT Solve Time | DT Iterations | DT Lap Time | Read |
|-----------|--------------:|--------------:|------------:|------|
| epoch 0 | 95.92 s | 636.0 | 16.09 s | much worse than baseline |
| best (epoch 3) | 47.54 s | 272.0 | 15.63 s | roughly equal to baseline |
| last | 43.34 s | 239.0 | 15.69 s | better than baseline |

Baseline on the same set:
- solve time: about `47.68 s`
- iterations: `267.0`
- lap time: `15.60 s`

Interpretation:
- removing the state-loss term helped nominal no-obstacle warm starts a lot
- the final checkpoint now beats baseline on this small no-obstacle set

#### One-obstacle benchmark (`3` scenarios)

Outputs:
- `results/warmstarts/eval/full_run_lambda0_epoch0_obs1/warmstart_eval_20260303_085600_summary.json`
- `results/warmstarts/eval/full_run_lambda0_best_obs1/warmstart_eval_20260303_085604_summary.json`
- `results/warmstarts/eval/full_run_lambda0_last_obs1/warmstart_eval_20260303_085610_summary.json`

Summary:

| Checkpoint | DT Solve Time | DT Iterations | DT Lap Time | Read |
|-----------|--------------:|--------------:|------------:|------|
| epoch 0 | 58.12 s | 348.0 | 15.97 s | worse than baseline |
| best (epoch 3) | 58.91 s | 346.7 | 15.92 s | worse than baseline |
| last | 65.16 s | 408.0 | 16.04 s | worst of the three |

Baseline on the same set:
- solve time: about `35.7-36.0 s`
- iterations: `174.3`
- lap time: `15.73 s`

Interpretation:
- obstacle-conditioned warm starts are still clearly worse than baseline
- the best obstacle-side checkpoint is early, not late
- the main remaining failure mode is now obstacle robustness, not nominal no-obstacle warm-start quality

### Rollout / wrapper diagnostics

To determine whether obstacle failures were coming only from the DT policy or
also from the rollout wrapper, diagnostic reruns were added for:
- `dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_best.pt`
- `dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_last.pt`

Diagnostics recorded:
- fallback count
- projection count
- projection total magnitude
- projection max-step magnitude

Diagnostic outputs:
- `dt/checkpoints/full_run_lambda0/warmstarts/eval/diag_best_noobs/`
- `dt/checkpoints/full_run_lambda0/warmstarts/eval/diag_best_obs1/`
- `dt/checkpoints/full_run_lambda0/warmstarts/eval/diag_last_noobs/`
- `dt/checkpoints/full_run_lambda0/warmstarts/eval/diag_last_obs1/`

Summary:

| Checkpoint | Scenario Set | Fallback Mean | Projection Count Mean | Projection Total Mean | Read |
|-----------|--------------|--------------:|----------------------:|----------------------:|------|
| best | no obstacles | 4.00 | 94.00 | 145.83 | wrapper intervenes heavily even in nominal case |
| best | 1 obstacle | 4.00 | 97.33 | 156.05 | obstacle case needs even more correction |
| last | no obstacles | 2.00 | 84.00 | 124.55 | nominal case is cleaner, but still heavily corrected |
| last | 1 obstacle | 2.67 | 88.00 | 125.50 | obstacle case still needs more correction and solves worse |

Interpretation:
- the wrapper is intervening a lot even when DT warm-starting works nominally
- obstacle cases generally increase projection pressure and worst-step correction size
- that means the obstacle bottleneck is not purely “the model is bad” or purely “the wrapper is bad”
- instead, the model is producing trajectories that already need heavy rescue, and obstacle cases amplify that rescue pressure

## Updated Conclusions

1. The original `lambda_x = 0.5` run was being harmed by the auxiliary state objective.
2. Removing the state-loss term materially improved no-obstacle warm-start behavior.
3. With `lambda_x = 0.0`, the final checkpoint is now better than baseline on the small no-obstacle benchmark.
4. Obstacle-conditioned warm starts remain clearly worse than baseline across the evaluated shortlist.
5. Rollout diagnostics show that the wrapper is intervening heavily even in nominal cases, and obstacle cases increase that intervention pressure further.
6. The project bottleneck has narrowed: the next work should focus on obstacle robustness and off-manifold recovery, not more generic training cleanup.

## Follow-Up Run: Weighted Repair Sampling

A third run tested naive global repair upweighting:
- run directory: `dt/checkpoints/full_run_lambda0_repairs`
- `lambda_x = 0.0`
- `repair_weight = 4.0`
- interrupted during epoch `12` after `11` full epochs
- accumulated train time at interruption: `10078.6 s` (`2 h 47 min 59 s`)

### Validation pattern

Best validation again happened immediately:

| Epoch | Val Loss | Val Action Loss |
|------:|---------:|----------------:|
| 1 | 0.04167 | 0.04167 |
| 2 | 0.04278 | 0.04278 |
| 3 | 0.04608 | 0.04608 |
| 4 | 0.04588 | 0.04588 |
| 5 | 0.04471 | 0.04471 |
| 6 | 0.04675 | 0.04675 |
| 7 | 0.04963 | 0.04963 |
| 8 | 0.05046 | 0.05046 |
| 9 | 0.04882 | 0.04882 |
| 10 | 0.05014 | 0.05014 |
| 11 | 0.05075 | 0.05075 |

Read:
- best `val_action_loss` was epoch `1`
- the same early-peak pattern remained
- validation alone did not show a win over `full_run_lambda0`

### Downstream benchmark result

This run was evaluated after fixing the checkpoint-layout stats lookup in `planning/dt_warmstart.py`, so the results below use the correct normalization stats.

Outputs are stored locally under:
- `dt/checkpoints/full_run_lambda0_repairs/warmstarts/eval/current_best_noobs/`
- `dt/checkpoints/full_run_lambda0_repairs/warmstarts/eval/current_best_obs1/`
- `dt/checkpoints/full_run_lambda0_repairs/warmstarts/eval/current_last_noobs/`
- `dt/checkpoints/full_run_lambda0_repairs/warmstarts/eval/current_last_obs1/`

Summary:

| Checkpoint | Scenario Set | DT Success | DT Solve Time | DT Iterations | Read |
|-----------|--------------|-----------:|--------------:|--------------:|------|
| best | no obstacles | 0/3 | n/a | n/a | catastrophic failure |
| best | 1 obstacle | 2/3 | 128.48 s | 617.0 | far worse than baseline |
| last | no obstacles | 3/3 | 126.43 s | 612.0 | far worse than baseline |
| last | 1 obstacle | 2/3 | 120.92 s | 553.5 | far worse than baseline |

Important clarification:
- DT warm-start generation itself was still accepted in these runs
- the failure is at the optimizer stage: IPOPT either fails or takes far more iterations

### Fair comparison against `full_run_lambda0` at epoch 11

To avoid comparing the weighted-repair run only against the fully completed `full_run_lambda0`, the same fixed benchmark sets were run on:
- `dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_epoch_0010.pt`

Outputs:
- `dt/checkpoints/full_run_lambda0/warmstarts/eval/epoch11_noobs/`
- `dt/checkpoints/full_run_lambda0/warmstarts/eval/epoch11_obs1/`

Results for `full_run_lambda0` epoch `11`:
- no obstacles: `3/3` success, `64.19 s`, `300.0` iterations
- 1 obstacle: `3/3` success, `290.46 s`, `372.7` iterations

Compared directly against the weighted-repair run state at epoch `11`:
- no obstacles: weighted-repair was worse
- 1 obstacle: weighted-repair was worse in success rate and worse in iterations

### Interpretation

This is a real negative result:
- naive global repair upweighting did not help obstacle robustness
- it also degraded nominal no-obstacle behavior
- the current best run remains `dt/checkpoints/full_run_lambda0`

The likely read is that simple repair oversampling changed the training distribution in a way that hurt the nominal manifold without solving the obstacle-conditioned distribution shift.

## Immediate Next Steps

The current recommended program is:

1. **Checkpoint selection by downstream benchmark**
   - keep using warm-start benchmark performance as the checkpoint-selection rule
   - feasibility first, then median IPOPT iterations or solve time
2. **Rollout / wrapper diagnostics**
   - use the newly added projection/fallback diagnostics when interpreting future runs
   - compare whether new data reduces wrapper intervention pressure, not just IPOPT iterations
3. **Expand the benchmark set**
   - keep the benchmark deterministic
   - grow beyond the current `3` no-obstacle + `3` one-obstacle cases once the diagnostics are in place
4. **Harder targeted data interventions**
   - skip generic repair upweighting and go directly to obstacle-specific data changes:
     - separate hard-repair shard
     - start-point bias toward low-clearance / near-obstacle cases
     - hotspot guidance from the `s` regions where projection or fallback spikes
     - perturb mainly `e` and `dpsi`
     - keep `uy` and `r` perturbations smaller and rarer
     - mixed horizons for the hard subset with current default mix:
       - `60% H=20`
       - `25% H=40`
       - `15% H=60`
     - save hardness and solver metadata with the shard
5. **Only then revisit sampling**
   - if targeted data exists, revisit weighting with a more selective policy rather than a global repair multiplier

### Next concrete experiment

The next concrete experiment should target obstacle robustness with a new hard-repair shard:

1. Keep the existing dataset unchanged.
2. Generate a separate hard-repair shard that:
   - focuses on obstacle scenarios first
   - biases starts toward low-clearance and near-obstacle cases
   - uses the diagnostic hotspot regions in `s` where projection/fallback pressure is high
   - for now, those hotspots are anchor points from bad obstacle scenarios, not a continuous heatmap
   - perturbs mainly `e` and `dpsi`
   - adds smaller and rarer `uy` / `r` perturbations
   - uses mixed horizons for the hard subset instead of a single new horizon:
     - `60% H=20`
     - `25% H=40`
     - `15% H=60`
   - stores hardness and solver metadata
3. Retrain with:
   - `lambda_x = 0.0`
   - existing shifts
   - existing repairs
   - the new hard-repair shard
   - no global repair multiplier
   - first intended training mix by sampled windows:
     - `75%` shifts
     - `10%` existing repairs
     - `15%` hard repairs
4. Current first-pass hard-repair scale:
   - target about `1200` hard repairs total across all tracks
   - with the current mixed-horizon default, this is about `3%` of the current dataset by transition count
4. Re-run the same fixed benchmark sets and the same rollout diagnostics.
5. Only if that still fails, consider more invasive additions later:
   - post-projection state training examples
   - broader multi-obstacle hard repairs
   - action-smoothing interventions
