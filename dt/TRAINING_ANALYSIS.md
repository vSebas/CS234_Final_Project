# DT Training Analysis

This note summarizes the completed `full_run1` Decision Transformer training run and the first downstream warm-start check.

## Run

Training run directory:
- `dt/checkpoints/full_run1`

Primary artifacts:
- `checkpoint_best.pt`
- `checkpoint_last.pt`
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
- `dt/checkpoints/full_run1/checkpoint_best.pt`

Reason:
- it corresponds to the best validation loss
- it also matches the best validation action loss in this run

Do not assume the final checkpoint is the best one for warm-starting.

## First Warm-Start Comparison

A first optimizer-level comparison was run on one fixed scenario with one obstacle using:
- `dt/checkpoints/full_run1/checkpoint_best.pt`

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
3. The current baseline DT can produce valid warm starts, but the first optimizer-level comparison was worse than baseline.
4. The next bottleneck is no longer loader/training stability; it is downstream warm-start usefulness.

## Recommended Next Steps

1. Run a small fixed benchmark set (`3-5` scenarios) with `checkpoint_best.pt` to check whether the first bad comparison is representative.
2. In the next training run, reduce the relative influence of the auxiliary state loss so the action head is prioritized more strongly.
3. Add early stopping or checkpoint selection based on `val_action_loss` or downstream warm-start performance, not only total validation loss.
4. Compare a few checkpoints directly in warm-start evaluation instead of assuming one scalar validation metric is enough.
