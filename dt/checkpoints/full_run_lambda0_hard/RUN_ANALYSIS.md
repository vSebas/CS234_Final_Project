# full_run_lambda0_hard

This run tested the first hard-repair data intervention.

## Configuration

- run directory:
  - `dt/checkpoints/full_run_lambda0_hard`
- objective:
  - `lambda_x = 0.0`
- source mix during training:
  - `75%` shifts
  - `10%` standard repairs
  - `15%` hard repairs
- `repair_weight = 1.0`
- dataset root:
  - `data/datasets`

## Purpose

This run was meant to test whether:
- a separate `*_repairs_hard` shard
- plus explicit source mixing

would improve obstacle-conditioned warm-starting without repeating the failure mode of global repair upweighting.

## Training Readout

Best validation checkpoint:
- epoch `2`
- `val_action_loss = 0.03181595686212873`

This was the best validation action loss seen across the DT runs so far.

However, the familiar pattern still appeared:
- validation peaked very early
- later epochs kept lowering train loss but did not improve validation

So the benchmarked checkpoint for this run was:
- `checkpoints/checkpoint_epoch_0001.pt`

## Downstream Benchmark

Benchmark protocol:
- fixed Oval benchmark
- `3` no-obstacle scenarios
- `3` one-obstacle scenarios
- per-track calibrated RTG

### No obstacles

Outputs:
- `warmstarts/eval/epoch2_noobs/warmstart_eval_20260304_100307.csv`
- `warmstarts/eval/epoch2_noobs/warmstart_eval_20260304_100307_summary.json`

Summary:
- baseline: `58.50 s`, `267` iterations, `3/3` success
- dt_warmstart: `108.95 s`, `584` iterations, `3/3` success

### One obstacle

Outputs:
- `warmstarts/eval/epoch2_obs1/warmstart_eval_20260304_100307.csv`
- `warmstarts/eval/epoch2_obs1/warmstart_eval_20260304_100307_summary.json`

Summary:
- baseline: `45.10 s`, `174.3` iterations, `3/3` success
- dt_warmstart: `86.99 s`, `447.0` iterations, `3/3` success

Diagnostics:
- no obstacles:
  - fallback mean `3.0`
  - projection count mean `84.0`
  - projection total magnitude mean `155.1`
- one obstacle:
  - fallback mean `2.0`
  - projection count mean `99.3`
  - projection total magnitude mean `173.4`

## Interpretation

This run is a downstream negative result.

What looked promising:
- best validation action loss improved substantially

What actually happened:
- warm-start quality on the fixed Oval benchmark was much worse than baseline
- and worse than the earlier `full_run_lambda0` run family

So this run reinforces the current lesson:
- validation loss is useful for triage
- but it is not a reliable proxy for actual warm-start usefulness

## Conclusion

Do not treat this run as an improvement over:
- `dt/checkpoints/full_run_lambda0`

Current status after this run:
- `full_run_lambda0` remains the best run family
- the hard-repair shard plus simple `75/10/15` source mixing did not solve the downstream problem
