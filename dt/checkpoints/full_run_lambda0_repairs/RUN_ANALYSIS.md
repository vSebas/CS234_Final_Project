# Full Run Lambda 0 Repairs

This directory is the self-contained bundle for the weighted-repair action-only DT run.

## Training

Key files in this directory:
- `config.json`
- `metrics.jsonl`
- `loss_curves.png`
- `val_loss_curves.png`
- `val_action_loss_only.png`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_last.pt`

Main training configuration:
- `lambda_x = 0.0`
- `repair_weight = 4.0`

Run status used for analysis:
- interrupted during epoch `12` after `11` completed epochs
- accumulated train time at interruption: about `2 h 47 min 59 s`

Best validation metrics:
- best `val_loss = val_action_loss`: epoch `1`
- validation worsened after the first epoch and did not recover through epoch `11`

Interpretation:
- naive global repair upweighting did not improve the validation proxy
- the same early-peak pattern from earlier runs remained

## Warm-Start Results

Local copies of warm-start evaluations are under:
- `warmstarts/eval/`

Important evaluation groups:
- `warmstarts/eval/current_best_noobs/`
- `warmstarts/eval/current_best_obs1/`
- `warmstarts/eval/current_last_noobs/`
- `warmstarts/eval/current_last_obs1/`

Main downstream result:
- this run is worse than `dt/checkpoints/full_run_lambda0`
- it is also worse than `full_run_lambda0` at epoch `11` in the direct fair comparison

Key reads:
- best checkpoint on no-obstacle set: `0/3` IPOPT successes
- best checkpoint on 1-obstacle set: `2/3` successes and very high iteration count
- last checkpoint on no-obstacle set: `3/3` successes, but `612` IPOPT iterations on average
- last checkpoint on 1-obstacle set: `2/3` successes and `553.5` IPOPT iterations on average

Important clarification:
- DT warm-start generation itself was accepted
- the failure is downstream: the optimizer either fails or needs much more work

## Conclusion

This run should be treated as a negative result.

What it showed:
- simple global repair oversampling is not the right fix
- it degraded nominal no-obstacle behavior
- it did not solve obstacle-conditioned warm-start quality

Current best run remains:
- `dt/checkpoints/full_run_lambda0`

Recommended use of this run:
- keep it as evidence that naive repair upweighting hurts performance
- do not use its checkpoints as the current best DT model

## Global Notes

The broader cross-run analysis is in:
- `dt/TRAINING_ANALYSIS.md`
