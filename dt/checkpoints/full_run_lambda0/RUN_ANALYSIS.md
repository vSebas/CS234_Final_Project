# Full Run Lambda 0

This directory is the self-contained bundle for the action-only DT training run.

## Training

Key files in this directory:
- `config.json`
- `metrics.jsonl`
- `loss_curves.png`
- `val_loss_curves.png`
- `action_loss_curves.png`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_last.pt`

Main training result:
- `lambda_x = 0.0`
- total training time: about `7 h 32 min 21 s`
- best validation checkpoint: epoch `3`
- later epochs overfit on validation, even though train loss kept falling

Best validation metrics:
- best `val_loss = val_action_loss`: epoch `3`
- later epochs improved train loss only, not validation quality

Interpretation:
- removing the auxiliary state-loss term materially changed downstream behavior
- the run improved nominal no-obstacle warm-start quality

## Warm-Start Results

Local copies of warm-start evaluations are under:
- `warmstarts/eval/`

Local copies of warm-start visualizations are under:
- `warmstarts/viz/`

Important evaluation groups:
- `warmstarts/eval/full_run_lambda0_epoch0_noobs/`
- `warmstarts/eval/full_run_lambda0_best_noobs/`
- `warmstarts/eval/full_run_lambda0_last_noobs/`
- `warmstarts/eval/full_run_lambda0_epoch0_obs1/`
- `warmstarts/eval/full_run_lambda0_best_obs1/`
- `warmstarts/eval/full_run_lambda0_last_obs1/`

Important visualization groups:
- `warmstarts/viz/full_run_lambda0_obs_compare/`
  - obstacle comparison for the best checkpoint
- `warmstarts/viz/full_run_lambda0_last_noobs_compare/`
  - no-obstacle comparison for the last checkpoint
- `warmstarts/viz/full_run_lambda0_last_obs_compare/`
  - obstacle comparison for the last checkpoint

Main conclusion for this run:
- the last checkpoint beat baseline on the small no-obstacle benchmark set
- obstacle-conditioned warm starts were still clearly worse than baseline
- the main remaining failure mode is obstacle robustness, not nominal racing-line initialization

Recommended use of this run:
- use `warmstarts/eval/` here when comparing checkpoint shortlist behavior
- use `warmstarts/viz/full_run_lambda0_last_noobs_compare/` as the nominal success example
- use `warmstarts/viz/full_run_lambda0_obs_compare/` and `warmstarts/viz/full_run_lambda0_last_obs_compare/` as the obstacle-failure examples

## Global Notes

The broader cross-run analysis is in:
- `dt/TRAINING_ANALYSIS.md`
