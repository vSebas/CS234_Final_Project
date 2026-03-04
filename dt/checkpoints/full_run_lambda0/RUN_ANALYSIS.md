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

## Rollout / Wrapper Diagnostics

Additional diagnostic reruns were executed to measure how much the warm-start
wrapper had to intervene during rollout.

Diagnostic CSVs:
- `warmstarts/eval/diag_best_noobs/`
- `warmstarts/eval/diag_best_obs1/`
- `warmstarts/eval/diag_last_noobs/`
- `warmstarts/eval/diag_last_obs1/`

Key readings:
- best checkpoint, no obstacles:
  - fallback mean `4.00`
  - projection count mean `94.00`
  - projection total magnitude mean `145.83`
- best checkpoint, 1 obstacle:
  - fallback mean `4.00`
  - projection count mean `97.33`
  - projection total magnitude mean `156.05`
  - projection max-step overall `5.88`
- last checkpoint, no obstacles:
  - fallback mean `2.00`
  - projection count mean `84.00`
  - projection total magnitude mean `124.55`
- last checkpoint, 1 obstacle:
  - fallback mean `2.67`
  - projection count mean `88.00`
  - projection total magnitude mean `125.50`
  - projection max-step overall `5.94`

Read:
- the wrapper intervenes heavily even in nominal no-obstacle cases
- obstacle cases generally require more projections and larger corrections
- this supports the idea that obstacle failures are partly a rollout/wrapper pressure problem, not only a pure policy-quality problem
- the DT policy still is not clean enough on its own, because even no-obstacle cases need substantial rescue

Main conclusion for this run:
- the last checkpoint beat baseline on the small no-obstacle benchmark set
- obstacle-conditioned warm starts were still clearly worse than baseline
- the main remaining failure mode is obstacle robustness, not nominal racing-line initialization
- after the later weighted-repair experiment failed, this remains the current best DT run

Recommended use of this run:
- use `warmstarts/eval/` here when comparing checkpoint shortlist behavior
- use `warmstarts/viz/full_run_lambda0_last_noobs_compare/` as the nominal success example
- use `warmstarts/viz/full_run_lambda0_obs_compare/` and `warmstarts/viz/full_run_lambda0_last_obs_compare/` as the obstacle-failure examples

## Global Notes

The broader cross-run analysis is in:
- `dt/TRAINING_ANALYSIS.md`
