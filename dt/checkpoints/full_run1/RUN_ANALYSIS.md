# Full Run 1

This directory is the self-contained bundle for the first full DT training run.

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
- `lambda_x = 0.5`
- total training time: about `1 h 54 min 34 s`
- best validation checkpoint: epoch `1`
- later epochs improved the auxiliary state term more than the action term

Best validation metrics:
- best `val_loss`: epoch `1`
- best `val_action_loss`: epoch `1`
- best `val_state_loss`: epoch `10`

Interpretation:
- training was numerically healthy
- the auxiliary state objective looked harmful for warm-start quality

## Warm-Start Results

Local copies of warm-start evaluations are under:
- `warmstarts/eval/`

Local copies of warm-start visualizations are under:
- `warmstarts/viz/`

Important result groups:
- `warmstarts/eval/full_run1_cmp_1/`
  - first single-scenario comparison
- `warmstarts/eval/full_run1_cmp_3/`
  - `3` fixed one-obstacle scenarios, pre-RTG-fix
- `warmstarts/eval/full_run1_cmp_3_noobs/`
  - `3` fixed no-obstacle scenarios, pre-RTG-fix
- `warmstarts/eval/full_run1_cmp_3_rtgfix/`
  - `3` fixed one-obstacle scenarios after per-track RTG calibration
- `warmstarts/eval/full_run1_cmp_3_noobs_rtgfix/`
  - `3` fixed no-obstacle scenarios after per-track RTG calibration
- `warmstarts/viz/full_run1_best/`
  - DT warm-start trajectory/state/control plots for the best checkpoint

Main conclusion for this run:
- DT warm-start was valid
- but it was worse than baseline on both the obstacle and no-obstacle benchmark sets
- RTG calibration helped the no-obstacle case somewhat, but did not solve the obstacle case

Recommended use of this run:
- use it as the baseline reference for the original two-head objective
- do not use it as the preferred obstacle warm-start model
- compare future runs against its local `warmstarts/eval/` copies, not against memory

## Global Notes

The broader cross-run analysis is in:
- `dt/TRAINING_ANALYSIS.md`
