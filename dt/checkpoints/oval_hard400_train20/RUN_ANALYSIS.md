# Oval Hard-400 Train-20 Run Analysis

## Run Summary

Run directory:
- `dt/checkpoints/oval_hard400_train20`

Config highlights:
- model: `4` layers, `4` heads, `d_model=128`
- objective: `lambda_x = 0.0` (action-only)
- epochs: `20`
- batch size: `64`
- train data shards:
  - `data/datasets/Oval_Track_260m_shifts`
  - `data/datasets/Oval_Track_260m_repairs`
  - `data/datasets/Oval_Track_260m_repairs_hard`
  - `data/datasets/Oval_Track_260m_repairs_postproj`

## Training Outcome

From `metrics.jsonl`:
- completed epochs: `20`
- best validation action loss: `0.007035` at epoch `8`
- final validation action loss: `0.013675` at epoch `19`
- final train loss: `0.001523`
- total train time: `2730.09 s` (`45m 30s`)

Top validation epochs (lowest `val_action_loss`):
1. epoch `8`: `0.007035`
2. epoch `13`: `0.008280`
3. epoch `12`: `0.009899`
4. epoch `10`: `0.010140`
5. epoch `3`: `0.012688`

Read:
- optimization is stable and convergent
- validation improves strongly early/mid training, then oscillates
- moderate overfitting appears after best epoch; checkpoint selection by downstream benchmark remains required

## Warm-Start Benchmark Results

### Fixed no-obstacle gate (`3` scenarios)
Source:
- `warmstarts/eval/oval_hard400_train20_best_noobs_20260306/warmstart_eval_20260306_151617_summary.json`

Results:
- baseline_retry: `3/3`, solve `40.38s`, iter `267.0`, lap `15.60s`
- dt_warmstart: `3/3`, solve `45.16s`, iter `311.0`, lap `15.66s`, total `46.25s`

### Fixed one-obstacle gate (`3` scenarios)
Source:
- `warmstarts/eval/oval_hard400_train20_best_obs1_20260306/warmstart_eval_20260306_152042_summary.json`

Results:
- baseline_retry: `3/3`, solve `30.95s`, iter `174.3`, lap `15.73s`
- dt_warmstart: `3/3`, solve `31.84s`, iter `196.3`, lap `15.95s`, total `32.98s`

### Smoke gate (`2` scenarios)
Source:
- `warmstarts/eval/smoke_20260306_135239/warmstart_eval_20260306_135239_summary.json`

Results:
- baseline_retry: `2/2`, solve `39.79s`, iter `250.0`
- dt_warmstart: `2/2`, solve `48.90s`, iter `338.0`

## Trajectory Quality Snapshot

Comparison plots (scenario 0, seed 42):
- `warmstarts/viz/oval_hard400_train20_traj_viz_20260306/noobs_scenario0_compare.png`
- `warmstarts/viz/oval_hard400_train20_traj_viz_20260306/obs1_scenario0_compare.png`

Observed behavior:
- trajectories are smooth and feasible in both no-obstacle and one-obstacle examples
- DT line choice is close to baseline with slightly more conservative pathing in some segments
- lap-time and solve-time gaps are small but consistently favor baseline on current gate

## Qualitative Analysis

### 1) Failure-mode shift vs previous runs

Compared to earlier runs (`full_run_lambda0`, `full_run_lambda0_hard`, `full_run_lambda0_repairs`), this run no longer shows the previous severe warm-start pathologies on the fixed Oval gate:
- no solver-failure pattern on the evaluated gate (`3/3` success for DT in noobs and obs1)
- no large iteration blow-up relative to prior DT runs
- no obvious unstable/garbage warm-start trajectories in the visual snapshots

Qualitatively, the regime shifted from “frequent heavy rescue / bad initialization” to “mostly usable initialization with a modest performance penalty.”

### 2) Where DT is still conservative/slower

Remaining behavior is not catastrophic but systematic:
- DT trajectories tend to be slightly more conservative in line choice in some segments (especially obstacle-conditioned segments and parts of the lower/right portions of the Oval snapshots).
- This is reflected in:
  - slightly higher lap-time objective vs baseline
  - slightly higher IPOPT iterations and solve time vs baseline

Interpretation:
- the policy appears to favor safer/easier-to-feasibilize trajectories rather than near-optimal baseline-like trajectories in difficult regions.

### 3) What this implies for post-projection data

The current gap profile matches a data-coverage issue more than a pure optimizer/model-capacity issue:
- large improvement already achieved with current architecture suggests core model pipeline is workable
- residual gap likely comes from undercoverage of wrapper-triggered, obstacle-conditioned states

So post-projection completion is expected to matter because it adds teacher labels exactly where current policy remains conservative or correction-heavy.

### 4) Confidence limits of the current gate

Current conclusions are directionally useful but not final:
- fixed gate size is still small (`3` noobs + `3` obs1 scenarios for decision comparison)
- qualitative behavior is consistent with the metric deltas, but this is still a narrow distribution slice

Required before strong claim:
- rerun the same comparison after post-proj completion and retraining
- include a larger deterministic gate to confirm this pattern holds beyond the small fixed set

## Comparison vs Previous Reference

Compared to `dt/checkpoints/full_run_lambda0_hard` benchmark reference (same fixed gates):
- no-obstacle DT:
  - solve time improved `78.26s -> 45.16s` (~`42%` faster)
  - iterations improved `584 -> 311` (~`47%` fewer)
- one-obstacle DT:
  - solve time improved `61.14s -> 31.84s` (~`48%` faster)
  - iterations improved `447 -> 196.3` (~`56%` fewer)

Interpretation:
- this run is a large downstream improvement over earlier checkpoints
- remaining gap is now a small baseline-vs-DT margin, not a catastrophic warm-start failure mode

## Artifacts

Training artifacts:
- `config.json`
- `metrics.jsonl`
- `loss_curves.png`
- `val_loss_curves.png`
- `val_action_loss_only.png`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_last.pt`

Warm-start eval artifacts:
- `warmstarts/eval/`

Warm-start viz artifacts:
- `warmstarts/viz/`

## Recommended Next Step

1. Complete `data/datasets/Oval_Track_260m_repairs_postproj` to target `1000`.
2. Retrain with the updated Oval-only shard mix.
3. Re-run the same fixed benchmark gates and compare against this run as the current baseline DT reference.
