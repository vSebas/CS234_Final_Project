# Decision Transformer Implementation Notes

This document describes the DT pipeline as it is currently implemented in this repo.

It is intentionally implementation-focused:
- what files exist
- what assumptions they make
- what data format they consume
- what has already been fixed
- what is still missing

## Scope

Current DT-related files:
- `dt/model.py`
- `dt/dataset.py`
- `dt/train.py`
- `dt/eval.py`
- `planning/dt_warmstart.py`
- `experiments/eval_warmstart.py`

## Dataset Interface

The current on-disk dataset schema is the source of truth.

Episode arrays:
- node-aligned arrays:
  - `s_m`
  - `X_full`
  - `U`
  - `pos_E`
  - `pos_N`
  - `yaw_world`
  - `kappa`
  - `half_width`
  - `grade`
  - `bank`
- transition-aligned arrays:
  - `dt`
  - `reward`
  - `rtg`

Episode metadata is stored in `manifest.jsonl` with:
- `episode_id`
- `episode_type`
- `base_id`
- `map_id`
- `map_hash`
- `solver_config`
- `solver_config_hash`
- `obstacles`
- `s_offset_m`
- `npz_path`

`dt/dataset.py` assumes:
- `X_full` has shape `(T+1, 8)`
- `U` has shape `(T+1, 2)`
- `rtg` has shape `(T,)`
- DT supervision uses the first `T` nodes, aligned with `rtg`

## DT Observation Construction

The DT does not consume the full optimizer state directly.

Per-step DT observation:
- vehicle/path observation:
  - `[ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world]`
- track features:
  - `[kappa, half_width]`
- obstacle features:
  - `M` padded slots of `[ds, de, r]`

Default dimensions in `dt/dataset.py`:
- vehicle/path state dim: `8`
- track feature dim: `2`
- obstacle slots: `8`
- obstacle feature dim per slot: `3`
- total augmented state dim: `34`

Obstacle features are derived online from the stored episode obstacle list.

## Dataset Loading

`dt/dataset.py` currently supports:
- one shard directory such as `data/datasets/Oval_Track_260m_shifts`
- a comma-separated list of shard directories
- a root directory such as `data/datasets` containing shard subdirectories with `manifest.jsonl`

This support is used by both:
- `dt/train.py`
- `dt/eval.py`

## Train/Validation Split

The loader now splits by `base_id`, not by raw episode index.

Reason:
- shift episodes generated from the same periodic base lap are highly correlated
- splitting by episode caused leakage between train and validation

Current behavior:
- load all episode metadata
- group episodes by `base_id`
- shuffle the base trajectories
- assign whole base groups to train or validation

Current limitation:
- split assignments are not yet persisted to disk as explicit train/val/test manifests
- the split is performed inside the loader at runtime

## Normalization

Normalization statistics are now computed from training episodes only.

Current behavior:
- split by `base_id`
- compute state/action/RTG statistics from the train split
- reuse those train-derived statistics for both train and validation

This fixed the earlier evaluation leak where statistics were fit on the full dataset before splitting.

## Obstacle Lookahead Consistency

Training and warm-start inference now use consistent lap wrap-around logic for obstacle lookahead.

Previously:
- training dropped obstacles with negative `ds`
- inference wrapped negative `ds` across the lap

Current behavior:
- both paths treat obstacles just past the start/finish line as ahead when appropriate

## DT Model

`dt/model.py` implements a causal transformer with:
- learned timestep embedding
- linear embeddings for RTG, augmented state, and action
- stacked token order:
  - `(RTG_k, state_k, action_k)`
- transformer blocks with:
  - causal self-attention
  - pre-layer norm
  - MLP block

Outputs:
- action head:
  - predicts `[delta, Fx]`
- state head:
  - predicts the next observation
  - `[ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world]`

Current default hyperparameters:
- layers: `4`
- heads: `4`
- `d_model`: `128`
- `d_ff`: `512`
- dropout: `0.1`
- context length: `30`

## Training

`dt/train.py` currently:
- builds train/validation dataloaders via `create_dataloaders()`
- constructs the DT model
- trains with:
  - action MSE
  - next-state MSE
  - total loss = `action_loss + lambda_x * state_loss`
- uses AdamW with warmup
- saves:
  - numbered checkpoints
  - `checkpoint_last.pt` for stable crash recovery
  - `dataset_stats.npz`
  - config JSON
  - `metrics.jsonl` with append-only structured training logs
  - TensorBoard event files under `logs/`

Resume behavior:
- default CLI mode is `--resume auto`
- resume resolution prefers `checkpoint_last.pt`
- if that is missing, it falls back to the latest `checkpoint_epoch_*.pt`
- `--resume none` disables resume
- `--resume path/to/checkpoint.pt` resumes from a specific checkpoint

Crash handling:
- the trainer refreshes `checkpoint_last.pt` at the end of each epoch
- if training is interrupted or crashes after an epoch completes, the next run can continue from that stable checkpoint
- if an exception bubbles out of the training loop, the trainer also writes a recovery `checkpoint_last.pt` before re-raising

Logging behavior:
- TensorBoard scalars are written to the run-local `logs/` directory
- `metrics.jsonl` is append-only and persists across resumes
- the structured log includes `resume`, `train_step`, `epoch_end`, `interrupted`, `crash`, and `complete` events

Current training target:
- action prediction at the state token
- next-state observation prediction at the action token

Current limitation:
- there is still no explicit balancing between shifts and repairs
- repairs are a small minority of the total data, so uniform sampling may underuse them

## Evaluation

`dt/eval.py`:
- loads a checkpoint and normalization stats
- loads a dataset using the same input semantics as training
- computes:
  - normalized action MSE
  - normalized state MSE
  - denormalized action RMSE

This is model-level evaluation, not yet the final optimizer-level benchmark.

## Warm-Start Integration

`planning/dt_warmstart.py` currently:
- loads a trained DT checkpoint and normalization stats
- builds DT augmented observations online
- predicts actions autoregressively
- clips actions to control bounds
- rolls out the vehicle/path state

Important update:
- rollout now uses model-consistent path dynamics via the actual vehicle model
- the earlier simplified kinematic placeholder has been replaced

Current rollout details:
- road geometry is queried from the current track
- path dynamics are evaluated with `vehicle.dynamics_dt_path_vec(...)`
- spatial rollout uses RK4 integration on `dx/ds`

## Warm-Start Validation

Warm-start validation currently checks:
- no NaNs
- positive forward motion
- track bounds
- obstacle clearance

Obstacle validation now uses the same effective radius convention as the current optimizer/dataset pipeline:
- `radius_m + margin_m + obstacle_clearance_m + vehicle_radius_m`

This matches the current repo state.

Future cleanup:
- when `margin_m` is removed from the pipeline, update the helper in `planning/dt_warmstart.py` to the new single-clearance convention

## Experiment Wiring

`experiments/eval_warmstart.py` now passes:
- `obstacle_clearance_m`

into DT warm-start generation so validator behavior is aligned with the optimizer’s obstacle setting in that experiment path.

Current caveat:
- if nonzero `vehicle_radius_m` is introduced in experiment configs, it should also be threaded through all evaluation call sites

## What Is Implemented vs Missing

Implemented:
- DT dataset loader
- DT model
- DT training script
- DT evaluation script
- multi-shard loading
- split-by-`base_id`
- train-only normalization stats
- consistent obstacle wrap-around features
- model-consistent warm-start rollout
- optimizer-consistent obstacle validation

Still missing or incomplete:
- persisted train/val/test split artifacts
- weighted sampling / balancing for repairs
- final benchmark results for baseline vs DT warm-start
- broader experiment/config cleanup for nonzero `vehicle_radius_m`
- final obstacle metadata simplification once `margin_m` is removed

## Recommended Next Work

1. Persist train/val/test splits to disk.
2. Add weighted sampling or shard balancing for repairs.
3. Run end-to-end DT-vs-baseline warm-start benchmarks.
4. Thread `vehicle_radius_m` through experiment configs if footprint inflation is used.
