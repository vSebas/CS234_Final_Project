#!/usr/bin/env bash
set -euo pipefail

# End-to-end loop:
# 1) export rollout traces from current checkpoint on frozen Oval scenarios
# 2) build/refresh post-projection repairs from those traces
# 3) fine-tune with improved model/config + cleaner source mix

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
checkpoint="${CHECKPOINT:-dt/checkpoints/oval_fatrop_clean_train20/checkpoints/checkpoint_best.pt}"
map_file="${MAP_FILE:-maps/Oval_Track_260m.mat}"

eval_solver="${EVAL_SOLVER:-ipopt}"
eval_noobs_seed="${EVAL_NOOBS_SEED:-42}"
eval_obs_seed="${EVAL_OBS_SEED:-43}"
eval_num_scenarios="${EVAL_NUM_SCENARIOS:-10}"
projection_mode="${PROJECTION_MODE:-off}"

postproj_target="${POSTPROJ_TARGET:-1000}"
postproj_solver="${POSTPROJ_SOLVER:-fatrop}"
postproj_output_suffix="${POSTPROJ_OUTPUT_SUFFIX:-repairs_postproj_fatrop_clean}"
postproj_max_trace_rows="${MAX_TRACE_ROWS:-0}"
postproj_timeout_s="${POSTPROJ_SOLVE_TIMEOUT_S:-10}"

train_mode="${TRAIN_MODE:-oval_fatrop_improved}"

mkdir -p results/dataset_runs
log_file="results/dataset_runs/oval_improved_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${log_file}") 2>&1

echo "[oval_improved_pipeline] checkpoint=${checkpoint}"
echo "[oval_improved_pipeline] map_file=${map_file}"
echo "[oval_improved_pipeline] eval_solver=${eval_solver}"
echo "[oval_improved_pipeline] projection_mode=${projection_mode}"
echo "[oval_improved_pipeline] postproj_solver=${postproj_solver}"
echo "[oval_improved_pipeline] postproj_target=${postproj_target}"
echo "[oval_improved_pipeline] postproj_output_suffix=${postproj_output_suffix}"
echo "[oval_improved_pipeline] train_mode=${train_mode}"

PYTHONPATH=. "${python_bin}" -u experiments/eval_warmstart.py \
  --checkpoint "${checkpoint}" \
  --map-file "${map_file}" \
  --num-scenarios "${eval_num_scenarios}" \
  --seed "${eval_noobs_seed}" \
  --min-obstacles 0 \
  --max-obstacles 0 \
  --solver "${eval_solver}" \
  --projection-mode "${projection_mode}" \
  --export-rollout-trace

PYTHONPATH=. "${python_bin}" -u experiments/eval_warmstart.py \
  --checkpoint "${checkpoint}" \
  --map-file "${map_file}" \
  --num-scenarios "${eval_num_scenarios}" \
  --seed "${eval_obs_seed}" \
  --min-obstacles 1 \
  --max-obstacles 4 \
  --solver "${eval_solver}" \
  --projection-mode "${projection_mode}" \
  --export-rollout-trace

trace_glob="dt/checkpoints/$(basename "$(dirname "$(dirname "${checkpoint}")")")/warmstarts/eval/Oval_Track_260m_obs*/warmstart_eval_*_rollout_trace.jsonl"
echo "[oval_improved_pipeline] trace_glob=${trace_glob}"

env \
  POSTPROJ_SOLVER="${postproj_solver}" \
  TRACE_JSONL="${trace_glob}" \
  OUTPUT_SUFFIX="${postproj_output_suffix}" \
  TOTAL_TARGET="${postproj_target}" \
  SINGLE_MAP_CAP=0 \
  MAX_TRACE_ROWS="${postproj_max_trace_rows}" \
  SOLVE_TIMEOUT_S="${postproj_timeout_s}" \
  ./data/run_postprojection_repairs_loop.sh

PYTHON_BIN="${python_bin}" ./dt/run_train.sh "${train_mode}"

echo "[oval_improved_pipeline] done"
