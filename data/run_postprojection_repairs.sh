#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
trace_jsonl="${TRACE_JSONL:-dt/checkpoints/full_run_lambda0/warmstarts/eval/postproj_trace_*/warmstart_eval_*_rollout_trace.jsonl}"
postproj_repairs="${POSTPROJ_REPAIRS:-600}"
postproj_per_map="${POSTPROJ_PER_MAP:-0}"
seed="${SEED:-0}"
output_root="${OUTPUT_ROOT:-data/datasets}"

mkdir -p results/dataset_runs
log_file="results/dataset_runs/postproj_repairs_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"

"${python_bin}" -u data/build_postprojection_repairs.py \
  --trace-jsonl "${trace_jsonl}" \
  --base-laps-dir data/base_laps \
  --output-root "${output_root}" \
  --num-segments "${postproj_repairs}" \
  --per-map-target "${postproj_per_map}" \
  --seed "${seed}" \
  --resume \
  2>&1 | tee "${log_file}"
