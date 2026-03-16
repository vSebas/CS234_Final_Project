#!/usr/bin/env bash
set -euo pipefail

wrapper_log_file="${WRAPPER_LOG_FILE:-results/dataset_runs/run_postproj_train_$(date +%Y%m%d_%H%M%S).log}"

postproj_repairs="${POSTPROJ_REPAIRS:-120}"
postproj_per_map="${POSTPROJ_PER_MAP:-0}"
max_attempts_factor="${MAX_ATTEMPTS_FACTOR:-4.0}"
max_trace_rows="${MAX_TRACE_ROWS:-0}"
nice_level="${NICE_LEVEL:-10}"
wall_time="${WALL_TIME:-30m}"
clear_cache_every="${CLEAR_CACHE_EVERY:-5}"
seed="${SEED:-0}"
python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
trace_jsonl="${TRACE_JSONL:-dt/checkpoints/full_run_lambda0/warmstarts/eval/postproj_trace_*/warmstart_eval_*_rollout_trace.jsonl}"

data_dir="${DATA_DIR:-data/datasets}"
output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run_lambda0_postproj}"
num_epochs="${NUM_EPOCHS:-20}"
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-4}"
shift_fraction="${SHIFT_FRACTION:-0.85}"
repair_fraction="${REPAIR_FRACTION:-0.10}"
hard_repair_fraction="${HARD_REPAIR_FRACTION:-0.00}"
postproj_repair_fraction="${POSTPROJ_REPAIR_FRACTION:-0.05}"

mkdir -p "$(dirname "${wrapper_log_file}")"
exec > >(tee -a "${wrapper_log_file}") 2>&1

echo "[dt/run_postproj_train] wrapper_log=${wrapper_log_file}"
echo "[dt/run_postproj_train] step=postproj_repairs"
echo "[dt/run_postproj_train] postproj_repairs=${postproj_repairs}"
echo "[dt/run_postproj_train] postproj_per_map=${postproj_per_map}"
echo "[dt/run_postproj_train] max_attempts_factor=${max_attempts_factor}"
echo "[dt/run_postproj_train] max_trace_rows=${max_trace_rows}"
echo "[dt/run_postproj_train] nice_level=${nice_level}"
echo "[dt/run_postproj_train] wall_time=${wall_time}"
echo "[dt/run_postproj_train] clear_cache_every=${clear_cache_every}"
echo "[dt/run_postproj_train] seed=${seed}"

PYTHON_BIN="${python_bin}" \
TRACE_JSONL="${trace_jsonl}" \
POSTPROJ_REPAIRS="${postproj_repairs}" \
POSTPROJ_PER_MAP="${postproj_per_map}" \
MAX_ATTEMPTS_FACTOR="${max_attempts_factor}" \
MAX_TRACE_ROWS="${max_trace_rows}" \
NICE_LEVEL="${nice_level}" \
WALL_TIME="${wall_time}" \
CLEAR_CACHE_EVERY="${clear_cache_every}" \
SEED="${seed}" \
OUTPUT_ROOT="${data_dir}" \
TOTAL_TARGET="${postproj_repairs}" \
./data/run_postprojection_repairs_loop.sh

echo "[dt/run_postproj_train] step=verify_postproj_repairs"
map_names=(Oval_Track_260m TRACK1_280m TRACK2 TRACK3_300m TRACK4_315m TRACK5_330m)
total_postproj_episodes=0
non_empty_postproj_shards=0
for map_name in "${map_names[@]}"; do
  manifest_path="${data_dir}/${map_name}_repairs_postproj/manifest.jsonl"
  if [[ -f "${manifest_path}" ]]; then
    episode_count="$(wc -l < "${manifest_path}")"
    total_postproj_episodes=$((total_postproj_episodes + episode_count))
    if [[ "${episode_count}" -gt 0 ]]; then
      non_empty_postproj_shards=$((non_empty_postproj_shards + 1))
    fi
    echo "[dt/run_postproj_train] postproj_shard=${map_name} episodes=${episode_count}"
  else
    echo "[dt/run_postproj_train] postproj_shard=${map_name} episodes=0 (missing manifest)"
  fi
done

echo "[dt/run_postproj_train] total_postproj_episodes=${total_postproj_episodes}"
if [[ "${non_empty_postproj_shards}" -eq 0 ]]; then
  echo "[dt/run_postproj_train] verification failed: no non-empty *_repairs_postproj shards were generated"
  exit 1
fi

echo "[dt/run_postproj_train] step=train"
echo "[dt/run_postproj_train] output_dir=${output_dir}"
echo "[dt/run_postproj_train] data_dir=${data_dir}"
echo "[dt/run_postproj_train] num_epochs=${num_epochs}"
echo "[dt/run_postproj_train] batch_size=${batch_size}"
echo "[dt/run_postproj_train] num_workers=${num_workers}"
echo "[dt/run_postproj_train] mix=${shift_fraction}/${repair_fraction}/${hard_repair_fraction}/${postproj_repair_fraction}"

PYTHON_BIN="${python_bin}" \
OUTPUT_DIR="${output_dir}" \
DATA_DIR="${data_dir}" \
NUM_EPOCHS="${num_epochs}" \
BATCH_SIZE="${batch_size}" \
NUM_WORKERS="${num_workers}" \
SHIFT_FRACTION="${shift_fraction}" \
REPAIR_FRACTION="${repair_fraction}" \
HARD_REPAIR_FRACTION="${hard_repair_fraction}" \
POSTPROJ_REPAIR_FRACTION="${postproj_repair_fraction}" \
./dt/run_train.sh full_action_postproj
