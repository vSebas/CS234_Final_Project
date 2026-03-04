#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/dataset_runs
wrapper_log_file="${WRAPPER_LOG_FILE:-results/dataset_runs/run_hard_repair_train_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${wrapper_log_file}") 2>&1

hard_repairs="${HARD_REPAIRS:-1200}"
hard_repair_chunk_size="${HARD_REPAIR_CHUNK_SIZE:-25}"
seed="${SEED:-0}"
output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run_lambda0_hard}"
data_dir="${DATA_DIR:-data/datasets}"
num_epochs="${NUM_EPOCHS:-40}"
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-4}"
context_length="${CONTEXT_LENGTH:-30}"
device="${DEVICE:-cuda}"
shift_fraction="${SHIFT_FRACTION:-0.75}"
repair_fraction="${REPAIR_FRACTION:-0.10}"
hard_repair_fraction="${HARD_REPAIR_FRACTION:-0.15}"
python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
map_names=(
  "Oval_Track_260m"
  "TRACK1_280m"
  "TRACK2"
  "TRACK3_300m"
  "TRACK4_315m"
  "TRACK5_330m"
)

echo "[dt/run_hard_repair_train] wrapper_log=${wrapper_log_file}"
echo "[dt/run_hard_repair_train] step=hard_repairs"
echo "[dt/run_hard_repair_train] hard_repairs=${hard_repairs}"
echo "[dt/run_hard_repair_train] hard_repair_chunk_size=${hard_repair_chunk_size}"
echo "[dt/run_hard_repair_train] seed=${seed}"

HARD_REPAIRS="${hard_repairs}" \
HARD_REPAIR_CHUNK_SIZE="${hard_repair_chunk_size}" \
SEED="${seed}" \
PYTHON_BIN="${python_bin}" \
./data/run_hard_repairs.sh

echo "[dt/run_hard_repair_train] step=verify_hard_repairs"
total_hard_episodes=0
present_shards=0
for map_name in "${map_names[@]}"; do
  shard_dir="data/datasets/${map_name}_repairs_hard"
  manifest_path="${shard_dir}/manifest.jsonl"
  if [[ -f "${manifest_path}" ]]; then
    episode_count=$(wc -l < "${manifest_path}")
    echo "[dt/run_hard_repair_train] hard_shard=${map_name} episodes=${episode_count}"
    total_hard_episodes=$((total_hard_episodes + episode_count))
    if [[ "${episode_count}" -gt 0 ]]; then
      present_shards=$((present_shards + 1))
    fi
  else
    echo "[dt/run_hard_repair_train] hard_shard=${map_name} episodes=0 (missing manifest)"
  fi
done

echo "[dt/run_hard_repair_train] total_hard_episodes=${total_hard_episodes}"
if [[ "${present_shards}" -eq 0 || "${total_hard_episodes}" -eq 0 ]]; then
  echo "[dt/run_hard_repair_train] verification failed: no non-empty *_repairs_hard shards were generated"
  exit 1
fi

echo "[dt/run_hard_repair_train] step=train"
echo "[dt/run_hard_repair_train] output_dir=${output_dir}"
echo "[dt/run_hard_repair_train] data_dir=${data_dir}"
echo "[dt/run_hard_repair_train] num_epochs=${num_epochs}"
echo "[dt/run_hard_repair_train] batch_size=${batch_size}"
echo "[dt/run_hard_repair_train] mix=${shift_fraction}/${repair_fraction}/${hard_repair_fraction}"

OUTPUT_DIR="${output_dir}" \
DATA_DIR="${data_dir}" \
NUM_EPOCHS="${num_epochs}" \
BATCH_SIZE="${batch_size}" \
NUM_WORKERS="${num_workers}" \
CONTEXT_LENGTH="${context_length}" \
DEVICE="${device}" \
PYTHON_BIN="${python_bin}" \
SHIFT_FRACTION="${shift_fraction}" \
REPAIR_FRACTION="${repair_fraction}" \
HARD_REPAIR_FRACTION="${hard_repair_fraction}" \
./dt/run_train.sh full_action_hard
