#!/usr/bin/env bash
set -euo pipefail

hard_repairs="${HARD_REPAIRS:-1200}"
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

echo "[dt/run_hard_repair_train] step=hard_repairs"
echo "[dt/run_hard_repair_train] hard_repairs=${hard_repairs}"
echo "[dt/run_hard_repair_train] seed=${seed}"

HARD_REPAIRS="${hard_repairs}" \
SEED="${seed}" \
PYTHON_BIN="${python_bin}" \
./data/run_hard_repairs.sh

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
