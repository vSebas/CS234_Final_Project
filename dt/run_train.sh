#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"

data_dir="${DATA_DIR:-data/datasets}"
context_length="${CONTEXT_LENGTH:-30}"
num_workers="${NUM_WORKERS:-4}"
device="${DEVICE:-cuda}"

case "${mode}" in
  smoke)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/smoke_full_root_gpu}"
    batch_size="${BATCH_SIZE:-16}"
    num_epochs="${NUM_EPOCHS:-1}"
    ;;
  full)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run1}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-10}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]"
    echo "Optional env overrides: OUTPUT_DIR, DATA_DIR, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, CONTEXT_LENGTH, DEVICE"
    exit 1
    ;;
esac

mkdir -p "${output_dir}"

echo "[dt/run_train] mode=${mode}"
echo "[dt/run_train] data_dir=${data_dir}"
echo "[dt/run_train] output_dir=${output_dir}"
echo "[dt/run_train] batch_size=${batch_size}"
echo "[dt/run_train] num_epochs=${num_epochs}"
echo "[dt/run_train] num_workers=${num_workers}"
echo "[dt/run_train] context_length=${context_length}"
echo "[dt/run_train] device=${device}"
echo "[dt/run_train] resume=auto"

python -u dt/train.py \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --context-length "${context_length}" \
  --batch-size "${batch_size}" \
  --num-epochs "${num_epochs}" \
  --num-workers "${num_workers}" \
  --device "${device}"
