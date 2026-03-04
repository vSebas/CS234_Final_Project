#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"

data_dir="${DATA_DIR:-data/datasets}"
context_length="${CONTEXT_LENGTH:-30}"
num_workers="${NUM_WORKERS:-4}"
device="${DEVICE:-cuda}"
lambda_x="${LAMBDA_X:-}"
repair_weight="${REPAIR_WEIGHT:-}"
shift_fraction="${SHIFT_FRACTION:-}"
repair_fraction="${REPAIR_FRACTION:-}"
hard_repair_fraction="${HARD_REPAIR_FRACTION:-}"
python_bin="${PYTHON_BIN:-python}"

case "${mode}" in
  smoke)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/smoke_full_root_gpu}"
    batch_size="${BATCH_SIZE:-16}"
    num_epochs="${NUM_EPOCHS:-1}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.5"
    fi
    ;;
  full)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run1}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-10}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.5"
    fi
    ;;
  full_action)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run_lambda0}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-40}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.0"
    fi
    ;;
  full_action_hard)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run_lambda0_hard}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-40}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.0"
    fi
    if [[ -z "${shift_fraction}" ]]; then
      shift_fraction="0.75"
    fi
    if [[ -z "${repair_fraction}" ]]; then
      repair_fraction="0.10"
    fi
    if [[ -z "${hard_repair_fraction}" ]]; then
      hard_repair_fraction="0.15"
    fi
    ;;
  *)
    echo "Usage: $0 [smoke|full|full_action|full_action_hard]"
    echo "Optional env overrides: OUTPUT_DIR, DATA_DIR, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, CONTEXT_LENGTH, DEVICE, LAMBDA_X, REPAIR_WEIGHT, SHIFT_FRACTION, REPAIR_FRACTION, HARD_REPAIR_FRACTION"
    exit 1
    ;;
esac

if [[ -z "${repair_weight}" ]]; then
  repair_weight="1.0"
fi

mkdir -p "${output_dir}"

echo "[dt/run_train] mode=${mode}"
echo "[dt/run_train] data_dir=${data_dir}"
echo "[dt/run_train] output_dir=${output_dir}"
echo "[dt/run_train] batch_size=${batch_size}"
echo "[dt/run_train] num_epochs=${num_epochs}"
echo "[dt/run_train] num_workers=${num_workers}"
echo "[dt/run_train] context_length=${context_length}"
echo "[dt/run_train] device=${device}"
echo "[dt/run_train] python_bin=${python_bin}"
echo "[dt/run_train] lambda_x=${lambda_x}"
echo "[dt/run_train] repair_weight=${repair_weight}"
if [[ -n "${shift_fraction}" || -n "${repair_fraction}" || -n "${hard_repair_fraction}" ]]; then
  echo "[dt/run_train] shift_fraction=${shift_fraction:-unset}"
  echo "[dt/run_train] repair_fraction=${repair_fraction:-unset}"
  echo "[dt/run_train] hard_repair_fraction=${hard_repair_fraction:-unset}"
fi
echo "[dt/run_train] resume=auto"

cmd=(
  "${python_bin}" -u dt/train.py
  --data-dir "${data_dir}"
  --output-dir "${output_dir}"
  --context-length "${context_length}"
  --batch-size "${batch_size}"
  --num-epochs "${num_epochs}"
  --num-workers "${num_workers}"
  --device "${device}"
  --lambda-x "${lambda_x}"
  --repair-weight "${repair_weight}"
)

if [[ -n "${shift_fraction}" || -n "${repair_fraction}" || -n "${hard_repair_fraction}" ]]; then
  cmd+=(
    --shift-fraction "${shift_fraction}"
    --repair-fraction "${repair_fraction}"
    --hard-repair-fraction "${hard_repair_fraction}"
  )
fi

"${cmd[@]}"
