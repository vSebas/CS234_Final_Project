#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun}"
DEVICE="${DEVICE:-cpu}"
AMP="${AMP:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "[postproj-ft2-pipeline] starting train -> benchmark pipeline"
echo "[postproj-ft2-pipeline] output_dir=${OUTPUT_DIR}"
echo "[postproj-ft2-pipeline] device=${DEVICE}"

OUTPUT_DIR="${OUTPUT_DIR}" \
DEVICE="${DEVICE}" \
AMP="${AMP}" \
NUM_WORKERS="${NUM_WORKERS}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash dt/run_train.sh oval_fatrop_improved_postproj_ft2

CKPT_B="${OUTPUT_DIR}/checkpoints/checkpoint_last.pt"
if [[ ! -f "${CKPT_B}" ]]; then
  echo "[postproj-ft2-pipeline] missing checkpoint after training: ${CKPT_B}" >&2
  exit 1
fi

echo "[postproj-ft2-pipeline] training finished; benchmarking CKPT_B=${CKPT_B}"
SKIP_EXISTING="${SKIP_EXISTING}" \
CKPT_B="${CKPT_B}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash experiments/run_fixed_gate_compare.sh

echo "[postproj-ft2-pipeline] pipeline complete"
