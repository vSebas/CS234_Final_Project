#!/usr/bin/env bash
set -euo pipefail

# GPU launcher for the corrected postproj_ft2 fine-tune + fixed-gate benchmark.
# Run from the project root:
#   bash dt/run_postproj_ft2_pipeline_gpu.sh

PYTHON_BIN="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

export PYTHON_BIN
export DEVICE
export AMP
export NUM_WORKERS
export OUTPUT_DIR
export SKIP_EXISTING

echo "[postproj-ft2-gpu] PYTHON_BIN=${PYTHON_BIN}"
echo "[postproj-ft2-gpu] DEVICE=${DEVICE}"
echo "[postproj-ft2-gpu] AMP=${AMP}"
echo "[postproj-ft2-gpu] NUM_WORKERS=${NUM_WORKERS}"
echo "[postproj-ft2-gpu] OUTPUT_DIR=${OUTPUT_DIR}"

bash dt/run_postproj_ft2_pipeline.sh
