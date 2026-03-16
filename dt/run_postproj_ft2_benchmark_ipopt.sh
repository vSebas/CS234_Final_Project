#!/usr/bin/env bash
set -euo pipefail

# Benchmark-only launcher for the finished postproj_ft2 rerun checkpoint,
# forcing the fixed-gate comparison to use IPOPT.
# Run from the project root:
#   bash dt/run_postproj_ft2_benchmark_ipopt.sh

PYTHON_BIN="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
SOLVER="${SOLVER:-ipopt}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CKPT_A="${CKPT_A:-dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/checkpoints/checkpoint_best.pt}"
CKPT_B="${CKPT_B:-dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/checkpoints/checkpoint_last.pt}"

export PYTHON_BIN
export SOLVER
export SKIP_EXISTING
export CKPT_A
export CKPT_B

echo "[postproj-ft2-benchmark-ipopt] PYTHON_BIN=${PYTHON_BIN}"
echo "[postproj-ft2-benchmark-ipopt] SOLVER=${SOLVER}"
echo "[postproj-ft2-benchmark-ipopt] CKPT_A=${CKPT_A}"
echo "[postproj-ft2-benchmark-ipopt] CKPT_B=${CKPT_B}"

bash experiments/run_fixed_gate_compare.sh
