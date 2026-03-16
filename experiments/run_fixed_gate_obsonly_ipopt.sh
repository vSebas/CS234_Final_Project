#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
MAP_FILE="${MAP_FILE:-maps/Oval_Track_260m.mat}"
N="${N:-120}"
NUM_SCENARIOS="${NUM_SCENARIOS:-3}"
SEED="${SEED:-44}"
TIMEOUT_S="${TIMEOUT_S:-300}"

CKPT_A="${CKPT_A:-dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/checkpoints/checkpoint_best.pt}"
CKPT_B="${CKPT_B:-dt/checkpoints/oval_fatrop_improved_postproj_ft/checkpoints/checkpoint_last.pt}"

for CKPT in "${CKPT_A}" "${CKPT_B}"; do
  for PMODE in off soft full; do
    echo "[obs-only-ipopt] ckpt=${CKPT} pmode=${PMODE} scenarios=${NUM_SCENARIOS} seed=${SEED}"
    timeout "${TIMEOUT_S}" env PYTHONPATH=. "${PYTHON_BIN}" -u experiments/eval_warmstart.py \
      --checkpoint "${CKPT}" \
      --map-file "${MAP_FILE}" \
      --N "${N}" \
      --solver ipopt \
      --projection-mode "${PMODE}" \
      --num-scenarios "${NUM_SCENARIOS}" \
      --seed "${SEED}" \
      --min-obstacles 1 \
      --max-obstacles 4 \
      --no-save-compare-plots || echo "[obs-only-ipopt] WARNING: failed or timed out"
  done
done
