#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
MAP_FILE="${MAP_FILE:-maps/Oval_Track_260m.mat}"
N="${N:-120}"
SOLVER="${SOLVER:-ipopt}"
NUM_SCENARIOS="${NUM_SCENARIOS:-10}"
SEED_NOOBS="${SEED_NOOBS:-52}"
SEED_OBS="${SEED_OBS:-44}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
SAVE_COMPARE_PLOTS="${SAVE_COMPARE_PLOTS:-0}"

CKPT_A="${CKPT_A:-dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/checkpoints/checkpoint_best.pt}"
CKPT_B="${CKPT_B:-dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/checkpoints/checkpoint_last.pt}"

plot_flag="--no-save-compare-plots"
if [[ "${SAVE_COMPARE_PLOTS}" == "1" ]]; then
  plot_flag="--save-compare-plots"
fi

run_eval() {
  local ckpt="$1"
  local pmode="$2"
  local min_obs="$3"
  local max_obs="$4"
  local seed="$5"

  echo "[projection-ablation] ckpt=${ckpt} pmode=${pmode} obs=${min_obs}-${max_obs} seed=${seed} scenarios=${NUM_SCENARIOS}"
  timeout "${TIMEOUT_S}" env PYTHONPATH=. "${PYTHON_BIN}" -u experiments/eval_warmstart.py \
    --checkpoint "${ckpt}" \
    --map-file "${MAP_FILE}" \
    --N "${N}" \
    --solver "${SOLVER}" \
    --projection-mode "${pmode}" \
    --num-scenarios "${NUM_SCENARIOS}" \
    --seed "${seed}" \
    --min-obstacles "${min_obs}" \
    --max-obstacles "${max_obs}" \
    ${plot_flag}
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[projection-ablation] WARNING rc=${rc} ckpt=${ckpt} pmode=${pmode} obs=${min_obs}-${max_obs}"
  else
    echo "[projection-ablation] DONE ckpt=${ckpt} pmode=${pmode} obs=${min_obs}-${max_obs}"
  fi
}

echo "[projection-ablation] started at $(date -Iseconds)"
for ckpt in "${CKPT_A}" "${CKPT_B}"; do
  for pmode in off soft full; do
    run_eval "${ckpt}" "${pmode}" 0 0 "${SEED_NOOBS}"
    run_eval "${ckpt}" "${pmode}" 1 4 "${SEED_OBS}"
  done
done
echo "[projection-ablation] finished at $(date -Iseconds)"
