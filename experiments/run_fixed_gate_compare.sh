#!/usr/bin/env bash
set -euo pipefail

# Fixed-gate DT warmstart comparison across checkpoints and projection modes.
# Defaults are tuned for the current Oval FATROP phase.

PYTHON_BIN="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
MAP_FILE="${MAP_FILE:-maps/Oval_Track_260m.mat}"
N="${N:-120}"
SOLVER="${SOLVER:-fatrop}"
NUM_SCENARIOS="${NUM_SCENARIOS:-10}"
SEED_NOOBS="${SEED_NOOBS:-52}"
SEED_OBS="${SEED_OBS:-44}"
TIMEOUT_S="${TIMEOUT_S:-1200}"
SAVE_COMPARE_PLOTS="${SAVE_COMPARE_PLOTS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CKPT_A="${CKPT_A:-dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/checkpoints/checkpoint_best.pt}"
CKPT_B="${CKPT_B:-dt/checkpoints/oval_fatrop_improved_postproj_ft/checkpoints/checkpoint_last.pt}"

if [[ ! -f "${CKPT_A}" ]]; then
  echo "[fixed-gate] missing checkpoint A: ${CKPT_A}" >&2
  exit 1
fi
if [[ ! -f "${CKPT_B}" ]]; then
  echo "[fixed-gate] missing checkpoint B: ${CKPT_B}" >&2
  exit 1
fi

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
  local ckpt_label
  local map_id
  local eval_dir

  ckpt_label="$(basename "$(dirname "$(dirname "$ckpt")")")"
  map_id="$(basename "${MAP_FILE%.*}")"
  eval_dir="dt/checkpoints/${ckpt_label}/warmstarts/eval/${map_id}_obs${min_obs}-${max_obs}_seed${seed}_N${N}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${eval_dir}/warmstart_eval_*_summary.json" > /dev/null; then
    echo "[fixed-gate] skip existing ckpt=${ckpt_label} pmode=${pmode} obs=${min_obs}-${max_obs} seed=${seed} dir=${eval_dir}"
    return 0
  fi

  echo "[fixed-gate] ckpt=${ckpt_label} pmode=${pmode} obs=${min_obs}-${max_obs} seed=${seed} scenarios=${NUM_SCENARIOS}"
  timeout "${TIMEOUT_S}" \
    env PYTHONPATH=. "${PYTHON_BIN}" -u experiments/eval_warmstart.py \
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
}

echo "[fixed-gate] started at $(date -Iseconds)"
for ckpt in "${CKPT_A}" "${CKPT_B}"; do
  for pmode in off soft full; do
    run_eval "${ckpt}" "${pmode}" 0 0 "${SEED_NOOBS}" || echo "[fixed-gate] WARNING: run failed/timed out"
    run_eval "${ckpt}" "${pmode}" 1 4 "${SEED_OBS}" || echo "[fixed-gate] WARNING: run failed/timed out"
  done
done
echo "[fixed-gate] finished at $(date -Iseconds)"
