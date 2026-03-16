#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
checkpoint="${CHECKPOINT:-dt/checkpoints/oval_fatrop_clean_train20/checkpoints/checkpoint_best.pt}"
map_file="${MAP_FILE:-maps/Oval_Track_260m.mat}"

num_scenarios_noobs="${NUM_SCENARIOS_NOOBS:-30}"
num_scenarios_obs="${NUM_SCENARIOS_OBS:-30}"
seed_noobs="${SEED_NOOBS:-42}"
seed_obs="${SEED_OBS:-43}"
eval_solver="${EVAL_SOLVER:-ipopt}"
projection_mode="${PROJECTION_MODE:-off}"

postproj_solver="${POSTPROJ_SOLVER:-fatrop}"
postproj_target="${POSTPROJ_TARGET:-1000}"
postproj_timeout_s="${POSTPROJ_SOLVE_TIMEOUT_S:-10}"
postproj_output_suffix="${POSTPROJ_OUTPUT_SUFFIX:-repairs_postproj_fatrop_clean}"

echo "[run_postproj_from_clean] checkpoint=${checkpoint}"
echo "[run_postproj_from_clean] map_file=${map_file}"
echo "[run_postproj_from_clean] eval_solver=${eval_solver} projection_mode=${projection_mode}"
echo "[run_postproj_from_clean] postproj_solver=${postproj_solver} target=${postproj_target}"

PYTHONPATH=. "${python_bin}" -u experiments/eval_warmstart.py \
  --checkpoint "${checkpoint}" \
  --map-file "${map_file}" \
  --num-scenarios "${num_scenarios_noobs}" \
  --seed "${seed_noobs}" \
  --min-obstacles 0 \
  --max-obstacles 0 \
  --solver "${eval_solver}" \
  --projection-mode "${projection_mode}" \
  --export-rollout-trace

PYTHONPATH=. "${python_bin}" -u experiments/eval_warmstart.py \
  --checkpoint "${checkpoint}" \
  --map-file "${map_file}" \
  --num-scenarios "${num_scenarios_obs}" \
  --seed "${seed_obs}" \
  --min-obstacles 1 \
  --max-obstacles 4 \
  --solver "${eval_solver}" \
  --projection-mode "${projection_mode}" \
  --export-rollout-trace

trace_glob="dt/checkpoints/oval_fatrop_clean_train20/warmstarts/eval/Oval_Track_260m_obs*/warmstart_eval_*_rollout_trace.jsonl"
echo "[run_postproj_from_clean] trace_glob=${trace_glob}"

env \
  POSTPROJ_SOLVER="${postproj_solver}" \
  TRACE_JSONL="${trace_glob}" \
  OUTPUT_SUFFIX="${postproj_output_suffix}" \
  TOTAL_TARGET="${postproj_target}" \
  SINGLE_MAP_CAP=0 \
  SOLVE_TIMEOUT_S="${postproj_timeout_s}" \
  ./data/run_postprojection_repairs_loop.sh

echo "[run_postproj_from_clean] done"
