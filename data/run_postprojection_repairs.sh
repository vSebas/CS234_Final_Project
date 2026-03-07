#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
trace_jsonl="${TRACE_JSONL:-dt/checkpoints/full_run_lambda0/warmstarts/eval/postproj_trace_*/warmstart_eval_*_rollout_trace.jsonl}"
postproj_repairs="${POSTPROJ_REPAIRS:-80}"
postproj_per_map="${POSTPROJ_PER_MAP:-0}"
seed="${SEED:-0}"
output_root="${OUTPUT_ROOT:-data/datasets}"
max_attempts_factor="${MAX_ATTEMPTS_FACTOR:-3.0}"
max_trace_rows="${MAX_TRACE_ROWS:-0}"
nice_level="${NICE_LEVEL:-10}"
wall_time="${WALL_TIME:-20m}"
clear_cache_every="${CLEAR_CACHE_EVERY:-5}"
single_map_cap="${SINGLE_MAP_CAP:-200}"
postproj_solver="${POSTPROJ_SOLVER:-ipopt}"

mkdir -p results/dataset_runs
log_file="results/dataset_runs/postproj_repairs_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"

# Safety guard: if traces are single-map, avoid huge default targets that can lock laptops for hours.
# Only run the check when the cap could actually fire (saves ~300ms Python cold-start per invocation).
if [[ "${single_map_cap}" != "0" && "${postproj_per_map}" == "0" && "${postproj_repairs}" -gt "${single_map_cap}" ]]; then
  # Count distinct map_file values using shell tools — reads only the first non-empty line of
  # each matched file (same logic as the old Python block, no interpreter startup overhead).
  shopt -s nullglob
  _trace_files=( ${trace_jsonl} )
  shopt -u nullglob
  if (( ${#_trace_files[@]} > 0 )); then
    map_count=$(head -qn1 "${_trace_files[@]}" 2>/dev/null \
      | grep -o '"map_file"[[:space:]]*:[[:space:]]*"[^"]*"' \
      | sed 's/.*"\([^"]*\)"/\1/' \
      | sort -u | wc -l | tr -d '[:space:]')
  else
    map_count=0
  fi
  if [[ "${map_count}" == "1" ]]; then
    echo "[run_postprojection_repairs] Single-map trace detected; capping POSTPROJ_REPAIRS ${postproj_repairs} -> ${single_map_cap} for safety."
    postproj_repairs="${single_map_cap}"
  fi
fi

# FATROP settings (only active when --solver fatrop).
export FATROP_PRESET="${FATROP_PRESET:-obstacle_fast}"
export FATROP_STRUCTURE_DETECTION="${FATROP_STRUCTURE_DETECTION:-auto}"
export FATROP_EXPAND="${FATROP_EXPAND:-0}"
export FATROP_STAGE_LOCAL_COST="${FATROP_STAGE_LOCAL_COST:-1}"
export FATROP_DYNAMICS_SCHEME="${FATROP_DYNAMICS_SCHEME:-euler}"
export FATROP_SMOOTH_CONTROLS="${FATROP_SMOOTH_CONTROLS:-1}"
export FATROP_CLOSURE_MODE="${FATROP_CLOSURE_MODE:-open}"
export FATROP_MAX_ITER="${FATROP_MAX_ITER:-800}"
export FATROP_TOL="${FATROP_TOL:-5e-3}"
export FATROP_ACCEPTABLE_TOL="${FATROP_ACCEPTABLE_TOL:-5e-3}"

cmd=(
  nice -n "${nice_level}" env
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  "${python_bin}" -u data/build_postprojection_repairs.py
  --trace-jsonl "${trace_jsonl}"
  --base-laps-dir data/base_laps
  --output-root "${output_root}"
  --num-segments "${postproj_repairs}"
  --per-map-target "${postproj_per_map}"
  --max-attempts-factor "${max_attempts_factor}"
  --max-trace-rows "${max_trace_rows}"
  --clear-cache-every "${clear_cache_every}"
  --solver "${postproj_solver}"
  --seed "${seed}"
  --resume
)

if [[ "${wall_time}" != "0" ]]; then
  cmd=(timeout "${wall_time}" "${cmd[@]}")
fi

"${cmd[@]}" \
  2>&1 | tee "${log_file}"
