#!/usr/bin/env bash
set -euo pipefail

# Repeatedly generate post-projection repairs until a target count is reached.
# This is the single maintained entrypoint (resume-safe).

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
map_name="${MAP_NAME:-Oval_Track_260m}"
output_root="${OUTPUT_ROOT:-data/datasets}"
output_suffix="${OUTPUT_SUFFIX:-repairs_postproj_fatrop_clean}"
manifest_path="${MANIFEST_PATH:-${output_root}/${map_name}_${output_suffix}/manifest.jsonl}"
base_laps_dir="${BASE_LAPS_DIR:-data/base_laps_fatrop_clean}"
trace_jsonl="${TRACE_JSONL:-dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260309_095838_rollout_trace.jsonl,dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260309_142322_rollout_trace.jsonl}"

# Accept legacy POSTPROJ_REPAIRS as alias, but canonical knob is TOTAL_TARGET.
total_target="${TOTAL_TARGET:-${POSTPROJ_REPAIRS:-1000}}"
per_map_target="${POSTPROJ_PER_MAP:-0}"
seed="${SEED:-0}"
postproj_solver="${POSTPROJ_SOLVER:-fatrop}"
max_attempts_factor="${MAX_ATTEMPTS_FACTOR:-3.0}"
max_trace_rows="${MAX_TRACE_ROWS:-20000}"
clear_cache_every="${CLEAR_CACHE_EVERY:-10}"
solve_timeout_s="${SOLVE_TIMEOUT_S:-12}"
long_horizon_prob="${LONG_HORIZON_PROB:-0.05}"
obs_subsamples="${OBS_SUBSAMPLES:-5}"
only_triggered="${ONLY_TRIGGERED:-1}"
nice_level="${NICE_LEVEL:-10}"
wall_time="${WALL_TIME:-25m}"

max_loops="${MAX_LOOPS:-40}"
max_no_progress_loops="${MAX_NO_PROGRESS_LOOPS:-5}"
sleep_seconds="${SLEEP_SECONDS:-2}"
single_map_cap="${SINGLE_MAP_CAP:-0}"

mkdir -p results/dataset_runs
wrapper_log="results/dataset_runs/postproj_repairs_loop_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${wrapper_log}") 2>&1

count_episodes() {
  if [[ -f "${manifest_path}" ]]; then
    wc -l < "${manifest_path}"
  else
    echo 0
  fi
}

echo "[postproj_loop] wrapper_log=${wrapper_log}"
echo "[postproj_loop] map_name=${map_name}"
echo "[postproj_loop] manifest_path=${manifest_path}"
echo "[postproj_loop] trace_jsonl=${trace_jsonl}"
echo "[postproj_loop] base_laps_dir=${base_laps_dir}"
echo "[postproj_loop] output_suffix=${output_suffix}"
echo "[postproj_loop] total_target=${total_target}"
echo "[postproj_loop] per_map_target=${per_map_target}"
echo "[postproj_loop] solver=${postproj_solver}"
echo "[postproj_loop] solve_timeout_s=${solve_timeout_s}"
echo "[postproj_loop] wall_time=${wall_time}"
echo "[postproj_loop] max_loops=${max_loops}"
echo "[postproj_loop] max_no_progress_loops=${max_no_progress_loops}"
echo "[postproj_loop] single_map_cap=${single_map_cap}"

# Safety guard for single-map traces.
if [[ "${single_map_cap}" != "0" && "${per_map_target}" == "0" && "${total_target}" -gt "${single_map_cap}" ]]; then
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
    echo "[postproj_loop] single-map trace detected; capping target ${total_target} -> ${single_map_cap}"
    total_target="${single_map_cap}"
  fi
fi

# FATROP defaults (only active when solver=fatrop).
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

loop=0
no_progress=0
prev_count="$(count_episodes)"

echo "[postproj_loop] start_count=${prev_count}"
if (( prev_count >= total_target )); then
  echo "[postproj_loop] already at/above target; nothing to do."
  exit 0
fi

while (( loop < max_loops )); do
  loop=$((loop + 1))
  echo "[postproj_loop] loop=${loop}/${max_loops} current=${prev_count}/${total_target}"

  cmd=(
    nice -n "${nice_level}" env
    OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    PYTHONPATH=.
    "${python_bin}" -u data/build_postprojection_repairs.py
    --trace-jsonl "${trace_jsonl}"
    --base-laps-dir "${base_laps_dir}"
    --output-root "${output_root}"
    --output-suffix "${output_suffix}"
    --num-segments "${total_target}"
    --per-map-target "${per_map_target}"
    --max-attempts-factor "${max_attempts_factor}"
    --max-trace-rows "${max_trace_rows}"
    --clear-cache-every "${clear_cache_every}"
    --solve-timeout-s "${solve_timeout_s}"
    --long-horizon-prob "${long_horizon_prob}"
    --obs-subsamples "${obs_subsamples}"
    --solver "${postproj_solver}"
    --seed "${seed}"
    --resume
  )
  if [[ "${only_triggered}" == "1" ]]; then
    cmd+=(--only-triggered)
  fi
  if [[ "${wall_time}" != "0" ]]; then
    cmd=(timeout "${wall_time}" "${cmd[@]}")
  fi
  "${cmd[@]}" || true

  curr_count="$(count_episodes)"
  delta=$((curr_count - prev_count))
  echo "[postproj_loop] loop=${loop} finished current=${curr_count}/${total_target} delta=${delta}"

  if (( curr_count >= total_target )); then
    echo "[postproj_loop] target reached."
    exit 0
  fi

  if (( curr_count <= prev_count )); then
    no_progress=$((no_progress + 1))
    echo "[postproj_loop] no progress (${no_progress}/${max_no_progress_loops})"
    if (( no_progress >= max_no_progress_loops )); then
      echo "[postproj_loop] stopping due to repeated no-progress loops."
      exit 2
    fi
  else
    no_progress=0
  fi

  prev_count="${curr_count}"
  sleep "${sleep_seconds}"
done

echo "[postproj_loop] reached max loops without hitting target."
exit 3
