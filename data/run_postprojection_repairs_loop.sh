#!/usr/bin/env bash
set -euo pipefail

# Repeatedly run post-projection repair generation until a total target count
# is reached (resume-safe), with guards against infinite no-progress loops.

map_name="${MAP_NAME:-Oval_Track_260m}"
output_root="${OUTPUT_ROOT:-data/datasets}"
manifest_path="${MANIFEST_PATH:-${output_root}/${map_name}_repairs_postproj/manifest.jsonl}"

total_target="${TOTAL_TARGET:-120}"
max_loops="${MAX_LOOPS:-50}"
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
echo "[postproj_loop] total_target=${total_target}"
echo "[postproj_loop] max_loops=${max_loops}"
echo "[postproj_loop] max_no_progress_loops=${max_no_progress_loops}"
echo "[postproj_loop] single_map_cap=${single_map_cap}"

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

  # Set absolute target for each run; generation is resume-safe.
  POSTPROJ_REPAIRS="${total_target}" SINGLE_MAP_CAP="${single_map_cap}" ./data/run_postprojection_repairs.sh

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
