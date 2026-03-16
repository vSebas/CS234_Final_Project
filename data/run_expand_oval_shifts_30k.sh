#!/usr/bin/env bash
set -euo pipefail

# One-command expansion of Oval FATROP clean base laps + shifts only
# (no hard repairs, no post-projection repairs).

base_laps_count="${BASE_LAPS_COUNT:-20}"
obstacle_laps_count="${OBSTACLE_LAPS_COUNT:-190}"
dataset_seed="${DATASET_SEED:-101}"
discretization_N="${DISCRETIZATION_N:-150}"
base_stage_wall_time="${BASE_STAGE_WALL_TIME:-20m}"
max_passes="${MAX_PASSES:-30}"
sleep_between_passes="${SLEEP_BETWEEN_PASSES:-2}"

echo "[run_expand_oval_shifts_30k] base_laps=${base_laps_count} obstacle_laps=${obstacle_laps_count} seed=${dataset_seed} N=${discretization_N}"
echo "[run_expand_oval_shifts_30k] running base-laps + shifts only"
echo "[run_expand_oval_shifts_30k] base_stage_wall_time=${base_stage_wall_time} max_passes=${max_passes}"

pass=0
while [ "${pass}" -lt "${max_passes}" ]; do
  pass=$((pass + 1))
  echo "[run_expand_oval_shifts_30k] pass ${pass}/${max_passes}"

  if [[ "${base_stage_wall_time}" != "0" ]]; then
    timeout "${base_stage_wall_time}" env \
      BASE_LAPS_COUNT="${base_laps_count}" \
      OBSTACLE_LAPS_COUNT="${obstacle_laps_count}" \
      DATASET_SEED="${dataset_seed}" \
      DISCRETIZATION_N="${discretization_N}" \
      RUN_HARD_REPAIRS=0 \
      RUN_POSTPROJ_REPAIRS=0 \
      ./data/run_full_dataset.sh oval_fatrop_clean || true
  else
    env \
      BASE_LAPS_COUNT="${base_laps_count}" \
      OBSTACLE_LAPS_COUNT="${obstacle_laps_count}" \
      DATASET_SEED="${dataset_seed}" \
      DISCRETIZATION_N="${discretization_N}" \
      RUN_HARD_REPAIRS=0 \
      RUN_POSTPROJ_REPAIRS=0 \
      ./data/run_full_dataset.sh oval_fatrop_clean || true
  fi

  base_dir="data/base_laps_fatrop_clean/Oval_Track_260m"
  noobs_count=$(find "${base_dir}" -maxdepth 1 -type f -name 'noobs_*.npz' | wc -l | tr -d '[:space:]')
  obs_count=$(find "${base_dir}" -maxdepth 1 -type f -name 'obs_*.npz' | wc -l | tr -d '[:space:]')
  echo "[run_expand_oval_shifts_30k] current base laps: noobs=${noobs_count}/${base_laps_count} obs=${obs_count}/${obstacle_laps_count}"

  if [ "${noobs_count}" -ge "${base_laps_count}" ] && [ "${obs_count}" -ge "${obstacle_laps_count}" ]; then
    echo "[run_expand_oval_shifts_30k] target base-lap counts reached."
    exit 0
  fi

  sleep "${sleep_between_passes}"
done

echo "[run_expand_oval_shifts_30k] reached MAX_PASSES without full target."
exit 2
