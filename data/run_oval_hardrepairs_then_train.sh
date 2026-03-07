#!/usr/bin/env bash
set -euo pipefail

# Run sequence:
# 1) Resume Oval hard-repair generation to HARD_TARGET (FATROP)
# 2) Start DT training for NUM_EPOCHS on Oval-only shards
#
# Usage:
#   ./data/run_oval_hardrepairs_then_train.sh
# Optional overrides:
#   HARD_TARGET=400 NUM_EPOCHS=20 TRAIN_OUTPUT_DIR=dt/checkpoints/oval_hard400_train20 ./data/run_oval_hardrepairs_then_train.sh

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
map_file="${MAP_FILE:-maps/Oval_Track_260m.mat}"
base_laps_dir="${BASE_LAPS_DIR:-data/base_laps}"

hard_output_dir="${HARD_OUTPUT_DIR:-data/datasets/Oval_Track_260m_repairs_hard}"
hard_target="${HARD_TARGET:-400}"
seed="${SEED:-0}"
H="${HORIZON:-20}"
max_attempts="${MAX_ATTEMPTS:-2400}"
solve_timeout_s="${SOLVE_TIMEOUT_S:-20}"
hard_pass_timeout_s="${HARD_PASS_TIMEOUT_S:-180}"
max_hard_passes="${MAX_HARD_PASSES:-50}"
max_no_progress_passes="${MAX_NO_PROGRESS_PASSES:-8}"

# Oval-only shards for training
train_data_dir="${TRAIN_DATA_DIR:-data/datasets/Oval_Track_260m_shifts,data/datasets/Oval_Track_260m_repairs,data/datasets/Oval_Track_260m_repairs_hard,data/datasets/Oval_Track_260m_repairs_postproj}"
train_output_dir="${TRAIN_OUTPUT_DIR:-dt/checkpoints/oval_hard400_train20}"
num_epochs="${NUM_EPOCHS:-20}"
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-4}"
context_length="${CONTEXT_LENGTH:-30}"
lambda_x="${LAMBDA_X:-0.0}"
resume_mode="${RESUME_MODE:-auto}"
log_file="${LOG_FILE:-results/dataset_runs/oval_hardrepairs_then_train_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "${log_file}")"
# Route all output to a stable log file while keeping terminal output in foreground runs.
exec > >(tee -a "${log_file}") 2>&1

echo "[oval-pipeline] started at $(date -Iseconds)"
echo "[oval-pipeline] log_file=${log_file}"
manifest_path="${hard_output_dir}/manifest.jsonl"
current_hard=0
if [[ -f "${manifest_path}" ]]; then
  current_hard=$(wc -l < "${manifest_path}")
fi

echo "[oval-pipeline] hard repairs current=${current_hard} target=${hard_target}"
if (( current_hard < hard_target )); then
  echo "[oval-pipeline] generating hard repairs to target with FATROP (chunked timeout mode)"
  pass=0
  no_progress=0
  while (( current_hard < hard_target && pass < max_hard_passes )); do
    pass=$((pass + 1))
    echo "[oval-pipeline] hard-pass ${pass}/${max_hard_passes} current=${current_hard}/${hard_target}"
    set +e
    timeout "${hard_pass_timeout_s}s" env \
      PYTHONPATH=. \
      FATROP_PRESET="${FATROP_PRESET:-obstacle_fast}" \
      FATROP_STRUCTURE_DETECTION="${FATROP_STRUCTURE_DETECTION:-auto}" \
      FATROP_EXPAND="${FATROP_EXPAND:-0}" \
      FATROP_STAGE_LOCAL_COST="${FATROP_STAGE_LOCAL_COST:-1}" \
      FATROP_DYNAMICS_SCHEME="${FATROP_DYNAMICS_SCHEME:-euler}" \
      FATROP_SMOOTH_CONTROLS="${FATROP_SMOOTH_CONTROLS:-1}" \
      FATROP_CLOSURE_MODE="${FATROP_CLOSURE_MODE:-open}" \
      FATROP_MAX_ITER="${FATROP_MAX_ITER:-800}" \
      FATROP_TOL="${FATROP_TOL:-5e-3}" \
      FATROP_ACCEPTABLE_TOL="${FATROP_ACCEPTABLE_TOL:-5e-3}" \
      "${python_bin}" -u data/build_repair_segments.py \
        --map-file "${map_file}" \
        --base-laps-dir "${base_laps_dir}" \
        --output-dir "${hard_output_dir}" \
        --num-segments "${hard_target}" \
        --seed "${seed}" \
        --H "${H}" \
        --hard-mode \
        --save-every 10 \
        --max-attempts "${max_attempts}" \
        --solve-timeout-s "${solve_timeout_s}" \
        --solver fatrop \
        --resume
    rc=$?
    set -e
    prev_hard="${current_hard}"
    if [[ -f "${manifest_path}" ]]; then
      current_hard=$(wc -l < "${manifest_path}")
    else
      current_hard=0
    fi
    echo "[oval-pipeline] hard-pass rc=${rc} new_count=${current_hard}/${hard_target}"
    if (( current_hard <= prev_hard )); then
      no_progress=$((no_progress + 1))
      echo "[oval-pipeline] no progress (${no_progress}/${max_no_progress_passes})"
      if (( no_progress >= max_no_progress_passes )); then
        echo "[oval-pipeline] stopping hard-repair stage due to repeated no-progress passes"
        break
      fi
    else
      no_progress=0
    fi
  done
else
  echo "[oval-pipeline] hard repairs already at/above target; skipping generation"
fi

final_hard=0
if [[ -f "${manifest_path}" ]]; then
  final_hard=$(wc -l < "${manifest_path}")
fi

echo "[oval-pipeline] hard repairs ready=${final_hard}"
echo "[oval-pipeline] starting training for ${num_epochs} epochs"

mkdir -p "${train_output_dir}"
"${python_bin}" -u dt/train.py \
  --data-dir "${train_data_dir}" \
  --output-dir "${train_output_dir}" \
  --context-length "${context_length}" \
  --batch-size "${batch_size}" \
  --num-epochs "${num_epochs}" \
  --num-workers "${num_workers}" \
  --lambda-x "${lambda_x}" \
  --resume "${resume_mode}"

echo "[oval-pipeline] finished at $(date -Iseconds)"
