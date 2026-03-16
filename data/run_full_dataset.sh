#!/usr/bin/env bash
set -euo pipefail

mode="${1:-standard}"

mkdir -p results/dataset_runs
log_file="results/dataset_runs/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"

map_files="maps/Oval_Track_260m.mat,maps/TRACK1_280m.mat,maps/TRACK2.mat,maps/TRACK3_300m.mat,maps/TRACK4_315m.mat,maps/TRACK5_330m.mat"

common_args=(
  --map-files "${map_files}"
  --base-laps 6
  --obstacle-laps 8
  --all-shifts
  --N 120
  --H 20
  --seed 0
  --resume
)

if [[ "${mode}" == "oval_fatrop_clean" ]]; then
  run_hard_repairs="${RUN_HARD_REPAIRS:-1}"
  run_postproj="${RUN_POSTPROJ_REPAIRS:-1}"
  hard_target="${HARD_TARGET:-400}"
  postproj_target="${POSTPROJ_TARGET:-1000}"
  solve_timeout_s="${SOLVE_TIMEOUT_S:-12}"
  postproj_output_suffix="${POSTPROJ_OUTPUT_SUFFIX:-repairs_postproj_fatrop_clean}"
  postproj_trace_jsonl="${POSTPROJ_TRACE_JSONL:-dt/checkpoints/*/warmstarts/eval/*/warmstart_eval_*_rollout_trace.jsonl}"
  base_laps_count="${BASE_LAPS_COUNT:-6}"
  obstacle_laps_count="${OBSTACLE_LAPS_COUNT:-15}"
  dataset_seed="${DATASET_SEED:-0}"
  discretization_N="${DISCRETIZATION_N:-150}"

  {
    echo "[run_full_dataset] Oval FATROP clean dataset (N=${discretization_N}): stage 1 base laps"
    echo "[run_full_dataset] base_laps=${base_laps_count} obstacle_laps=${obstacle_laps_count} seed=${dataset_seed}"
    env FATROP_PRESET=obstacle_fast \
      FATROP_STRUCTURE_DETECTION=auto \
      FATROP_EXPAND=0 \
      FATROP_STAGE_LOCAL_COST=1 \
      FATROP_DYNAMICS_SCHEME=euler \
      FATROP_SMOOTH_CONTROLS=1 \
      FATROP_CLOSURE_MODE=open \
      FATROP_MAX_ITER=800 \
      FATROP_TOL=5e-3 \
      FATROP_ACCEPTABLE_TOL=5e-3 \
      PYTHONPATH=. \
      /home/saveas/.conda/envs/DT_trajopt/bin/python -u data/build_base_laps.py \
      --map-files maps/Oval_Track_260m.mat \
      --output-dir data/base_laps_fatrop_clean \
      --solver fatrop \
      --N "${discretization_N}" \
      --ux-min 5.0 \
      --base-laps "${base_laps_count}" \
      --obstacle-laps "${obstacle_laps_count}" \
      --min-obstacles 1 \
      --max-obstacles 4 \
      --solve-timeout-s "${solve_timeout_s}" \
      --seed "${dataset_seed}" \
      --resume

    echo "[run_full_dataset] Oval FATROP clean dataset (N=${discretization_N}): stage 2 shifts"
    PYTHONPATH=. /home/saveas/.conda/envs/DT_trajopt/bin/python -u data/make_shift_episodes.py \
      --map-file maps/Oval_Track_260m.mat \
      --base-laps-dir data/base_laps_fatrop_clean \
      --output-dir data/datasets/Oval_Track_260m_shifts_fatrop_clean \
      --all-shifts \
      --seed "${dataset_seed}" \
      --resume

    if [[ "${run_hard_repairs}" == "1" ]]; then
      echo "[run_full_dataset] Oval FATROP clean dataset (N=${discretization_N}): stage 3 hard repairs"
      env FATROP_PRESET=obstacle_fast \
        FATROP_STRUCTURE_DETECTION=auto \
        FATROP_EXPAND=0 \
        FATROP_STAGE_LOCAL_COST=1 \
        FATROP_DYNAMICS_SCHEME=euler \
        FATROP_SMOOTH_CONTROLS=1 \
        FATROP_CLOSURE_MODE=open \
        FATROP_MAX_ITER=800 \
        FATROP_TOL=5e-3 \
        FATROP_ACCEPTABLE_TOL=5e-3 \
        PYTHONPATH=. \
        /home/saveas/.conda/envs/DT_trajopt/bin/python -u data/build_repair_segments.py \
        --map-file maps/Oval_Track_260m.mat \
        --base-laps-dir data/base_laps_fatrop_clean \
        --output-dir data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean \
        --num-segments "${hard_target}" \
        --seed "${dataset_seed}" \
        --H 20 \
        --hard-mode \
        --ux-min 5.0 \
        --save-every 10 \
        --solve-timeout-s "${solve_timeout_s}" \
        --solver fatrop \
        --resume
    else
      echo "[run_full_dataset] skipping hard repairs (RUN_HARD_REPAIRS=${run_hard_repairs})"
    fi

    if [[ "${run_postproj}" == "1" ]]; then
      echo "[run_full_dataset] Oval FATROP clean dataset (N=${discretization_N}): stage 4 post-projection repairs"
      echo "[run_full_dataset] using trace glob: ${postproj_trace_jsonl}"
      env POSTPROJ_SOLVER=fatrop \
        SOLVE_TIMEOUT_S="${solve_timeout_s}" \
        TRACE_JSONL="${postproj_trace_jsonl}" \
        OUTPUT_SUFFIX="${postproj_output_suffix}" \
        TOTAL_TARGET="${postproj_target}" \
        SINGLE_MAP_CAP=0 \
        ./data/run_postprojection_repairs_loop.sh
    else
      echo "[run_full_dataset] skipping post-projection repairs (RUN_POSTPROJ_REPAIRS=${run_postproj})"
    fi
  } 2>&1 | tee "${log_file}"
  exit 0
fi

{
  echo "[run_full_dataset] Stage 1: base laps + shifts on multiple cores"
  python -u data/build_dataset.py \
    "${common_args[@]}" \
    --repair-segments 0 \
    --parallel \
    --max-workers 3

  echo "[run_full_dataset] Stage 2: repairs on a single core"
  python -u data/build_dataset.py \
    "${common_args[@]}" \
    --repair-segments 500

  if [[ "${mode}" == "hard_repairs" ]]; then
    hotspot_json="data/hotspots/all_tracks_hotspots.json"
    if [[ ! -f "${hotspot_json}" ]]; then
      echo "[run_full_dataset] Building hotspot JSON for all tracks"
      ./data/build_all_hotspots.sh
    fi

    echo "[run_full_dataset] Stage 3: hard repairs on a single core"
    python -u data/build_dataset.py \
      "${common_args[@]}" \
      --repair-segments 0 \
      --hard-repair-segments 1200 \
      --hard-repair-hotspot-json "${hotspot_json}"
  fi
} 2>&1 | tee "${log_file}"
