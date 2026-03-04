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
    hotspot_json="data/hotspots/Oval_Track_260m_hotspots.json"
    if [[ ! -f "${hotspot_json}" ]]; then
      echo "[run_full_dataset] Building hotspot JSON for Oval hard repairs"
      /home/saveas/.conda/envs/DT_trajopt/bin/python data/build_hotspot_json.py \
        --csv dt/checkpoints/full_run_lambda0/warmstarts/eval/diag_best_obs1/warmstart_eval_20260303_212627.csv \
        --map-file maps/Oval_Track_260m.mat \
        --seed 42 \
        --num-scenarios 3 \
        --min-obstacles 1 \
        --max-obstacles 1 \
        --output-json "${hotspot_json}"
    fi

    echo "[run_full_dataset] Stage 3: hard repairs on a single core"
    python -u data/build_dataset.py \
      "${common_args[@]}" \
      --repair-segments 0 \
      --hard-repair-segments 1200 \
      --hard-repair-hotspot-json "${hotspot_json}"
  fi
} 2>&1 | tee "${log_file}"
