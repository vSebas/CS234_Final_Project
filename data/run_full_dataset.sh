#!/usr/bin/env bash
set -euo pipefail

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
} 2>&1 | tee "${log_file}"
