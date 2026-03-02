#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/dataset_runs
log_file="results/dataset_runs/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"

python -u data/build_dataset.py \
  --map-files maps/Oval_Track_260m.mat,maps/TRACK1_280m.mat,maps/TRACK2.mat,maps/TRACK3_300m.mat,maps/TRACK4_315m.mat,maps/TRACK5_330m.mat \
  --base-laps 6 \
  --obstacle-laps 8 \
  --all-shifts \
  --repair-segments 500 \
  --N 120 \
  --H 20 \
  --seed 0 \
  --resume \
  # --parallel \
  # --max-workers 8 \
  2>&1 | tee "${log_file}"
