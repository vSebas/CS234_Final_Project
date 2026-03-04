#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
hotspot_json="${HOTSPOT_JSON:-data/hotspots/all_tracks_hotspots.json}"
hard_repairs="${HARD_REPAIRS:-1200}"
hard_repair_chunk_size="${HARD_REPAIR_CHUNK_SIZE:-25}"
seed="${SEED:-0}"

map_files="${MAP_FILES:-maps/Oval_Track_260m.mat,maps/TRACK1_280m.mat,maps/TRACK2.mat,maps/TRACK3_300m.mat,maps/TRACK4_315m.mat,maps/TRACK5_330m.mat}"

mkdir -p results/dataset_runs
log_file="results/dataset_runs/hard_repairs_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"

if [[ ! -f "${hotspot_json}" ]]; then
  ./data/build_all_hotspots.sh
fi

"${python_bin}" -u data/build_dataset.py \
  --map-files "${map_files}" \
  --base-laps 6 \
  --obstacle-laps 8 \
  --all-shifts \
  --N 120 \
  --H 20 \
  --seed "${seed}" \
  --resume \
  --repair-segments 0 \
  --hard-repair-segments "${hard_repairs}" \
  --hard-repair-chunk-size "${hard_repair_chunk_size}" \
  --hard-repair-hotspot-json "${hotspot_json}" \
  2>&1 | tee "${log_file}"
