#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
hotspot_json="${HOTSPOT_JSON:-data/hotspots/all_tracks_hotspots.json}"
hard_repairs="${HARD_REPAIRS:-1200}"
hard_repair_chunk_size="${HARD_REPAIR_CHUNK_SIZE:-25}"
seed="${SEED:-0}"

map_files="${MAP_FILES:-maps/Oval_Track_260m.mat,maps/TRACK1_280m.mat,maps/TRACK2.mat,maps/TRACK3_300m.mat,maps/TRACK4_315m.mat,maps/TRACK5_330m.mat}"

mkdir -p results/dataset_runs
log_file="results/dataset_runs/hard_repairs_fatrop_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"

if [[ ! -f "${hotspot_json}" ]]; then
  ./data/build_all_hotspots.sh
fi

# FATROP settings that are robust for repair-form trajectories.
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
  --hard-repair-solver fatrop \
  2>&1 | tee "${log_file}"
