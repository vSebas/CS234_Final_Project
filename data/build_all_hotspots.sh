#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
checkpoint="${CHECKPOINT:-dt/checkpoints/full_run_lambda0/checkpoints/checkpoint_best.pt}"
output_json="${OUTPUT_JSON:-data/hotspots/all_tracks_hotspots.json}"
seed="${SEED:-42}"
num_scenarios="${NUM_SCENARIOS:-3}"

map_files=(
  maps/Oval_Track_260m.mat
  maps/TRACK1_280m.mat
  maps/TRACK2.mat
  maps/TRACK3_300m.mat
  maps/TRACK4_315m.mat
  maps/TRACK5_330m.mat
)

rm -f "${output_json}"
mkdir -p "$(dirname "${output_json}")"

for map_file in "${map_files[@]}"; do
  map_id="$(basename "${map_file}" .mat)"
  eval_dir="dt/checkpoints/full_run_lambda0/warmstarts/eval/hotspots_${map_id}"

  "${python_bin}" experiments/eval_warmstart.py \
    --checkpoint "${checkpoint}" \
    --map-file "${map_file}" \
    --num-scenarios "${num_scenarios}" \
    --seed "${seed}" \
    --min-obstacles 1 \
    --max-obstacles 1 \
    --output-dir "${eval_dir}"

  csv_path="$(find "${eval_dir}" -maxdepth 1 -name 'warmstart_eval_*.csv' | sort | tail -n 1)"
  if [[ -z "${csv_path}" ]]; then
    echo "Failed to locate eval CSV for ${map_id}" >&2
    exit 1
  fi

  "${python_bin}" data/build_hotspot_json.py \
    --csv "${csv_path}" \
    --map-file "${map_file}" \
    --seed "${seed}" \
    --num-scenarios "${num_scenarios}" \
    --min-obstacles 1 \
    --max-obstacles 1 \
    --output-json "${output_json}"
done

echo "Saved merged hotspot JSON to ${output_json}"
