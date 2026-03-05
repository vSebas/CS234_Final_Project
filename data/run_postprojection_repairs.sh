#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/saveas/.conda/envs/DT_trajopt/bin/python}"
trace_jsonl="${TRACE_JSONL:-dt/checkpoints/full_run_lambda0/warmstarts/eval/postproj_trace_*/warmstart_eval_*_rollout_trace.jsonl}"
postproj_repairs="${POSTPROJ_REPAIRS:-80}"
postproj_per_map="${POSTPROJ_PER_MAP:-0}"
seed="${SEED:-0}"
output_root="${OUTPUT_ROOT:-data/datasets}"
max_attempts_factor="${MAX_ATTEMPTS_FACTOR:-3.0}"
max_trace_rows="${MAX_TRACE_ROWS:-0}"
nice_level="${NICE_LEVEL:-10}"
wall_time="${WALL_TIME:-20m}"
clear_cache_every="${CLEAR_CACHE_EVERY:-5}"
single_map_cap="${SINGLE_MAP_CAP:-200}"

mkdir -p results/dataset_runs
log_file="results/dataset_runs/postproj_repairs_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"

# Safety guard: if traces are single-map, avoid huge default targets that can lock laptops for hours.
map_count="$("${python_bin}" - "${trace_jsonl}" << 'PY'
import glob
import json
import sys

pattern = sys.argv[1]
paths = sorted(glob.glob(pattern))
maps = set()
for p in paths:
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                m = row.get("map_file")
                if m:
                    maps.add(m)
                break
    except Exception:
        pass
print(len(maps))
PY
)"

if [[ "${single_map_cap}" != "0" && "${postproj_per_map}" == "0" && "${map_count}" == "1" && "${postproj_repairs}" -gt "${single_map_cap}" ]]; then
  echo "[run_postprojection_repairs] Single-map trace detected; capping POSTPROJ_REPAIRS ${postproj_repairs} -> ${single_map_cap} for safety."
  postproj_repairs="${single_map_cap}"
fi

cmd=(
  nice -n "${nice_level}" env
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  "${python_bin}" -u data/build_postprojection_repairs.py
  --trace-jsonl "${trace_jsonl}"
  --base-laps-dir data/base_laps
  --output-root "${output_root}"
  --num-segments "${postproj_repairs}"
  --per-map-target "${postproj_per_map}"
  --max-attempts-factor "${max_attempts_factor}"
  --max-trace-rows "${max_trace_rows}"
  --clear-cache-every "${clear_cache_every}"
  --seed "${seed}"
  --resume
)

if [[ "${wall_time}" != "0" ]]; then
  cmd=(timeout "${wall_time}" "${cmd[@]}")
fi

"${cmd[@]}" \
  2>&1 | tee "${log_file}"
