#!/usr/bin/env python3
"""
Build hotspot JSON files for hard-repair generation.

Current source:
- deterministic obstacle benchmark scenarios from experiments/eval_warmstart.py
- weighted by DT warm-start failure pressure from a diagnostic CSV

This is a proxy for true per-step projection/fallback hotspot logging. It uses
the obstacle s-positions from the benchmark scenarios that produced the worst
DT warm-start outcomes, so hard-repair sampling can be biased toward those
track regions immediately.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from experiments.eval_warmstart import EvalConfig, sample_obstacles
from utils.world import World


def build_hotspots_from_diag_csv(
    csv_path: Path,
    map_file: Path,
    seed: int,
    num_scenarios: int,
    min_obstacles: int,
    max_obstacles: int,
) -> List[float]:
    world = World(str(map_file), map_file.stem, diagnostic_plotting=False)
    cfg = EvalConfig(
        map_file=str(map_file),
        num_scenarios=num_scenarios,
        seed=seed,
        min_obstacles=min_obstacles,
        max_obstacles=max_obstacles,
    )

    # Reconstruct the deterministic obstacle placements used by eval_warmstart.
    rng = np.random.default_rng(seed)
    scenario_obstacles: Dict[int, List[Dict]] = {}
    for scenario_id in range(num_scenarios):
        num_obs = int(rng.integers(min_obstacles, max_obstacles + 1))
        scenario_obstacles[scenario_id] = sample_obstacles(rng, world, num_obs, cfg)

    hotspot_scores: List[tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method") != "dt_warmstart":
                continue
            scenario_id = int(row["scenario_id"])
            success = row.get("success", "False") == "True"
            fallback_count = float(row.get("ws_fallback_count", 0) or 0)
            projection_count = float(row.get("ws_projection_count", 0) or 0)
            projection_total = float(row.get("ws_projection_total_magnitude", 0) or 0)

            severity = fallback_count + 0.1 * projection_count + 0.01 * projection_total
            if not success:
                severity += 100.0

            for obs in scenario_obstacles.get(scenario_id, []):
                hotspot_scores.append((float(obs["s_m"]), severity))

    if not hotspot_scores:
        return []

    # Weight obstacle locations by severity, then keep them ordered along s.
    hotspot_scores.sort(key=lambda x: (-x[1], x[0]))
    selected = hotspot_scores[: min(len(hotspot_scores), 12)]
    s_values = sorted(float(s) for s, _ in selected)

    # Deduplicate nearby hotspots to avoid collapsing on the same obstacle location.
    deduped: List[float] = []
    for s in s_values:
        if not deduped or min(abs(s - prev) for prev in deduped) > 5.0:
            deduped.append(s)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hotspot JSON from diagnostic eval CSVs.")
    parser.add_argument("--csv", type=str, required=True, help="Diagnostic eval CSV for dt_warmstart.")
    parser.add_argument("--map-file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-scenarios", type=int, default=3)
    parser.add_argument("--min-obstacles", type=int, default=1)
    parser.add_argument("--max-obstacles", type=int, default=1)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    map_file = Path(args.map_file)
    output_json = Path(args.output_json)

    hotspots = build_hotspots_from_diag_csv(
        csv_path=csv_path,
        map_file=map_file,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
    )

    payload = {}
    if output_json.exists():
        with output_json.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            payload.update(existing)
    payload[map_file.stem] = hotspots
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved hotspot JSON to {output_json}")
    print(f"Map: {map_file.stem}")
    print(f"Hotspots (s_m): {hotspots}")


if __name__ == "__main__":
    main()
