#!/usr/bin/env python3
"""
Visualize hotspot anchors on a track map.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from world.world import World


def hotspot_global_xy(world: World, s_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_mod = np.mod(s_values, world.length_m)
    east = np.asarray(world.posE_m_interp_fcn(s_mod), dtype=float).squeeze()
    north = np.asarray(world.posN_m_interp_fcn(s_mod), dtype=float).squeeze()
    return east, north


def plot_hotspots(map_file: Path, hotspot_json: Path, output_dir: Path) -> Path:
    map_id = map_file.stem
    with hotspot_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    hotspots = np.asarray(payload.get(map_id, []), dtype=float)

    world = World(str(map_file), map_id, diagnostic_plotting=False)
    fig, ax = plt.subplots(figsize=(8, 8))

    inner = world.data["inner_bounds_m"]
    outer = world.data["outer_bounds_m"]
    ax.plot(inner[:, 0], inner[:, 1], color="gray", linewidth=1.5, label="Track bounds")
    ax.plot(outer[:, 0], outer[:, 1], color="gray", linewidth=1.5)
    ax.plot(
        world.data["posE_m"],
        world.data["posN_m"],
        color="black",
        linewidth=0.7,
        linestyle="--",
        alpha=0.6,
        label="Centerline",
    )

    if hotspots.size > 0:
        east, north = hotspot_global_xy(world, hotspots)
        ax.scatter(east, north, s=80, c="tab:orange", edgecolors="black", linewidths=0.7, zorder=5, label="Hotspots")
        for i, (e, n, s_val) in enumerate(zip(np.atleast_1d(east), np.atleast_1d(north), hotspots)):
            ax.text(float(e), float(n), f"{i+1}", fontsize=8, ha="left", va="bottom")

    ax.set_title(f"{map_id}: hotspot anchors")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{map_id}_hotspots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize hotspot anchors on track maps.")
    parser.add_argument("--hotspot-json", type=str, required=True)
    parser.add_argument("--map-file", type=str, default=None, help="Single map to plot. If omitted, plot all maps present in the hotspot JSON.")
    parser.add_argument("--output-dir", type=str, default="results/trajectory_optimization/hotspots")
    args = parser.parse_args()

    hotspot_json = Path(args.hotspot_json)
    output_dir = Path(args.output_dir)

    if args.map_file:
        map_files = [Path(args.map_file)]
    else:
        with hotspot_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        map_files = [project_root / "maps" / f"{map_id}.mat" for map_id in payload.keys()]

    for map_file in map_files:
        out_path = plot_hotspots(map_file, hotspot_json, output_dir)
        print(out_path)


if __name__ == "__main__":
    main()
