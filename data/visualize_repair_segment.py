#!/usr/bin/env python3
"""
Visualize a generated repair segment using the standard trajectory report outputs.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from planning import OptimizationResult
from utils.visualization import TrajectoryVisualizer, create_animation
from world.world import World


def _load_manifest_entry(npz_path: Path) -> dict:
    manifest_path = npz_path.parents[1] / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for repair episode: {manifest_path}")

    npz_resolved = npz_path.resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            candidate = Path(row["npz_path"])
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            if candidate == npz_resolved:
                return row

    raise ValueError(f"No manifest entry found for repair episode: {npz_path}")


def _attach_obstacles_to_world(world: World, obstacles: list[dict]) -> None:
    if not obstacles:
        for key in (
            "obstacles_s_m",
            "obstacles_e_m",
            "obstacles_radius_m",
            "obstacles_margin_m",
            "obstacles_radius_tilde_m",
        ):
            world.data.pop(key, None)
        return

    s_vals = np.array([float(obs["s_m"]) for obs in obstacles], dtype=float)
    e_vals = np.array([float(obs["e_m"]) for obs in obstacles], dtype=float)
    r_vals = np.array([float(obs["radius_m"]) for obs in obstacles], dtype=float)
    m_vals = np.array([float(obs.get("margin_m", 0.0)) for obs in obstacles], dtype=float)
    world.data["obstacles_s_m"] = s_vals
    world.data["obstacles_e_m"] = e_vals
    world.data["obstacles_radius_m"] = r_vals
    world.data["obstacles_margin_m"] = m_vals
    world.data["obstacles_radius_tilde_m"] = r_vals + m_vals


def _convert_to_global(world: World, s_m: np.ndarray, e_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    east, north, _ = world.map_match_vectorized(s_m, e_m)
    return np.asarray(east, dtype=float), np.asarray(north, dtype=float)


def _plot_repair_context(
    world: World,
    base_npz_path: Path,
    repair_result: OptimizationResult,
    episode_id: str,
    output_dir: Path,
) -> Path:
    base_data = np.load(base_npz_path, allow_pickle=True)
    base_s = base_data["s_m"].astype(float)
    base_X = base_data["X_full"].astype(float).T
    repair_s = repair_result.s_m
    repair_X = repair_result.X

    base_e = base_X[6, :]
    repair_e = repair_X[6, :]
    base_east, base_north = _convert_to_global(world, base_s, base_e)
    repair_east, repair_north = _convert_to_global(world, repair_s, repair_e)

    fig, ax = plt.subplots(figsize=(8, 8))
    inner = world.data["inner_bounds_m"]
    outer = world.data["outer_bounds_m"]
    ax.plot(inner[:, 0], inner[:, 1], color="gray", linewidth=1.5, label="Track bounds")
    ax.plot(outer[:, 0], outer[:, 1], color="gray", linewidth=1.5)
    ax.plot(
        world.data["posE_m"],
        world.data["posN_m"],
        color="black",
        linewidth=0.6,
        linestyle="--",
        alpha=0.5,
        label="Centerline",
    )

    for i, (east_m, north_m, radius_m, radius_tilde_m) in enumerate(TrajectoryVisualizer(world)._get_obstacle_plot_data()):
        ax.add_patch(plt.Circle((east_m, north_m), radius_m, color="tab:red", alpha=0.22))
        ax.add_patch(
            plt.Circle(
                (east_m, north_m),
                radius_tilde_m,
                fill=False,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
            )
        )
        if i == 0:
            ax.plot([], [], color="tab:red", label="Obstacle / safety")

    ax.plot(base_east, base_north, color="0.65", linewidth=2.0, label="Base trajectory", zorder=2)
    ax.plot(repair_east, repair_north, color="tab:blue", linewidth=2.8, label="Repair segment", zorder=4)
    ax.scatter(repair_east[0], repair_north[0], s=70, c="green", marker="o", label="Repair start", zorder=5)
    ax.scatter(repair_east[-1], repair_north[-1], s=80, c="red", marker="x", label="Repair end", zorder=5)

    ax.set_title(f"{episode_id}: full base trajectory with repair overlay")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    filepath = output_dir / f"{episode_id}_context.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def _build_result(npz_path: Path) -> tuple[OptimizationResult, dict]:
    data = np.load(npz_path, allow_pickle=True)
    manifest_entry = _load_manifest_entry(npz_path)

    X = data["X_full"].astype(float).T
    U = data["U"].astype(float).T
    s_m = data["s_m"].astype(float)
    kappa = data["kappa"].astype(float)
    grade = data["grade"].astype(float)
    bank = data["bank"].astype(float)
    t_final = float(X[5, -1])

    result = OptimizationResult(
        success=True,
        s_m=s_m,
        X=X,
        U=U,
        cost=t_final,
        iterations=-1,
        solve_time=-1.0,
        k_psi=kappa,
        theta=grade,
        phi=bank,
        max_obstacle_slack=0.0,
        min_obstacle_clearance=float("inf"),
    )
    return result, manifest_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a repair segment episode.")
    parser.add_argument("--episode-npz", type=str, required=True, help="Path to repair episode .npz file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults under results/trajectory_optimization/repairs/<map_id>/<episode_id>.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Animation frame rate.")
    args = parser.parse_args()

    npz_path = Path(args.episode_npz)
    if not npz_path.is_absolute():
        npz_path = (project_root / npz_path).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Repair episode not found: {npz_path}")

    result, manifest_entry = _build_result(npz_path)
    map_id = manifest_entry["map_id"]
    episode_id = manifest_entry["episode_id"]
    base_id = manifest_entry["base_id"]
    map_file = project_root / "maps" / f"{map_id}.mat"
    if not map_file.exists():
        raise FileNotFoundError(f"Map file not found for repair episode: {map_file}")
    base_npz_path = project_root / "data" / "base_laps" / map_id / f"{base_id}.npz"
    if not base_npz_path.exists():
        raise FileNotFoundError(f"Base lap not found for repair episode: {base_npz_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else project_root / "results" / "trajectory_optimization" / "repairs" / map_id / episode_id
    )
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    world = World(str(map_file), map_id, diagnostic_plotting=False)
    _attach_obstacles_to_world(world, manifest_entry.get("obstacles", []))

    visualizer = TrajectoryVisualizer(world, output_dir=str(output_dir))
    plots = visualizer.generate_full_report(result, prefix=episode_id)
    context_plot = _plot_repair_context(world, base_npz_path, result, episode_id, output_dir)
    anim = create_animation(visualizer, result, filename=f"{episode_id}_animation.gif", fps=args.fps)

    print(f"Episode: {episode_id}")
    print(f"Map: {map_id}")
    print(f"Output dir: {output_dir}")
    for key, path in plots.items():
        print(f"{key}: {path}")
    print(f"context: {context_plot}")
    print(f"animation: {anim}")


if __name__ == "__main__":
    main()
