#!/usr/bin/env python3
"""
IPOPT trajectory optimization demo (production path).

Runs a single-stage solve with combined objective terms:
- time objective
- smoothness regularization
- obstacle slack penalty
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml
from planning import ObstacleCircle, TrajectoryOptimizer
from utils.visualization import TrajectoryVisualizer, create_animation
from world.world import World


class TeeStream:
    """Write to terminal and log file at the same time."""
    def __init__(self, terminal_stream, log_file):
        self.terminal_stream = terminal_stream
        self.log_file = log_file

    def write(self, data):
        self.terminal_stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.terminal_stream.flush()
        self.log_file.flush()


def create_vehicle():
    """Create vehicle model from YAML configuration."""
    config_file = project_root / "models" / "config" / "vehicle_params_gti.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Vehicle config not found at {config_file}. "
            "Please ensure models/config/vehicle_params_gti.yaml exists."
        )
    return load_vehicle_from_yaml(config_file)


def load_obstacles_from_world(world: World):
    """Load obstacle metadata from .mat map if present."""
    data = world.data
    required = {
        "obstacles_s_m",
        "obstacles_e_m",
        "obstacles_radius_m",
        "obstacles_margin_m",
    }
    if not required.issubset(set(data.keys())):
        return []

    s_vals = np.atleast_1d(data["obstacles_s_m"]).astype(float)
    e_vals = np.atleast_1d(data["obstacles_e_m"]).astype(float)
    r_vals = np.atleast_1d(data["obstacles_radius_m"]).astype(float)
    m_vals = np.atleast_1d(data["obstacles_margin_m"]).astype(float)
    if not (len(s_vals) == len(e_vals) == len(r_vals) == len(m_vals)):
        raise ValueError("Obstacle arrays in map have inconsistent lengths.")

    obstacles = []
    for s_m, e_m, r_m, m_m in zip(s_vals, e_vals, r_vals, m_vals):
        obstacles.append(
            ObstacleCircle(
                s_m=float(s_m),
                e_m=float(e_m),
                radius_m=float(r_m),
                margin_m=float(m_m),
            )
        )
    return obstacles


def run_demo():
    output_dir = project_root / "results" / "trajectory_optimization"
    output_dir.mkdir(exist_ok=True)
    log_file = output_dir / "run_trajopt_demo_output.log"

    with open(log_file, "w", encoding="utf-8") as lf:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, lf)
        sys.stderr = TeeStream(original_stderr, lf)
        try:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] Logging to: {log_file}")
            print("=" * 70)
            print("IPOPT TRAJECTORY OPTIMIZATION DEMO (SINGLE STAGE)")
            print("=" * 70)

            map_file = Path(os.environ.get("TRACK_MAP", str(project_root / "maps" / "Medium_Oval_Map_260m_Obstacles.mat")))
            if not map_file.exists():
                map_file = project_root / "maps" / "Medium_Oval_Map_260m.mat"
            if not map_file.exists():
                print(f"Error: Map file not found at {map_file}")
                print("Run: python create_oval_track.py")
                return

            matplotlib.use("Agg")

            print(f"\n1. Loading track: {map_file}")
            world = World(str(map_file), "Oval", diagnostic_plotting=False)
            print(f"   Track length: {world.length_m:.1f} m")

            print("\n2. Creating vehicle model...")
            vehicle = create_vehicle()
            print(f"   Vehicle: {vehicle}")

            obstacles = load_obstacles_from_world(world)
            print(f"\n3. Loaded obstacles from map: {len(obstacles)}")

            N = int(os.environ.get("N", "120"))
            ds_m = world.length_m / N
            extra_clearance_m = float(os.environ.get("OBS_EXTRA_CLEAR_M", "0.5"))
            obstacle_subsamples = int(os.environ.get("OBS_SUBSAMPLES", "7"))
            track_buffer_m = float(os.environ.get("TRACK_BUFFER_M", "0.0"))
            print(f"\n4. Setup:")
            print(f"   N = {N}")
            print(f"   ds = {ds_m:.2f} m")
            print(f"   Extra obstacle clearance = {extra_clearance_m:.2f} m")
            print(f"   Obstacle subsamples/segment = {obstacle_subsamples}")
            print(f"   Track buffer = {track_buffer_m:.2f} m")

            # Update world metadata so visualizations show the enforced obstacle radius.
            if "obstacles_radius_m" in world.data:
                r = np.atleast_1d(np.asarray(world.data["obstacles_radius_m"], dtype=float))
                m = np.atleast_1d(np.asarray(world.data.get("obstacles_margin_m", np.zeros_like(r)), dtype=float))
                world.data["obstacles_radius_tilde_m"] = r + m + extra_clearance_m
            if "obstacles_ENR_m" in world.data:
                enr = np.atleast_2d(np.asarray(world.data["obstacles_ENR_m"], dtype=float))
                if enr.shape[1] == 3:
                    required = enr.copy()
                    margin = np.atleast_1d(np.asarray(world.data.get("obstacles_margin_m", np.zeros(required.shape[0])), dtype=float))
                    required[:, 2] = required[:, 2] + margin + extra_clearance_m
                    world.data["obstacles_ENR_required_m"] = required

            optimizer = TrajectoryOptimizer(vehicle, world)
            visualizer = TrajectoryVisualizer(world, output_dir=str(output_dir))

            print("\n" + "=" * 70)
            print("SINGLE STAGE: COMBINED OBJECTIVE")
            print("=" * 70)
            result = optimizer.solve(
                N=N,
                ds_m=ds_m,
                stage="time",
                track_buffer_m=track_buffer_m,
                obstacles=obstacles,
                obstacle_window_m=float(os.environ.get("OBS_WINDOW_M", "30.0")),
                obstacle_clearance_m=extra_clearance_m,
                obstacle_subsamples_per_segment=obstacle_subsamples,
                obstacle_slack_weight=float(os.environ.get("OBS_SLACK_W", "1e5")),
                smoothness_weight=float(os.environ.get("SMOOTHNESS_W", "1.0")),
                ux_min=3.0,
                convergent_lap=True,
                verbose=True,
            )
            print(f"  Success: {result.success}")
            print(f"  Cost: {result.cost:.4f}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Solve time: {result.solve_time:.2f} s")
            print(f"  Max obstacle slack: {result.max_obstacle_slack:.4e}")
            print(f"  Min obstacle clearance: {result.min_obstacle_clearance:.4f} m")

            print("\nGenerating plots...")
            plots = visualizer.generate_full_report(result, prefix="ipopt_single_stage")
            for _, path in plots.items():
                print(f"  Saved: {path}")

            try:
                print("\nGenerating animation...")
                anim = create_animation(visualizer, result, filename="ipopt_single_stage_animation.gif", fps=15)
                print(f"  Saved: {anim}")
            except Exception as err:
                print(f"  Animation skipped: {err}")

            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"{'Run':<25} {'Iters':<8} {'Time [s]':<10} {'Cost [s]':<10} {'Max Sigma':<12} {'Min Clear [m]':<14} {'Success'}")
            print("-" * 95)
            print(f"{'Single stage':<25} {result.iterations:<8} {result.solve_time:<10.2f} {result.cost:<10.4f} {result.max_obstacle_slack:<12.3e} {result.min_obstacle_clearance:<14.4f} {result.success}")
            print(f"\nAll outputs saved to: {output_dir}")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    run_demo()
