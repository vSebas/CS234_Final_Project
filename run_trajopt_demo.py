#!/usr/bin/env python3
"""
IPOPT trajectory optimization demo (production path).

Runs a two-stage solve:
1) Stage A: feasibility-first
2) Stage B: minimum-time refinement (warm-started from Stage A)
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
            print("IPOPT TRAJECTORY OPTIMIZATION DEMO (STAGE A/B)")
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

            N = int(os.environ.get("N", "60"))
            ds_m = world.length_m / N
            print(f"\n4. Setup:")
            print(f"   N = {N}")
            print(f"   ds = {ds_m:.2f} m")

            optimizer = TrajectoryOptimizer(vehicle, world)
            visualizer = TrajectoryVisualizer(world, output_dir=str(output_dir))

            print("\n" + "=" * 70)
            print("STAGE A: FEASIBILITY")
            print("=" * 70)
            stage_a = optimizer.solve(
                N=N,
                ds_m=ds_m,
                stage="feas",
                obstacles=obstacles,
                obstacle_window_m=float(os.environ.get("OBS_WINDOW_M", "30.0")),
                obstacle_slack_weight=float(os.environ.get("OBS_SLACK_W_FEAS", "1e4")),
                smoothness_weight=float(os.environ.get("SMOOTHNESS_W_FEAS", "2.0")),
                time_weight_feas=float(os.environ.get("TIME_W_FEAS", "1e-2")),
                ux_min=3.0,
                convergent_lap=True,
                verbose=True,
            )
            print(f"  Success: {stage_a.success}")
            print(f"  Cost: {stage_a.cost:.4f}")
            print(f"  Iterations: {stage_a.iterations}")
            print(f"  Solve time: {stage_a.solve_time:.2f} s")
            print(f"  Max obstacle slack: {stage_a.max_obstacle_slack:.4e}")
            print(f"  Min obstacle clearance: {stage_a.min_obstacle_clearance:.4f} m")

            print("\nGenerating Stage A plots...")
            plots_a = visualizer.generate_full_report(stage_a, prefix="ipopt_stage_a")
            for _, path in plots_a.items():
                print(f"  Saved: {path}")

            print("\n" + "=" * 70)
            print("STAGE B: MIN-TIME REFINEMENT")
            print("=" * 70)
            stage_b = optimizer.solve(
                N=N,
                ds_m=ds_m,
                stage="time",
                X_init=stage_a.X,
                U_init=stage_a.U,
                obstacles=obstacles,
                obstacle_window_m=float(os.environ.get("OBS_WINDOW_M", "30.0")),
                obstacle_slack_weight=float(os.environ.get("OBS_SLACK_W_TIME", "1e6")),
                smoothness_weight=float(os.environ.get("SMOOTHNESS_W_TIME", "0.5")),
                ux_min=3.0,
                convergent_lap=True,
                verbose=True,
            )
            print(f"  Success: {stage_b.success}")
            print(f"  Cost: {stage_b.cost:.4f}")
            print(f"  Iterations: {stage_b.iterations}")
            print(f"  Solve time: {stage_b.solve_time:.2f} s")
            print(f"  Max obstacle slack: {stage_b.max_obstacle_slack:.4e}")
            print(f"  Min obstacle clearance: {stage_b.min_obstacle_clearance:.4f} m")

            print("\nGenerating Stage B plots...")
            plots_b = visualizer.generate_full_report(stage_b, prefix="ipopt_stage_b")
            for _, path in plots_b.items():
                print(f"  Saved: {path}")

            print("\nGenerating comparison plot...")
            comp = visualizer.plot_comparison(
                {"Stage A (feasibility)": stage_a, "Stage B (time)": stage_b},
                filename="ipopt_stage_comparison.png",
                title="IPOPT Stage A/B Comparison",
            )
            print(f"  Saved: {comp}")

            try:
                print("\nGenerating animation (Stage B)...")
                anim = create_animation(visualizer, stage_b, filename="ipopt_stage_b_animation.gif", fps=15)
                print(f"  Saved: {anim}")
            except Exception as err:
                print(f"  Animation skipped: {err}")

            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"{'Run':<25} {'Iters':<8} {'Time [s]':<10} {'Cost [s]':<10} {'Max Sigma':<12} {'Min Clear [m]':<14} {'Success'}")
            print("-" * 95)
            print(f"{'Stage A (feas)':<25} {stage_a.iterations:<8} {stage_a.solve_time:<10.2f} {stage_a.cost:<10.4f} {stage_a.max_obstacle_slack:<12.3e} {stage_a.min_obstacle_clearance:<14.4f} {stage_a.success}")
            print(f"{'Stage B (time)':<25} {stage_b.iterations:<8} {stage_b.solve_time:<10.2f} {stage_b.cost:<10.4f} {stage_b.max_obstacle_slack:<12.3e} {stage_b.min_obstacle_clearance:<14.4f} {stage_b.success}")
            print(f"\nAll outputs saved to: {output_dir}")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    run_demo()
