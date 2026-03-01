#!/usr/bin/env python3
"""
IPOPT trajectory optimization demo (production path).

Runs a single-stage solve with combined objective terms:
- time objective
- control-difference regularization (lambda_u)
- obstacle slack penalty
"""

import argparse
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
from planning import ObstacleCircle, OptimizationResult, TrajectoryOptimizer
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


def run_python_solver_direct(
    optimizer: TrajectoryOptimizer,
    N: int,
    ds_m: float,
    lambda_u: float,
    ux_min: float,
    ux_max,
    track_buffer_m: float,
    eps_s: float,
    eps_kappa: float,
    obstacles,
    obstacle_window_m: float,
    obstacle_clearance_m: float,
    obstacle_use_slack: bool,
    obstacle_enforce_midpoints: bool,
    obstacle_subsamples_per_segment: int,
    obstacle_slack_weight: float,
    vehicle_radius_m: float,
    convergent_lap: bool,
    verbose: bool,
) -> OptimizationResult:
    return optimizer.solve(
        N=N,
        ds_m=ds_m,
        lambda_u=lambda_u,
        ux_min=ux_min,
        ux_max=ux_max,
        track_buffer_m=track_buffer_m,
        obstacles=obstacles,
        obstacle_window_m=obstacle_window_m,
        obstacle_clearance_m=obstacle_clearance_m,
        obstacle_use_slack=obstacle_use_slack,
        obstacle_enforce_midpoints=obstacle_enforce_midpoints,
        obstacle_subsamples_per_segment=obstacle_subsamples_per_segment,
        obstacle_slack_weight=obstacle_slack_weight,
        obstacle_aware_init=True,
        obstacle_init_sigma_m=8.0,
        obstacle_init_margin_m=0.3,
        vehicle_radius_m=vehicle_radius_m,
        eps_s=eps_s,
        eps_kappa=eps_kappa,
        convergent_lap=convergent_lap,
        verbose=verbose,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run IPOPT trajectory optimization demo.")
    parser.add_argument("--map-file", type=str, default=None, help="Path to .mat map file.")
    parser.add_argument("--n", type=int, default=None, help="Number of spatial nodes.")
    parser.add_argument("--obs-subsamples", type=int, default=None, help="Obstacle midpoint subsamples per segment.")
    parser.add_argument("--obs-enforce-midpoints", action=argparse.BooleanOptionalAction, default=None, help="Enable midpoint obstacle constraints.")
    parser.add_argument("--obs-use-slack", action=argparse.BooleanOptionalAction, default=None, help="Enable obstacle slack variables.")
    parser.add_argument("--obs-slack-weight", type=float, default=None, help="Penalty weight for obstacle slack.")
    parser.add_argument("--track-buffer-m", type=float, default=None, help="Track boundary inward buffer [m].")
    parser.add_argument("--lambda-u", type=float, default=None, help="Control-difference regularization weight.")
    parser.add_argument("--obstacle-clearance-m", type=float, default=None, help="Extra obstacle clearance [m].")
    parser.add_argument("--vehicle-radius-m", type=float, default=None, help="Vehicle footprint radius [m].")
    parser.add_argument("--eps-s", type=float, default=None, help="Minimum forward progress.")
    parser.add_argument("--eps-kappa", type=float, default=None, help="Minimum Frenet non-singularity margin.")
    parser.add_argument("--accept-min-clearance-m", type=float, default=None, help="Acceptance gate min dense clearance [m].")
    parser.add_argument("--accept-max-slack", type=float, default=None, help="Acceptance gate max obstacle slack.")
    parser.add_argument("--obs-window-m", type=float, default=None, help="Along-track obstacle activation window [m].")
    return parser.parse_args()


def run_demo(args):
    output_dir = project_root / "results" / "trajectory_optimization" / "nlp"
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

            # CLI arg first, then MAP_FILE/TRACK_MAP env vars.
            map_env = args.map_file or os.environ.get("MAP_FILE") or os.environ.get("TRACK_MAP")
            if map_env:
                map_file = Path(map_env)
                if not map_file.is_absolute():
                    if map_file.exists():
                        map_file = map_file.resolve()
                    else:
                        map_file = (project_root / map_file).resolve()
                if not map_file.exists():
                    print(f"Error: Requested map file not found: {map_file}")
                    print("Set MAP_FILE to a valid .mat path.")
                    return
            else:
                map_file = project_root / "maps" / "Medium_Oval_Map_260m.mat"
                if not map_file.exists():
                    print(f"Error: Default map file not found at {map_file}")
                    print("Run: python create_tracks.py --preset all")
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
            optimizer = TrajectoryOptimizer(vehicle, world)

            # Paper-aligned default discretization (N=260 -> ds=1 m on the 260 m oval).
            N = int(args.n if args.n is not None else os.environ.get("N", "260"))
            ds_m = world.length_m / N
            obstacle_subsamples = int(args.obs_subsamples if args.obs_subsamples is not None else os.environ.get("OBS_SUBSAMPLES", "7"))
            obstacle_enforce_midpoints = (
                args.obs_enforce_midpoints
                if args.obs_enforce_midpoints is not None
                else (os.environ.get("OBS_ENFORCE_MIDPOINTS", "0") == "1")
            )
            obstacle_use_slack = (
                args.obs_use_slack
                if args.obs_use_slack is not None
                else (os.environ.get("OBS_USE_SLACK", "0") == "1")
            )
            obstacle_slack_weight = float(args.obs_slack_weight if args.obs_slack_weight is not None else os.environ.get("OBS_SLACK_WEIGHT", "1e4"))
            track_buffer_m = float(args.track_buffer_m if args.track_buffer_m is not None else os.environ.get("TRACK_BUFFER_M", "0.0"))
            lambda_u = float(args.lambda_u if args.lambda_u is not None else os.environ.get("LAMBDA_U", "0.001"))
            base_clearance_m = float(args.obstacle_clearance_m if args.obstacle_clearance_m is not None else os.environ.get("OBSTACLE_CLEARANCE_M", "0.0"))
            obstacle_aware_init = os.environ.get("OBSTACLE_AWARE_INIT", "1") != "0"
            obstacle_init_sigma_m = float(os.environ.get("OBSTACLE_INIT_SIGMA_M", "8.0"))
            obstacle_init_margin_m = float(os.environ.get("OBSTACLE_INIT_MARGIN_M", "0.3"))
            vehicle_radius_m = float(args.vehicle_radius_m if args.vehicle_radius_m is not None else os.environ.get("VEHICLE_RADIUS_M", "0.0"))
            eps_s = float(args.eps_s if args.eps_s is not None else os.environ.get("EPS_S", "0.1"))
            eps_kappa = float(args.eps_kappa if args.eps_kappa is not None else os.environ.get("EPS_KAPPA", "0.05"))
            print(f"\n4. Setup:")
            print(f"   N = {N}")
            print(f"   ds = {ds_m:.2f} m")
            print(f"   Obstacle subsamples/segment = {obstacle_subsamples}")
            print(f"   Obstacle enforce midpoints = {obstacle_enforce_midpoints}")
            print(f"   Obstacle use slack = {obstacle_use_slack}")
            print(f"   Track buffer = {track_buffer_m:.2f} m")
            print(f"   lambda_u = {lambda_u:.4g}")
            print(f"   vehicle_radius_m = {vehicle_radius_m:.2f}")
            print(f"   eps_s = {eps_s:.3f}, eps_kappa = {eps_kappa:.3f}")

            visualizer = TrajectoryVisualizer(world, output_dir=str(output_dir))
            print("   Solver path = Python/CasADi (IPOPT)")

            # Deterministic retry schedule for acceptance-gated runs.
            # Each fallback step increases conservatism/resolution in a fixed order.
            acceptance_min_clearance_m = float(
                args.accept_min_clearance_m
                if args.accept_min_clearance_m is not None
                else os.environ.get("ACCEPT_MIN_CLEARANCE_M", "-0.001")
            )
            acceptance_max_slack = float(
                args.accept_max_slack
                if args.accept_max_slack is not None
                else os.environ.get("ACCEPT_MAX_SLACK", "0.0")
            )
            obstacle_window_m = float(args.obs_window_m if args.obs_window_m is not None else os.environ.get("OBS_WINDOW_M", "30.0"))

            retry_N_1 = int(os.environ.get("RETRY_N_1", str(max(N, 160))))
            retry_subsamples_2 = int(os.environ.get("RETRY_SUBSAMPLES_2", str(max(obstacle_subsamples, 11))))
            retry_clearance_3 = float(os.environ.get("RETRY_CLEARANCE_3", str(max(base_clearance_m, 0.10))))
            retry_N_3 = int(os.environ.get("RETRY_N_3", str(max(retry_N_1, 180))))
            retry_subsamples_3 = int(os.environ.get("RETRY_SUBSAMPLES_3", str(max(retry_subsamples_2, 13))))

            attempt_configs = [
                ("baseline", N, obstacle_subsamples, base_clearance_m),
                ("retry_higher_N", retry_N_1, obstacle_subsamples, base_clearance_m),
                ("retry_higher_subsamples", retry_N_1, retry_subsamples_2, base_clearance_m),
                ("retry_more_conservative", retry_N_3, retry_subsamples_3, retry_clearance_3),
            ]

            def is_accepted(res) -> bool:
                return (
                    bool(res.success)
                    and float(res.max_obstacle_slack) <= acceptance_max_slack
                    and float(res.min_obstacle_clearance) >= acceptance_min_clearance_m
                )

            attempt_results = []
            accepted_result = None
            accepted_name = None

            print("\n" + "=" * 70)
            print("SINGLE STAGE: COMBINED OBJECTIVE (ACCEPTANCE-GATED)")
            print("=" * 70)
            print(f"Acceptance gates: success=True, max_slack<={acceptance_max_slack:.3e}, min_clearance>={acceptance_min_clearance_m:.4f} m")
            if acceptance_min_clearance_m < 0:
                print("Acceptance policy: practical epsilon mode (allows tiny negative dense clearance).")
            else:
                print("Acceptance policy: strict mode (no negative dense clearance).")
            print(
                "Initializer: "
                f"obstacle_aware_init={obstacle_aware_init}, "
                f"sigma={obstacle_init_sigma_m:.2f} m, "
                f"margin={obstacle_init_margin_m:.2f} m"
            )

            for idx, (attempt_name, n_i, subs_i, clear_i) in enumerate(attempt_configs, start=1):
                ds_i = world.length_m / n_i
                print(f"\nAttempt {idx}/{len(attempt_configs)}: {attempt_name}")
                print(f"  N={n_i}, ds={ds_i:.3f} m, subsamples={subs_i}, obstacle_clearance_m={clear_i:.3f}")
                result_i = run_python_solver_direct(
                    optimizer=optimizer,
                    N=n_i,
                    ds_m=ds_i,
                    lambda_u=lambda_u,
                    ux_min=3.0,
                    ux_max=None,
                    track_buffer_m=track_buffer_m,
                    eps_s=eps_s,
                    eps_kappa=eps_kappa,
                    obstacles=obstacles,
                    obstacle_window_m=obstacle_window_m,
                    obstacle_clearance_m=clear_i,
                    obstacle_use_slack=obstacle_use_slack,
                    obstacle_enforce_midpoints=obstacle_enforce_midpoints,
                    obstacle_subsamples_per_segment=subs_i,
                    obstacle_slack_weight=obstacle_slack_weight,
                    vehicle_radius_m=vehicle_radius_m,
                    convergent_lap=True,
                    verbose=True,
                )
                accepted_i = is_accepted(result_i)
                attempt_results.append((attempt_name, n_i, subs_i, clear_i, result_i, accepted_i))
                print(f"  Success: {result_i.success}")
                print(f"  Cost: {result_i.cost:.4f}")
                print(f"  Iterations: {result_i.iterations}")
                print(f"  Solve time: {result_i.solve_time:.2f} s")
                print(f"  Max obstacle slack: {result_i.max_obstacle_slack:.4e}")
                print(f"  Min obstacle clearance: {result_i.min_obstacle_clearance:.4f} m")
                print(f"  Accepted: {accepted_i}")
                if accepted_i:
                    accepted_result = result_i
                    accepted_name = attempt_name
                    break

            # If no attempt passes gates, keep the last attempt result for diagnostics/artifacts.
            if accepted_result is None:
                attempt_name, _, _, _, result, _ = attempt_results[-1]
                accepted_name = f"{attempt_name} (failed acceptance)"
                print("\nNo attempt passed acceptance gates; using last attempt outputs for diagnostics.")
            else:
                result = accepted_result
                print(f"\nAccepted attempt: {accepted_name}")

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
            print(f"Selected attempt: {accepted_name}")
            print(f"{'Attempt':<26} {'N':<6} {'Subs':<6} {'Clear':<8} {'Iters':<8} {'Time [s]':<10} {'Cost [s]':<10} {'Max Sigma':<12} {'Min Clear [m]':<14} {'Accepted'}")
            print("-" * 125)
            for attempt_name, n_i, subs_i, clear_i, res_i, accepted_i in attempt_results:
                print(f"{attempt_name:<26} {n_i:<6} {subs_i:<6} {clear_i:<8.3f} {res_i.iterations:<8} {res_i.solve_time:<10.2f} {res_i.cost:<10.4f} {res_i.max_obstacle_slack:<12.3e} {res_i.min_obstacle_clearance:<14.4f} {accepted_i}")
            print(f"\nAll outputs saved to: {output_dir}")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    run_demo(parse_args())
