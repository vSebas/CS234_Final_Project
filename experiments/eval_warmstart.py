#!/usr/bin/env python3
"""
Warm-start evaluation: compare baseline initializer vs DT warm-start.

Based on PLAN.md Section 5:
- Metrics: IPOPT success rate, acceptance rate, solve time, iterations, lap time
- Baselines: (1) baseline init, (2) baseline + retry, (3) DT warm-start + IPOPT

Usage:
    python experiments/eval_warmstart.py \
        --checkpoint dt/checkpoints/checkpoint_best.pt \
        --map-file maps/Oval_Track_260m.mat \
        --num-scenarios 50 \
        --seed 42
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer
from planning.dt_warmstart import DTWarmStarter, load_warmstarter
from world.world import World


@dataclass
class ScenarioResult:
    """Result for a single scenario."""
    scenario_id: int
    method: str  # 'baseline', 'baseline_retry', 'dt_warmstart'

    # Solve outcome
    success: bool
    accepted: bool
    rejection_reason: Optional[str]

    # Performance
    solve_time_s: float
    ipopt_iterations: int
    lap_time_s: float

    # Warm-start specific
    warmstart_time_s: float = 0.0
    warmstart_accepted: bool = True

    # Obstacle info
    num_obstacles: int = 0
    min_clearance_m: float = float('inf')

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    checkpoint_path: Optional[str] = None
    map_file: str = "maps/Oval_Track_260m.mat"
    num_scenarios: int = 50
    seed: int = 42
    output_dir: str = "results/warmstarts/eval"

    # Optimizer settings
    N: int = 120
    lambda_u: float = 0.005
    ux_min: float = 3.0
    track_buffer_m: float = 0.0
    eps_s: float = 0.1
    eps_kappa: float = 0.05

    # Obstacle settings
    min_obstacles: int = 0
    max_obstacles: int = 4
    obstacle_radius_min: float = 0.8
    obstacle_radius_max: float = 1.5
    obstacle_margin: float = 0.3
    obstacle_clearance: float = 0.3

    # Retry settings
    max_retries: int = 3


def sample_obstacles(
    rng: np.random.Generator,
    world: World,
    num_obstacles: int,
    config: EvalConfig,
) -> List[Dict]:
    """Sample random obstacles on the track."""
    if num_obstacles == 0:
        return []

    obstacles = []
    track_length = world.length_m
    min_spacing = 20.0  # Minimum spacing between obstacles

    placed_s = []
    for _ in range(num_obstacles):
        # Try to find valid placement
        for attempt in range(50):
            s = rng.uniform(0, track_length)

            # Check spacing from existing obstacles
            valid = True
            for ps in placed_s:
                ds = abs(s - ps)
                ds = min(ds, track_length - ds)  # Handle wrap
                if ds < min_spacing:
                    valid = False
                    break

            if valid:
                break
        else:
            continue  # Skip if can't place

        # Get track width at this location
        half_width = 0.5 * float(world.track_width_m_LUT(np.array([s])))

        # Random lateral position (with margin from edges)
        edge_margin = config.obstacle_radius_max + config.obstacle_margin
        e_max = half_width - edge_margin
        if e_max <= 0:
            e_max = half_width * 0.5
        e = rng.uniform(-e_max, e_max)

        # Random radius
        r = rng.uniform(config.obstacle_radius_min, config.obstacle_radius_max)

        obstacles.append({
            # Keep the optimizer-facing schema canonical here. The DT warm-starter
            # accepts these names and also supports the older aliases internally.
            "s_m": float(s),
            "e_m": float(e),
            "radius_m": float(r),
            "margin_m": float(config.obstacle_margin),
        })
        placed_s.append(s)

    return obstacles


def run_baseline_solve(
    optimizer: TrajectoryOptimizer,
    config: EvalConfig,
    obstacles: List[Dict],
    verbose: bool = False,
) -> Tuple[bool, Dict]:
    """Run optimizer with baseline initialization."""
    ds_m = optimizer.world.length_m / config.N

    t_start = time.time()
    result = optimizer.solve(
        N=config.N,
        ds_m=ds_m,
        lambda_u=config.lambda_u,
        ux_min=config.ux_min,
        track_buffer_m=config.track_buffer_m,
        obstacles=obstacles if obstacles else None,
        obstacle_window_m=30.0,
        obstacle_clearance_m=config.obstacle_clearance,
        obstacle_use_slack=False,
        obstacle_enforce_midpoints=False,
        eps_s=config.eps_s,
        eps_kappa=config.eps_kappa,
        convergent_lap=True,
        verbose=verbose,
    )
    solve_time = time.time() - t_start

    return result.success, {
        "solve_time_s": solve_time,
        "ipopt_iterations": result.iterations,
        "lap_time_s": result.cost if result.success else float('inf'),
        "min_clearance_m": getattr(result, 'min_obstacle_clearance', float('inf')),
    }


def run_dt_warmstart_solve(
    optimizer: TrajectoryOptimizer,
    warmstarter: DTWarmStarter,
    config: EvalConfig,
    obstacles: List[Dict],
    verbose: bool = False,
) -> Tuple[bool, bool, Dict]:
    """Run optimizer with DT warm-start."""
    ds_m = optimizer.world.length_m / config.N

    # Generate warm-start
    x0 = np.array([config.ux_min + 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ws_result = warmstarter.generate_warmstart(
        N=config.N,
        ds_m=ds_m,
        x0=x0,
        obstacles=obstacles,
        obstacle_clearance_m=config.obstacle_clearance,
        vehicle_radius_m=0.0,
    )

    warmstart_accepted = ws_result.success

    if not warmstart_accepted:
        # Fall back to baseline
        success, metrics = run_baseline_solve(optimizer, config, obstacles, verbose)
        metrics["warmstart_time_s"] = ws_result.inference_time_s
        metrics["warmstart_accepted"] = False
        return success, False, metrics

    # Run IPOPT with warm-start
    t_start = time.time()
    result = optimizer.solve(
        N=config.N,
        ds_m=ds_m,
        lambda_u=config.lambda_u,
        ux_min=config.ux_min,
        track_buffer_m=config.track_buffer_m,
        obstacles=obstacles if obstacles else None,
        obstacle_window_m=30.0,
        obstacle_clearance_m=config.obstacle_clearance,
        obstacle_use_slack=False,
        obstacle_enforce_midpoints=False,
        eps_s=config.eps_s,
        eps_kappa=config.eps_kappa,
        convergent_lap=True,
        X_init=ws_result.X_init,
        U_init=ws_result.U_init,
        verbose=verbose,
    )
    solve_time = time.time() - t_start

    return result.success, True, {
        "solve_time_s": solve_time,
        "ipopt_iterations": result.iterations,
        "lap_time_s": result.cost if result.success else float('inf'),
        "min_clearance_m": getattr(result, 'min_obstacle_clearance', float('inf')),
        "warmstart_time_s": ws_result.inference_time_s,
        "warmstart_accepted": True,
    }


def evaluate_scenario(
    scenario_id: int,
    optimizer: TrajectoryOptimizer,
    warmstarter: Optional[DTWarmStarter],
    obstacles: List[Dict],
    config: EvalConfig,
    verbose: bool = False,
) -> List[ScenarioResult]:
    """Evaluate all methods on a single scenario."""
    results = []

    # 1. Baseline
    success, metrics = run_baseline_solve(optimizer, config, obstacles, verbose)
    results.append(ScenarioResult(
        scenario_id=scenario_id,
        method="baseline",
        success=success,
        accepted=success,
        rejection_reason=None if success else "solver_failed",
        num_obstacles=len(obstacles),
        **metrics,
    ))

    # 2. Baseline with retry (if failed)
    if not success and config.max_retries > 0:
        for retry in range(config.max_retries):
            success_retry, metrics_retry = run_baseline_solve(optimizer, config, obstacles, verbose)
            if success_retry:
                results.append(ScenarioResult(
                    scenario_id=scenario_id,
                    method="baseline_retry",
                    success=True,
                    accepted=True,
                    rejection_reason=None,
                    num_obstacles=len(obstacles),
                    **metrics_retry,
                ))
                break
        else:
            # All retries failed
            results.append(ScenarioResult(
                scenario_id=scenario_id,
                method="baseline_retry",
                success=False,
                accepted=False,
                rejection_reason="all_retries_failed",
                num_obstacles=len(obstacles),
                **metrics,
            ))
    else:
        # No retry needed or no retries configured
        results.append(ScenarioResult(
            scenario_id=scenario_id,
            method="baseline_retry",
            success=success,
            accepted=success,
            rejection_reason=None if success else "solver_failed",
            num_obstacles=len(obstacles),
            **metrics,
        ))

    # 3. DT warm-start (if available)
    if warmstarter is not None:
        success_dt, ws_accepted, metrics_dt = run_dt_warmstart_solve(
            optimizer, warmstarter, config, obstacles, verbose
        )
        metrics_dt = dict(metrics_dt)
        metrics_dt.pop("warmstart_accepted", None)
        results.append(ScenarioResult(
            scenario_id=scenario_id,
            method="dt_warmstart",
            success=success_dt,
            accepted=success_dt,
            rejection_reason=None if success_dt else "solver_failed",
            warmstart_accepted=ws_accepted,
            num_obstacles=len(obstacles),
            **metrics_dt,
        ))

    return results


def compute_summary_stats(results: List[ScenarioResult]) -> Dict:
    """Compute summary statistics for each method."""
    methods = set(r.method for r in results)
    summary = {}

    for method in methods:
        method_results = [r for r in results if r.method == method]
        n = len(method_results)

        successes = [r for r in method_results if r.success]
        n_success = len(successes)

        summary[method] = {
            "n_scenarios": n,
            "success_rate": n_success / n if n > 0 else 0,
            "n_success": n_success,
            "n_failed": n - n_success,
        }

        if successes:
            solve_times = [r.solve_time_s for r in successes]
            iterations = [r.ipopt_iterations for r in successes]
            lap_times = [r.lap_time_s for r in successes]

            summary[method].update({
                "solve_time_mean": np.mean(solve_times),
                "solve_time_std": np.std(solve_times),
                "solve_time_median": np.median(solve_times),
                "iterations_mean": np.mean(iterations),
                "iterations_std": np.std(iterations),
                "iterations_median": np.median(iterations),
                "lap_time_mean": np.mean(lap_times),
                "lap_time_std": np.std(lap_times),
            })

            # DT-specific metrics
            if method == "dt_warmstart":
                ws_times = [r.warmstart_time_s for r in method_results]
                ws_accepted = sum(1 for r in method_results if r.warmstart_accepted)
                summary[method].update({
                    "warmstart_time_mean": np.mean(ws_times),
                    "warmstart_acceptance_rate": ws_accepted / n if n > 0 else 0,
                    "total_time_mean": np.mean(solve_times) + np.mean(ws_times),
                })

    return summary


def print_summary(summary: Dict) -> None:
    """Print summary table."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    methods = ["baseline", "baseline_retry", "dt_warmstart"]
    methods = [m for m in methods if m in summary]

    # Header
    print(f"\n{'Method':<20} {'Success':<12} {'Solve Time':<15} {'Iterations':<15} {'Lap Time':<12}")
    print("-" * 80)

    for method in methods:
        s = summary[method]
        success_str = f"{s['n_success']}/{s['n_scenarios']} ({100*s['success_rate']:.1f}%)"

        if s['n_success'] > 0:
            time_str = f"{s['solve_time_mean']:.2f} +/- {s['solve_time_std']:.2f}s"
            iter_str = f"{s['iterations_mean']:.1f} +/- {s['iterations_std']:.1f}"
            lap_str = f"{s['lap_time_mean']:.2f}s"
        else:
            time_str = "N/A"
            iter_str = "N/A"
            lap_str = "N/A"

        print(f"{method:<20} {success_str:<12} {time_str:<15} {iter_str:<15} {lap_str:<12}")

    # DT-specific
    if "dt_warmstart" in summary and summary["dt_warmstart"]["n_success"] > 0:
        s = summary["dt_warmstart"]
        print(f"\nDT Warm-start Details:")
        print(f"  Warm-start inference time: {s['warmstart_time_mean']:.3f}s")
        print(f"  Warm-start acceptance rate: {100*s['warmstart_acceptance_rate']:.1f}%")
        print(f"  Total time (ws + solve): {s['total_time_mean']:.2f}s")

    # Speedup calculation
    if "baseline" in summary and "dt_warmstart" in summary:
        b = summary["baseline"]
        d = summary["dt_warmstart"]
        if b["n_success"] > 0 and d["n_success"] > 0:
            time_speedup = b["solve_time_mean"] / d["solve_time_mean"]
            iter_reduction = 1 - d["iterations_mean"] / b["iterations_mean"]
            print(f"\nSpeedup (DT vs Baseline):")
            print(f"  Solve time speedup: {time_speedup:.2f}x")
            print(f"  Iteration reduction: {100*iter_reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate warm-start methods")
    parser.add_argument("--checkpoint", type=str, default=None, help="DT checkpoint (optional)")
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--num-scenarios", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/warmstarts/eval")
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--min-obstacles", type=int, default=0)
    parser.add_argument("--max-obstacles", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        map_file=args.map_file,
        num_scenarios=args.num_scenarios,
        seed=args.seed,
        output_dir=args.output_dir,
        N=args.N,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
    )

    # Setup output
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Warm-Start Evaluation")
    print("=" * 60)
    print(f"Map: {config.map_file}")
    print(f"Scenarios: {config.num_scenarios}")
    print(f"Obstacles: {config.min_obstacles}-{config.max_obstacles}")
    print(f"DT checkpoint: {config.checkpoint_path or 'None (baseline only)'}")
    print()

    # Load world and vehicle
    world = World(config.map_file, Path(config.map_file).stem, diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    optimizer = TrajectoryOptimizer(vehicle, world)

    # Load DT warm-starter if checkpoint provided
    warmstarter = None
    if config.checkpoint_path:
        try:
            warmstarter = load_warmstarter(config.checkpoint_path, vehicle, world)
            print(f"Loaded DT warm-starter")
        except Exception as e:
            print(f"Warning: Could not load DT checkpoint: {e}")
            print("Running baseline evaluation only.")

    # Run evaluation
    rng = np.random.default_rng(config.seed)
    all_results = []

    for scenario_id in range(config.num_scenarios):
        # Sample obstacles
        num_obs = rng.integers(config.min_obstacles, config.max_obstacles + 1)
        obstacles = sample_obstacles(rng, world, num_obs, config)

        print(f"Scenario {scenario_id + 1}/{config.num_scenarios}: {len(obstacles)} obstacles...", end=" ", flush=True)

        # Evaluate
        results = evaluate_scenario(
            scenario_id, optimizer, warmstarter, obstacles, config, config.verbose if hasattr(config, 'verbose') else False
        )
        all_results.extend(results)

        # Quick status
        baseline_result = next(r for r in results if r.method == "baseline")
        status = "OK" if baseline_result.success else "FAIL"
        print(f"{status} ({baseline_result.solve_time_s:.1f}s)")

    # Compute summary
    summary = compute_summary_stats(all_results)
    print_summary(summary)

    # Save results
    results_csv = output_dir / f"warmstart_eval_{timestamp}.csv"
    with open(results_csv, "w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].to_dict().keys())
            writer.writeheader()
            for r in all_results:
                writer.writerow(r.to_dict())
    print(f"\nDetailed results saved to: {results_csv}")

    summary_json = output_dir / f"warmstart_eval_{timestamp}_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
