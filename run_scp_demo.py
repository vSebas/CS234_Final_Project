#!/usr/bin/env python3
"""
SCP Solver Demo and Visualization

This script demonstrates:
1. Running the direct collocation optimizer (baseline)
2. Running the SCP solver with cold start
3. Running the SCP solver with warm start (from direct collocation)
4. Comparing results and generating visualizations

All outputs are saved to the results/ directory.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer, SCPSolver, SCPParams
from world.world import World
from utils.visualization import TrajectoryVisualizer, create_animation


def create_vehicle():
    """
    Create vehicle model from VW Golf GTI YAML configuration.

    Uses parameters from: Aggarwal & Gerdes, "Friction-Robust Autonomous Racing
    Using Trajectory Optimization Over Multiple Models", IEEE OJCS 2025.
    """
    config_file = project_root  / "models" / "config" / "vehicle_params_gti.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Vehicle config not found at {config_file}. "
            "Please ensure models/config/vehicle_params_gti.yaml exists."
        )

    return load_vehicle_from_yaml(config_file)


def run_demo():
    """Run the full demo."""
    print("=" * 70)
    print("SCP SOLVER DEMO WITH VISUALIZATION")
    print("=" * 70)

    # Create output directory
    output_dir = project_root / "results" / "trajectory_optimization"
    output_dir.mkdir(exist_ok=True)

    # Load track
    map_file = project_root / "maps" / "Medium_Oval_Map_260m.mat"
    if not map_file.exists():
        print(f"Error: Map file not found at {map_file}")
        print("Please run create_oval_track.py first")
        return

    print("\n1. Loading track...")
    # Disable diagnostic plotting in World
    import matplotlib
    matplotlib.use('Agg')

    world = World(str(map_file), "Oval", diagnostic_plotting=False)
    print(f"   Track length: {world.length_m:.1f} m")

    # Create vehicle
    print("\n2. Creating vehicle model...")
    vehicle = create_vehicle()
    print(f"   Vehicle: {vehicle}")

    # Create visualizer
    visualizer = TrajectoryVisualizer(world, output_dir=str(output_dir))

    # Optimization parameters
    N = 100  # Number of discretization points
    ds_m = world.length_m / N  # Step size to cover exactly one lap

    print(f"\n3. Optimization setup:")
    print(f"   N = {N} points")
    print(f"   ds = {ds_m:.2f} m")
    print(f"   Total distance = {N * ds_m:.1f} m")

    # =========================================================================
    # Step 1: Run Direct Collocation (baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Direct Collocation Optimizer (IPOPT)")
    print("=" * 70)

    dc_optimizer = TrajectoryOptimizer(vehicle, world)

    print("Solving...")
    dc_result = dc_optimizer.solve(
        N=N,
        ds_m=ds_m,
        ux_min=3.0,
        convergent_lap=True,
        verbose=True
    )

    print(f"\nDirect Collocation Result:")
    print(f"  Success: {dc_result.success}")
    print(f"  Lap time: {dc_result.cost:.4f} s")
    print(f"  Iterations: {dc_result.iterations}")
    print(f"  Solve time: {dc_result.solve_time:.2f} s")

    # Generate visualizations for DC result
    print("\nGenerating Direct Collocation visualizations...")
    dc_plots = visualizer.generate_full_report(dc_result, prefix="dc")
    for name, path in dc_plots.items():
        print(f"  Saved: {path}")

    # =========================================================================
    # Step 2: Run SCP with Cold Start
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SCP Solver (Cold Start)")
    print("=" * 70)

    scp_params = SCPParams(
        max_iterations=30,
        tr_radius_init=3.0,
        convergence_tol=1e-3,
        verbose=True
    )

    scp_solver = SCPSolver(vehicle, world, params=scp_params)

    print("Solving with cold start (no warm-start)...")
    scp_cold_result = scp_solver.solve(
        N=N,
        ds_m=ds_m,
        X_init=None,  # No warm-start
        U_init=None,
        ux_min=3.0,
        convergent_lap=True
    )

    print(f"\nSCP Cold Start Result:")
    print(f"  Success: {scp_cold_result.success}")
    print(f"  Converged: {scp_cold_result.converged}")
    print(f"  Lap time: {scp_cold_result.cost:.4f} s")
    print(f"  Iterations: {scp_cold_result.iterations}")
    print(f"  Solve time: {scp_cold_result.solve_time:.2f} s")
    print(f"  Termination: {scp_cold_result.termination_reason}")

    # Generate visualizations
    print("\nGenerating SCP cold start visualizations...")
    scp_cold_plots = visualizer.generate_full_report(scp_cold_result, prefix="scp_cold")
    for name, path in scp_cold_plots.items():
        print(f"  Saved: {path}")

    # =========================================================================
    # Step 3: Run SCP with Warm Start (from DC)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SCP Solver (Warm Start from Direct Collocation)")
    print("=" * 70)

    print("Solving with warm start from DC result...")
    scp_warm_result = scp_solver.solve(
        N=N,
        ds_m=ds_m,
        X_init=dc_result.X,  # Warm-start from DC
        U_init=dc_result.U,
        ux_min=3.0,
        convergent_lap=True
    )

    print(f"\nSCP Warm Start Result:")
    print(f"  Success: {scp_warm_result.success}")
    print(f"  Converged: {scp_warm_result.converged}")
    print(f"  Lap time: {scp_warm_result.cost:.4f} s")
    print(f"  Iterations: {scp_warm_result.iterations}")
    print(f"  Solve time: {scp_warm_result.solve_time:.2f} s")
    print(f"  Termination: {scp_warm_result.termination_reason}")

    # Generate visualizations
    print("\nGenerating SCP warm start visualizations...")
    scp_warm_plots = visualizer.generate_full_report(scp_warm_result, prefix="scp_warm")
    for name, path in scp_warm_plots.items():
        print(f"  Saved: {path}")

    # =========================================================================
    # Step 4: Comparison and Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Comparison and Analysis")
    print("=" * 70)

    # Comparison plot
    print("\nGenerating comparison plots...")
    comparison_path = visualizer.plot_comparison(
        {
            'Direct Collocation': dc_result,
            'SCP Cold Start': scp_cold_result,
            'SCP Warm Start': scp_warm_result,
        },
        filename="method_comparison.png",
        title="Trajectory Optimization Method Comparison"
    )
    print(f"  Saved: {comparison_path}")

    # Warm-start analysis
    analysis_path = visualizer.plot_warm_start_analysis(
        scp_cold_result,
        scp_warm_result,
        filename="warm_start_analysis.png",
        title="Warm-Start Effectiveness Analysis"
    )
    print(f"  Saved: {analysis_path}")

    # Create animation (optional - can be slow)
    try:
        print("\nGenerating trajectory animation...")
        anim_path = create_animation(
            visualizer,
            dc_result,
            filename="trajectory_animation.gif",
            fps=15
        )
        print(f"  Saved: {anim_path}")
    except Exception as e:
        print(f"  Animation skipped: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Iterations':<12} {'Time [s]':<12} {'Cost [s]':<12} {'Success'}")
    print("-" * 70)
    print(f"{'Direct Collocation':<25} {dc_result.iterations:<12} {dc_result.solve_time:<12.2f} {dc_result.cost:<12.4f} {dc_result.success}")
    print(f"{'SCP (Cold Start)':<25} {scp_cold_result.iterations:<12} {scp_cold_result.solve_time:<12.2f} {scp_cold_result.cost:<12.4f} {scp_cold_result.success}")
    print(f"{'SCP (Warm Start)':<25} {scp_warm_result.iterations:<12} {scp_warm_result.solve_time:<12.2f} {scp_warm_result.cost:<12.4f} {scp_warm_result.success}")

    # Calculate speedup
    if scp_cold_result.iterations > 0 and scp_warm_result.iterations > 0:
        iteration_speedup = scp_cold_result.iterations / scp_warm_result.iterations
        time_speedup = scp_cold_result.solve_time / scp_warm_result.solve_time
        print(f"\nWarm-start speedup:")
        print(f"  Iteration reduction: {iteration_speedup:.2f}x ({scp_cold_result.iterations} -> {scp_warm_result.iterations})")
        print(f"  Time reduction: {time_speedup:.2f}x ({scp_cold_result.solve_time:.2f}s -> {scp_warm_result.solve_time:.2f}s)")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nDemo complete!")


if __name__ == "__main__":
    run_demo()
