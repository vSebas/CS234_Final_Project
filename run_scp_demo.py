#!/usr/bin/env python3
"""
SCP Solver Demo and Visualization

This script demonstrates:
1. Running the direct collocation optimizer (baseline, IPOPT)
2. Running the SCP solver with cold start (IPOPT - robust for ill-conditioned)
3. Running the SCP solver with warm start (OSQP - fast convex QP solver)
4. Comparing results and generating visualizations

QP Solver Strategy:
- Cold start uses IPOPT: More robust for ill-conditioned problems
- Warm start uses OSQP: Faster for well-conditioned convex QPs

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
    # Step 2: Run SCP with Cold Start (IPOPT - robust for ill-conditioned)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SCP Solver (Cold Start) - Using IPOPT")
    print("=" * 70)

    # Use IPOPT for cold-start (more robust for ill-conditioned problems)
    scp_params_cold = SCPParams(
        max_iterations=30,
        tr_radius_init=3.0,
        convergence_tol=1e-3,
        qp_solver='ipopt',  # Robust for cold-start
        verbose=True
    )

    scp_solver_cold = SCPSolver(vehicle, world, params=scp_params_cold)

    print("Solving with cold start (no warm-start, IPOPT solver)...")
    scp_cold_result = scp_solver_cold.solve(
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
    print(f"  Feasible: {scp_cold_result.feasible}")
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
    # Step 3: Run SCP with Warm Start (from DC) - Using OSQP
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SCP Solver (Warm Start from Direct Collocation) - Using OSQP")
    print("=" * 70)

    # Use OSQP for warm-start (faster convex QP solver, works well with good init)
    scp_params_warm = SCPParams(
        max_iterations=30,
        tr_radius_init=3.0,
        convergence_tol=1e-3,
        qp_solver='osqp',  # Fast convex QP solver for warm-start
        verbose=True
    )

    scp_solver_warm = SCPSolver(vehicle, world, params=scp_params_warm)

    print("Solving with warm start from DC result (OSQP solver)...")
    scp_warm_result = scp_solver_warm.solve(
        N=N,
        ds_m=ds_m,
        X_init=dc_result.X,  # Warm-start from DC
        U_init=dc_result.U,
        ux_min=3.0,
        convergent_lap=True
    )

    early_exit_used = scp_warm_result.iterations == 0 and scp_warm_result.feasible
    print(f"\nSCP Warm Start Result:")
    print(f"  Success: {scp_warm_result.success}")
    print(f"  Converged: {scp_warm_result.converged}")
    print(f"  Feasible: {scp_warm_result.feasible}")
    print(f"  Early exit: {early_exit_used}")
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

    print(f"\n{'Method':<30} {'Solver':<8} {'Iters':<6} {'Time[s]':<8} {'Cost[s]':<9} {'Conv':<6} {'Feas':<6} {'Early'}")
    print("-" * 90)
    print(f"{'Direct Collocation':<30} {'IPOPT':<8} {dc_result.iterations:<6} {dc_result.solve_time:<8.2f} {dc_result.cost:<9.4f} {'N/A':<6} {'N/A':<6} {'N/A'}")
    print(f"{'SCP (Cold Start)':<30} {'IPOPT':<8} {scp_cold_result.iterations:<6} {scp_cold_result.solve_time:<8.2f} {scp_cold_result.cost:<9.4f} {str(scp_cold_result.converged):<6} {str(scp_cold_result.feasible):<6} {'No'}")
    warm_early = "Yes" if (scp_warm_result.iterations == 0 and scp_warm_result.feasible) else "No"
    print(f"{'SCP (Warm Start)':<30} {'OSQP':<8} {scp_warm_result.iterations:<6} {scp_warm_result.solve_time:<8.2f} {scp_warm_result.cost:<9.4f} {str(scp_warm_result.converged):<6} {str(scp_warm_result.feasible):<6} {warm_early}")

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
