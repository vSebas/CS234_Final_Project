#!/usr/bin/env python3
"""
SCP Solver Demo and Visualization

This script demonstrates:
1. Running the direct collocation optimizer (baseline, IPOPT)
2. Running the SCP solver with cold start (IPOPT - robust for ill-conditioned)
3. Comparing results and generating visualizations

Solver Strategy:
- Direct collocation uses IPOPT
- SCP cold start uses IPOPT

All outputs are saved to the results/ directory.
"""

import os
import sys
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
    # Use a lighter default for SCP runtime; override with SCP_N if needed.
    N = int(os.environ.get("SCP_N", "60"))
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

    # Use IPOPT-backed convex subproblems for robust SCP iterations.
    # OSQP path is currently broken (systematic subproblem failures).
    scp_params_cold = SCPParams(
        max_iterations=120,
        max_solve_time_s=120.0,
        tr_radius_init=1.0,
        tr_radius_min=0.005,
        tr_shrink_factor=0.5,
        tr_expand_factor=1.4,
        virtual_control_weight=5e4,
        defect_penalty_weight=5e4,
        constraint_tol=5e-2,
        virtual_control_tol=1e-2,
        convergence_tol=1e-3,
        defect_switch_tol=5e-2,
        early_exit_on_feasible=True,
        qp_solver='ipopt',
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
    # Step 3: Comparison and Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Comparison and Analysis")
    print("=" * 70)

    # Comparison plot
    print("\nGenerating comparison plots...")
    comparison_path = visualizer.plot_comparison(
        {
            'Direct Collocation': dc_result,
            'SCP Cold Start': scp_cold_result,
        },
        filename="method_comparison.png",
        title="Trajectory Optimization Method Comparison"
    )
    print(f"  Saved: {comparison_path}")

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

    print(f"\n{'Method':<30} {'Solver':<8} {'Iters':<8} {'Time [s]':<10} {'Cost [s]':<10} {'Success'}")
    print("-" * 75)
    print(f"{'Direct Collocation':<30} {'IPOPT':<8} {dc_result.iterations:<8} {dc_result.solve_time:<10.2f} {dc_result.cost:<10.4f} {dc_result.success}")
    print(f"{'SCP (Cold Start)':<30} {'IPOPT':<8} {scp_cold_result.iterations:<8} {scp_cold_result.solve_time:<10.2f} {scp_cold_result.cost:<10.4f} {scp_cold_result.success}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nDemo complete!")


if __name__ == "__main__":
    run_demo()
