#!/usr/bin/env python3
"""
Integration test: End-to-end DT warm-start -> IPOPT solve.

This script tests the full pipeline:
1. Load trained DT model
2. Generate warm-start trajectory
3. Pass to IPOPT and solve
4. Verify solution quality

Run without a checkpoint to test baseline only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def test_baseline_solve():
    """Test baseline IPOPT solve (no DT)."""
    print("\n" + "=" * 60)
    print("TEST: Baseline IPOPT Solve")
    print("=" * 60)

    from models import load_vehicle_from_yaml
    from planning import TrajectoryOptimizer
    from world.world import World

    # Load components
    map_file = "maps/Oval_Track_260m.mat"
    world = World(map_file, "Oval_Track_260m", diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    optimizer = TrajectoryOptimizer(vehicle, world)

    # Solve
    N = 120
    ds_m = world.length_m / N

    print(f"Solving N={N}, ds={ds_m:.2f}m...")
    result = optimizer.solve(
        N=N,
        ds_m=ds_m,
        lambda_u=0.005,
        ux_min=3.0,
        convergent_lap=True,
        verbose=False,
    )

    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Solve time: {result.solve_time:.2f}s")
    print(f"  Lap time: {result.cost:.2f}s")

    assert result.success, "Baseline solve failed!"
    print("PASSED")
    return True


def test_dt_model():
    """Test DT model forward pass."""
    print("\n" + "=" * 60)
    print("TEST: DT Model Forward Pass")
    print("=" * 60)

    import torch
    from dt.model import DecisionTransformer, DTConfig

    config = DTConfig()
    model = DecisionTransformer(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"State aug dim: {config.state_aug_dim}")

    # Test forward
    batch_size = 2
    seq_len = config.context_length
    states = torch.randn(batch_size, seq_len, config.state_aug_dim)
    actions = torch.randn(batch_size, seq_len, config.act_dim)
    rtg = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    action_preds, state_preds = model(states, actions, rtg, timesteps)

    assert action_preds.shape == (batch_size, seq_len, config.act_dim)
    assert state_preds.shape == (batch_size, seq_len, config.state_dim)

    print(f"  action_preds shape: {action_preds.shape}")
    print(f"  state_preds shape: {state_preds.shape}")
    print("PASSED")
    return True


def test_dt_warmstart(checkpoint_path: str):
    """Test DT warm-start generation."""
    print("\n" + "=" * 60)
    print("TEST: DT Warm-Start Generation")
    print("=" * 60)

    from models import load_vehicle_from_yaml
    from planning.dt_warmstart import load_warmstarter
    from world.world import World

    # Load components
    map_file = "maps/Oval_Track_260m.mat"
    world = World(map_file, "Oval_Track_260m", diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")

    # Load warm-starter
    print(f"Loading checkpoint: {checkpoint_path}")
    warmstarter = load_warmstarter(checkpoint_path, vehicle, world)

    # Generate warm-start
    N = 120
    ds_m = world.length_m / N
    x0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"Generating warm-start for N={N}...")
    result = warmstarter.generate_warmstart(
        N,
        ds_m,
        x0,
        obstacle_clearance_m=0.0,
        vehicle_radius_m=0.0,
    )

    print(f"  Success: {result.success}")
    print(f"  Rejection reason: {result.rejection_reason}")
    print(f"  Inference time: {result.inference_time_s:.3f}s")
    print(f"  X_init shape: {result.X_init.shape}")
    print(f"  U_init shape: {result.U_init.shape}")

    # Check shapes
    assert result.X_init.shape == (8, N + 1), f"Wrong X_init shape: {result.X_init.shape}"
    assert result.U_init.shape == (2, N + 1), f"Wrong U_init shape: {result.U_init.shape}"

    print("PASSED")
    return result


def test_warmstart_to_ipopt(checkpoint_path: str):
    """Test full pipeline: DT warm-start -> IPOPT solve."""
    print("\n" + "=" * 60)
    print("TEST: DT Warm-Start -> IPOPT Solve")
    print("=" * 60)

    from models import load_vehicle_from_yaml
    from planning import TrajectoryOptimizer
    from planning.dt_warmstart import load_warmstarter
    from world.world import World

    # Load components
    map_file = "maps/Oval_Track_260m.mat"
    world = World(map_file, "Oval_Track_260m", diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    optimizer = TrajectoryOptimizer(vehicle, world)
    warmstarter = load_warmstarter(checkpoint_path, vehicle, world)

    N = 120
    ds_m = world.length_m / N

    # Generate warm-start
    x0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ws_result = warmstarter.generate_warmstart(
        N,
        ds_m,
        x0,
        obstacle_clearance_m=0.0,
        vehicle_radius_m=0.0,
    )

    if not ws_result.success:
        print(f"  Warm-start generation failed: {ws_result.rejection_reason}")
        print("  Falling back to baseline...")
        X_init, U_init = None, None
    else:
        print(f"  Warm-start generated in {ws_result.inference_time_s:.3f}s")
        X_init, U_init = ws_result.X_init, ws_result.U_init

    # Solve with IPOPT
    print("  Running IPOPT...")
    result = optimizer.solve(
        N=N,
        ds_m=ds_m,
        lambda_u=0.005,
        ux_min=3.0,
        convergent_lap=True,
        X_init=X_init,
        U_init=U_init,
        verbose=False,
    )

    print(f"  IPOPT Success: {result.success}")
    print(f"  IPOPT Iterations: {result.iterations}")
    print(f"  IPOPT Solve time: {result.solve_time:.2f}s")
    print(f"  Lap time: {result.cost:.2f}s")

    if result.success:
        print("PASSED")
    else:
        print("FAILED (IPOPT did not converge)")

    return result.success


def main():
    parser = argparse.ArgumentParser(description="Integration tests")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="DT checkpoint path (optional, skip DT tests if not provided)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline IPOPT test")
    args = parser.parse_args()

    results = {}

    # Test baseline
    if not args.skip_baseline:
        try:
            results["baseline"] = test_baseline_solve()
        except Exception as e:
            print(f"FAILED: {e}")
            results["baseline"] = False

    # Test DT model
    try:
        results["dt_model"] = test_dt_model()
    except Exception as e:
        print(f"FAILED: {e}")
        results["dt_model"] = False

    # Test DT warm-start (if checkpoint provided)
    if args.checkpoint:
        try:
            results["dt_warmstart"] = test_dt_warmstart(args.checkpoint) is not None
        except Exception as e:
            print(f"FAILED: {e}")
            results["dt_warmstart"] = False

        try:
            results["full_pipeline"] = test_warmstart_to_ipopt(args.checkpoint)
        except Exception as e:
            print(f"FAILED: {e}")
            results["full_pipeline"] = False
    else:
        print("\n[Skipping DT warm-start tests - no checkpoint provided]")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
