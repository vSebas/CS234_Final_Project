#!/usr/bin/env python3
"""
Rockit-style FATROP trajectory optimization runner.

This script is intentionally standalone (separate entrypoint) and uses the
stage-structured OCP formulation that scales better with FATROP than the older
IPOPT-style collocation path.

Note:
- This does not require the external `rockit` Python package at runtime.
- It keeps a dedicated CLI for "Rockit + FATROP" experiments while reusing the
  validated native FATROP formulation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer
from world.world import World

from experiments.run_fatrop_native_trajopt import _load_obstacles_from_world, solve_fatrop_native


def _set_default_env_if_missing(key: str, value: str) -> None:
    if key not in os.environ:
        os.environ[key] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--compare-ipopt", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Rockit-style default profile for FATROP experiments.
    _set_default_env_if_missing("FATROP_PRESET", "obstacle_fast")
    _set_default_env_if_missing("FATROP_STRUCTURE_DETECTION", "none")
    _set_default_env_if_missing("FATROP_EXPAND", "0")
    _set_default_env_if_missing("FATROP_STAGE_LOCAL_COST", "1")
    _set_default_env_if_missing("FATROP_DYNAMICS_SCHEME", "trapezoidal")
    _set_default_env_if_missing("FATROP_CLOSURE_MODE", "soft")
    _set_default_env_if_missing("FATROP_CLOSURE_SOFT_WEIGHT", "100")
    _set_default_env_if_missing("FATROP_MAX_ITER", "800")

    map_file = Path(args.map_file)
    if not map_file.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    world = World(str(map_file), map_file.stem, diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    obstacles = _load_obstacles_from_world(world)
    ds_m = float(world.length_m / int(args.N))

    if args.compare_ipopt:
        base = TrajectoryOptimizer(vehicle, world).solve(
            N=int(args.N),
            ds_m=ds_m,
            lambda_u=0.005,
            ux_min=0.5,
            track_buffer_m=0.0,
            eps_s=0.1,
            eps_kappa=0.05,
            obstacles=obstacles,
            obstacle_clearance_m=0.0,
            obstacle_use_slack=False,
            obstacle_enforce_midpoints=False,
            obstacle_subsamples_per_segment=5,
            convergent_lap=False,
            verbose=False,
        )
        print(
            f"[ipopt] success={base.success} iterations={base.iterations} "
            f"cost={base.cost:.6f} solve_time={base.solve_time:.3f}s"
        )

    out = solve_fatrop_native(
        vehicle=vehicle,
        world=world,
        N=int(args.N),
        ds_m=ds_m,
        obstacles=obstacles,
        lambda_u=0.005,
        ux_min=0.5,
        track_buffer_m=0.0,
        obstacle_clearance_m=0.0,
        eps_s=0.1,
        eps_kappa=0.05,
        verbose=bool(args.verbose),
    )
    print(
        f"[rockit-fatrop] success={out.success} iterations={out.iterations} "
        f"cost={out.cost:.6f} solve_time={out.solve_time:.3f}s "
        f"build_time={getattr(out, 'build_time_s', float('nan')):.3f}s "
        f"total_time={getattr(out, 'total_time_s', float('nan')):.3f}s "
        f"min_clearance={out.min_obstacle_clearance:.4f}"
    )


if __name__ == "__main__":
    main()
