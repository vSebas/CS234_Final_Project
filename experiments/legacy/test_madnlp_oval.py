#!/usr/bin/env python3
"""
Oval parity smoke test for MadNLP/ExaModels backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer
from experiments.legacy.run_madnlp_trajopt import solve_madnlp_trajopt
from utils.world import World


def _load_obstacles_from_world(world: World):
    req = ("obstacles_s_m", "obstacles_e_m", "obstacles_radius_m", "obstacles_margin_m")
    data = getattr(world, "data", {})
    if any(k not in data for k in req):
        return []
    s_vals = np.atleast_1d(data["obstacles_s_m"]).astype(float)
    e_vals = np.atleast_1d(data["obstacles_e_m"]).astype(float)
    r_vals = np.atleast_1d(data["obstacles_radius_m"]).astype(float)
    m_vals = np.atleast_1d(data["obstacles_margin_m"]).astype(float)
    n = int(min(len(s_vals), len(e_vals), len(r_vals), len(m_vals)))
    out = []
    for i in range(n):
        out.append(
            {
                "s_m": float(s_vals[i]),
                "e_m": float(e_vals[i]),
                "radius_m": float(r_vals[i]),
                "margin_m": float(m_vals[i]),
            }
        )
    return out


def _run_once(backend: str, N: int, world: World):
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    ds_m = float(world.length_m / N)
    obstacles = _load_obstacles_from_world(world)
    if backend == "madnlp_exa":
        return solve_madnlp_trajopt(
            vehicle=vehicle,
            world=world,
            N=N,
            ds_m=ds_m,
            obstacles=obstacles,
            lambda_u=0.005,
            ux_min=0.5,
            track_buffer_m=0.0,
            obstacle_clearance_m=0.0,
            eps_s=0.1,
            eps_kappa=0.05,
            convergent_lap=True,
            verbose=False,
        )
    optimizer = TrajectoryOptimizer(vehicle, world)
    return optimizer.solve(
        N=N,
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
        convergent_lap=True,
        verbose=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--N", type=int, default=120)
    args = parser.parse_args()

    map_file = Path(args.map_file)
    if not map_file.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    world = World(str(map_file), map_file.stem, diagnostic_plotting=False)

    print("Running baseline (casadi_ipopt)...")
    base = _run_once("casadi_ipopt", int(args.N), world)
    print(
        f"  success={base.success} iterations={base.iterations} "
        f"cost={base.cost:.6f} solve_time={base.solve_time:.3f}s"
    )

    print("Running experimental (madnlp_exa)...")
    mad = _run_once("madnlp_exa", int(args.N), world)
    print(
        f"  success={mad.success} iterations={mad.iterations} "
        f"cost={mad.cost:.6f} solve_time={mad.solve_time:.3f}s"
    )
    if base.X.shape == mad.X.shape and base.U.shape == mad.U.shape:
        dx = float(np.max(np.abs(base.X - mad.X)))
        du = float(np.max(np.abs(base.U - mad.U)))
        print(f"  max|ΔX|={dx:.6e} max|ΔU|={du:.6e}")


if __name__ == "__main__":
    main()
