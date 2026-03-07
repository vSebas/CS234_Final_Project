#!/usr/bin/env python3
"""
Generate baseline-vs-DT warm-start trajectory comparison plots with explicit lap-time annotations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from experiments.eval_warmstart import EvalConfig, sample_obstacles
from models import load_vehicle_from_yaml
from planning import TrajectoryOptimizer
from planning.dt_warmstart import load_warmstarter
from utils.world import World


def _get_scenario_obstacles(
    world: World,
    map_file: str,
    seed: int,
    num_scenarios: int,
    scenario_id: int,
    num_obstacles: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    cfg = EvalConfig(
        map_file=map_file,
        num_scenarios=num_scenarios,
        seed=seed,
        min_obstacles=num_obstacles,
        max_obstacles=num_obstacles,
    )
    for sid in range(num_scenarios):
        n = int(rng.integers(num_obstacles, num_obstacles + 1))
        obs = sample_obstacles(rng, world, n, cfg)
        if sid == scenario_id:
            return obs
    return []


def _plot_case(
    world: World,
    label: str,
    scenario_id: int,
    seed: int,
    obstacles: List[Dict],
    base_result,
    dt_result,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    inner = world.data["inner_bounds_m"]
    outer = world.data["outer_bounds_m"]
    ax.plot(inner[:, 0], inner[:, 1], color="0.6", lw=1.2)
    ax.plot(outer[:, 0], outer[:, 1], color="0.6", lw=1.2)
    ax.plot(world.data["posE_m"], world.data["posN_m"], color="0.8", lw=0.8, ls="--")

    for obs in obstacles:
        e_arr, n_arr, _ = world.map_match_vectorized(
            np.array([obs["s_m"]]), np.array([obs["e_m"]])
        )
        e = float(e_arr[0])
        n = float(n_arr[0])
        r = float(obs["radius_m"])
        r_safe = r + float(obs.get("margin_m", 0.0))
        ax.add_patch(
            Circle((e, n), r, edgecolor="tab:red", facecolor="tab:red", alpha=0.25, lw=1.2)
        )
        ax.add_patch(
            Circle((e, n), r_safe, edgecolor="tab:red", facecolor="none", ls="--", lw=1.0)
        )

    be, bn, _ = world.map_match_vectorized(base_result.s_m, base_result.X[6, :])
    de, dn, _ = world.map_match_vectorized(dt_result.s_m, dt_result.X[6, :])
    ax.plot(
        be,
        bn,
        color="tab:blue",
        lw=2.0,
        label=f"baseline ({base_result.cost:.2f}s, {base_result.iterations} it)",
    )
    ax.plot(
        de,
        dn,
        color="tab:orange",
        lw=2.0,
        label=f"DT warmstart ({dt_result.cost:.2f}s, {dt_result.iterations} it)",
    )
    ax.scatter([be[0]], [bn[0]], c="green", s=35, zorder=5)

    lap_delta = float(dt_result.cost - base_result.cost)
    solve_delta = float(dt_result.solve_time - base_result.solve_time)
    iter_delta = int(dt_result.iterations - base_result.iterations)
    metrics_text = (
        f"Lap time: baseline {base_result.cost:.2f}s | DT {dt_result.cost:.2f}s | delta {lap_delta:+.2f}s\n"
        f"Solve time: baseline {base_result.solve_time:.2f}s | DT {dt_result.solve_time:.2f}s | delta {solve_delta:+.2f}s\n"
        f"Iterations: baseline {base_result.iterations} | DT {dt_result.iterations} | delta {iter_delta:+d}"
    )
    ax.text(
        0.02,
        0.02,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.6"},
    )

    ax.set_aspect("equal")
    ax.set_title(
        f"Oval trajectory compare: {label} (scenario {scenario_id}, seed {seed})\n"
        f"baseline {base_result.cost:.2f}s vs DT {dt_result.cost:.2f}s"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot warm-start trajectory comparisons with explicit lap-time annotations.")
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--map-file", default="maps/Oval_Track_260m.mat", type=str)
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--num-scenarios", default=3, type=int)
    ap.add_argument("--scenario-id", default=0, type=int)
    ap.add_argument("--N", default=120, type=int)
    ap.add_argument("--target-lap-time", default=15.517, type=float)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = World(args.map_file, Path(args.map_file).stem, diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
    optimizer = TrajectoryOptimizer(vehicle, world)
    warmstarter = load_warmstarter(args.checkpoint, vehicle, world, device="cpu")

    cfg = EvalConfig(
        checkpoint_path=args.checkpoint,
        map_file=args.map_file,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
        N=args.N,
    )
    ds_m = world.length_m / float(args.N)
    x0 = np.array([cfg.ux_min + 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    summary = {"cases": []}
    for num_obs, label in [(0, "noobs"), (1, "obs1")]:
        obstacles = _get_scenario_obstacles(
            world=world,
            map_file=args.map_file,
            seed=args.seed,
            num_scenarios=args.num_scenarios,
            scenario_id=args.scenario_id,
            num_obstacles=num_obs,
        )
        baseline = optimizer.solve(
            N=args.N,
            ds_m=ds_m,
            lambda_u=cfg.lambda_u,
            ux_min=cfg.ux_min,
            track_buffer_m=cfg.track_buffer_m,
            obstacles=obstacles if obstacles else None,
            obstacle_window_m=30.0,
            obstacle_clearance_m=cfg.obstacle_clearance,
            obstacle_use_slack=False,
            obstacle_enforce_midpoints=False,
            eps_s=cfg.eps_s,
            eps_kappa=cfg.eps_kappa,
            convergent_lap=True,
            verbose=False,
        )
        ws = warmstarter.generate_warmstart(
            N=args.N,
            ds_m=ds_m,
            x0=x0,
            target_lap_time=args.target_lap_time,
            obstacles=obstacles,
            obstacle_clearance_m=cfg.obstacle_clearance,
            vehicle_radius_m=0.0,
        )
        dt = optimizer.solve(
            N=args.N,
            ds_m=ds_m,
            lambda_u=cfg.lambda_u,
            ux_min=cfg.ux_min,
            track_buffer_m=cfg.track_buffer_m,
            obstacles=obstacles if obstacles else None,
            obstacle_window_m=30.0,
            obstacle_clearance_m=cfg.obstacle_clearance,
            obstacle_use_slack=False,
            obstacle_enforce_midpoints=False,
            eps_s=cfg.eps_s,
            eps_kappa=cfg.eps_kappa,
            convergent_lap=True,
            X_init=ws.X_init,
            U_init=ws.U_init,
            verbose=False,
        )

        out_png = out_dir / f"{label}_scenario{args.scenario_id}_compare.png"
        _plot_case(
            world=world,
            label=label,
            scenario_id=args.scenario_id,
            seed=args.seed,
            obstacles=obstacles,
            base_result=baseline,
            dt_result=dt,
            out_path=out_png,
        )
        summary["cases"].append(
            {
                "label": label,
                "plot": str(out_png),
                "baseline": {
                    "success": bool(baseline.success),
                    "solve_time_s": float(baseline.solve_time),
                    "iters": int(baseline.iterations),
                    "lap_s": float(baseline.cost),
                },
                "dt": {
                    "success": bool(dt.success),
                    "solve_time_s": float(dt.solve_time),
                    "iters": int(dt.iterations),
                    "lap_s": float(dt.cost),
                    "fallback_count": int(getattr(ws, "fallback_count", 0)),
                    "projection_count": int(getattr(ws, "projection_count", 0)),
                },
                "obstacles": obstacles,
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)
    for case in summary["cases"]:
        print(case["label"], case["plot"])


if __name__ == "__main__":
    main()
