#!/usr/bin/env python3
"""
Warm-start evaluation: compare baseline initializer vs DT warm-start.

Based on PLAN.md Section 5:
- Metrics: IPOPT success rate, acceptance rate, solve time, iterations, lap time
- Baselines: (1) baseline init, (2) baseline + retry, (3) DT warm-start + IPOPT

Usage:
    python experiments/eval_warmstart.py \
        --checkpoint dt/checkpoints/full_run1/checkpoints/checkpoint_best.pt \
        --map-file maps/Oval_Track_260m.mat \
        --num-scenarios 50 \
        --seed 42
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml
from planning import ObstacleCircle
from planning import TrajectoryOptimizer
from planning.dt_warmstart import DTWarmStarter, load_warmstarter
from experiments.run_fatrop_native_trajopt import solve_fatrop_native
from utils.world import World


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
    ws_fallback_count: int = 0
    ws_projection_count: int = 0
    ws_projection_total_magnitude: float = 0.0
    ws_projection_max_magnitude: float = 0.0

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
    output_dir: Optional[str] = None
    dataset_root: str = "data/datasets"
    target_lap_time_s: Optional[float] = None

    # Optimizer settings
    N: int = 120
    lambda_u: float = 0.005
    ux_min: float = 3.0
    track_buffer_m: float = 0.0
    eps_s: float = 0.1
    eps_kappa: float = 0.05
    solver: str = "ipopt"

    # Obstacle settings
    min_obstacles: int = 0
    max_obstacles: int = 4
    obstacle_radius_min: float = 0.8
    obstacle_radius_max: float = 1.5
    obstacle_margin: float = 0.3
    obstacle_clearance: float = 0.3

    # Retry settings
    max_retries: int = 3
    export_rollout_trace: bool = False
    trace_projection_thresh: float = 0.1
    trace_clearance_thresh: float = 0.2
    trace_random_keep_prob: float = 0.01
    trace_max_keep_per_scenario: int = 200
    save_compare_plots: bool = True


def resolve_default_output_dir(config: EvalConfig) -> Path:
    """
    Resolve default output directory for warm-start evaluations.

    Priority:
    1) explicit config.output_dir
    2) checkpoint-local: dt/checkpoints/<run>/warmstarts/eval/<tag>
    3) fallback: results/warmstarts/eval/<tag>
    """
    if config.output_dir:
        return Path(config.output_dir)

    map_id = Path(config.map_file).stem
    tag = (
        f"{map_id}_obs{int(config.min_obstacles)}-{int(config.max_obstacles)}_"
        f"seed{int(config.seed)}_N{int(config.N)}"
    )

    if config.checkpoint_path:
        ckpt = Path(config.checkpoint_path)
        # Expected checkpoint layout:
        # dt/checkpoints/<run>/checkpoints/checkpoint_*.pt
        if ckpt.parent.name == "checkpoints":
            run_dir = ckpt.parent.parent
            return run_dir / "warmstarts" / "eval" / tag

    return Path("results/warmstarts/eval") / tag


def infer_track_target_lap_time_s(map_file: str, dataset_root: str) -> Optional[float]:
    """
    Estimate a track-specific target lap time from the generated dataset.

    Prefer no-obstacle shift episodes for the same map. Fall back to all shift
    episodes for the map if no no-obstacle subset is present.
    """
    map_id = Path(map_file).stem
    manifest_path = Path(dataset_root) / f"{map_id}_shifts" / "manifest.jsonl"
    if not manifest_path.exists():
        return None

    nominal_times: List[float] = []
    fallback_times: List[float] = []

    with open(manifest_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            npz_path = Path(rec["npz_path"])
            if not npz_path.is_absolute():
                npz_path = Path(".") / npz_path
            if not npz_path.exists():
                continue

            with np.load(npz_path, allow_pickle=False) as data:
                rtg = np.asarray(data["rtg"]).reshape(-1)
                if rtg.size == 0:
                    continue
                lap_time_s = float(-rtg[0])

            fallback_times.append(lap_time_s)

            base_id = str(rec.get("base_id", ""))
            obstacles = rec.get("obstacles", [])
            is_nominal = base_id.startswith("noobs_") or len(obstacles) == 0
            if is_nominal:
                nominal_times.append(lap_time_s)

    if nominal_times:
        return float(np.median(np.asarray(nominal_times, dtype=float)))
    if fallback_times:
        return float(np.median(np.asarray(fallback_times, dtype=float)))
    return None


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
) -> Tuple[bool, Dict, object]:
    """Run optimizer with baseline initialization."""
    ds_m = optimizer.world.length_m / config.N
    obstacles_norm = [ObstacleCircle(**obs) for obs in obstacles] if obstacles else None

    t_start = time.time()
    if config.solver == "fatrop":
        result = solve_fatrop_native(
            vehicle=optimizer.vehicle,
            world=optimizer.world,
            N=config.N,
            ds_m=ds_m,
            obstacles=obstacles_norm,
            lambda_u=config.lambda_u,
            ux_min=config.ux_min,
            track_buffer_m=config.track_buffer_m,
            obstacle_window_m=30.0,
            obstacle_clearance_m=config.obstacle_clearance,
            eps_s=config.eps_s,
            eps_kappa=config.eps_kappa,
            verbose=verbose,
        )
    else:
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
    }, result


def run_dt_warmstart_solve(
    optimizer: TrajectoryOptimizer,
    warmstarter: DTWarmStarter,
    config: EvalConfig,
    obstacles: List[Dict],
    verbose: bool = False,
) -> Tuple[bool, bool, Dict, List[Dict], object, object]:
    """Run optimizer with DT warm-start."""
    ds_m = optimizer.world.length_m / config.N

    # Generate warm-start
    x0 = np.array([config.ux_min + 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ws_result = warmstarter.generate_warmstart(
        N=config.N,
        ds_m=ds_m,
        x0=x0,
        target_lap_time=config.target_lap_time_s,
        obstacles=obstacles,
        obstacle_clearance_m=config.obstacle_clearance,
        vehicle_radius_m=0.0,
        collect_rollout_trace=config.export_rollout_trace,
    )

    warmstart_accepted = ws_result.success
    rollout_trace = ws_result.rollout_trace or []

    if not warmstart_accepted:
        # Fall back to baseline
        success, metrics, fallback_result = run_baseline_solve(optimizer, config, obstacles, verbose)
        metrics["warmstart_time_s"] = ws_result.inference_time_s
        metrics["warmstart_accepted"] = False
        metrics["ws_fallback_count"] = ws_result.fallback_count
        metrics["ws_projection_count"] = ws_result.projection_count
        metrics["ws_projection_total_magnitude"] = ws_result.projection_total_magnitude
        metrics["ws_projection_max_magnitude"] = ws_result.projection_max_magnitude
        return success, False, metrics, rollout_trace, fallback_result, ws_result

    # Run trajectory optimizer with warm-start
    t_start = time.time()
    if config.solver == "fatrop":
        obstacles_norm = [ObstacleCircle(**obs) for obs in obstacles] if obstacles else None
        result = solve_fatrop_native(
            vehicle=optimizer.vehicle,
            world=optimizer.world,
            N=config.N,
            ds_m=ds_m,
            obstacles=obstacles_norm,
            lambda_u=config.lambda_u,
            ux_min=config.ux_min,
            track_buffer_m=config.track_buffer_m,
            obstacle_window_m=30.0,
            obstacle_clearance_m=config.obstacle_clearance,
            eps_s=config.eps_s,
            eps_kappa=config.eps_kappa,
            X_init=ws_result.X_init,
            U_init=ws_result.U_init,
            verbose=verbose,
        )
    else:
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
        "ws_fallback_count": ws_result.fallback_count,
        "ws_projection_count": ws_result.projection_count,
        "ws_projection_total_magnitude": ws_result.projection_total_magnitude,
        "ws_projection_max_magnitude": ws_result.projection_max_magnitude,
    }, rollout_trace, result, ws_result


def _save_compare_plot(
    optimizer: TrajectoryOptimizer,
    output_dir: Path,
    scenario_id: int,
    num_obstacles: int,
    obstacles: List[Dict],
    baseline_result,
    dt_result,
    ws_result,
) -> Optional[Path]:
    """Save baseline vs DT (including DT_init) trajectory comparison plot."""
    if baseline_result is None or dt_result is None or ws_result is None:
        return None
    if getattr(ws_result, "X_init", None) is None:
        return None
    if not bool(getattr(baseline_result, "success", False)):
        return None
    if not bool(getattr(dt_result, "success", False)):
        return None

    world = optimizer.world
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
        ax.add_patch(Circle((e, n), r, edgecolor="tab:red", facecolor="tab:red", alpha=0.25, lw=1.2))
        ax.add_patch(Circle((e, n), r_safe, edgecolor="tab:red", facecolor="none", ls="--", lw=1.0))

    be, bn, _ = world.map_match_vectorized(baseline_result.s_m, baseline_result.X[6, :])
    ie, inn, _ = world.map_match_vectorized(dt_result.s_m, ws_result.X_init[6, :])
    de, dn, _ = world.map_match_vectorized(dt_result.s_m, dt_result.X[6, :])
    ax.plot(
        be, bn, color="tab:blue", lw=2.0,
        label=f"baseline ({baseline_result.cost:.2f}s, {baseline_result.iterations} it)",
    )
    ax.plot(
        ie, inn, color="tab:green", lw=1.8, ls="--", alpha=0.95,
        label="DT_init warm-start (pre-solve)",
    )
    ax.plot(
        de, dn, color="tab:orange", lw=2.0,
        label=f"DT warmstart ({dt_result.cost:.2f}s, {dt_result.iterations} it)",
    )
    ax.scatter([be[0]], [bn[0]], c="green", s=35, zorder=5)

    lap_delta = float(dt_result.cost - baseline_result.cost)
    solve_delta = float(dt_result.solve_time - baseline_result.solve_time)
    iter_delta = int(dt_result.iterations - baseline_result.iterations)
    metrics_text = (
        f"Lap time: baseline {baseline_result.cost:.2f}s | DT {dt_result.cost:.2f}s | delta {lap_delta:+.2f}s\n"
        f"Solve time: baseline {baseline_result.solve_time:.2f}s | DT {dt_result.solve_time:.2f}s | delta {solve_delta:+.2f}s\n"
        f"Iterations: baseline {baseline_result.iterations} | DT {dt_result.iterations} | delta {iter_delta:+d}"
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
    ax.set_title(f"Scenario {scenario_id} (obs={num_obstacles})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    out_path = output_dir / f"scenario{scenario_id:03d}_obs{num_obstacles}_compare.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_scenario(
    scenario_id: int,
    optimizer: TrajectoryOptimizer,
    warmstarter: Optional[DTWarmStarter],
    obstacles: List[Dict],
    config: EvalConfig,
    verbose: bool = False,
) -> Tuple[List[ScenarioResult], List[Dict]]:
    """Evaluate all methods on a single scenario."""
    results = []
    collected_trace_rows: List[Dict] = []

    # 1. Baseline
    success, metrics, baseline_raw = run_baseline_solve(optimizer, config, obstacles, verbose)
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
            success_retry, metrics_retry, _ = run_baseline_solve(optimizer, config, obstacles, verbose)
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
        success_dt, ws_accepted, metrics_dt, rollout_trace, dt_raw, ws_obj = run_dt_warmstart_solve(
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

        if config.save_compare_plots and config.output_dir:
            try:
                compare_dir = Path(config.output_dir) / "compare_plots"
                compare_dir.mkdir(parents=True, exist_ok=True)
                _save_compare_plot(
                    optimizer=optimizer,
                    output_dir=compare_dir,
                    scenario_id=scenario_id,
                    num_obstacles=len(obstacles),
                    obstacles=obstacles,
                    baseline_result=baseline_raw,
                    dt_result=dt_raw,
                    ws_result=ws_obj,
                )
            except Exception:
                pass

        if config.export_rollout_trace:
            kept = 0
            for row in rollout_trace:
                projection_mag = float(row.get("projection_mag", 0.0))
                clearance_proxy = float(row.get("clearance_proxy_m", float("inf")))
                fallback_used = bool(row.get("fallback_used", False))
                trigger = (
                    fallback_used
                    or projection_mag >= float(config.trace_projection_thresh)
                    or clearance_proxy <= float(config.trace_clearance_thresh)
                )
                random_keep = np.random.random() < float(config.trace_random_keep_prob)
                if not (trigger or random_keep):
                    continue
                if kept >= int(config.trace_max_keep_per_scenario):
                    break
                trace_row = {
                    "scenario_id": int(scenario_id),
                    "method": "dt_warmstart",
                    "map_file": str(config.map_file),
                    "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else "",
                    "num_obstacles": int(len(obstacles)),
                    "triggered": bool(trigger),
                    "random_keep": bool(random_keep),
                    **row,
                }
                collected_trace_rows.append(trace_row)
                kept += 1

    return results, collected_trace_rows


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
            "success_rate": float(n_success / n) if n > 0 else 0.0,
            "n_success": n_success,
            "n_failed": n - n_success,
        }

        if successes:
            solve_times = [r.solve_time_s for r in successes]
            iterations = [r.ipopt_iterations for r in successes]
            lap_times = [r.lap_time_s for r in successes]

            summary[method].update({
                "solve_time_mean": float(np.mean(solve_times)),
                "solve_time_std": float(np.std(solve_times)),
                "solve_time_median": float(np.median(solve_times)),
                "iterations_mean": float(np.mean(iterations)),
                "iterations_std": float(np.std(iterations)),
                "iterations_median": float(np.median(iterations)),
                "lap_time_mean": float(np.mean(lap_times)),
                "lap_time_std": float(np.std(lap_times)),
            })

            # DT-specific metrics
            if method == "dt_warmstart":
                ws_times = [r.warmstart_time_s for r in method_results]
                ws_accepted = sum(1 for r in method_results if r.warmstart_accepted)
                ws_fallback_counts = [r.ws_fallback_count for r in method_results]
                ws_projection_counts = [r.ws_projection_count for r in method_results]
                ws_projection_totals = [r.ws_projection_total_magnitude for r in method_results]
                ws_projection_maxes = [r.ws_projection_max_magnitude for r in method_results]
                summary[method].update({
                    "warmstart_time_mean": float(np.mean(ws_times)),
                    "warmstart_acceptance_rate": float(ws_accepted / n) if n > 0 else 0.0,
                    "total_time_mean": float(np.mean(solve_times) + np.mean(ws_times)),
                    "fallback_count_mean": float(np.mean(ws_fallback_counts)),
                    "fallback_count_max": int(np.max(ws_fallback_counts)),
                    "projection_count_mean": float(np.mean(ws_projection_counts)),
                    "projection_count_max": int(np.max(ws_projection_counts)),
                    "projection_total_magnitude_mean": float(np.mean(ws_projection_totals)),
                    "projection_max_magnitude_mean": float(np.mean(ws_projection_maxes)),
                    "projection_max_magnitude_max": float(np.max(ws_projection_maxes)),
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
        print(
            f"  Fallback count: mean {s['fallback_count_mean']:.2f}, "
            f"max {int(s['fallback_count_max'])}"
        )
        print(
            f"  Projection count: mean {s['projection_count_mean']:.2f}, "
            f"max {int(s['projection_count_max'])}"
        )
        print(
            f"  Projection magnitude: mean total {s['projection_total_magnitude_mean']:.3f}, "
            f"mean max-step {s['projection_max_magnitude_mean']:.3f}, "
            f"max-step overall {s['projection_max_magnitude_max']:.3f}"
        )

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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory override. Default behavior saves under "
            "dt/checkpoints/<run>/warmstarts/eval/<map_obs_seed_N>/ when --checkpoint "
            "matches the checkpoint layout."
        ),
    )
    parser.add_argument("--dataset-root", type=str, default="data/datasets")
    parser.add_argument(
        "--target-lap-time",
        type=float,
        default=None,
        help="Explicit DT target lap time in seconds. If omitted, use a per-track dataset-calibrated target.",
    )
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--solver", type=str, choices=("ipopt", "fatrop"), default="ipopt")
    parser.add_argument("--min-obstacles", type=int, default=0)
    parser.add_argument("--max-obstacles", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--export-rollout-trace", action="store_true")
    parser.add_argument("--trace-projection-thresh", type=float, default=0.1)
    parser.add_argument("--trace-clearance-thresh", type=float, default=0.2)
    parser.add_argument("--trace-random-keep-prob", type=float, default=0.01)
    parser.add_argument("--trace-max-keep-per-scenario", type=int, default=200)
    parser.add_argument("--save-compare-plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        map_file=args.map_file,
        num_scenarios=args.num_scenarios,
        seed=args.seed,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        target_lap_time_s=args.target_lap_time,
        N=args.N,
        solver=args.solver,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
        export_rollout_trace=args.export_rollout_trace,
        trace_projection_thresh=args.trace_projection_thresh,
        trace_clearance_thresh=args.trace_clearance_thresh,
        trace_random_keep_prob=args.trace_random_keep_prob,
        trace_max_keep_per_scenario=args.trace_max_keep_per_scenario,
        save_compare_plots=bool(args.save_compare_plots),
    )

    # Setup output
    output_dir = resolve_default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Warm-Start Evaluation")
    print("=" * 60)
    print(f"Map: {config.map_file}")
    print(f"Scenarios: {config.num_scenarios}")
    print(f"Obstacles: {config.min_obstacles}-{config.max_obstacles}")
    print(f"DT checkpoint: {config.checkpoint_path or 'None (baseline only)'}")
    print(f"Solver: {config.solver}")
    print(f"Output dir: {output_dir}")
    print()

    if config.target_lap_time_s is None:
        config.target_lap_time_s = infer_track_target_lap_time_s(
            config.map_file,
            config.dataset_root,
        )
    if config.target_lap_time_s is not None:
        print(f"DT target lap time: {config.target_lap_time_s:.3f}s")
    else:
        print("DT target lap time: default heuristic (dataset-calibrated target unavailable)")
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
    all_trace_rows: List[Dict] = []

    for scenario_id in range(config.num_scenarios):
        # Sample obstacles
        num_obs = rng.integers(config.min_obstacles, config.max_obstacles + 1)
        obstacles = sample_obstacles(rng, world, num_obs, config)

        print(f"Scenario {scenario_id + 1}/{config.num_scenarios}: {len(obstacles)} obstacles...", end=" ", flush=True)

        # Evaluate
        results, trace_rows = evaluate_scenario(
            scenario_id, optimizer, warmstarter, obstacles, config, config.verbose if hasattr(config, 'verbose') else False
        )
        all_results.extend(results)
        all_trace_rows.extend(trace_rows)

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

    if config.export_rollout_trace and all_trace_rows:
        trace_jsonl = output_dir / f"warmstart_eval_{timestamp}_rollout_trace.jsonl"
        with open(trace_jsonl, "w", encoding="utf-8") as f:
            for row in all_trace_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Rollout trace saved to: {trace_jsonl} ({len(all_trace_rows)} rows)")


if __name__ == "__main__":
    main()
