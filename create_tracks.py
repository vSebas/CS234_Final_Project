#!/usr/bin/env python3
"""
Track generation entrypoint (pruned).

Supported tracks:
- Medium_Oval_Map_260m
- MAP1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.spatial import ConvexHull


def _compute_bounds_from_centerline(
    posE_m: np.ndarray,
    posN_m: np.ndarray,
    psi_rad: np.ndarray,
    width_right_m: np.ndarray,
    width_left_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    posE_m = np.asarray(posE_m, dtype=float)
    posN_m = np.asarray(posN_m, dtype=float)
    psi_rad = np.asarray(psi_rad, dtype=float)
    wr = np.asarray(width_right_m, dtype=float)
    wl = np.asarray(width_left_m, dtype=float)

    normal_E = -np.sin(psi_rad)
    normal_N = np.cos(psi_rad)

    center_E = float(np.mean(posE_m))
    center_N = float(np.mean(posN_m))
    prev_n = None
    for i in range(len(normal_E)):
        n = np.array([normal_E[i], normal_N[i]], dtype=float)
        radial = np.array([posE_m[i] - center_E, posN_m[i] - center_N], dtype=float)
        if np.dot(n, radial) < 0.0:
            n = -n
        if prev_n is not None and np.dot(n, prev_n) < 0.0:
            n = -n
        normal_E[i], normal_N[i] = n[0], n[1]
        prev_n = n

    inner = np.zeros((len(posE_m), 3), dtype=float)
    outer = np.zeros((len(posE_m), 3), dtype=float)
    inner[:, 0] = posE_m - wr * normal_E
    inner[:, 1] = posN_m - wr * normal_N
    outer[:, 0] = posE_m + wl * normal_E
    outer[:, 1] = posN_m + wl * normal_N
    return inner, outer


def create_oval_track(
    output_filename: str,
    total_length: float = 260.0,
    track_width: float = 10.0,
    turn_radius: float = 18.0,
    num_points: int = 520,
) -> dict:
    turn_length = np.pi * turn_radius
    straight_length = (total_length - 2.0 * turn_length) / 2.0
    if straight_length <= 0:
        raise ValueError("Invalid oval dimensions; straight length must be positive.")

    s_m = np.linspace(0, total_length, num_points, endpoint=False)
    ds = s_m[1] - s_m[0]

    posE_m = np.zeros(num_points)
    posN_m = np.zeros(num_points)
    posU_m = np.zeros(num_points)
    psi_rad = np.zeros(num_points)
    curvature = np.zeros(num_points)

    s1_end = straight_length
    s2_end = straight_length + turn_length
    s3_end = 2.0 * straight_length + turn_length

    c1 = np.array([straight_length, turn_radius])
    c2 = np.array([0.0, turn_radius])

    for i, s in enumerate(s_m):
        if s < s1_end:
            posE_m[i] = s
            posN_m[i] = 0.0
            psi_rad[i] = 0.0
            curvature[i] = 0.0
        elif s < s2_end:
            a = (s - s1_end) / turn_radius
            posE_m[i] = c1[0] + turn_radius * np.sin(a)
            posN_m[i] = c1[1] - turn_radius * np.cos(a)
            psi_rad[i] = a
            curvature[i] = 1.0 / turn_radius
        elif s < s3_end:
            d = s - s2_end
            posE_m[i] = straight_length - d
            posN_m[i] = 2.0 * turn_radius
            psi_rad[i] = np.pi
            curvature[i] = 0.0
        else:
            a = (s - s3_end) / turn_radius
            posE_m[i] = c2[0] - turn_radius * np.sin(a)
            posN_m[i] = c2[1] + turn_radius * np.cos(a)
            psi_rad[i] = np.pi + a
            curvature[i] = 1.0 / turn_radius

    psi_s = curvature.copy()
    psi_ss = np.gradient(psi_s, ds)

    grade = np.zeros(num_points)
    bank = np.zeros(num_points)
    track_width_m = np.full(num_points, track_width)

    hw = np.full(num_points, 0.5 * track_width, dtype=float)
    inner, outer = _compute_bounds_from_centerline(posE_m, posN_m, psi_rad, hw, hw)

    data = {
        "s_m": s_m,
        "length_m": total_length,
        "gpsXYZRef_m": np.array([0.0, 0.0, 0.0]),
        "posE_m": posE_m,
        "posN_m": posN_m,
        "posU_m": posU_m,
        "psi_rad": psi_rad,
        "psi_s_radpm": psi_s,
        "psi_ss_radpm2": psi_ss,
        "grade_rad": grade,
        "grade_s_radpm": grade.copy(),
        "grade_ss_radpm2": grade.copy(),
        "bank_rad": bank,
        "bank_s_radpm": bank.copy(),
        "bank_ss_radpm2": bank.copy(),
        "track_width_m": track_width_m,
        "inner_bounds_m": inner,
        "outer_bounds_m": outer,
    }
    sio.savemat(output_filename, data)
    return data


def _frenet_to_en_from_map(
    data: dict,
    s_query_m: np.ndarray,
    e_query_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    s_ref = np.asarray(data["s_m"], dtype=float)
    length_m = float(data["length_m"])
    posE_ref = np.asarray(data["posE_m"], dtype=float)
    posN_ref = np.asarray(data["posN_m"], dtype=float)
    psi_ref = np.asarray(data["psi_rad"], dtype=float)

    s_mod = np.mod(np.asarray(s_query_m, dtype=float), length_m)
    e_arr = np.asarray(e_query_m, dtype=float)

    s_ext = np.r_[s_ref, length_m]
    posE_ext = np.r_[posE_ref, posE_ref[0]]
    posN_ext = np.r_[posN_ref, posN_ref[0]]
    psi_ext = np.r_[psi_ref, psi_ref[0]]

    posE_cl = np.interp(s_mod, s_ext, posE_ext)
    posN_cl = np.interp(s_mod, s_ext, posN_ext)
    psi_cl = np.interp(s_mod, s_ext, psi_ext)

    posE = posE_cl - e_arr * np.sin(psi_cl)
    posN = posN_cl + e_arr * np.cos(psi_cl)
    return posE, posN


def create_oval_track_with_obstacles(
    output_filename: str,
    total_length: float = 260.0,
    track_width: float = 10.0,
    turn_radius: float = 18.0,
    num_points: int = 520,
) -> dict:
    data = create_oval_track(
        output_filename=output_filename,
        total_length=total_length,
        track_width=track_width,
        turn_radius=turn_radius,
        num_points=num_points,
    )

    # Fixed obstacle set for reproducible obstacle-avoidance experiments.
    # These are intentionally larger and close to centerline so the optimizer
    # must make visible avoidance maneuvers.
    obstacles_s_m = np.array([36.0, 86.0, 138.0, 196.0, 238.0], dtype=float)
    obstacles_e_m = np.array([0.35, -0.30, 0.20, -0.25, 0.30], dtype=float)
    obstacles_radius_m = np.array([2.20, 2.10, 2.25, 2.15, 2.05], dtype=float)
    obstacles_margin_m = np.array([0.55, 0.55, 0.55, 0.55, 0.55], dtype=float)
    obstacles_radius_tilde_m = obstacles_radius_m + obstacles_margin_m

    obs_east, obs_north = _frenet_to_en_from_map(data, obstacles_s_m, obstacles_e_m)
    obstacles_ENR_m = np.column_stack([obs_east, obs_north, obstacles_radius_m])
    obstacles_ENR_tilde_m = np.column_stack([obs_east, obs_north, obstacles_radius_tilde_m])

    data["obstacles_s_m"] = obstacles_s_m
    data["obstacles_e_m"] = obstacles_e_m
    data["obstacles_radius_m"] = obstacles_radius_m
    data["obstacles_margin_m"] = obstacles_margin_m
    data["obstacles_radius_tilde_m"] = obstacles_radius_tilde_m
    data["obstacles_ENR_m"] = obstacles_ENR_m
    data["obstacles_ENR_tilde_m"] = obstacles_ENR_tilde_m

    sio.savemat(output_filename, data)
    return data


def _bezier_point(t: float, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    a = (1.0 - t) * p0 + t * p1
    b = (1.0 - t) * p1 + t * p2
    return (1.0 - t) * a + t * b


def _quadratic_bezier_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, num: int) -> np.ndarray:
    t_vals = np.linspace(0.0, 1.0, num)
    return np.vstack([_bezier_point(float(t), p0, p1, p2) for t in t_vals])


def _isaac_interpolate_large_gaps(points: np.ndarray, num_samples: int = 10, gap_threshold: float = 7.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        d = float(np.linalg.norm(p2 - p1))
        if d > gap_threshold:
            for t in np.linspace(0.0, 1.0, num_samples):
                out.append((1.0 - t) * p1 + t * p2)
        else:
            out.append(p2)
    return np.asarray(out, dtype=float)


def _isaac_offset_track(points: np.ndarray, offset_distance: float = 5.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    out = np.zeros_like(pts, dtype=float)
    center = np.mean(pts, axis=0)
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        tangent = pts[next_idx] - pts[prev_idx]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-12:
            tangent = np.array([1.0, 0.0], dtype=float)
        else:
            tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        direction = pts[i] - center
        if np.dot(normal, direction) < 0.0:
            normal = -normal
        out[i] = pts[i] + offset_distance * normal
    return out


def _resample_closed_polyline(points: np.ndarray, num_samples: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    d = np.diff(np.vstack([pts, pts[0]]), axis=0)
    ds = np.sqrt((d * d).sum(axis=1))
    s_nodes = np.r_[0.0, np.cumsum(ds)]
    x_nodes = np.r_[pts[:, 0], pts[0, 0]]
    y_nodes = np.r_[pts[:, 1], pts[0, 1]]
    s_t = np.linspace(0.0, s_nodes[-1], num_samples, endpoint=False)
    x = np.interp(s_t, s_nodes, x_nodes)
    y = np.interp(s_t, s_nodes, y_nodes)
    return np.column_stack([x, y])


def _filter_near_duplicate_points(points: np.ndarray, min_step: float = 1e-3) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - out[-1]) >= min_step:
            out.append(pts[i])
    if len(out) > 2 and np.linalg.norm(np.asarray(out[-1]) - np.asarray(out[0])) < min_step:
        out = out[:-1]
    return np.asarray(out, dtype=float)


def _build_map_from_inner_offset(
    inner_raw: np.ndarray,
    output_filename: str,
    target_length_m: float,
    num_samples: int,
    offset_distance: float,
    label: str,
    target_width_m: float | None = None,
) -> dict:
    inner_raw = _filter_near_duplicate_points(inner_raw, min_step=1e-2)
    if len(inner_raw) < 6:
        raise ValueError(f"{label}: too few distinct points after filtering.")
    outer_raw = _isaac_offset_track(inner_raw, offset_distance=offset_distance)

    inner = _resample_closed_polyline(inner_raw, num_samples)
    outer = _resample_closed_polyline(outer_raw, num_samples)
    center = 0.5 * (inner + outer)
    posE_m = center[:, 0]
    posN_m = center[:, 1]

    dE_c = np.diff(np.r_[posE_m, posE_m[0]])
    dN_c = np.diff(np.r_[posN_m, posN_m[0]])
    L = float(np.sqrt(dE_c * dE_c + dN_c * dN_c).sum())
    if L <= 0.0:
        raise ValueError(f"{label}: invalid centerline length.")
    sc = target_length_m / L
    inner *= sc
    outer *= sc
    posE_m = 0.5 * (inner[:, 0] + outer[:, 0])
    posN_m = 0.5 * (inner[:, 1] + outer[:, 1])

    if target_width_m is not None and target_width_m > 0.0:
        half = 0.5 * (outer - inner)
        cur_w = np.linalg.norm(outer - inner, axis=1)
        w_mean = float(np.mean(cur_w))
        if w_mean > 1e-9:
            gain = target_width_m / w_mean
            center = 0.5 * (inner + outer)
            inner = center - gain * half
            outer = center + gain * half
            posE_m = center[:, 0]
            posN_m = center[:, 1]

    cE = float(np.mean(posE_m))
    cN = float(np.mean(posN_m))
    inner[:, 0] -= cE
    outer[:, 0] -= cE
    posE_m -= cE
    inner[:, 1] -= cN
    outer[:, 1] -= cN
    posN_m -= cN

    dE = np.diff(np.r_[posE_m, posE_m[0]])
    dN = np.diff(np.r_[posN_m, posN_m[0]])
    ds_seg = np.sqrt(dE * dE + dN * dN)
    length_m = float(ds_seg.sum())
    s_m = np.r_[0.0, np.cumsum(ds_seg[:-1])]
    ds = length_m / num_samples

    dE_ds = np.gradient(posE_m, ds)
    dN_ds = np.gradient(posN_m, ds)
    psi_rad = np.unwrap(np.arctan2(-dE_ds, dN_ds))
    psi_s = np.gradient(psi_rad, ds)
    psi_ss = np.gradient(psi_s, ds)

    width = np.linalg.norm(outer - inner, axis=1)
    z = np.zeros(num_samples)
    inner3 = np.zeros((num_samples, 3), dtype=float)
    outer3 = np.zeros((num_samples, 3), dtype=float)
    inner3[:, :2] = inner
    outer3[:, :2] = outer

    data = {
        "s_m": s_m,
        "length_m": length_m,
        "gpsXYZRef_m": np.array([0.0, 0.0, 0.0]),
        "posE_m": posE_m,
        "posN_m": posN_m,
        "posU_m": z,
        "psi_rad": psi_rad,
        "psi_s_radpm": psi_s,
        "psi_ss_radpm2": psi_ss,
        "grade_rad": z.copy(),
        "grade_s_radpm": z.copy(),
        "grade_ss_radpm2": z.copy(),
        "bank_rad": z.copy(),
        "bank_s_radpm": z.copy(),
        "bank_ss_radpm2": z.copy(),
        "track_width_m": width,
        "inner_bounds_m": inner3,
        "outer_bounds_m": outer3,
    }
    sio.savemat(output_filename, data)
    print(
        f"{label}: length={length_m:.2f} m, width_mean={float(np.mean(width)):.2f} m, "
        f"width_min={float(np.min(width)):.2f} m, width_max={float(np.max(width)):.2f} m, samples={num_samples}"
    )
    return data


def _isaac_create_track_points(
    seed: int = 23,
    num_points: int = 10,
    x_bounds: tuple[float, float] = (0.0, 100.0),
    y_bounds: tuple[float, float] = (0.0, 100.0),
    corner_cells: int = 15,
) -> np.ndarray:
    rs = np.random.RandomState(seed)

    x_values = rs.uniform(x_bounds[0], x_bounds[1], num_points)
    y_values = rs.uniform(y_bounds[0], y_bounds[1], num_points)
    points = np.column_stack((x_values, y_values))
    hull = ConvexHull(points)
    hull_verts = points[hull.vertices]

    center = np.mean(hull_verts, axis=0)
    idx = int(rs.randint(0, len(hull_verts) - 2))
    mid = 0.5 * (hull_verts[idx] + hull_verts[(idx + 1) % len(hull_verts)])
    scale = float(rs.uniform(0.1, 1.0))
    displaced = center + scale * (mid - center)
    hull_verts = np.insert(hull_verts, idx + 1, displaced, axis=0)

    inner = []
    outer = []
    for i in range(len(hull_verts)):
        p = hull_verts[i]
        q = hull_verts[(i + 1) % len(hull_verts)]
        rp_in = float(rs.uniform(0.1, 0.4))
        rp_out = float(rs.uniform(0.6, 0.9))
        inner.append((1.0 - rp_in) * p + rp_in * q)
        outer.append((1.0 - rp_out) * p + rp_out * q)
    inner = np.asarray(inner, dtype=float)
    outer = np.asarray(outer, dtype=float)

    curves: list[np.ndarray] = []
    for i, p in enumerate(hull_verts):
        prev_outer = outer[i - 1] if i > 0 else outer[-1]
        seg = _quadratic_bezier_curve(prev_outer, p, inner[i], corner_cells)
        curves.extend(seg.tolist())
    curves.append(outer[-1].tolist())
    curves = np.asarray(curves, dtype=float)
    curves = _isaac_interpolate_large_gaps(curves, num_samples=10, gap_threshold=7.0)
    return curves


def create_isaac_style_track(
    output_filename: str,
    seed: int = 23,
    target_length_m: float = 300.0,
    track_width_m: float = 10.0,
    num_samples: int = 520,
) -> dict:
    inner_raw = _isaac_create_track_points(seed=seed)
    return _build_map_from_inner_offset(
        inner_raw=inner_raw,
        output_filename=output_filename,
        target_length_m=target_length_m,
        num_samples=num_samples,
        offset_distance=5.0,
        label=f"Generated Isaac-style track (reference-faithful, seed={seed})",
        target_width_m=track_width_m,
    )


def create_overview_plot(maps_dir: Path):
    files = sorted(p for p in maps_dir.glob("*.mat"))
    n = len(files)
    cols = min(2, max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n)))

    for i, p in enumerate(files):
        ax = axes[i]
        d = sio.loadmat(p, squeeze_me=True)
        c = colors[i % len(colors)]
        inner = np.asarray(d["inner_bounds_m"])
        outer = np.asarray(d["outer_bounds_m"])
        center_e = np.asarray(d["posE_m"])
        center_n = np.asarray(d["posN_m"])
        ax.plot(inner[:, 0], inner[:, 1], lw=1.5, color=c, alpha=0.95, label="Inner")
        ax.plot(outer[:, 0], outer[:, 1], lw=1.5, color=c, alpha=0.95, label="Outer")
        ax.plot(center_e, center_n, lw=0.9, color=c, alpha=0.45, linestyle="--", label="Centerline")
        ax.set_title(p.stem)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Track Bounds Overview", fontsize=16)
    fig.tight_layout()
    out = maps_dir / "tracks_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Track generation (pruned)")
    parser.add_argument(
        "--preset",
        type=str,
        default="all",
        choices=["medium_oval", "isaac_style", "overview", "all"],
    )
    parser.add_argument(
        "--with-obstacles",
        action="store_true",
        help="Also generate an obstacle variant for the medium oval map.",
    )
    args = parser.parse_args()

    maps = Path("maps")
    maps.mkdir(parents=True, exist_ok=True)

    if args.preset in ("medium_oval", "all"):
        out = maps / "Medium_Oval_Map_260m.mat"
        create_oval_track(str(out))
        print(f"Saved {out}")
        if args.with_obstacles:
            out_obs = maps / "Medium_Oval_Map_260m_Obstacles.mat"
            create_oval_track_with_obstacles(str(out_obs))
            print(f"Saved {out_obs}")

    if args.preset in ("isaac_style", "all"):
        out = maps / "MAP1.mat"
        create_isaac_style_track(str(out), seed=23, target_length_m=300.0, track_width_m=10.0)
        print(f"Saved {out}")

    if args.preset in ("overview", "all"):
        out = create_overview_plot(maps)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
