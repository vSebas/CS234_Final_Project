#!/usr/bin/env python3
"""
Track generation entrypoint.

Supported tracks:
- Oval_Track_260m: Standard oval (260m, 6m width, R=18m turns)
- TRACK1_280m: D-shaped (R_tight=12m, R_wide=24m)
- TRACK2_280m: D-shaped + serpentine on the straight
- TRACK3_300m: Serpentine (continuous curves, no straights)
- TRACK4_330m: Technical (procedural, sharp corners)
- TRACK5_350m: Isaac-style (reference-faithful)
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
    track_width: float = 6.0,
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
    track_width: float = 6.0,
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


def add_obstacles_to_map(
    input_filename: str,
    output_filename: str,
    num_obstacles: int | None = None,
    seed: int | None = None,
) -> dict:
    data = sio.loadmat(input_filename, squeeze_me=True)
    length_m = float(np.atleast_1d(data["length_m"]).item())
    width = np.atleast_1d(data["track_width_m"]).astype(float)
    width_min = float(np.min(width))

    if num_obstacles is None:
        num_obstacles = max(5, int(round(length_m / 60.0)))

    # Deterministic per-file obstacles unless explicitly overridden.
    if seed is None:
        seed = abs(hash(Path(input_filename).stem)) % (2**32)
    rng = np.random.RandomState(seed)

    # Place obstacles away from each other and away from boundaries.
    margin = 0.55
    radius_min, radius_max = 1.6, 2.4
    e_max = max(0.0, 0.5 * width_min - (radius_max + margin))
    if e_max <= 0:
        raise ValueError("Track too narrow for obstacles with current radius/margin.")

    s_base = np.linspace(0.1 * length_m, 0.9 * length_m, num_obstacles, endpoint=True)
    s_jitter = rng.uniform(-0.05 * length_m / num_obstacles, 0.05 * length_m / num_obstacles, size=num_obstacles)
    obstacles_s_m = np.mod(s_base + s_jitter, length_m)
    obstacles_e_m = rng.uniform(-e_max, e_max, size=num_obstacles)
    obstacles_radius_m = rng.uniform(radius_min, radius_max, size=num_obstacles)
    obstacles_margin_m = np.full(num_obstacles, margin, dtype=float)
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


def create_d_shaped_track(
    output_filename: str,
    total_length: float = 260.0,
    track_width: float = 6.0,
    tight_turn_radius: float = 12.0,
    wide_turn_radius: float = 24.0,
    num_points: int = 520,
) -> dict:
    """
    D-shaped track with asymmetric turns (teardrop/egg shape).

    One end has a tight hairpin (small radius), the other has a sweeping
    wide turn (large radius). Both turns touch y=0 at their bottom.

    Geometry (counterclockwise traversal, like the oval):
    - Segment 1: Bottom straight from (0, 0) to (H, 0), heading East (psi=0)
    - Segment 2: Wide turn (right), 180° CCW, center at (H, R_wide)
    - Segment 3: Top diagonal from (H, 2*R_wide) to (0, 2*R_tight)
    - Segment 4: Tight turn (left), 180° CCW, center at (0, R_tight)
    """
    R_t = tight_turn_radius
    R_w = wide_turn_radius

    # Arc lengths
    tight_arc = np.pi * R_t
    wide_arc = np.pi * R_w

    # Height difference at top: top of wide turn is at 2*R_w, top of tight turn at 2*R_t
    delta_h = 2.0 * R_w - 2.0 * R_t

    # Solve for horizontal separation H
    # Total = H + wide_arc + diagonal + tight_arc
    # diagonal = sqrt(H^2 + delta_h^2)
    # Let S = total - wide_arc - tight_arc
    # S = H + sqrt(H^2 + delta_h^2)
    # => H = (S^2 - delta_h^2) / (2*S)
    S = total_length - tight_arc - wide_arc
    if S <= abs(delta_h):
        raise ValueError("Invalid D-shape: not enough length for straights.")

    H = (S * S - delta_h * delta_h) / (2.0 * S)
    if H <= 0:
        raise ValueError("Invalid D-shape: negative horizontal separation.")

    bottom_len = H
    diag_len = np.sqrt(H * H + delta_h * delta_h)

    # Segment boundaries
    s1 = bottom_len
    s2 = s1 + wide_arc
    s3 = s2 + diag_len
    # s4 = total_length

    s_m = np.linspace(0, total_length, num_points, endpoint=False)
    ds = s_m[1] - s_m[0]

    posE_m = np.zeros(num_points)
    posN_m = np.zeros(num_points)
    posU_m = np.zeros(num_points)
    curvature = np.zeros(num_points)

    for i, s in enumerate(s_m):
        if s < s1:
            # Segment 1: bottom straight, going East
            posE_m[i] = s
            posN_m[i] = 0.0
            curvature[i] = 0.0

        elif s < s2:
            # Segment 2: wide turn, 180° CCW starting from (H, 0)
            a = (s - s1) / R_w  # 0 to pi
            posE_m[i] = H + R_w * np.sin(a)
            posN_m[i] = R_w * (1.0 - np.cos(a))
            curvature[i] = 1.0 / R_w

        elif s < s3:
            # Segment 3: top diagonal from (H, 2*R_w) to (0, 2*R_t)
            t = (s - s2) / diag_len
            posE_m[i] = H * (1.0 - t)
            posN_m[i] = 2.0 * R_w - t * delta_h
            curvature[i] = 0.0

        else:
            # Segment 4: tight turn, 180° CCW starting from (0, 2*R_t)
            # Center at (0, R_t), similar to oval's second (left) turn
            a = (s - s3) / R_t  # 0 to pi
            posE_m[i] = -R_t * np.sin(a)
            posN_m[i] = R_t * (1.0 + np.cos(a))
            curvature[i] = 1.0 / R_t

    # Compute heading from path gradient (robust, handles corners smoothly)
    # Use central differences for interior, one-sided at boundaries
    dE = np.zeros(num_points)
    dN = np.zeros(num_points)

    # Central differences for interior points
    for i in range(1, num_points - 1):
        dE[i] = posE_m[i + 1] - posE_m[i - 1]
        dN[i] = posN_m[i + 1] - posN_m[i - 1]

    # Handle boundary with wrap-around (closed track)
    dE[0] = posE_m[1] - posE_m[-1]
    dN[0] = posN_m[1] - posN_m[-1]
    dE[-1] = posE_m[0] - posE_m[-2]
    dN[-1] = posN_m[0] - posN_m[-2]

    # Heading: psi = atan2(dN, dE) in the convention where psi=0 is East
    psi_rad = np.arctan2(dN, dE)
    psi_rad = np.unwrap(psi_rad)

    psi_s = curvature.copy()
    psi_ss = np.gradient(psi_s, ds)

    grade = np.zeros(num_points)
    bank = np.zeros(num_points)
    track_width_m = np.full(num_points, track_width)

    # Compute bounds directly from heading (more robust for asymmetric tracks)
    # For heading psi: tangent = (cos(psi), sin(psi))
    # Left normal (perpendicular, 90° CCW) = (-sin(psi), cos(psi))
    # For counterclockwise track, left side is OUTER
    hw = 0.5 * track_width
    inner = np.zeros((num_points, 3), dtype=float)
    outer = np.zeros((num_points, 3), dtype=float)

    for i in range(num_points):
        # Left normal (outer for counterclockwise track)
        left_E = -np.sin(psi_rad[i])
        left_N = np.cos(psi_rad[i])
        # Outer = centerline + hw * left_normal
        outer[i, 0] = posE_m[i] + hw * left_E
        outer[i, 1] = posN_m[i] + hw * left_N
        # Inner = centerline - hw * left_normal
        inner[i, 0] = posE_m[i] - hw * left_E
        inner[i, 1] = posN_m[i] - hw * left_N

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
    print(
        f"D-shaped track: length={total_length:.1f}m, width={track_width:.1f}m, "
        f"R_tight={R_t:.1f}m, R_wide={R_w:.1f}m, H={H:.1f}m"
    )
    return data


def create_d_shaped_serpentine_track(
    output_filename: str,
    total_length: float = 260.0,
    track_width: float = 6.0,
    tight_turn_radius: float = 12.0,
    wide_turn_radius: float = 24.0,
    serp_amp_m: float = 1.5,
    serp_waves: int = 3,
    num_points: int = 520,
) -> dict:
    """
    D-shaped track with a serpentine added to the bottom straight.

    The serpentine is a smooth lateral offset applied to the bottom straight.
    The whole polyline is then rescaled to match total_length and resampled
    to uniform arc length.
    """
    if serp_waves < 1:
        raise ValueError("serp_waves must be >= 1")

    R_t = tight_turn_radius
    R_w = wide_turn_radius

    tight_arc = np.pi * R_t
    wide_arc = np.pi * R_w
    delta_h = 2.0 * R_w - 2.0 * R_t

    S = total_length - tight_arc - wide_arc
    if S <= abs(delta_h):
        raise ValueError("Invalid D-shape: not enough length for straights.")

    H = (S * S - delta_h * delta_h) / (2.0 * S)
    if H <= 0:
        raise ValueError("Invalid D-shape: negative horizontal separation.")

    bottom_len = H
    diag_len = np.sqrt(H * H + delta_h * delta_h)

    s1 = bottom_len
    s2 = s1 + wide_arc
    s3 = s2 + diag_len

    # Build a dense polyline for resampling.
    dense = 2000
    pts = []

    # Segment 1: bottom straight with serpentine (x in [0, H], y offset)
    n1 = max(int(dense * (bottom_len / total_length)), 200)
    u = np.linspace(0.0, 1.0, n1, endpoint=False)
    # Windowed sine keeps zero offset + zero slope at endpoints
    offset = serp_amp_m * np.sin(2.0 * np.pi * serp_waves * u) * np.sin(np.pi * u)
    x1 = bottom_len * u
    y1 = offset
    pts.append(np.column_stack([x1, y1]))

    # Segment 2: wide 180° CCW turn from (H, 0) to (H, 2*R_w)
    n2 = max(int(dense * (wide_arc / total_length)), 200)
    a = np.linspace(0.0, np.pi, n2, endpoint=False)
    x2 = H + R_w * np.sin(a)
    y2 = R_w * (1.0 - np.cos(a))
    pts.append(np.column_stack([x2, y2]))

    # Segment 3: top diagonal from (H, 2*R_w) to (0, 2*R_t)
    n3 = max(int(dense * (diag_len / total_length)), 200)
    t = np.linspace(0.0, 1.0, n3, endpoint=False)
    x3 = H * (1.0 - t)
    y3 = 2.0 * R_w - t * delta_h
    pts.append(np.column_stack([x3, y3]))

    # Segment 4: tight 180° CCW turn from (0, 2*R_t) to (0, 0)
    n4 = max(int(dense * (tight_arc / total_length)), 200)
    a = np.linspace(0.0, np.pi, n4, endpoint=False)
    x4 = -R_t * np.sin(a)
    y4 = R_t * (1.0 + np.cos(a))
    pts.append(np.column_stack([x4, y4]))

    pts = np.vstack(pts)

    # Close the loop
    pts = np.vstack([pts, pts[0]])

    # Rescale to target length
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_raw = np.r_[0.0, np.cumsum(d)]
    raw_length = s_raw[-1]
    scale = total_length / raw_length
    pts *= scale
    s_raw *= scale
    raw_length *= scale

    # Center the track
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])

    # Uniform resample to num_points
    s_m = np.linspace(0.0, raw_length, num_points, endpoint=False)
    posE_m = np.interp(s_m, s_raw, pts[:, 0])
    posN_m = np.interp(s_m, s_raw, pts[:, 1])
    posU_m = np.zeros(num_points)

    ds = s_m[1] - s_m[0]
    dE = np.zeros(num_points)
    dN = np.zeros(num_points)
    for i in range(1, num_points - 1):
        dE[i] = posE_m[i + 1] - posE_m[i - 1]
        dN[i] = posN_m[i + 1] - posN_m[i - 1]
    dE[0] = posE_m[1] - posE_m[-1]
    dN[0] = posN_m[1] - posN_m[-1]
    dE[-1] = posE_m[0] - posE_m[-2]
    dN[-1] = posN_m[0] - posN_m[-2]

    psi_rad = np.arctan2(dN, dE)
    psi_rad = np.unwrap(psi_rad)
    psi_s = np.gradient(psi_rad, ds)
    psi_ss = np.gradient(psi_s, ds)

    grade = np.zeros(num_points)
    bank = np.zeros(num_points)
    track_width_m = np.full(num_points, track_width)

    hw = 0.5 * track_width
    inner = np.zeros((num_points, 3), dtype=float)
    outer = np.zeros((num_points, 3), dtype=float)
    for i in range(num_points):
        left_E = -np.sin(psi_rad[i])
        left_N = np.cos(psi_rad[i])
        outer[i, 0] = posE_m[i] + hw * left_E
        outer[i, 1] = posN_m[i] + hw * left_N
        inner[i, 0] = posE_m[i] - hw * left_E
        inner[i, 1] = posN_m[i] - hw * left_N

    data = {
        "s_m": s_m,
        "length_m": raw_length,
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
    print(
        "D-shaped serpentine track: "
        f"length={raw_length:.1f}m, width={track_width:.1f}m, "
        f"R_tight={R_t:.1f}m, R_wide={R_w:.1f}m, "
        f"serp_amp={serp_amp_m:.2f}m, waves={serp_waves}"
    )
    return data


def create_serpentine_track(
    output_filename: str,
    total_length: float = 260.0,
    track_width: float = 6.0,
    num_points: int = 520,
) -> dict:
    """
    U-shaped serpentine track with continuous flowing curves.

    Overall U-shape with S-curves along both legs:
    - Left leg going down with alternating curves
    - Tight hairpin at the bottom
    - Right leg going up with alternating curves
    - Wide turn at top to close the loop
    """
    # Define turns: (radius, arc_angle_degrees, direction)
    # direction: 1 = left (CCW), -1 = right (CW)
    # Creates a U-shape: down the left, hairpin, up the right, close at top
    turns = [
        # Top-left, starting heading South
        (15.0, 60, -1),   # curve right (heading SW)
        (12.0, 70, 1),    # curve left (heading S)
        (10.0, 50, -1),   # curve right
        (14.0, 60, 1),    # curve left (heading SE)
        # Bottom hairpin
        (8.0, 160, 1),    # tight left hairpin (now heading NW)
        # Going up the right side
        (12.0, 50, -1),   # curve right
        (10.0, 60, 1),    # curve left
        (14.0, 70, -1),   # curve right
        (12.0, 50, 1),    # curve left (heading N)
        # Top connection back to start
        (20.0, 100, 1),   # wide left to close
    ]

    # Note: final scaling is done after spline smoothing

    # Generate track by tracing arcs
    # Start at origin, heading East (psi = 0)
    raw_points = []
    x, y = 0.0, 0.0
    psi = 0.0  # current heading

    n_per_turn = 200
    for radius, arc_deg, direction in turns:
        arc_rad = np.radians(arc_deg)
        n_pts = max(int(n_per_turn * arc_deg / 90), 20)

        # Center of the arc
        # For left turn (CCW): center is to the left of current heading
        # For right turn (CW): center is to the right
        cx = x - direction * radius * np.sin(psi)
        cy = y + direction * radius * np.cos(psi)

        # Starting angle on the circle (from center to current position)
        start_angle = np.arctan2(y - cy, x - cx)

        # Trace the arc
        for i in range(n_pts):
            t = i / n_pts
            angle = start_angle + direction * arc_rad * t
            px = cx + radius * np.cos(angle)
            py = cy + radius * np.sin(angle)
            raw_points.append([px, py])

        # Update position and heading after this turn
        end_angle = start_angle + direction * arc_rad
        x = cx + radius * np.cos(end_angle)
        y = cy + radius * np.sin(end_angle)
        psi = psi + direction * arc_rad

    raw_points = np.array(raw_points)

    # Smooth with spline for perfect continuity
    from scipy import interpolate
    pts_x = np.r_[raw_points[:, 0], raw_points[0, 0]]
    pts_y = np.r_[raw_points[:, 1], raw_points[0, 1]]

    tck, u = interpolate.splprep([pts_x, pts_y], s=0, per=True)
    spline_pts = 2000
    xi, yi = interpolate.splev(np.linspace(0, 1, spline_pts), tck)

    # Compute arc length
    dx = np.diff(np.r_[xi, xi[0]])
    dy = np.diff(np.r_[yi, yi[0]])
    ds_raw = np.sqrt(dx**2 + dy**2)
    s_raw = np.r_[0, np.cumsum(ds_raw[:-1])]
    raw_length = s_raw[-1] + ds_raw[-1]

    # Scale to target length
    scale = total_length / raw_length
    xi *= scale
    yi *= scale

    # Recompute arc length after scaling
    dx = np.diff(np.r_[xi, xi[0]])
    dy = np.diff(np.r_[yi, yi[0]])
    ds_raw = np.sqrt(dx**2 + dy**2)
    s_raw = np.r_[0, np.cumsum(ds_raw[:-1])]
    computed_length = s_raw[-1] + ds_raw[-1]

    # Center the track
    xi -= np.mean(xi)
    yi -= np.mean(yi)

    # Resample to uniform arc length
    s_m = np.linspace(0, computed_length, num_points, endpoint=False)
    s_extended = np.r_[s_raw, computed_length]
    xi_ext = np.r_[xi, xi[0]]
    yi_ext = np.r_[yi, yi[0]]

    posE_m = np.interp(s_m, s_extended, xi_ext)
    posN_m = np.interp(s_m, s_extended, yi_ext)
    posU_m = np.zeros(num_points)

    ds = s_m[1] - s_m[0]

    # Compute heading from path gradient
    dE = np.zeros(num_points)
    dN = np.zeros(num_points)
    for i in range(1, num_points - 1):
        dE[i] = posE_m[i + 1] - posE_m[i - 1]
        dN[i] = posN_m[i + 1] - posN_m[i - 1]
    dE[0] = posE_m[1] - posE_m[-1]
    dN[0] = posN_m[1] - posN_m[-1]
    dE[-1] = posE_m[0] - posE_m[-2]
    dN[-1] = posN_m[0] - posN_m[-2]

    psi_rad = np.arctan2(dN, dE)
    psi_rad = np.unwrap(psi_rad)

    psi_s = np.gradient(psi_rad, ds)
    psi_ss = np.gradient(psi_s, ds)

    # Compute min corner radius
    max_curv = np.max(np.abs(psi_s))
    min_radius = 1.0 / max_curv if max_curv > 1e-6 else float('inf')

    grade = np.zeros(num_points)
    bank = np.zeros(num_points)
    track_width_m = np.full(num_points, track_width)

    # Compute bounds directly from heading
    hw = 0.5 * track_width
    inner = np.zeros((num_points, 3), dtype=float)
    outer = np.zeros((num_points, 3), dtype=float)

    for i in range(num_points):
        left_E = -np.sin(psi_rad[i])
        left_N = np.cos(psi_rad[i])
        outer[i, 0] = posE_m[i] + hw * left_E
        outer[i, 1] = posN_m[i] + hw * left_N
        inner[i, 0] = posE_m[i] - hw * left_E
        inner[i, 1] = posN_m[i] - hw * left_N

    data = {
        "s_m": s_m,
        "length_m": computed_length,
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
    print(
        f"Serpentine track: length={computed_length:.1f}m, width={track_width:.1f}m, "
        f"min_corner_R={min_radius:.1f}m, 6 turns"
    )
    return data


def create_hairpin_track(
    output_filename: str,
    total_length: float = 260.0,
    track_width: float = 6.0,
    hairpin_radius: float = 10.0,
    closing_radius: float = 25.0,
    num_points: int = 520,
) -> dict:
    """
    Hairpin track: two parallel straights with tight hairpin and wider closing turn.

    Shape like a paperclip:
    - Straight going right
    - Tight 180° hairpin at the end
    - Straight returning left (parallel, below)
    - Wider 180° turn to close

    Parameters:
        hairpin_radius: Radius of the tight hairpin (default 10m)
        closing_radius: Radius of the wider closing turn (default 25m)
    """
    # For a 180° turn between parallel straights, separation must be 2 * radius.
    # If closing_radius != hairpin_radius, the track will not close smoothly.
    if abs(closing_radius - hairpin_radius) > 1e-6:
        print(
            "WARNING: create_hairpin_track requires closing_radius == hairpin_radius "
            "for a closed 180° paperclip. Forcing closing_radius to hairpin_radius."
        )
        closing_radius = hairpin_radius

    # Calculate straight length
    # Total = 2 * straight + hairpin_arc + closing_arc
    hairpin_arc = np.pi * hairpin_radius
    closing_arc = np.pi * closing_radius
    straight_length = (total_length - hairpin_arc - closing_arc) / 2.0

    if straight_length <= 0:
        raise ValueError("Track parameters result in non-positive straight length")

    # Build track analytically (no spline needed for simple geometry)
    s_m = np.linspace(0, total_length, num_points, endpoint=False)

    posE_m = np.zeros(num_points)
    posN_m = np.zeros(num_points)
    posU_m = np.zeros(num_points)
    psi_rad = np.zeros(num_points)
    curvature = np.zeros(num_points)

    # Section boundaries
    s1 = straight_length                      # End of first straight
    s2 = s1 + hairpin_arc                     # End of hairpin
    s3 = s2 + straight_length                 # End of second straight
    # s4 = s3 + closing_arc = total_length    # End of closing turn

    # Hairpin center (at right end)
    hc_x = straight_length
    hc_y = -hairpin_radius

    # Closing turn center (at left end)
    cc_x = 0.0
    cc_y = -closing_radius

    for i, s in enumerate(s_m):
        if s < s1:
            # First straight (going right, y=0)
            posE_m[i] = s
            posN_m[i] = 0.0
            psi_rad[i] = 0.0
            curvature[i] = 0.0
        elif s < s2:
            # Hairpin (180° right turn, curving down then left)
            arc_s = s - s1
            phi = np.pi / 2 + arc_s / hairpin_radius
            posE_m[i] = hc_x + hairpin_radius * np.cos(phi)
            posN_m[i] = hc_y + hairpin_radius * np.sin(phi)
            psi_rad[i] = phi - np.pi / 2
            curvature[i] = -1.0 / hairpin_radius
        elif s < s3:
            # Second straight (going left, y = -2*hairpin_radius)
            dist = s - s2
            posE_m[i] = straight_length - dist
            posN_m[i] = -2.0 * hairpin_radius
            psi_rad[i] = np.pi
            curvature[i] = 0.0
        else:
            # Closing turn (180° right turn, curving up then right)
            arc_s = s - s3
            phi = 3 * np.pi / 2 - arc_s / closing_radius
            posE_m[i] = cc_x + closing_radius * np.cos(phi)
            posN_m[i] = cc_y + closing_radius * np.sin(phi)
            psi_rad[i] = phi - np.pi / 2
            curvature[i] = -1.0 / closing_radius

    # Center the track
    posE_m -= np.mean(posE_m)
    posN_m -= np.mean(posN_m)

    # Unwrap heading
    psi_rad = np.unwrap(psi_rad)

    ds = s_m[1] - s_m[0]
    psi_s = np.gradient(psi_rad, ds)
    psi_ss = np.gradient(psi_s, ds)

    grade = np.zeros(num_points)
    bank = np.zeros(num_points)
    track_width_m = np.full(num_points, track_width)

    # Compute bounds: left of heading = outer (for CCW track)
    hw = 0.5 * track_width
    inner = np.zeros((num_points, 3), dtype=float)
    outer = np.zeros((num_points, 3), dtype=float)

    for i in range(num_points):
        left_E = -np.sin(psi_rad[i])
        left_N = np.cos(psi_rad[i])
        outer[i, 0] = posE_m[i] + hw * left_E
        outer[i, 1] = posN_m[i] + hw * left_N
        inner[i, 0] = posE_m[i] - hw * left_E
        inner[i, 1] = posN_m[i] - hw * left_N

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
    print(
        f"Hairpin track: length={total_length:.1f}m, width={track_width:.1f}m, "
        f"hairpin_R={hairpin_radius:.1f}m, closing_R={closing_radius:.1f}m"
    )
    return data


def create_technical_track(
    output_filename: str,
    seed: int = 42,
    target_length_m: float = 220.0,
    track_width: float = 6.0,
    num_points: int = 520,
) -> dict:
    """
    Procedural technical circuit with sharp corners (no self-intersection).

    Uses convex hull + outward-only midpoint displacement + spline smoothing.
    Outward displacement guarantees no self-intersection.

    Parameters:
        seed: Random seed for reproducibility
        target_length_m: Target track length in meters
        track_width: Track width in meters
    """
    from scipy import interpolate

    rng = np.random.RandomState(seed)

    # Generate random points for convex hull
    n_random = rng.randint(6, 10)
    raw_points = []
    for _ in range(n_random * 5):
        x = rng.uniform(15, 85)
        y = rng.uniform(15, 85)
        too_close = False
        for p in raw_points:
            if np.sqrt((p[0] - x)**2 + (p[1] - y)**2) < 15:
                too_close = True
                break
        if not too_close:
            raw_points.append([x, y])
        if len(raw_points) >= n_random:
            break

    raw_points = np.array(raw_points)

    # Get convex hull
    hull = ConvexHull(raw_points)
    hull_points = np.array([raw_points[i] for i in hull.vertices])

    # Compute center of hull
    center = np.mean(hull_points, axis=0)

    # Create track by adding midpoints with OUTWARD displacement only
    # This guarantees no self-intersection (track stays star-convex)
    track_points = []
    for i in range(len(hull_points)):
        next_i = (i + 1) % len(hull_points)
        p1 = hull_points[i]
        p2 = hull_points[next_i]

        # Add vertex (possibly with small random displacement outward)
        vertex_disp = rng.uniform(0, 8)
        outward = p1 - center
        outward_norm = outward / (np.linalg.norm(outward) + 1e-9)
        track_points.append(p1 + vertex_disp * outward_norm)

        # Add midpoint with outward displacement
        mid = (p1 + p2) / 2
        outward_mid = mid - center
        outward_mid_norm = outward_mid / (np.linalg.norm(outward_mid) + 1e-9)

        # Random displacement: mostly outward, small tangent component
        disp_mag = rng.uniform(5, 20)
        track_points.append(mid + disp_mag * outward_mid_norm)

    track_points = np.array(track_points)

    # Smooth with spline
    x = np.r_[track_points[:, 0], track_points[0, 0]]
    y = np.r_[track_points[:, 1], track_points[0, 1]]

    tck, u = interpolate.splprep([x, y], s=0, per=True)
    spline_pts = 2000
    xi, yi = interpolate.splev(np.linspace(0, 1, spline_pts), tck)

    # Compute raw arc length
    dx = np.diff(np.r_[xi, xi[0]])
    dy = np.diff(np.r_[yi, yi[0]])
    ds_raw = np.sqrt(dx**2 + dy**2)
    s_raw = np.r_[0, np.cumsum(ds_raw[:-1])]
    raw_length = s_raw[-1] + ds_raw[-1]

    # Scale to target length
    scale = target_length_m / raw_length
    xi *= scale
    yi *= scale

    # Center the track
    xi -= np.mean(xi)
    yi -= np.mean(yi)

    # Recompute arc length
    dx = np.diff(np.r_[xi, xi[0]])
    dy = np.diff(np.r_[yi, yi[0]])
    ds_raw = np.sqrt(dx**2 + dy**2)
    s_raw = np.r_[0, np.cumsum(ds_raw[:-1])]
    total_length = s_raw[-1] + ds_raw[-1]

    # Resample to uniform arc length
    s_m = np.linspace(0, total_length, num_points, endpoint=False)
    s_extended = np.r_[s_raw, total_length]
    xi_ext = np.r_[xi, xi[0]]
    yi_ext = np.r_[yi, yi[0]]

    posE_m = np.interp(s_m, s_extended, xi_ext)
    posN_m = np.interp(s_m, s_extended, yi_ext)
    posU_m = np.zeros(num_points)

    ds = s_m[1] - s_m[0]

    # Compute heading from path gradient
    dE = np.zeros(num_points)
    dN = np.zeros(num_points)
    for i in range(1, num_points - 1):
        dE[i] = posE_m[i + 1] - posE_m[i - 1]
        dN[i] = posN_m[i + 1] - posN_m[i - 1]
    dE[0] = posE_m[1] - posE_m[-1]
    dN[0] = posN_m[1] - posN_m[-1]
    dE[-1] = posE_m[0] - posE_m[-2]
    dN[-1] = posN_m[0] - posN_m[-2]

    psi_rad = np.arctan2(dN, dE)
    psi_rad = np.unwrap(psi_rad)

    psi_s = np.gradient(psi_rad, ds)
    psi_ss = np.gradient(psi_s, ds)

    # Compute min corner radius from curvature
    curvature = np.abs(psi_s)
    max_curv = np.max(curvature)
    min_radius = 1.0 / max_curv if max_curv > 1e-6 else float('inf')

    grade = np.zeros(num_points)
    bank = np.zeros(num_points)
    track_width_m = np.full(num_points, track_width)

    # Compute bounds directly from heading
    hw = 0.5 * track_width
    inner = np.zeros((num_points, 3), dtype=float)
    outer = np.zeros((num_points, 3), dtype=float)

    for i in range(num_points):
        left_E = -np.sin(psi_rad[i])
        left_N = np.cos(psi_rad[i])
        outer[i, 0] = posE_m[i] + hw * left_E
        outer[i, 1] = posN_m[i] + hw * left_N
        inner[i, 0] = posE_m[i] - hw * left_E
        inner[i, 1] = posN_m[i] - hw * left_N

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
    print(
        f"Technical track (seed={seed}): length={total_length:.1f}m, width={track_width:.1f}m, "
        f"min_corner_R={min_radius:.1f}m"
    )
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


def create_overview_plot(maps_dir: Path, include_obstacles: bool = False, output_name: str = "tracks_overview.png"):
    if include_obstacles:
        files = sorted(p for p in maps_dir.glob("*_Obstacles.mat"))
    else:
        files = sorted(p for p in maps_dir.glob("*.mat") if "Obstacles" not in p.name)
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
        ax.plot(inner[:, 0], inner[:, 1], lw=1.6, color=c, alpha=0.95, linestyle="-", label="Inner/Outer")
        ax.plot(outer[:, 0], outer[:, 1], lw=1.6, color=c, alpha=0.95, linestyle="-", label="_nolegend_")
        ax.plot(center_e, center_n, lw=0.9, color=c, alpha=0.45, linestyle=":", label="Centerline")
        # Plot obstacles if present.
        if "obstacles_ENR_m" in d:
            obs = np.asarray(d["obstacles_ENR_m"])
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            for j, (oe, on, r) in enumerate(obs):
                circ = plt.Circle((oe, on), r, fill=False, lw=1.2, color="black", alpha=0.8)
                ax.add_patch(circ)
                if j == 0:
                    circ.set_label("Obstacle")
        ax.set_title(p.stem)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    title = "Track Bounds Overview (Obstacles)" if include_obstacles else "Track Bounds Overview"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    out = maps_dir / output_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Track generation")
    parser.add_argument(
        "--preset",
        type=str,
        default="all",
        choices=[
            "medium_oval",
            "d_shaped",
            "d_shaped_serpentine",
            "serpentine",
            "u_shaped",
            "technical",
            "isaac_style",
            "overview",
            "all",
        ],
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
        out = maps / "Oval_Track_260m.mat"
        create_oval_track(str(out), track_width=6.0)
        print(f"Saved {out}")
        if args.with_obstacles:
            out_obs = maps / "Oval_Track_260m_Obstacles.mat"
            create_oval_track_with_obstacles(str(out_obs), track_width=6.0)
            print(f"Saved {out_obs}")

    if args.preset in ("d_shaped", "all"):
        out = maps / "TRACK1_280m.mat"
        create_d_shaped_track(
            str(out),
            total_length=280.0,
            track_width=6.0,
            tight_turn_radius=12.0,
            wide_turn_radius=24.0,
        )
        print(f"Saved {out}")

    if args.preset in ("d_shaped_serpentine", "all"):
        out = maps / "TRACK2_280m.mat"
        create_d_shaped_serpentine_track(
            str(out),
            total_length=280.0,
            track_width=6.0,
            tight_turn_radius=12.0,
            wide_turn_radius=24.0,
            serp_amp_m=1.5,
            serp_waves=3,
            num_points=520,
        )
        print(f"Saved {out}")

    if args.preset in ("serpentine", "all"):
        out = maps / "TRACK3_300m.mat"
        create_serpentine_track(
            str(out),
            total_length=300.0,
            track_width=6.0,
        )
        print(f"Saved {out}")

    if args.preset in ("technical", "all"):
        out = maps / "TRACK4_330m.mat"
        create_technical_track(
            str(out),
            seed=42,
            target_length_m=330.0,
            track_width=6.0,
        )
        print(f"Saved {out}")

    if args.preset in ("isaac_style", "all"):
        out = maps / "TRACK5_350m.mat"
        create_isaac_style_track(str(out), seed=23, target_length_m=350.0, track_width_m=10.0)
        print(f"Saved {out}")

    if args.preset in ("overview", "all"):
        out = create_overview_plot(maps)
        print(f"Saved {out}")
        # Build obstacle variants for all base maps and render an obstacles overview.
        base_maps = sorted(p for p in maps.glob("*.mat") if "Obstacles" not in p.name)
        for p in base_maps:
            out_obs = p.with_name(p.stem + "_Obstacles.mat")
            add_obstacles_to_map(str(p), str(out_obs))
        out_obs_png = create_overview_plot(maps, include_obstacles=True, output_name="tracks_overview_obstacles.png")
        print(f"Saved {out_obs_png}")


if __name__ == "__main__":
    main()
