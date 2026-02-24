#!/usr/bin/env python3
"""
Create an oval track .mat with static obstacle metadata.

Outputs:
- maps/Medium_Oval_Map_260m_Obstacles.mat
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

from create_oval_track import create_oval_track


def _interp_periodic(values_s: np.ndarray, values: np.ndarray, query_s: np.ndarray, length_m: float) -> np.ndarray:
    """Periodic 1D interpolation on track arc-length s."""
    s_mod = np.mod(query_s, length_m)
    s_ext = np.concatenate([values_s, [values_s[0] + length_m]])
    v_ext = np.concatenate([values, [values[0]]])
    return np.interp(s_mod, s_ext, v_ext)


def _map_match_flat_track(track_data: dict, s_vals: np.ndarray, e_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (s, e) to EN for flat tracks using centerline + left-normal offset.
    """
    s_grid = track_data["s_m"]
    length_m = float(track_data["length_m"])

    x_cl = _interp_periodic(s_grid, track_data["posE_m"], s_vals, length_m)
    y_cl = _interp_periodic(s_grid, track_data["posN_m"], s_vals, length_m)
    psi = _interp_periodic(s_grid, np.unwrap(track_data["psi_rad"]), s_vals, length_m)

    # Left normal in EN coordinates.
    x_obs = x_cl - e_vals * np.sin(psi)
    y_obs = y_cl + e_vals * np.cos(psi)
    return x_obs, y_obs


def create_oval_obstacle_track(
    output_filename: str = "maps/Medium_Oval_Map_260m_Obstacles.mat",
    total_length: float = 260.0,
    track_width: float = 10.0,
    turn_radius: float = 30.0,
    num_points: int = 520,
) -> dict:
    """
    Create a 260m oval track with obstacle metadata.
    """
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    track_data = create_oval_track(
        total_length=total_length,
        track_width=track_width,
        turn_radius=turn_radius,
        num_points=num_points,
        output_filename=str(output_path),
    )

    # Static obstacle scenario in Frenet coordinates: (s [m], e [m], radius [m], margin [m]).
    obstacle_spec = np.array(
        [
            [35.0, 1.2, 1.0, 0.7],   # bottom straight, left of centerline
            [92.0, -1.1, 0.9, 0.7],  # right turn entry, right of centerline
            [148.0, 0.8, 1.1, 0.7],  # top straight
            [214.0, -1.4, 1.0, 0.7], # left turn exit
        ],
        dtype=float,
    )

    s_obs = obstacle_spec[:, 0]
    e_obs = obstacle_spec[:, 1]
    r_obs = obstacle_spec[:, 2]
    margin_obs = obstacle_spec[:, 3]
    r_tilde = r_obs + margin_obs

    e_obs_en, n_obs_en = _map_match_flat_track(track_data, s_obs, e_obs)

    track_data["obstacle_count"] = np.array([len(obstacle_spec)], dtype=np.int32)
    track_data["obstacles_s_m"] = s_obs
    track_data["obstacles_e_m"] = e_obs
    track_data["obstacles_radius_m"] = r_obs
    track_data["obstacles_margin_m"] = margin_obs
    track_data["obstacles_radius_tilde_m"] = r_tilde
    track_data["obstacles_ENR_m"] = np.column_stack([e_obs_en, n_obs_en, r_obs])
    track_data["obstacles_ENR_tilde_m"] = np.column_stack([e_obs_en, n_obs_en, r_tilde])

    sio.savemat(str(output_path), track_data)
    print(f"Saved obstacle track to: {output_path}")
    print("Obstacles (E, N, R_tilde):")
    for i in range(len(obstacle_spec)):
        print(f"  {i}: ({e_obs_en[i]:.2f}, {n_obs_en[i]:.2f}, {r_tilde[i]:.2f})")

    return track_data


if __name__ == "__main__":
    create_oval_obstacle_track()
