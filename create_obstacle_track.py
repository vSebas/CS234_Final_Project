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
from world.world import World


def create_oval_obstacle_track(
    output_filename: str = "maps/Medium_Oval_Map_260m_Obstacles.mat",
    total_length: float = 260.0,
    track_width: float = 10.0,
    turn_radius: float = 18.0,
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
            [35.0, 1.2, 1.6, 0.8],   # bottom straight, left of centerline
            [92.0, -1.1, 1.5, 0.8],  # right turn entry, right of centerline
            [148.0, 0.8, 1.7, 0.8],  # top straight
            [214.0, -1.4, 1.6, 0.8], # left turn exit
        ],
        dtype=float,
    )

    s_obs = obstacle_spec[:, 0]
    e_obs = obstacle_spec[:, 1]
    r_obs = obstacle_spec[:, 2]
    margin_obs = obstacle_spec[:, 3]
    r_tilde = r_obs + margin_obs

    # Use the same map-matching path as optimizer/visualization for consistency.
    world = World(str(output_path), "Oval", diagnostic_plotting=False)
    e_obs_en, n_obs_en, _ = world.map_match_vectorized(s_obs, e_obs)

    track_data["obstacle_count"] = np.array([len(obstacle_spec)], dtype=np.int32)
    track_data["obstacles_s_m"] = s_obs
    track_data["obstacles_e_m"] = e_obs
    track_data["obstacles_radius_m"] = r_obs
    track_data["obstacles_margin_m"] = margin_obs
    track_data["obstacles_radius_tilde_m"] = r_tilde
    # Do not store duplicated EN obstacle metadata; keep Frenet fields as source-of-truth.
    # EN positions are derived via world.map_match_vectorized by optimizer/visualizer.

    sio.savemat(str(output_path), track_data)
    print(f"Saved obstacle track to: {output_path}")
    print("Obstacles (E, N, R_tilde):")
    for i in range(len(obstacle_spec)):
        print(f"  {i}: ({e_obs_en[i]:.2f}, {n_obs_en[i]:.2f}, {r_tilde[i]:.2f})")

    return track_data


if __name__ == "__main__":
    create_oval_obstacle_track()
