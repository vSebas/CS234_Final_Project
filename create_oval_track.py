"""
Create a synthetic oval track .mat file for multimodel-trajectory-optimization.

The track is a 260m oval with:
- Two straight sections
- Two semicircular turns
- Flat surface (no grade/bank)
- 10m track width
"""

import numpy as np
import scipy.io as sio


def create_oval_track(
    total_length=260.0,
    track_width=10.0,
    turn_radius=18.0,
    num_points=520,
    output_filename="Medium_Oval_Map_260m.mat"
):
    """
    Create an oval track and save as .mat file.

    The oval consists of:
    - Two straights of length L_straight
    - Two semicircles of radius turn_radius

    Total length = 2 * L_straight + 2 * pi * turn_radius
    """

    # Calculate straight length
    turn_length = np.pi * turn_radius  # Length of one semicircle
    straight_length = (total_length - 2 * turn_length) / 2

    if straight_length < 0:
        raise ValueError(f"Turn radius too large for total length. Max radius: {total_length / (2 * np.pi):.1f}m")

    print(f"Track parameters:")
    print(f"  Total length: {total_length} m")
    print(f"  Straight sections: {straight_length:.1f} m each")
    print(f"  Turn radius: {turn_radius} m")
    print(f"  Turn length: {turn_length:.1f} m each")
    print(f"  Track width: {track_width} m")

    # Generate s coordinates (arc length)
    s_m = np.linspace(0, total_length, num_points, endpoint=False)
    ds = s_m[1] - s_m[0]

    # Initialize arrays
    posE_m = np.zeros(num_points)  # East (x)
    posN_m = np.zeros(num_points)  # North (y)
    posU_m = np.zeros(num_points)  # Up (z) - flat track
    psi_rad = np.zeros(num_points)  # Heading
    curvature = np.zeros(num_points)  # Curvature (1/radius)

    # Define track segments
    # Segment 1: Bottom straight (heading = 0, going East)
    # Segment 2: Right turn (semicircle, turning left/counterclockwise)
    # Segment 3: Top straight (heading = pi, going West)
    # Segment 4: Left turn (semicircle, turning left/counterclockwise)

    s1_end = straight_length
    s2_end = straight_length + turn_length
    s3_end = 2 * straight_length + turn_length
    s4_end = total_length

    # Center positions for the turns
    turn1_center = np.array([straight_length, turn_radius])
    turn2_center = np.array([0, turn_radius])

    for i, s in enumerate(s_m):
        if s < s1_end:
            # Bottom straight: going in +x direction
            posE_m[i] = s
            posN_m[i] = 0
            psi_rad[i] = 0
            curvature[i] = 0

        elif s < s2_end:
            # Right semicircle turn
            angle = (s - s1_end) / turn_radius  # Angle traveled (0 to pi)
            posE_m[i] = turn1_center[0] + turn_radius * np.sin(angle)
            posN_m[i] = turn1_center[1] - turn_radius * np.cos(angle)
            psi_rad[i] = angle  # Heading increases from 0 to pi
            curvature[i] = 1.0 / turn_radius

        elif s < s3_end:
            # Top straight: going in -x direction
            dist_along = s - s2_end
            posE_m[i] = straight_length - dist_along
            posN_m[i] = 2 * turn_radius
            psi_rad[i] = np.pi
            curvature[i] = 0

        else:
            # Left semicircle turn
            angle = (s - s3_end) / turn_radius  # Angle traveled (0 to pi)
            posE_m[i] = turn2_center[0] - turn_radius * np.sin(angle)
            posN_m[i] = turn2_center[1] + turn_radius * np.cos(angle)
            psi_rad[i] = np.pi + angle  # Heading from pi to 2*pi
            curvature[i] = 1.0 / turn_radius

    # Wrap heading to [-pi, pi] or keep continuous
    # For this application, keep it continuous and let the code handle wrapping

    # Compute derivatives of heading (curvature and its derivative)
    psi_s_radpm = curvature.copy()  # d(psi)/ds = curvature
    psi_ss_radpm2 = np.gradient(psi_s_radpm, ds)  # d^2(psi)/ds^2

    # Road grade and bank (flat track)
    grade_rad = np.zeros(num_points)
    grade_s_radpm = np.zeros(num_points)
    grade_ss_radpm2 = np.zeros(num_points)

    bank_rad = np.zeros(num_points)
    bank_s_radpm = np.zeros(num_points)
    bank_ss_radpm2 = np.zeros(num_points)

    # Track width (constant)
    track_width_m = track_width * np.ones(num_points)

    # Compute track boundaries
    # Normal vector to the path (perpendicular to heading, pointing left)
    normal_E = -np.sin(psi_rad)
    normal_N = np.cos(psi_rad)

    half_width = track_width / 2

    # Inner boundary (right side of track, looking in direction of travel)
    inner_bounds_m = np.zeros((num_points, 3))
    inner_bounds_m[:, 0] = posE_m - half_width * normal_E
    inner_bounds_m[:, 1] = posN_m - half_width * normal_N
    inner_bounds_m[:, 2] = posU_m

    # Outer boundary (left side)
    outer_bounds_m = np.zeros((num_points, 3))
    outer_bounds_m[:, 0] = posE_m + half_width * normal_E
    outer_bounds_m[:, 1] = posN_m + half_width * normal_N
    outer_bounds_m[:, 2] = posU_m

    # GPS reference (arbitrary, set to origin)
    gpsXYZRef_m = np.array([0.0, 0.0, 0.0])

    # Create the data dictionary
    track_data = {
        's_m': s_m,
        'length_m': total_length,
        'gpsXYZRef_m': gpsXYZRef_m,
        'posE_m': posE_m,
        'posN_m': posN_m,
        'posU_m': posU_m,
        'psi_rad': psi_rad,
        'psi_s_radpm': psi_s_radpm,
        'psi_ss_radpm2': psi_ss_radpm2,
        'grade_rad': grade_rad,
        'grade_s_radpm': grade_s_radpm,
        'grade_ss_radpm2': grade_ss_radpm2,
        'bank_rad': bank_rad,
        'bank_s_radpm': bank_s_radpm,
        'bank_ss_radpm2': bank_ss_radpm2,
        'track_width_m': track_width_m,
        'inner_bounds_m': inner_bounds_m,
        'outer_bounds_m': outer_bounds_m,
    }

    # Save to .mat file
    sio.savemat(output_filename, track_data)
    print(f"\nSaved track to: {output_filename}")

    return track_data


def plot_track(track_data):
    """Visualize the track."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot boundaries
    inner = track_data['inner_bounds_m']
    outer = track_data['outer_bounds_m']

    ax.plot(inner[:, 0], inner[:, 1], 'b-', linewidth=2, label='Inner boundary')
    ax.plot(outer[:, 0], outer[:, 1], 'r-', linewidth=2, label='Outer boundary')

    # Plot centerline
    ax.plot(track_data['posE_m'], track_data['posN_m'], 'k--', linewidth=1, label='Centerline')

    # Mark start/finish
    ax.plot(track_data['posE_m'][0], track_data['posN_m'][0], 'go', markersize=15, label='Start/Finish')

    # Add direction arrows
    n_arrows = 8
    indices = np.linspace(0, len(track_data['s_m'])-1, n_arrows, dtype=int)
    for idx in indices:
        x = track_data['posE_m'][idx]
        y = track_data['posN_m'][idx]
        psi = track_data['psi_rad'][idx]
        dx = 3 * np.cos(psi)
        dy = 3 * np.sin(psi)
        ax.arrow(x, y, dx, dy, head_width=1.5, head_length=1, fc='green', ec='green')

    ax.set_xlabel('East [m]', fontsize=12)
    ax.set_ylabel('North [m]', fontsize=12)
    ax.set_title(f"Oval Track - {track_data['length_m']:.0f}m", fontsize=14)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('oval_track_preview.png', dpi=150)
    print("Saved preview to: oval_track_preview.png")
    plt.show()


if __name__ == "__main__":
    # Create the track
    track_data = create_oval_track(
        total_length=260.0,
        track_width=10.0,
        turn_radius=18.0,
        num_points=520,
        output_filename="maps/Medium_Oval_Map_260m.mat"
    )

    # Visualize and save preview
    plot_track(track_data)
