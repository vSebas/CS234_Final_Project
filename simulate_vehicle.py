#!/usr/bin/env python3
"""
Vehicle Dynamics Simulation and Visualization

Simulate the vehicle model with user-specified throttle and steering inputs,
then visualize the resulting motion as a video/gif.

Usage:
    python simulate_vehicle.py --scenario constant_turn
    python simulate_vehicle.py --scenario lane_change
    python simulate_vehicle.py --scenario acceleration
    python simulate_vehicle.py --throttle 2.0 --steering 5.0 --duration 10
    python simulate_vehicle.py --interactive

The script supports predefined maneuvers or custom constant inputs.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import casadi as ca

# Add project root to path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import load_vehicle_from_yaml


@dataclass
class SimulationResult:
    """Container for simulation results."""
    t: np.ndarray          # Time [s]
    ux: np.ndarray         # Longitudinal velocity [m/s]
    uy: np.ndarray         # Lateral velocity [m/s]
    r: np.ndarray          # Yaw rate [rad/s]
    east: np.ndarray       # East position [m]
    north: np.ndarray      # North position [m]
    psi: np.ndarray        # Heading [rad]
    delta: np.ndarray      # Steering angle [rad]
    fx: np.ndarray         # Throttle force [kN]
    speed: np.ndarray      # Total speed [m/s]
    beta: np.ndarray       # Sideslip angle [rad]


def create_dynamics_function(vehicle):
    """
    Create a CasADi function for numerical integration.

    Returns a function f(x, u) -> dx/dt
    State x: [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi]
    Control u: [delta, fx]
    """
    # Symbolic variables
    ux = ca.SX.sym('ux')
    uy = ca.SX.sym('uy')
    r = ca.SX.sym('r')
    dfz_long = ca.SX.sym('dfz_long')
    dfz_lat = ca.SX.sym('dfz_lat')
    t = ca.SX.sym('t')
    east = ca.SX.sym('east')
    north = ca.SX.sym('north')
    psi = ca.SX.sym('psi')

    delta = ca.SX.sym('delta')
    fx = ca.SX.sym('fx')

    # State and control vectors
    x = ca.vertcat(ux, uy, r, dfz_long, dfz_lat, t, east, north, psi)
    u = ca.vertcat(delta, fx)

    # Get dynamics (no road geometry - flat ground)
    dux, duy, dr, ddfz_long, ddfz_lat, dt, deast, dnorth, dpsi = \
        vehicle.temporal_global_dynamics(
            ux, uy, r, dfz_long, dfz_lat,
            t, east, north, psi,
            delta, fx,
            theta_rad=0.0, phi_rad=0.0
        )

    # State derivative vector
    dx = ca.vertcat(dux, duy, dr, ddfz_long, ddfz_lat, dt, deast, dnorth, dpsi)

    # Create CasADi function
    f = ca.Function('dynamics', [x, u], [dx], ['x', 'u'], ['dx'])

    return f


def rk4_step(f, x, u, dt):
    """
    Fourth-order Runge-Kutta integration step.
    """
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def simulate(
    vehicle,
    control_fn: Callable[[float], Tuple[float, float]],
    duration: float = 10.0,
    dt: float = 0.01,
    initial_speed: float = 10.0,
    initial_heading: float = 0.0
) -> SimulationResult:
    """
    Simulate the vehicle with time-varying control inputs.

    Args:
        vehicle: SingleTrackModel instance
        control_fn: Function (t) -> (delta_rad, fx_kn)
        duration: Simulation duration [s]
        dt: Time step [s]
        initial_speed: Initial forward speed [m/s]
        initial_heading: Initial heading [rad] (0 = North)

    Returns:
        SimulationResult with trajectory data
    """
    # Create dynamics function
    f = create_dynamics_function(vehicle)

    # Number of steps
    N = int(duration / dt)

    # Preallocate arrays
    t_arr = np.zeros(N + 1)
    ux_arr = np.zeros(N + 1)
    uy_arr = np.zeros(N + 1)
    r_arr = np.zeros(N + 1)
    east_arr = np.zeros(N + 1)
    north_arr = np.zeros(N + 1)
    psi_arr = np.zeros(N + 1)
    delta_arr = np.zeros(N + 1)
    fx_arr = np.zeros(N + 1)

    # Initial state: [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi]
    x = np.array([initial_speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initial_heading])

    # Store initial state
    t_arr[0] = 0.0
    ux_arr[0] = x[0]
    uy_arr[0] = x[1]
    r_arr[0] = x[2]
    east_arr[0] = x[6]
    north_arr[0] = x[7]
    psi_arr[0] = x[8]
    delta_arr[0], fx_arr[0] = control_fn(0.0)

    # Simulation loop
    for i in range(N):
        t = i * dt
        delta, fx = control_fn(t)
        u = np.array([delta, fx])

        # RK4 integration
        x = np.array(rk4_step(f, x, u, dt)).flatten()

        # Clamp speed to prevent numerical issues (allow near-zero but not negative)
        x[0] = max(x[0], 0.01)  # Minimum forward speed

        # Store results
        t_arr[i + 1] = t + dt
        ux_arr[i + 1] = x[0]
        uy_arr[i + 1] = x[1]
        r_arr[i + 1] = x[2]
        east_arr[i + 1] = x[6]
        north_arr[i + 1] = x[7]
        psi_arr[i + 1] = x[8]
        delta_arr[i + 1] = delta
        fx_arr[i + 1] = fx

    # Compute derived quantities
    speed = np.sqrt(ux_arr**2 + uy_arr**2)
    beta = np.arctan2(uy_arr, ux_arr)

    return SimulationResult(
        t=t_arr, ux=ux_arr, uy=uy_arr, r=r_arr,
        east=east_arr, north=north_arr, psi=psi_arr,
        delta=delta_arr, fx=fx_arr,
        speed=speed, beta=beta
    )


# =============================================================================
# Predefined Scenarios
# =============================================================================

def constant_control(delta_deg: float, fx_kn: float):
    """Create a constant control function."""
    delta_rad = np.deg2rad(delta_deg)
    def control_fn(t):
        return delta_rad, fx_kn
    return control_fn


def lane_change_control(amplitude_deg: float = 10.0, period: float = 2.0, fx_kn: float = 0.0):
    """
    Sinusoidal steering for lane change maneuver.

    A sine wave steering input creates a lane change motion.
    """
    amplitude_rad = np.deg2rad(amplitude_deg)
    def control_fn(t):
        delta = amplitude_rad * np.sin(2 * np.pi * t / period)
        return delta, fx_kn
    return control_fn


def step_steer_control(
    step_time: float = 2.0,
    delta_deg: float = 10.0,
    fx_kn: float = 0.0,
    ramp_time: float = 0.2
):
    """
    Step steering input (with optional ramp for smoothness).
    """
    delta_rad = np.deg2rad(delta_deg)
    def control_fn(t):
        if t < step_time:
            delta = 0.0
        elif t < step_time + ramp_time:
            # Linear ramp
            delta = delta_rad * (t - step_time) / ramp_time
        else:
            delta = delta_rad
        return delta, fx_kn
    return control_fn


def acceleration_control(fx_start: float = 0.0, fx_end: float = 5.0, ramp_time: float = 2.0):
    """
    Acceleration from rest (or cruise) to full throttle.
    """
    def control_fn(t):
        if t < ramp_time:
            fx = fx_start + (fx_end - fx_start) * t / ramp_time
        else:
            fx = fx_end
        return 0.0, fx
    return control_fn


def braking_control(fx_brake: float = -8.0, brake_time: float = 2.0):
    """
    Sudden braking maneuver.
    """
    def control_fn(t):
        if t < brake_time:
            return 0.0, 0.0
        else:
            return 0.0, fx_brake
    return control_fn


def slalom_control(
    period: float = 2.0,
    amplitude_deg: float = 12.0,
    fx_kn: float = 0.5
):
    """
    Slalom-like steering for weaving maneuver.

    Uses a fixed period for steering oscillation regardless of speed.
    """
    amplitude_rad = np.deg2rad(amplitude_deg)

    def control_fn(t):
        delta = amplitude_rad * np.sin(2 * np.pi * t / period)
        return delta, fx_kn
    return control_fn


# =============================================================================
# Visualization
# =============================================================================

def draw_vehicle(ax, east, north, psi, length=4.5, width=1.8, color='blue', alpha=1.0):
    """Draw a rectangle representing the vehicle."""
    # Vehicle corners in body frame
    corners = np.array([
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2],
        [-length/2, -width/2]
    ])

    # Rotation matrix (heading from North, CCW positive)
    # In ENU: x=East, y=North, heading from North
    # Body x (forward) points in direction psi from North
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[-s, -c], [c, -s]])  # ENU transformation

    # Transform to global frame
    corners_global = (R @ corners.T).T + np.array([east, north])

    # Create polygon
    poly = patches.Polygon(corners_global, closed=True, facecolor=color,
                          edgecolor='black', alpha=alpha, linewidth=1.5)
    ax.add_patch(poly)

    # Draw heading arrow
    arrow_len = length * 0.6
    arrow_east = east - arrow_len * np.sin(psi)
    arrow_north = north + arrow_len * np.cos(psi)
    ax.arrow(east, north, arrow_east - east, arrow_north - north,
             head_width=0.5, head_length=0.3, fc='red', ec='red')


def plot_trajectory(result: SimulationResult, output_path: Optional[str] = None):
    """Create a static plot of the trajectory."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Trajectory (bird's eye view)
    ax = axes[0, 0]
    ax.plot(result.east, result.north, 'b-', linewidth=2, label='Trajectory')
    ax.plot(result.east[0], result.north[0], 'go', markersize=10, label='Start')
    ax.plot(result.east[-1], result.north[-1], 'ro', markersize=10, label='End')
    # Draw vehicle at final position
    draw_vehicle(ax, result.east[-1], result.north[-1], result.psi[-1], alpha=0.7)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('Trajectory (Bird\'s Eye View)')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    # 2. Speed
    ax = axes[0, 1]
    ax.plot(result.t, result.speed * 3.6, 'b-', linewidth=2)  # Convert to km/h
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Speed [km/h]')
    ax.set_title('Vehicle Speed')
    ax.grid(True)

    # 3. Heading
    ax = axes[0, 2]
    ax.plot(result.t, np.rad2deg(result.psi), 'b-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Heading [deg]')
    ax.set_title('Heading (from North, CCW+)')
    ax.grid(True)

    # 4. Steering angle
    ax = axes[1, 0]
    ax.plot(result.t, np.rad2deg(result.delta), 'g-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Steering [deg]')
    ax.set_title('Steering Angle')
    ax.grid(True)

    # 5. Throttle/Brake force
    ax = axes[1, 1]
    ax.plot(result.t, result.fx, 'r-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [kN]')
    ax.set_title('Longitudinal Force (+ = throttle, - = brake)')
    ax.grid(True)

    # 6. Yaw rate and sideslip
    ax = axes[1, 2]
    ax.plot(result.t, np.rad2deg(result.r), 'b-', linewidth=2, label='Yaw rate [deg/s]')
    ax.plot(result.t, np.rad2deg(result.beta), 'r--', linewidth=2, label='Sideslip [deg]')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Yaw Rate and Sideslip')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved trajectory plot to: {output_path}")

    return fig


def create_animation(
    result: SimulationResult,
    output_path: str = "vehicle_animation.gif",
    fps: int = 30,
    skip_frames: int = 1,
    trail_length: int = 50
):
    """
    Create an animated GIF of the vehicle motion.

    Args:
        result: Simulation result
        output_path: Output file path (.gif)
        fps: Frames per second
        skip_frames: Skip every N frames (for faster rendering)
        trail_length: Number of past positions to show as trail
    """
    # Subsample data
    indices = np.arange(0, len(result.t), skip_frames)

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Compute plot bounds with margin
    margin = 20
    east_min, east_max = result.east.min() - margin, result.east.max() + margin
    north_min, north_max = result.north.min() - margin, result.north.max() + margin

    # Make aspect ratio equal
    east_range = east_max - east_min
    north_range = north_max - north_min
    max_range = max(east_range, north_range)
    east_center = (east_max + east_min) / 2
    north_center = (north_max + north_min) / 2

    def init():
        ax.set_xlim(east_center - max_range/2, east_center + max_range/2)
        ax.set_ylim(north_center - max_range/2, north_center + max_range/2)
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return []

    def animate(frame_idx):
        ax.clear()
        init()

        i = indices[frame_idx]
        t = result.t[i]

        # Draw trail
        trail_start = max(0, i - trail_length * skip_frames)
        ax.plot(result.east[trail_start:i+1], result.north[trail_start:i+1],
               'b-', linewidth=2, alpha=0.5)

        # Draw full trajectory (faded)
        ax.plot(result.east, result.north, 'b-', linewidth=1, alpha=0.2)

        # Draw vehicle
        draw_vehicle(ax, result.east[i], result.north[i], result.psi[i],
                    color='royalblue', alpha=0.9)

        # Draw start position marker
        ax.plot(result.east[0], result.north[0], 'go', markersize=10, label='Start')

        # Info text
        speed_kmh = result.speed[i] * 3.6
        steer_deg = np.rad2deg(result.delta[i])
        ax.set_title(f'Time: {t:.2f}s | Speed: {speed_kmh:.1f} km/h | '
                    f'Steering: {steer_deg:.1f}° | Throttle: {result.fx[i]:.2f} kN')

        return []

    # Create animation
    n_frames = len(indices)
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=1000/fps, blit=True)

    # Save as GIF
    print(f"Creating animation with {n_frames} frames...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"Saved animation to: {output_path}")

    plt.close(fig)
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulate vehicle dynamics with custom control inputs"
    )

    # Scenario selection
    parser.add_argument('--scenario', type=str, default=None,
                       choices=['constant_turn', 'lane_change', 'step_steer',
                               'acceleration', 'braking', 'slalom'],
                       help='Predefined driving scenario')

    # Custom constant control
    parser.add_argument('--throttle', type=float, default=0.0,
                       help='Constant throttle force [kN] (use with --steering)')
    parser.add_argument('--steering', type=float, default=0.0,
                       help='Constant steering angle [deg] (use with --throttle)')

    # Simulation parameters
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Simulation duration [s]')
    parser.add_argument('--initial-speed', type=float, default=0.0,
                       help='Initial forward speed [m/s]')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Simulation time step [s]')

    # Output options
    parser.add_argument('--output-dir', type=str, default='results/dynamic_simulations',
                       help='Output directory')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation generation')
    parser.add_argument('--fps', type=int, default=30,
                       help='Animation FPS')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load vehicle
    print("Loading vehicle model...")
    config_file = project_root / "models" / "config" / "vehicle_params_gti.yaml"
    vehicle = load_vehicle_from_yaml(config_file)
    print(f"  Vehicle: {vehicle.params.name}")

    # Select control function based on scenario
    if args.scenario == 'constant_turn':
        print("\nScenario: Constant Turn (steady-state cornering)")
        control_fn = constant_control(delta_deg=10.0, fx_kn=0.5)
        args.initial_speed = 15.0  # Need speed for cornering

    elif args.scenario == 'lane_change':
        print("\nScenario: Lane Change (sinusoidal steering)")
        control_fn = lane_change_control(amplitude_deg=12.0, period=3.0, fx_kn=0.5)
        args.initial_speed = 15.0  # Need speed for lane change

    elif args.scenario == 'step_steer':
        print("\nScenario: Step Steer (sudden steering input)")
        control_fn = step_steer_control(step_time=2.0, delta_deg=15.0, fx_kn=0.5)
        args.initial_speed = 15.0  # Need speed for steering response

    elif args.scenario == 'acceleration':
        print("\nScenario: Acceleration (throttle ramp)")
        control_fn = acceleration_control(fx_start=0.0, fx_end=6.0, ramp_time=2.0)
        args.initial_speed = 0.1  # Start slower

    elif args.scenario == 'braking':
        print("\nScenario: Braking (hard deceleration)")
        control_fn = braking_control(fx_brake=-10.0, brake_time=2.0)
        args.initial_speed = 25.0  # Start faster

    elif args.scenario == 'slalom':
        print("\nScenario: Slalom (weaving maneuver)")
        control_fn = slalom_control(period=2.0, amplitude_deg=12.0, fx_kn=0.5)
        args.initial_speed = 10.0  # Need initial speed for weaving

    else:
        # Custom constant control
        print(f"\nCustom Control: steering={args.steering}°, throttle={args.throttle} kN")
        control_fn = constant_control(delta_deg=args.steering, fx_kn=args.throttle)

    # Run simulation
    print(f"\nSimulating for {args.duration}s at dt={args.dt}s...")
    print(f"  Initial speed: {args.initial_speed} m/s ({args.initial_speed * 3.6:.1f} km/h)")

    result = simulate(
        vehicle,
        control_fn=control_fn,
        duration=args.duration,
        dt=args.dt,
        initial_speed=args.initial_speed,
        initial_heading=0.0  # Start heading North
    )

    # Print summary
    print(f"\nSimulation Results:")
    print(f"  Final position: ({result.east[-1]:.1f}, {result.north[-1]:.1f}) m")
    print(f"  Final heading: {np.rad2deg(result.psi[-1]):.1f}°")
    print(f"  Final speed: {result.speed[-1] * 3.6:.1f} km/h")
    print(f"  Max speed: {result.speed.max() * 3.6:.1f} km/h")
    print(f"  Max yaw rate: {np.rad2deg(np.abs(result.r).max()):.1f} deg/s")
    print(f"  Max sideslip: {np.rad2deg(np.abs(result.beta).max()):.1f}°")

    # Generate static plot
    scenario_name = args.scenario or "custom"
    plot_path = output_dir / f"sim_{scenario_name}_trajectory.png"
    plot_trajectory(result, str(plot_path))

    # Generate animation
    if not args.no_animation:
        anim_path = output_dir / f"sim_{scenario_name}_animation.gif"
        # Adjust skip_frames based on duration and dt
        total_frames = int(args.duration / args.dt)
        target_frames = 200  # Target ~200 frames for reasonable file size
        skip = max(1, total_frames // target_frames)
        create_animation(result, str(anim_path), fps=args.fps, skip_frames=skip)

    print(f"\nDone! Outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
