"""
Visualization utilities for trajectory optimization results.

All functions save outputs to files instead of displaying them.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import os


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize_trajectory: tuple = (8, 8)
    figsize_states: tuple = (12, 8)
    figsize_controls: tuple = (10, 5)
    figsize_convergence: tuple = (10, 6)

    dpi: int = 150
    line_width: float = 1.5
    marker_size: float = 3

    cmap_name: str = 'viridis'

    # Colors
    track_color: str = 'gray'
    centerline_color: str = 'black'
    trajectory_color: str = 'tab:blue'
    reference_color: str = 'tab:orange'


class TrajectoryVisualizer:
    """
    Visualization class for trajectory optimization results.

    All plots are saved to files, not displayed.
    """

    def __init__(
        self,
        world,
        output_dir: str = "results",
        config: Optional[PlotConfig] = None
    ):
        """
        Initialize visualizer.

        Args:
            world: World/track instance
            output_dir: Directory to save outputs
            config: Plot configuration
        """
        self.world = world
        self.output_dir = Path(output_dir)
        self.config = config or PlotConfig()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _convert_to_global(
        self,
        s_m: np.ndarray,
        e_m: np.ndarray
    ) -> tuple:
        """Convert path coordinates to global (East, North)."""
        return self.world.map_match_vectorized(s_m, e_m)

    def _get_obstacle_plot_data(self) -> List[tuple]:
        """
        Extract obstacle data as tuples (east, north, radius, radius_tilde).

        Supports either ENR fields or Frenet fields stored in world.data.
        """
        data = self.world.data
        obstacles: List[tuple] = []

        if "obstacles_ENR_m" in data:
            enr = np.atleast_2d(np.asarray(data["obstacles_ENR_m"], dtype=float))
            if enr.shape[1] == 3:
                if "obstacles_ENR_tilde_m" in data:
                    enr_tilde = np.atleast_2d(np.asarray(data["obstacles_ENR_tilde_m"], dtype=float))
                    for i in range(min(len(enr), len(enr_tilde))):
                        obstacles.append((float(enr[i, 0]), float(enr[i, 1]), float(enr[i, 2]), float(enr_tilde[i, 2])))
                    return obstacles
                if "obstacles_margin_m" in data:
                    margins = np.atleast_1d(np.asarray(data["obstacles_margin_m"], dtype=float))
                    for i in range(min(len(enr), len(margins))):
                        r = float(enr[i, 2])
                        obstacles.append((float(enr[i, 0]), float(enr[i, 1]), r, r + float(margins[i])))
                    return obstacles
                for i in range(len(enr)):
                    r = float(enr[i, 2])
                    obstacles.append((float(enr[i, 0]), float(enr[i, 1]), r, r))
                return obstacles

        required_frenet = {"obstacles_s_m", "obstacles_e_m", "obstacles_radius_m"}
        if required_frenet.issubset(set(data.keys())):
            s_vals = np.atleast_1d(np.asarray(data["obstacles_s_m"], dtype=float))
            e_vals = np.atleast_1d(np.asarray(data["obstacles_e_m"], dtype=float))
            r_vals = np.atleast_1d(np.asarray(data["obstacles_radius_m"], dtype=float))
            if "obstacles_radius_tilde_m" in data:
                rt_vals = np.atleast_1d(np.asarray(data["obstacles_radius_tilde_m"], dtype=float))
            elif "obstacles_margin_m" in data:
                margins = np.atleast_1d(np.asarray(data["obstacles_margin_m"], dtype=float))
                rt_vals = r_vals + margins
            else:
                rt_vals = r_vals

            east, north, _ = self.world.map_match_vectorized(s_vals, e_vals)
            for i in range(min(len(east), len(r_vals), len(rt_vals))):
                obstacles.append((float(east[i]), float(north[i]), float(r_vals[i]), float(rt_vals[i])))

        return obstacles

    def plot_trajectory_overhead(
        self,
        result,
        filename: str = "trajectory_overhead.png",
        title: Optional[str] = None,
        show_colorbar: bool = True,
        color_by: str = 'velocity'  # 'velocity', 'time', 'index'
    ) -> str:
        """
        Plot top-down view of trajectory on track.

        Args:
            result: Optimization result with X, U, s_m attributes
            filename: Output filename
            title: Plot title
            show_colorbar: Whether to show colorbar
            color_by: Variable to color trajectory by

        Returns:
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_trajectory)

        # Get track boundaries
        s_track = self.world.data['s_m']
        inner = self.world.data['inner_bounds_m']
        outer = self.world.data['outer_bounds_m']

        # Plot track boundaries
        ax.plot(inner[:, 0], inner[:, 1],
                color=self.config.track_color, linewidth=1.5, label='Track bounds')
        ax.plot(outer[:, 0], outer[:, 1],
                color=self.config.track_color, linewidth=1.5)

        # Plot centerline
        ax.plot(self.world.data['posE_m'], self.world.data['posN_m'],
                color=self.config.centerline_color, linewidth=0.5,
                linestyle='--', alpha=0.5, label='Centerline')

        # Plot obstacles if available
        obstacles = self._get_obstacle_plot_data()
        for i, (east_m, north_m, radius_m, radius_tilde_m) in enumerate(obstacles):
            show_label_obs = (i == 0)
            obs_circle = Circle(
                (east_m, north_m),
                radius_m,
                edgecolor='tab:red',
                facecolor='tab:red',
                alpha=0.25,
                linewidth=1.5,
                label='Obstacle' if show_label_obs else None,
                zorder=3,
            )
            ax.add_patch(obs_circle)
            safe_circle = Circle(
                (east_m, north_m),
                radius_tilde_m,
                edgecolor='tab:red',
                facecolor='none',
                linestyle='--',
                linewidth=1.0,
                alpha=0.9,
                label='Obstacle safety' if show_label_obs else None,
                zorder=3,
            )
            ax.add_patch(safe_circle)

        # Convert trajectory to global coordinates
        s_m = result.s_m
        e_m = result.X[6, :]  # e is at index 6

        x_traj, y_traj, _ = self._convert_to_global(s_m, e_m)

        # Color by specified variable
        if color_by == 'velocity':
            color_values = result.X[0, :]  # ux
            cbar_label = 'Velocity [m/s]'
        elif color_by == 'time':
            color_values = result.X[5, :]  # t
            cbar_label = 'Time [s]'
        else:  # index
            color_values = np.arange(len(s_m))
            cbar_label = 'Node index'

        # Plot trajectory with color
        points = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        norm = plt.Normalize(color_values.min(), color_values.max())
        lc = LineCollection(segments, cmap=self.config.cmap_name, norm=norm)
        lc.set_array(color_values[:-1])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        if show_colorbar:
            cbar = fig.colorbar(line, ax=ax, shrink=0.8)
            cbar.set_label(cbar_label)

        # Mark start/finish
        ax.scatter(x_traj[0], y_traj[0], s=100, c='green', marker='o',
                   zorder=10, label='Start')
        ax.scatter(x_traj[-1], y_traj[-1], s=100, c='red', marker='x',
                   zorder=10, label='Finish')

        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')

        if title:
            ax.set_title(title)
        else:
            cost_str = f", Lap time: {result.cost:.2f}s" if hasattr(result, 'cost') else ""
            ax.set_title(f"Trajectory{cost_str}")

        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return str(filepath)

    def plot_states(
        self,
        result,
        filename: str = "states.png",
        title: Optional[str] = None
    ) -> str:
        """
        Plot state trajectories vs arc length.

        Args:
            result: Optimization result
            filename: Output filename
            title: Plot title

        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(4, 2, figsize=self.config.figsize_states)

        s_m = result.s_m
        X = result.X

        # State labels
        state_labels = [
            (r'$u_x$ [m/s]', 0),
            (r'$u_y$ [m/s]', 1),
            (r'$r$ [rad/s]', 2),
            (r'$\Delta F_{z,\mathrm{long}}$ [kN]', 3),
            (r'$\Delta F_{z,\mathrm{lat}}$ [kN]', 4),
            (r'$t$ [s]', 5),
            (r'$e$ [m]', 6),
            (r'$\Delta\psi$ [rad]', 7),
        ]

        for i, (label, idx) in enumerate(state_labels):
            row = i // 2
            col = i % 2
            ax = axes[row, col]

            ax.plot(s_m, X[idx, :], color=self.config.trajectory_color,
                    linewidth=self.config.line_width)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

            if row == 3:
                ax.set_xlabel('Arc length $s$ [m]')

        if title:
            fig.suptitle(title)

        fig.tight_layout()

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return str(filepath)

    def plot_controls(
        self,
        result,
        filename: str = "controls.png",
        title: Optional[str] = None
    ) -> str:
        """
        Plot control inputs vs arc length.

        Args:
            result: Optimization result
            filename: Output filename
            title: Plot title

        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(2, 1, figsize=self.config.figsize_controls,
                                  sharex=True)

        s_m = result.s_m
        U = result.U

        # Steering angle
        axes[0].plot(s_m, np.degrees(U[0, :]),
                     color=self.config.trajectory_color,
                     linewidth=self.config.line_width)
        axes[0].set_ylabel(r'Steering $\delta$ [deg]')
        axes[0].grid(True, alpha=0.3)

        # Longitudinal force
        axes[1].plot(s_m, U[1, :],
                     color=self.config.trajectory_color,
                     linewidth=self.config.line_width)
        axes[1].set_ylabel(r'Force $F_x$ [kN]')
        axes[1].set_xlabel('Arc length $s$ [m]')
        axes[1].grid(True, alpha=0.3)

        if title:
            fig.suptitle(title)

        fig.tight_layout()

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return str(filepath)

    def plot_scp_convergence(
        self,
        result,
        filename: str = "scp_convergence.png",
        title: Optional[str] = None
    ) -> str:
        """
        Plot SCP convergence history.

        Args:
            result: SCPResult with iteration history
            filename: Output filename
            title: Plot title

        Returns:
            Path to saved file
        """
        if not hasattr(result, 'iteration_history') or not result.iteration_history:
            print("Warning: No iteration history in result")
            return ""

        fig, axes = plt.subplots(3, 1, figsize=self.config.figsize_convergence,
                                  sharex=True)

        iterations = np.arange(len(result.iteration_history))

        # Cost history
        axes[0].plot(iterations, result.iteration_history, 'o-',
                     color=self.config.trajectory_color,
                     linewidth=self.config.line_width,
                     markersize=self.config.marker_size)
        axes[0].set_ylabel('Cost (lap time) [s]')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Cost Convergence')

        # Trust region history
        if result.tr_radius_history:
            axes[1].plot(iterations, result.tr_radius_history, 's-',
                         color='tab:orange',
                         linewidth=self.config.line_width,
                         markersize=self.config.marker_size)
            axes[1].set_ylabel('Trust Region Radius')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Trust Region Evolution')

        # Constraint violation
        if result.constraint_violation_history:
            axes[2].plot(iterations, result.constraint_violation_history, '^-',
                         color='tab:red',
                         linewidth=self.config.line_width,
                         markersize=self.config.marker_size)
            axes[2].set_ylabel('Max Constraint Violation')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title('Constraint Satisfaction')

        axes[-1].set_xlabel('SCP Iteration')

        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'SCP Convergence ({result.iterations} iterations, '
                         f'{result.solve_time:.2f}s)')

        fig.tight_layout()

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return str(filepath)

    def plot_comparison(
        self,
        results: Dict[str, object],
        filename: str = "comparison.png",
        title: Optional[str] = None
    ) -> str:
        """
        Compare multiple optimization results.

        Args:
            results: Dict mapping name -> result object
            filename: Output filename
            title: Plot title

        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # Top-down trajectory comparison
        ax = axes[0, 0]

        # Track
        inner = self.world.data['inner_bounds_m']
        outer = self.world.data['outer_bounds_m']
        ax.plot(inner[:, 0], inner[:, 1], color=self.config.track_color, linewidth=1)
        ax.plot(outer[:, 0], outer[:, 1], color=self.config.track_color, linewidth=1)

        for i, (name, result) in enumerate(results.items()):
            s_m = result.s_m
            e_m = result.X[6, :]
            x_traj, y_traj, _ = self._convert_to_global(s_m, e_m)
            ax.plot(x_traj, y_traj, color=colors[i], linewidth=1.5, label=name)

        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Trajectories')

        # Velocity comparison
        ax = axes[0, 1]
        for i, (name, result) in enumerate(results.items()):
            ax.plot(result.s_m, result.X[0, :], color=colors[i],
                    linewidth=1.5, label=name)
        ax.set_xlabel('Arc length [m]')
        ax.set_ylabel('Velocity [m/s]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Velocity Profiles')

        # Steering comparison
        ax = axes[1, 0]
        for i, (name, result) in enumerate(results.items()):
            ax.plot(result.s_m, np.degrees(result.U[0, :]), color=colors[i],
                    linewidth=1.5, label=name)
        ax.set_xlabel('Arc length [m]')
        ax.set_ylabel('Steering [deg]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Steering Angle')

        # Performance bar chart
        ax = axes[1, 1]
        names = list(results.keys())
        costs = [r.cost for r in results.values()]
        iterations = [getattr(r, 'iterations', 0) for r in results.values()]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, costs, width, label='Cost [s]', color='tab:blue')
        ax.set_ylabel('Cost (lap time) [s]', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, iterations, width, label='Iterations', color='tab:orange')
        ax2.set_ylabel('Iterations', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title('Performance Comparison')

        if title:
            fig.suptitle(title)

        fig.tight_layout()

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return str(filepath)

    def plot_warm_start_analysis(
        self,
        cold_result,
        warm_result,
        filename: str = "warm_start_analysis.png",
        title: Optional[str] = None
    ) -> str:
        """
        Analyze warm-start effectiveness.

        Args:
            cold_result: Result from cold start (no warm-start)
            warm_result: Result from warm-started solve
            filename: Output filename
            title: Plot title

        Returns:
            Path to saved file
        """
        fig = plt.figure(figsize=(14, 10))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Trajectory comparison
        ax1 = fig.add_subplot(gs[0, :2])

        inner = self.world.data['inner_bounds_m']
        outer = self.world.data['outer_bounds_m']
        ax1.plot(inner[:, 0], inner[:, 1], color=self.config.track_color, linewidth=1)
        ax1.plot(outer[:, 0], outer[:, 1], color=self.config.track_color, linewidth=1)

        # Cold start
        s_m = cold_result.s_m
        e_m = cold_result.X[6, :]
        x_cold, y_cold, _ = self._convert_to_global(s_m, e_m)
        ax1.plot(x_cold, y_cold, 'b-', linewidth=1.5, label='Cold start')

        # Warm start
        s_m = warm_result.s_m
        e_m = warm_result.X[6, :]
        x_warm, y_warm, _ = self._convert_to_global(s_m, e_m)
        ax1.plot(x_warm, y_warm, 'r--', linewidth=1.5, label='Warm start')

        ax1.set_xlabel('East [m]')
        ax1.set_ylabel('North [m]')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.set_title('Trajectory Comparison')

        # Metrics bar chart
        ax2 = fig.add_subplot(gs[0, 2])
        metrics = ['Iterations', 'Time [s]', 'Cost [s]']
        cold_vals = [cold_result.iterations, cold_result.solve_time, cold_result.cost]
        warm_vals = [warm_result.iterations, warm_result.solve_time, warm_result.cost]

        x = np.arange(len(metrics))
        width = 0.35

        ax2.bar(x - width/2, cold_vals, width, label='Cold', color='tab:blue')
        ax2.bar(x + width/2, warm_vals, width, label='Warm', color='tab:red')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_title('Performance Metrics')

        # Convergence comparison
        ax3 = fig.add_subplot(gs[1, :])
        if hasattr(cold_result, 'iteration_history') and cold_result.iteration_history:
            ax3.plot(cold_result.iteration_history, 'b-o', label='Cold start',
                     markersize=4)
        if hasattr(warm_result, 'iteration_history') and warm_result.iteration_history:
            ax3.plot(warm_result.iteration_history, 'r-s', label='Warm start',
                     markersize=4)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Cost [s]')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Convergence Comparison')

        # State difference
        ax4 = fig.add_subplot(gs[2, 0])
        state_diff = np.abs(cold_result.X - warm_result.X)
        ax4.semilogy(cold_result.s_m, np.max(state_diff, axis=0), 'k-')
        ax4.set_xlabel('Arc length [m]')
        ax4.set_ylabel('Max |state difference|')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('State Trajectory Difference')

        # Velocity comparison
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(cold_result.s_m, cold_result.X[0, :], 'b-', label='Cold')
        ax5.plot(warm_result.s_m, warm_result.X[0, :], 'r--', label='Warm')
        ax5.set_xlabel('Arc length [m]')
        ax5.set_ylabel('Velocity [m/s]')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Velocity Profile')

        # Summary text
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')

        speedup = cold_result.iterations / max(warm_result.iterations, 1)
        time_savings = (cold_result.solve_time - warm_result.solve_time)
        cost_diff = warm_result.cost - cold_result.cost

        summary_text = (
            f"Warm-Start Analysis\n"
            f"{'='*30}\n\n"
            f"Cold start:\n"
            f"  Iterations: {cold_result.iterations}\n"
            f"  Time: {cold_result.solve_time:.2f}s\n"
            f"  Cost: {cold_result.cost:.4f}s\n\n"
            f"Warm start:\n"
            f"  Iterations: {warm_result.iterations}\n"
            f"  Time: {warm_result.solve_time:.2f}s\n"
            f"  Cost: {warm_result.cost:.4f}s\n\n"
            f"Improvement:\n"
            f"  Iteration speedup: {speedup:.2f}x\n"
            f"  Time saved: {time_savings:.2f}s\n"
            f"  Cost diff: {cost_diff:+.4f}s"
        )

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                 fontfamily='monospace', fontsize=10, verticalalignment='top')

        if title:
            fig.suptitle(title, fontsize=14)

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

        return str(filepath)

    def generate_full_report(
        self,
        result,
        prefix: str = "opt"
    ) -> Dict[str, str]:
        """
        Generate all visualization plots for a result.

        Args:
            result: Optimization result
            prefix: Filename prefix

        Returns:
            Dict mapping plot type -> filepath
        """
        filepaths = {}

        filepaths['trajectory'] = self.plot_trajectory_overhead(
            result, f"{prefix}_trajectory.png"
        )

        filepaths['states'] = self.plot_states(
            result, f"{prefix}_states.png"
        )

        filepaths['controls'] = self.plot_controls(
            result, f"{prefix}_controls.png"
        )

        if hasattr(result, 'iteration_history') and result.iteration_history:
            filepaths['convergence'] = self.plot_scp_convergence(
                result, f"{prefix}_convergence.png"
            )

        return filepaths


def create_animation(
    visualizer: TrajectoryVisualizer,
    result,
    filename: str = "trajectory_animation.gif",
    fps: int = 10
) -> str:
    """
    Create animation of trajectory evolution.

    Args:
        visualizer: TrajectoryVisualizer instance
        result: Optimization result
        filename: Output filename
        fps: Frames per second

    Returns:
        Path to saved animation
    """
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=visualizer.config.figsize_trajectory)

    # Track
    world = visualizer.world
    inner = world.data['inner_bounds_m']
    outer = world.data['outer_bounds_m']
    ax.plot(inner[:, 0], inner[:, 1], color='gray', linewidth=1.5)
    ax.plot(outer[:, 0], outer[:, 1], color='gray', linewidth=1.5)

    # Obstacles (static overlays)
    for east_m, north_m, radius_m, radius_tilde_m in visualizer._get_obstacle_plot_data():
        ax.add_patch(Circle((east_m, north_m), radius_m, edgecolor='tab:red',
                            facecolor='tab:red', alpha=0.25, linewidth=1.5, zorder=3))
        ax.add_patch(Circle((east_m, north_m), radius_tilde_m, edgecolor='tab:red',
                            facecolor='none', linestyle='--', linewidth=1.0, alpha=0.9, zorder=3))

    # Get trajectory
    s_m = result.s_m
    e_m = result.X[6, :]
    x_traj, y_traj, _ = visualizer._convert_to_global(s_m, e_m)

    # Set axis limits
    margin = 10
    ax.set_xlim(x_traj.min() - margin, x_traj.max() + margin)
    ax.set_ylim(y_traj.min() - margin, y_traj.max() + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')

    # Initialize line and point
    line, = ax.plot([], [], 'b-', linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=10)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text

    def animate(i):
        line.set_data(x_traj[:i+1], y_traj[:i+1])
        point.set_data([x_traj[i]], [y_traj[i]])

        t = result.X[5, i]
        v = result.X[0, i]
        time_text.set_text(f't = {t:.2f}s, v = {v:.1f} m/s')

        return line, point, time_text

    n_frames = len(x_traj)
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000/fps, blit=True
    )

    filepath = visualizer.output_dir / filename
    anim.save(str(filepath), writer='pillow', fps=fps)
    plt.close(fig)

    return str(filepath)
