#!/usr/bin/env python3
"""
Interactive track drawing tool.

Usage:
    python draw_track.py [--output maps/MyTrack.mat] [--length 260] [--width 6]

Instructions:
    1. LEFT CLICK to place waypoints (minimum 4 recommended)
    2. RIGHT CLICK to remove last waypoint
    3. Press 'c' to close the loop and preview the track
    4. Press 'r' to reset and start over
    5. Press 's' to save the track
    6. Press 'q' to quit

The track will be smoothed with a spline and scaled to the target length.
"""

# Set interactive backend BEFORE any other imports
import matplotlib
matplotlib.use('TkAgg')

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


class TrackDrawer:
    def __init__(self, output_file: str, target_length: float, track_width: float, load_file: str = None):
        self.output_file = output_file
        self.target_length = target_length
        self.track_width = track_width

        self.waypoints = []
        self.preview_line = None
        self.bounds_inner = None
        self.bounds_outer = None
        self.closed = False
        self.connecting_line = None

        # Drag state
        self.dragging_idx = None
        self.drag_threshold = 10  # pixels

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        # Disable default matplotlib key bindings that conflict
        for key in ['s', 'q']:
            if key in plt.rcParams['keymap.save']:
                plt.rcParams['keymap.save'].remove(key)
            if key in plt.rcParams['keymap.quit']:
                plt.rcParams['keymap.quit'].remove(key)
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('East [m]')
        self.ax.set_ylabel('North [m]')

        # Waypoint markers
        self.waypoint_scatter = self.ax.scatter([], [], s=100, c='red', zorder=10, picker=True)
        self.waypoint_numbers = []

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Load existing track if specified (after figure setup)
        if load_file:
            self.load_track(load_file)
            self.update_display()
            # Auto-preview if loaded
            if len(self.waypoints) >= 3:
                self.closed = True
                self.preview_track()

        self.update_title()

    def update_title(self):
        status = "CLOSED - Drag points or press 's' to save" if self.closed else f"{len(self.waypoints)} points"
        self.ax.set_title(
            f"Track Drawer | {status}\n"
            f"Click: add | Drag: move | Right-click: remove | c: close | r: reset | s: save | q: quit"
        )

    def find_nearest_waypoint(self, x, y, threshold_data=5.0):
        """Find waypoint within threshold distance. Returns index or None."""
        if not self.waypoints:
            return None
        for i, (wx, wy) in enumerate(self.waypoints):
            dist = np.sqrt((x - wx)**2 + (y - wy)**2)
            if dist < threshold_data:
                return i
        return None

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            # Check if clicking near existing waypoint (to drag)
            idx = self.find_nearest_waypoint(event.xdata, event.ydata)
            if idx is not None:
                self.dragging_idx = idx
                return

            # Otherwise add new point (if not closed)
            if self.closed:
                print("Track is closed. Press 'r' to reset, drag points to edit, or 's' to save.")
                return
            self.waypoints.append((event.xdata, event.ydata))
            self.update_display()

        elif event.button == 3:  # Right click - remove last point
            if self.waypoints:
                self.waypoints.pop()
                self.closed = False
                self.clear_preview()
                self.update_display()

    def on_release(self, event):
        """Handle mouse release - stop dragging."""
        if self.dragging_idx is not None:
            self.dragging_idx = None
            # Update preview if track is closed
            if self.closed and len(self.waypoints) >= 3:
                self.preview_track()

    def on_motion(self, event):
        """Handle mouse motion - drag waypoint."""
        if self.dragging_idx is None or event.inaxes != self.ax:
            return
        # Update waypoint position
        self.waypoints[self.dragging_idx] = (event.xdata, event.ydata)
        self.update_display()
        # Live preview update while dragging (if closed)
        if self.closed:
            self.preview_track()

    def on_key(self, event):
        if event.key == 'c':  # Close and preview
            if len(self.waypoints) >= 3:
                self.closed = True
                self.preview_track()
            else:
                print("Need at least 3 waypoints to close the track.")

        elif event.key == 'r':  # Reset
            self.waypoints = []
            self.closed = False
            self.clear_preview()
            self.update_display()

        elif event.key == 's':  # Save
            if self.closed and len(self.waypoints) >= 3:
                # Prompt for filename
                print("\n" + "="*40)
                name = input("Enter track name (or press Enter for default): ").strip()
                if name:
                    # Add .mat extension if not present
                    if not name.endswith('.mat'):
                        name += '.mat'
                    self.output_file = str(Path(self.output_file).parent / name)
                print(f"Saving to: {self.output_file}")
                self.save_track()
            else:
                print("Close the track first (press 'c') before saving.")

        elif event.key == 'q':  # Quit
            plt.close(self.fig)

    def update_display(self):
        # Update waypoint markers
        if self.waypoints:
            pts = np.array(self.waypoints)
            self.waypoint_scatter.set_offsets(pts)

            # Update numbers
            for txt in self.waypoint_numbers:
                txt.remove()
            self.waypoint_numbers = []
            for i, (x, y) in enumerate(self.waypoints):
                txt = self.ax.text(x + 2, y + 2, str(i + 1), fontsize=10, color='red')
                self.waypoint_numbers.append(txt)

            # Draw lines connecting waypoints
            if hasattr(self, 'connecting_line') and self.connecting_line:
                self.connecting_line.remove()
            if len(self.waypoints) > 1:
                pts_closed = np.vstack([pts, pts[0]]) if self.closed else pts
                self.connecting_line, = self.ax.plot(
                    pts_closed[:, 0], pts_closed[:, 1],
                    'r--', alpha=0.5, lw=1
                )
            else:
                self.connecting_line = None
        else:
            self.waypoint_scatter.set_offsets(np.empty((0, 2)))
            for txt in self.waypoint_numbers:
                txt.remove()
            self.waypoint_numbers = []
            if hasattr(self, 'connecting_line') and self.connecting_line:
                self.connecting_line.remove()
                self.connecting_line = None

        self.update_title()
        self.fig.canvas.draw_idle()

    def clear_preview(self):
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
        if self.bounds_inner:
            self.bounds_inner.remove()
            self.bounds_inner = None
        if self.bounds_outer:
            self.bounds_outer.remove()
            self.bounds_outer = None

    def preview_track(self):
        """Generate and display track preview."""
        self.clear_preview()

        try:
            pts = np.array(self.waypoints)
            pts_closed = np.vstack([pts, pts[0]])

            # Fit spline
            tck, u = interpolate.splprep(
                [pts_closed[:, 0], pts_closed[:, 1]],
                s=0,
                per=True,
            )

            # Sample
            t_dense = np.linspace(0, 1, 500, endpoint=False)
            xi, yi = interpolate.splev(t_dense, tck)

            # Compute arc length (no scaling - use natural size)
            dx = np.diff(np.r_[xi, xi[0]])
            dy = np.diff(np.r_[yi, yi[0]])
            ds_raw = np.sqrt(dx**2 + dy**2)
            raw_length = np.sum(ds_raw)

            # Compute heading for bounds
            dE = np.gradient(xi)
            dN = np.gradient(yi)
            psi = np.arctan2(dN, dE)

            hw = 0.5 * self.track_width
            left_E = -np.sin(psi)
            left_N = np.cos(psi)

            inner_x = xi - hw * left_E
            inner_y = yi - hw * left_N
            outer_x = xi + hw * left_E
            outer_y = yi + hw * left_N

            # Plot
            self.preview_line, = self.ax.plot(
                np.r_[xi, xi[0]], np.r_[yi, yi[0]],
                'b-', lw=1.5, alpha=0.7, label='Centerline'
            )
            self.bounds_inner, = self.ax.plot(
                np.r_[inner_x, inner_x[0]], np.r_[inner_y, inner_y[0]],
                'g-', lw=1, alpha=0.5
            )
            self.bounds_outer, = self.ax.plot(
                np.r_[outer_x, outer_x[0]], np.r_[outer_y, outer_y[0]],
                'g-', lw=1, alpha=0.5
            )

            # Auto-fit view
            margin = 20
            self.ax.set_xlim(np.min(outer_x) - margin, np.max(outer_x) + margin)
            self.ax.set_ylim(np.min(outer_y) - margin, np.max(outer_y) + margin)

            print(f"Preview: track length = {raw_length:.1f}m")

        except Exception as e:
            print(f"Preview failed: {e}")

        self.update_title()
        self.fig.canvas.draw_idle()

    def load_track(self, filepath: str):
        """Load waypoints from an existing track file."""
        import scipy.io as sio

        try:
            data = sio.loadmat(filepath, squeeze_me=True)
            length = float(np.atleast_1d(data['length_m']).item())
            width = np.atleast_1d(data['track_width_m'])
            self.track_width = float(np.mean(width))
            self.target_length = length

            # Check if original waypoints are saved
            if 'waypoints_xy' in data:
                wp = np.asarray(data['waypoints_xy'])
                self.waypoints = [(wp[i, 0], wp[i, 1]) for i in range(len(wp))]
                print(f"Loaded {len(self.waypoints)} original waypoints from {filepath}")
            else:
                # Fall back to sampling from centerline
                posE = np.asarray(data['posE_m'])
                posN = np.asarray(data['posN_m'])
                n_waypoints = max(8, int(length / 25))
                indices = np.linspace(0, len(posE) - 1, n_waypoints, dtype=int)
                self.waypoints = [(posE[i], posN[i]) for i in indices]
                print(f"Sampled {len(self.waypoints)} waypoints from {filepath} (no original waypoints saved)")

            print(f"Track length: {length:.1f}m, width: {self.track_width:.1f}m")

        except Exception as e:
            print(f"Failed to load track: {e}")

    def save_track(self):
        """Save the track to file."""
        try:
            # Import here to avoid backend conflict at module load time
            from create_tracks import create_track_from_waypoints, create_overview_plot
            create_track_from_waypoints(
                self.output_file,
                self.waypoints,
                total_length=None,  # Use natural size from waypoints
                track_width=self.track_width,
            )
            print(f"\nTrack saved to: {self.output_file}")
            print(f"Waypoints: {self.waypoints}")

            # Save PNG of this track
            track_png = Path(self.output_file).with_suffix('.png')
            self.fig.savefig(track_png, dpi=150, bbox_inches='tight')
            print(f"Track image saved: {track_png}")

            # Regenerate overview
            maps_dir = Path(self.output_file).parent
            overview_path = create_overview_plot(maps_dir)
            print(f"Overview updated: {overview_path}")
        except Exception as e:
            print(f"Save failed: {e}")

    def run(self):
        print("\n" + "="*60)
        print("TRACK DRAWER")
        print("="*60)
        print(f"Output: {self.output_file}")
        print(f"Target length: {self.target_length}m")
        print(f"Track width: {self.track_width}m")
        print("\nInstructions:")
        print("  LEFT CLICK  - Add waypoint")
        print("  RIGHT CLICK - Remove last waypoint")
        print("  'c'         - Close loop and preview")
        print("  'r'         - Reset")
        print("  's'         - Save track")
        print("  'q'         - Quit")
        print("="*60 + "\n")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive track drawing tool")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="maps",
        help="Output file or directory (default: maps, auto-generates timestamped filename)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load an existing track to modify (e.g., maps/Oval_Track_260m.mat)"
    )
    parser.add_argument(
        "--length", "-l",
        type=float,
        default=260.0,
        help="Target track length in meters (default: 260, ignored if --load)"
    )
    parser.add_argument(
        "--width", "-w",
        type=float,
        default=6.0,
        help="Track width in meters (default: 6, ignored if --load)"
    )
    args = parser.parse_args()

    # Handle output path
    output_path = Path(args.output)
    if output_path.is_dir() or not output_path.suffix:
        # If output is a directory, generate a filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output) / f"Custom_Track_{timestamp}.mat"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    drawer = TrackDrawer(str(output_path), args.length, args.width, load_file=args.load)
    drawer.run()


if __name__ == "__main__":
    main()
