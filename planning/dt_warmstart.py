#!/usr/bin/env python3
"""
Decision Transformer warm-start integration for IPOPT trajectory optimization.

Based on PLAN.md Section 4:
- Load trained DT model
- Run dynamics-in-the-loop rollout to generate warm-start trajectory
- Validate warm-start before passing to IPOPT
- Integration with TrajectoryOptimizer.solve()
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dt.model import DecisionTransformer, DTConfig
from dt.dataset import DatasetStats


@dataclass
class WarmStartResult:
    """Result of DT warm-start generation."""
    X_init: np.ndarray          # (n_states, N+1)
    U_init: np.ndarray          # (n_controls, N+1)
    success: bool
    rejection_reason: Optional[str] = None
    rtg_used: float = 0.0
    inference_time_s: float = 0.0


class DTWarmStarter:
    """
    Decision Transformer warm-start generator.

    Generates initial trajectories for IPOPT using a trained DT model.
    Uses dynamics-in-the-loop rollout where DT predicts actions,
    then dynamics are propagated to get next state.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vehicle_model,
        world,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize warm-starter.

        Args:
            checkpoint_path: Path to trained DT checkpoint
            vehicle_model: Vehicle dynamics model (SingleTrackModel)
            world: Track/world object
            device: Torch device
        """
        self.device = torch.device(device)
        self.vehicle = vehicle_model
        self.world = world

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Reconstruct model config
        model_config_dict = checkpoint.get("model_config", {})
        self.model_config = DTConfig(**model_config_dict)

        # Build and load model
        self.model = DecisionTransformer(self.model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization stats
        stats_path = Path(checkpoint_path).parent / "dataset_stats.npz"
        if stats_path.exists():
            self.stats = DatasetStats.load(stats_path)
        else:
            print(f"Warning: No dataset_stats.npz found at {stats_path}")
            self.stats = None

        # Control bounds (from vehicle/optimizer config)
        self.delta_max = 0.5   # rad
        self.fx_min = -10.0    # kN
        self.fx_max = 5.0      # kN

        print(f"Loaded DT warm-starter from {checkpoint_path}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def _normalize_state(self, state_aug: np.ndarray) -> np.ndarray:
        """Normalize augmented state."""
        if self.stats is not None:
            return (state_aug - self.stats.state_mean) / self.stats.state_std
        return state_aug

    def _normalize_rtg(self, rtg: float) -> float:
        """Normalize return-to-go."""
        if self.stats is not None:
            return (rtg - self.stats.rtg_mean) / self.stats.rtg_std
        return rtg

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action."""
        if self.stats is not None:
            return action * self.stats.action_std + self.stats.action_mean
        return action

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to control bounds."""
        delta = np.clip(action[0], -self.delta_max, self.delta_max)
        fx = np.clip(action[1], self.fx_min, self.fx_max)
        return np.array([delta, fx])

    def _get_track_features(self, s: float) -> np.ndarray:
        """Get track features at arc-length s."""
        s_mod = s % self.world.length_m
        kappa = float(self.world.psi_s_radpm_LUT(np.array([s_mod])))
        half_width = 0.5 * float(self.world.track_width_m_LUT(np.array([s_mod])))
        return np.array([kappa, half_width], dtype=np.float32)

    def _get_obstacle_features(
        self,
        s: float,
        e: float,
        obstacles: List[Dict],
    ) -> np.ndarray:
        """Get obstacle features at current position."""
        M = self.model_config.obstacle_slots
        feat_dim = self.model_config.obstacle_feat_dim
        lookahead = 60.0  # m

        obs_feats = np.zeros((M, feat_dim), dtype=np.float32)

        if not obstacles:
            return obs_feats.flatten()

        # Find obstacles ahead
        ahead_obs = []
        for obs in obstacles:
            s_obs = obs.get("s_obs", obs.get("s", 0))
            e_obs = obs.get("e_obs", obs.get("e", 0))
            r_obs = obs.get("r_obs", obs.get("r", 1.0))

            ds = s_obs - s
            # Handle wrap-around
            if ds < 0:
                ds += self.world.length_m

            if 0 < ds <= lookahead:
                ahead_obs.append({
                    "ds": ds,
                    "de": e_obs - e,
                    "r": r_obs,
                })

        # Sort by distance and take closest M
        ahead_obs.sort(key=lambda x: x["ds"])
        for i, obs in enumerate(ahead_obs[:M]):
            obs_feats[i, 0] = obs["ds"]
            obs_feats[i, 1] = obs["de"]
            obs_feats[i, 2] = obs["r"]

        return obs_feats.flatten()

    def _get_global_pose(self, s: float, e: float, dpsi: float) -> Dict[str, float]:
        """Compute global position from Frenet coordinates."""
        s_mod = s % self.world.length_m
        posE_cl = float(self.world.posE_m_interp_fcn(np.array([s_mod])))
        posN_cl = float(self.world.posN_m_interp_fcn(np.array([s_mod])))
        psi_cl = float(self.world.psi_rad_interp_fcn(np.array([s_mod])))

        posE = posE_cl - e * np.sin(psi_cl)
        posN = posN_cl + e * np.cos(psi_cl)
        yaw_world = psi_cl + dpsi

        return {"pos_E": posE, "pos_N": posN, "yaw_world": yaw_world}

    def _build_state_aug(
        self,
        state: np.ndarray,  # [ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world]
        s: float,
        obstacles: List[Dict],
    ) -> np.ndarray:
        """Build augmented state for DT input."""
        track_feats = self._get_track_features(s)
        obs_feats = self._get_obstacle_features(s, state[3], obstacles)  # e is index 3
        return np.concatenate([state, track_feats, obs_feats]).astype(np.float32)

    def _dynamics_step(
        self,
        x_full: np.ndarray,  # Full state: [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi]
        u: np.ndarray,       # Control: [delta, fx]
        s: float,            # Current arc-length
        ds: float,           # Step size
    ) -> Tuple[np.ndarray, float]:
        """
        Propagate dynamics one spatial step.

        Returns:
            x_next: Next full state
            dt: Time taken for step
        """
        # Get curvature at current position
        s_mod = s % self.world.length_m
        kappa = float(self.world.psi_s_radpm_LUT(np.array([s_mod])))

        # Extract states
        ux, uy, r = x_full[0], x_full[1], x_full[2]
        dfz_long, dfz_lat = x_full[3], x_full[4]
        t, e, dpsi = x_full[5], x_full[6], x_full[7]

        # Compute sdot (spatial velocity)
        one_minus_kappa_e = 1.0 - kappa * e
        sdot = (ux * np.cos(dpsi) - uy * np.sin(dpsi)) / max(0.1, one_minus_kappa_e)
        sdot = max(0.1, sdot)  # Ensure positive progress

        # Time for this step
        dt = ds / sdot

        # Simple Euler integration of path kinematics
        e_dot = ux * np.sin(dpsi) + uy * np.cos(dpsi)
        dpsi_dot = r - kappa * sdot

        # For simplicity, keep vehicle states roughly constant over short spatial step
        # (A more accurate approach would integrate the full dynamics)
        x_next = np.array([
            ux,  # Could integrate dvx/dt but keep simple
            uy,
            r,
            dfz_long,
            dfz_lat,
            t + dt,
            e + e_dot * dt,
            dpsi + dpsi_dot * dt,
        ])

        return x_next, dt

    @torch.no_grad()
    def generate_warmstart(
        self,
        N: int,
        ds_m: float,
        x0: np.ndarray,
        obstacles: Optional[List[Dict]] = None,
        target_lap_time: Optional[float] = None,
    ) -> WarmStartResult:
        """
        Generate warm-start trajectory using DT.

        Args:
            N: Number of spatial nodes
            ds_m: Spatial step size
            x0: Initial full state [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi]
            obstacles: List of obstacle dicts
            target_lap_time: Target lap time for RTG conditioning

        Returns:
            WarmStartResult with X_init, U_init, and validation info
        """
        import time
        t_start = time.time()

        obstacles = obstacles or []
        n_states = len(x0)
        n_controls = 2

        # Initialize arrays
        X_init = np.zeros((n_states, N + 1))
        U_init = np.zeros((n_controls, N + 1))
        X_init[:, 0] = x0

        # Estimate RTG
        if target_lap_time is not None:
            rtg_0 = -target_lap_time
        else:
            # Use default based on track length and speed
            avg_speed = max(5.0, x0[0])
            estimated_time = self.world.length_m / avg_speed
            rtg_0 = -estimated_time * 0.9  # Slightly ambitious

        # Context buffers
        K = self.model_config.context_length
        states_buffer = []
        actions_buffer = []
        rtg_buffer = []
        timesteps_buffer = []

        # Current state for rollout
        x_full = x0.copy()
        s = 0.0
        rtg = rtg_0

        # Rollout
        for k in range(N):
            # Build DT observation: [ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world]
            pose = self._get_global_pose(s, x_full[6], x_full[7])
            state_obs = np.array([
                x_full[0],  # ux
                x_full[1],  # uy
                x_full[2],  # r
                x_full[6],  # e
                x_full[7],  # dpsi
                pose["pos_E"],
                pose["pos_N"],
                pose["yaw_world"],
            ], dtype=np.float32)

            # Build augmented state
            state_aug = self._build_state_aug(state_obs, s, obstacles)

            # Add to buffers
            states_buffer.append(state_aug)
            rtg_buffer.append(rtg)
            timesteps_buffer.append(k)

            # Truncate buffers to context length
            if len(states_buffer) > K:
                states_buffer = states_buffer[-K:]
                actions_buffer = actions_buffer[-(K-1):]  # One less action
                rtg_buffer = rtg_buffer[-K:]
                timesteps_buffer = timesteps_buffer[-K:]

            # Prepare tensors for model
            states_t = torch.tensor(
                np.stack(states_buffer), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            # Normalize states
            if self.stats is not None:
                states_t = (states_t - torch.tensor(self.stats.state_mean, device=self.device)) / \
                           torch.tensor(self.stats.state_std, device=self.device)

            # Actions buffer (pad with zeros for current step)
            if actions_buffer:
                actions_np = np.stack(actions_buffer + [np.zeros(n_controls)])
            else:
                actions_np = np.zeros((1, n_controls))
            actions_t = torch.tensor(actions_np, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Normalize actions
            if self.stats is not None:
                actions_t = (actions_t - torch.tensor(self.stats.action_mean, device=self.device)) / \
                            torch.tensor(self.stats.action_std, device=self.device)

            # RTG
            rtg_np = np.array(rtg_buffer, dtype=np.float32)
            if self.stats is not None:
                rtg_np = (rtg_np - self.stats.rtg_mean) / self.stats.rtg_std
            rtg_t = torch.tensor(rtg_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

            # Timesteps
            timesteps_t = torch.tensor(timesteps_buffer, dtype=torch.long).unsqueeze(0).to(self.device)

            # Pad to context length if needed
            T = states_t.shape[1]
            if T < K:
                pad_len = K - T
                states_t = torch.cat([
                    torch.zeros(1, pad_len, states_t.shape[-1], device=self.device),
                    states_t
                ], dim=1)
                actions_t = torch.cat([
                    torch.zeros(1, pad_len, n_controls, device=self.device),
                    actions_t
                ], dim=1)
                rtg_t = torch.cat([
                    torch.zeros(1, pad_len, 1, device=self.device),
                    rtg_t
                ], dim=1)
                timesteps_t = torch.cat([
                    torch.zeros(1, pad_len, device=self.device, dtype=torch.long),
                    timesteps_t
                ], dim=1)

            # Forward pass
            action_preds, _ = self.model(states_t, actions_t, rtg_t, timesteps_t)

            # Get last action prediction
            action_pred = action_preds[0, -1].cpu().numpy()

            # Denormalize and clip
            if self.stats is not None:
                action_pred = action_pred * self.stats.action_std + self.stats.action_mean
            action_pred = self._clip_action(action_pred)

            # Store action
            U_init[:, k] = action_pred
            actions_buffer.append(action_pred)

            # Propagate dynamics
            x_next, dt = self._dynamics_step(x_full, action_pred, s, ds_m)
            X_init[:, k + 1] = x_next

            # Update state and progress
            x_full = x_next
            s += ds_m
            rtg += dt  # RTG is negative time remaining, so add dt

        # Copy last action
        U_init[:, N] = U_init[:, N - 1]

        inference_time = time.time() - t_start

        # Validate warm-start
        success, reason = self._validate_warmstart(X_init, U_init, obstacles)

        return WarmStartResult(
            X_init=X_init,
            U_init=U_init,
            success=success,
            rejection_reason=reason,
            rtg_used=rtg_0,
            inference_time_s=inference_time,
        )

    def _validate_warmstart(
        self,
        X_init: np.ndarray,
        U_init: np.ndarray,
        obstacles: List[Dict],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate warm-start trajectory.

        Checks:
        - No NaNs
        - Track bounds
        - Forward progress (sdot > 0)
        - Obstacle clearance

        Returns:
            (success, rejection_reason)
        """
        # Check for NaNs
        if np.any(np.isnan(X_init)) or np.any(np.isnan(U_init)):
            return False, "NaN in trajectory"

        # Check forward progress (ux > 0)
        if np.any(X_init[0, :] < 0.1):
            return False, "Negative/zero forward velocity"

        # Check track bounds
        N = X_init.shape[1] - 1
        ds_m = self.world.length_m / N
        for k in range(N + 1):
            s = k * ds_m
            e = X_init[6, k]
            s_mod = s % self.world.length_m
            half_width = 0.5 * float(self.world.track_width_m_LUT(np.array([s_mod])))
            if abs(e) > half_width:
                return False, f"Track bounds violated at node {k}: |e|={abs(e):.2f} > hw={half_width:.2f}"

        # Check obstacle clearance (if obstacles present)
        if obstacles:
            for k in range(N + 1):
                s = k * ds_m
                e = X_init[6, k]
                pose = self._get_global_pose(s, e, X_init[7, k])

                for obs in obstacles:
                    s_obs = obs.get("s_obs", obs.get("s", 0))
                    e_obs = obs.get("e_obs", obs.get("e", 0))
                    r_obs = obs.get("r_obs", obs.get("r", 1.0))

                    # Convert obstacle to global
                    obs_pose = self._get_global_pose(s_obs, e_obs, 0)

                    dist = np.sqrt(
                        (pose["pos_E"] - obs_pose["pos_E"])**2 +
                        (pose["pos_N"] - obs_pose["pos_N"])**2
                    )
                    if dist < r_obs:
                        return False, f"Obstacle collision at node {k}"

        return True, None


def load_warmstarter(
    checkpoint_path: str,
    vehicle_model,
    world,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DTWarmStarter:
    """Load a DTWarmStarter from checkpoint."""
    return DTWarmStarter(checkpoint_path, vehicle_model, world, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DT warm-start")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--map-file", type=str, default="maps/Oval_Track_260m.mat")
    parser.add_argument("--N", type=int, default=120)
    args = parser.parse_args()

    from models import load_vehicle_from_yaml
    from world.world import World

    # Load world and vehicle
    world = World(args.map_file, Path(args.map_file).stem, diagnostic_plotting=False)
    vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")

    # Load warm-starter
    warmstarter = load_warmstarter(args.checkpoint, vehicle, world)

    # Generate warm-start
    ds_m = world.length_m / args.N
    x0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state

    result = warmstarter.generate_warmstart(args.N, ds_m, x0)

    print(f"\nWarm-start result:")
    print(f"  Success: {result.success}")
    print(f"  Rejection reason: {result.rejection_reason}")
    print(f"  Inference time: {result.inference_time_s:.3f}s")
    print(f"  RTG used: {result.rtg_used:.2f}")
    print(f"  X_init shape: {result.X_init.shape}")
    print(f"  U_init shape: {result.U_init.shape}")
