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
import casadi as ca

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
    fallback_count: int = 0
    projection_count: int = 0
    projection_total_magnitude: float = 0.0
    projection_max_magnitude: float = 0.0


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
        checkpoint_dir = Path(checkpoint_path).parent
        stats_candidates = [
            checkpoint_dir / "dataset_stats.npz",
            checkpoint_dir.parent / "dataset_stats.npz",
        ]
        self.stats = None
        for stats_path in stats_candidates:
            if stats_path.exists():
                self.stats = DatasetStats.load(stats_path)
                break
        if self.stats is None:
            print(
                "Warning: No dataset_stats.npz found at any of: "
                + ", ".join(str(p) for p in stats_candidates)
            )

        # Control bounds (from vehicle/optimizer config)
        self.delta_max = 0.5   # rad
        self.fx_min = -10.0    # kN
        self.fx_max = 5.0      # kN

        print(f"Loaded DT warm-starter from {checkpoint_path}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    @staticmethod
    def _effective_obstacle_radius(
        obs: Dict,
        obstacle_clearance_m: float,
        vehicle_radius_m: float,
    ) -> float:
        """
        Current optimizer-consistent obstacle radius.

        Matches the active dataset/optimizer convention:
        radius_m + margin_m + obstacle_clearance_m + vehicle_radius_m

        When the repo collapses margin into a single clearance term, this helper
        is the only place that needs to change on the warm-start side.
        """
        radius_m = float(obs.get("radius_m", obs.get("r_obs", obs.get("r", 1.0))))
        margin_m = float(obs.get("margin_m", 0.0))
        return radius_m + margin_m + float(obstacle_clearance_m) + float(vehicle_radius_m)

    @staticmethod
    def _obstacle_frenet(obs: Dict) -> Tuple[float, float, float]:
        """Read obstacle Frenet coordinates/radius from canonical or legacy keys."""
        s_m = float(obs.get("s_m", obs.get("s_obs", obs.get("s", 0.0))))
        e_m = float(obs.get("e_m", obs.get("e_obs", obs.get("e", 0.0))))
        radius_m = float(obs.get("radius_m", obs.get("r_obs", obs.get("r", 1.0))))
        return s_m, e_m, radius_m

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

    def _get_road_geometry(self, s: float) -> Tuple[float, float, float]:
        """Get (kappa, theta, phi) at arc-length s."""
        s_mod = s % self.world.length_m
        kappa = float(self.world.psi_s_radpm_LUT(np.array([s_mod])))
        theta = float(self.world.grade_rad_LUT(np.array([s_mod]))) if hasattr(self.world, "grade_rad_LUT") else 0.0
        phi = float(self.world.bank_rad_LUT(np.array([s_mod]))) if hasattr(self.world, "bank_rad_LUT") else 0.0
        return kappa, theta, phi

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
            s_obs, e_obs, r_obs = self._obstacle_frenet(obs)

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

    def _dx_ds(
        self,
        x_full: np.ndarray,  # Full state: [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi]
        u: np.ndarray,       # Control: [delta, fx]
        s: float,            # Current arc-length
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluate spatial path dynamics using the actual vehicle model.

        Returns:
            dx_ds: Spatial derivative of the full path state
            s_dot: Arc length rate [m/s]
        """
        kappa, theta, phi = self._get_road_geometry(s)
        dx_dt, s_dot = self.vehicle.dynamics_dt_path_vec(
            ca.DM(x_full),
            ca.DM(u),
            kappa,
            theta,
            phi,
        )
        s_dot_float = max(0.1, float(s_dot))
        dx_ds = np.array(dx_dt / s_dot_float, dtype=np.float64).reshape(-1)
        return dx_ds, s_dot_float

    def _dynamics_step(
        self,
        x_full: np.ndarray,
        u: np.ndarray,
        s: float,
        ds: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Propagate one spatial step with RK4 on the model-consistent path dynamics.
        """
        k1, sdot_1 = self._dx_ds(x_full, u, s)
        k2, _ = self._dx_ds(x_full + 0.5 * ds * k1, u, s + 0.5 * ds)
        k3, _ = self._dx_ds(x_full + 0.5 * ds * k2, u, s + 0.5 * ds)
        k4, _ = self._dx_ds(x_full + ds * k3, u, s + ds)

        x_next = x_full + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        dt = ds / sdot_1

        # Preserve the spatial rollout clock exactly with the integrated step time.
        x_next[5] = x_full[5] + dt

        return x_next, dt

    def _state_is_reasonable(self, x_full: np.ndarray, s: float) -> bool:
        """Screen for rollout states that have already blown up numerically."""
        if not np.isfinite(x_full).all():
            return False

        s_mod = s % self.world.length_m
        half_width = 0.5 * float(self.world.track_width_m_LUT(np.array([s_mod])))

        ux, uy, r = float(x_full[0]), float(x_full[1]), float(x_full[2])
        e, dpsi = float(x_full[6]), float(x_full[7])

        if ux < 0.1 or ux > 80.0:
            return False
        if abs(uy) > 20.0 or abs(r) > 5.0:
            return False
        if abs(e) > max(2.0 * half_width, half_width + 5.0):
            return False
        if abs(dpsi) > np.pi:
            return False
        return True

    def _project_state_to_track(
        self,
        x_full: np.ndarray,
        s: float,
        obstacles: Optional[List[Dict]] = None,
        obstacle_clearance_m: float = 0.0,
        vehicle_radius_m: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """Project a rollout state back to a conservative in-track envelope."""
        x_before = x_full.copy()
        x_proj = x_full.copy()
        s_mod = s % self.world.length_m
        half_width = 0.5 * float(self.world.track_width_m_LUT(np.array([s_mod])))
        e_limit = max(0.5, 0.9 * half_width)

        x_proj[0] = float(np.clip(x_proj[0], 5.0, 25.0))
        x_proj[1] = float(np.clip(x_proj[1], -2.0, 2.0))
        x_proj[2] = float(np.clip(x_proj[2], -1.0, 1.0))
        e_proj = float(np.clip(x_proj[6], -e_limit, e_limit))

        if obstacles:
            longitudinal_window = max(4.0, 3.0 * (self.world.length_m / 120.0))
            for obs in obstacles:
                s_obs, e_obs, _ = self._obstacle_frenet(obs)
                ds = s_obs - s
                if ds < -0.5 * self.world.length_m:
                    ds += self.world.length_m
                elif ds > 0.5 * self.world.length_m:
                    ds -= self.world.length_m

                if abs(ds) > longitudinal_window:
                    continue

                required = self._effective_obstacle_radius(
                    obs,
                    obstacle_clearance_m=obstacle_clearance_m,
                    vehicle_radius_m=vehicle_radius_m,
                )
                if abs(e_proj - e_obs) < required:
                    left_target = e_obs + required
                    right_target = e_obs - required
                    left_feasible = abs(left_target) <= e_limit
                    right_feasible = abs(right_target) <= e_limit

                    if left_feasible and right_feasible:
                        e_proj = left_target if abs(left_target) < abs(right_target) else right_target
                    elif left_feasible:
                        e_proj = left_target
                    elif right_feasible:
                        e_proj = right_target
                    else:
                        e_proj = float(np.clip(e_proj, -e_limit, e_limit))

        x_proj[6] = e_proj
        x_proj[7] = float(np.clip(x_proj[7], -0.75, 0.75))
        projection_mag = float(np.linalg.norm(x_proj - x_before))
        return x_proj, projection_mag

    def _fallback_step(
        self,
        x_full: np.ndarray,
        u: np.ndarray,
        s: float,
        ds: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Conservative rollout fallback used when the full model step becomes unstable.

        This keeps the warm-start numerically usable for IPOPT rather than letting
        the trajectory explode to NaNs. The fallback also nudges the trajectory
        back toward the centerline instead of passively preserving a diverging
        lateral state.
        """
        kappa, _, _ = self._get_road_geometry(s)
        s_next = s + ds
        s_next_mod = s_next % self.world.length_m
        half_width = 0.5 * float(self.world.track_width_m_LUT(np.array([s_next_mod])))
        ux = float(np.clip(x_full[0], 5.0, 30.0))
        e = float(x_full[6])
        dpsi = float(x_full[7])

        dt = ds / max(ux, 0.5)

        # Simple centerline-recovery law in spatial coordinates.
        dpsi_cmd = np.clip(0.55 * dpsi - 0.08 * e - ds * kappa, -0.45, 0.45)
        e_next = e + ds * np.sin(dpsi_cmd)
        e_limit = max(0.5, 0.8 * half_width)
        e_next = float(np.clip(e_next, -e_limit, e_limit))
        dpsi_next = float(np.clip(dpsi_cmd, -0.6, 0.6))

        x_next = x_full.copy()
        x_next[0] = float(np.clip(ux + 0.02 * np.clip(u[1], -2.0, 2.0), 5.0, 25.0))
        x_next[1] = float(np.clip(0.35 * x_full[1], -1.5, 1.5))
        x_next[2] = float(np.clip((dpsi_next - dpsi) / max(dt, 1e-3), -0.8, 0.8))
        x_next[3] = 0.0
        x_next[4] = 0.0
        x_next[5] = x_full[5] + dt
        x_next[6] = e_next
        x_next[7] = dpsi_next
        return x_next, dt

    @torch.no_grad()
    def generate_warmstart(
        self,
        N: int,
        ds_m: float,
        x0: np.ndarray,
        obstacles: Optional[List[Dict]] = None,
        target_lap_time: Optional[float] = None,
        obstacle_clearance_m: float = 0.0,
        vehicle_radius_m: float = 0.0,
    ) -> WarmStartResult:
        """
        Generate warm-start trajectory using DT.

        Args:
            N: Number of spatial nodes
            ds_m: Spatial step size
            x0: Initial full state [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi]
            obstacles: List of obstacle dicts
            target_lap_time: Target lap time for RTG conditioning
            obstacle_clearance_m: Extra clearance beyond obstacle radius + margin
            vehicle_radius_m: Conservative vehicle footprint radius

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
        fallback_count = 0
        projection_count = 0
        projection_total_magnitude = 0.0
        projection_max_magnitude = 0.0
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

            if not np.isfinite(action_pred).all():
                action_pred = np.zeros_like(action_pred)

            # Store action
            U_init[:, k] = action_pred
            actions_buffer.append(action_pred)

            # Propagate dynamics
            x_next, dt = self._dynamics_step(x_full, action_pred, s, ds_m)
            if (not np.isfinite(x_next).all()) or (not np.isfinite(dt)) or (not self._state_is_reasonable(x_next, s + ds_m)):
                fallback_count += 1
                x_next, dt = self._fallback_step(x_full, action_pred, s, ds_m)
                if (not np.isfinite(x_next).all()) or (not np.isfinite(dt)) or (not self._state_is_reasonable(x_next, s + ds_m)):
                    return WarmStartResult(
                        X_init=X_init,
                        U_init=U_init,
                        success=False,
                        rejection_reason=f"Rollout unstable at step {k}",
                        rtg_used=rtg_0,
                        inference_time_s=time.time() - t_start,
                        fallback_count=fallback_count,
                        projection_count=projection_count,
                        projection_total_magnitude=projection_total_magnitude,
                        projection_max_magnitude=projection_max_magnitude,
                    )
            x_next, projection_mag = self._project_state_to_track(
                x_next,
                s + ds_m,
                obstacles=obstacles,
                obstacle_clearance_m=obstacle_clearance_m,
                vehicle_radius_m=vehicle_radius_m,
            )
            if projection_mag > 1e-9:
                projection_count += 1
                projection_total_magnitude += projection_mag
                projection_max_magnitude = max(projection_max_magnitude, projection_mag)
            X_init[:, k + 1] = x_next

            # Update state and progress
            x_full = x_next
            s += ds_m
            rtg += dt  # RTG is negative time remaining, so add dt

        # Copy last action
        U_init[:, N] = U_init[:, N - 1]

        inference_time = time.time() - t_start

        # Validate warm-start
        success, reason = self._validate_warmstart(
            X_init,
            U_init,
            obstacles,
            obstacle_clearance_m=obstacle_clearance_m,
            vehicle_radius_m=vehicle_radius_m,
        )

        return WarmStartResult(
            X_init=X_init,
            U_init=U_init,
            success=success,
            rejection_reason=reason,
            rtg_used=rtg_0,
            inference_time_s=inference_time,
            fallback_count=fallback_count,
            projection_count=projection_count,
            projection_total_magnitude=projection_total_magnitude,
            projection_max_magnitude=projection_max_magnitude,
        )

    def _validate_warmstart(
        self,
        X_init: np.ndarray,
        U_init: np.ndarray,
        obstacles: List[Dict],
        obstacle_clearance_m: float = 0.0,
        vehicle_radius_m: float = 0.0,
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
                    s_obs, e_obs, r_obs = self._obstacle_frenet(obs)

                    # Convert obstacle to global
                    obs_pose = self._get_global_pose(s_obs, e_obs, 0)

                    dist = np.sqrt(
                        (pose["pos_E"] - obs_pose["pos_E"])**2 +
                        (pose["pos_N"] - obs_pose["pos_N"])**2
                    )
                    required_radius = self._effective_obstacle_radius(
                        obs,
                        obstacle_clearance_m=obstacle_clearance_m,
                        vehicle_radius_m=vehicle_radius_m,
                    )
                    if dist < required_radius:
                        return False, (
                            f"Obstacle collision at node {k}: "
                            f"dist={dist:.3f} < required_radius={required_radius:.3f}"
                        )

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

    result = warmstarter.generate_warmstart(
        args.N,
        ds_m,
        x0,
        obstacle_clearance_m=0.0,
        vehicle_radius_m=0.0,
    )

    print(f"\nWarm-start result:")
    print(f"  Success: {result.success}")
    print(f"  Rejection reason: {result.rejection_reason}")
    print(f"  Inference time: {result.inference_time_s:.3f}s")
    print(f"  RTG used: {result.rtg_used:.2f}")
    print(f"  X_init shape: {result.X_init.shape}")
    print(f"  U_init shape: {result.U_init.shape}")
