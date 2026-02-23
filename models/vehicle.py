"""
Single-Track Vehicle Model - Unified Best-of-Both-Worlds

Combines features from both models-main and multimodel-trajectory-optimization:
- From models-main: lateral weight transfer (dfz_lat), clean structure, load-dependent tires
- From multimodel: road geometry (grade, bank), time state, brake yaw moment

Reference: Aggarwal & Gerdes, "Friction-Robust Autonomous Racing Using Trajectory
Optimization Over Multiple Models", IEEE Open Journal of Control Systems, 2025.

State vectors:
- Dynamics:  [ux, uy, r, dfz_long, dfz_lat] (5 states)
- Global:    [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi] (9 states)
- Path:      [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi] (8 states)

Control: [delta, fx_kn] (2 inputs)

Road geometry (passed separately): [theta, phi, k_psi]
- theta: road grade [rad] (positive = uphill)
- phi: road bank [rad] (positive = banked right)
- k_psi: path curvature [1/m]
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING
import numpy as np
from scipy import constants
from yaml import safe_load

import casadi as ca

if TYPE_CHECKING:
    from .tire import FialaBrushTire


G_MPS2 = 9.81  # Gravitational acceleration
BETA_SOFTPLUS = 2.0  # Smoothing parameter for braking-only force extraction


# =============================================================================
# Vehicle Parameters - Extended from models-main
# =============================================================================

@dataclass(frozen=True)
class VehicleParams:
    """
    Vehicle parameters - immutable dataclass.

    Extended to include all parameters from both original implementations.
    """

    name: str

    # Mass and inertia
    m_kg: float
    iz_kgm2: float

    # Vehicle geometry
    a_m: float          # CG to front axle [m]
    b_m: float          # CG to rear axle [m]
    track_f_m: float    # front track width [m]
    track_r_m: float    # rear track width [m]
    h_com_m: float      # center of mass height [m]
    h_rc_f_m: float     # front roll center height [m]
    h_rc_r_m: float     # rear roll center height [m]

    # Drag properties
    cd0_n: float        # rolling resistance [N]
    cd1_nspm: float     # linear drag [N*s/m]
    cd2_ns2pm2: float   # quadratic drag [N*s^2/m^2]

    # Actuator properties
    drive_f_frac: float
    drive_r_frac: float
    brake_f_frac: float
    brake_r_frac: float

    p_eng_max_w: float          # max engine power [W]
    max_delta_deg: float        # max steering angle [deg]
    max_delta_dot_degps: float  # max steering rate [deg/s]
    max_fx_kn: float            # max driving force [kN]
    min_fx_kn: float            # max braking force [kN] (negative)
    max_fx_dot_knps: float      # max force rate [kN/s]
    min_fx_dot_knps: float      # min force rate [kN/s]

    # Load transfer properties (from models-main)
    tau_long_weight_transfer_s: float   # longitudinal weight transfer time constant
    tau_lat_weight_transfer_s: float    # lateral weight transfer time constant
    roll_rate_radpmps2: float           # roll rate [rad/m/s^2]
    gamma_none: float                   # lateral load transfer distribution front/rear

    # Roll stiffness (optional, from paper)
    K_phi_f_knmprad: float = 81.0       # front roll stiffness [kN*m/rad]
    K_phi_r_knmprad: float = 38.0       # rear roll stiffness [kN*m/rad]

    # Miscellaneous (optional, with defaults)
    tire_radius_m: float = 0.33
    J_kgm2: float = 1.0                 # axle inertia
    braking_n_to_bar_front: float = 1.0
    braking_n_to_bar_rear: float = 1.0

    @property
    def l_m(self) -> float:
        """Wheelbase [m]."""
        return self.a_m + self.b_m

    @property
    def wf_n(self) -> float:
        """Front axle static weight [N]."""
        return self.m_kg * constants.g * (self.b_m / self.l_m)

    @property
    def wr_n(self) -> float:
        """Rear axle static weight [N]."""
        return self.m_kg * constants.g * (self.a_m / self.l_m)

    @property
    def max_delta_rad(self) -> float:
        return np.radians(self.max_delta_deg)

    @property
    def max_delta_dot_radps(self) -> float:
        return np.radians(self.max_delta_dot_degps)

    @property
    def track_avg_m(self) -> float:
        """Average track width [m]."""
        return 0.5 * (self.track_f_m + self.track_r_m)

    @property
    def h_rc_avg_m(self) -> float:
        """Roll center height at CG [m]."""
        return (self.h_rc_f_m * self.b_m + self.h_rc_r_m * self.a_m) / self.l_m

    @property
    def h_roll_arm_m(self) -> float:
        """Roll arm: vertical distance from CG to roll axis [m]."""
        return self.h_com_m - self.h_rc_avg_m

    @staticmethod
    def load_from_yaml(yaml_file: Union[str, Path]) -> VehicleParams:
        """
        Load vehicle parameters from YAML file.

        Args:
            yaml_file: Path to YAML config file

        Returns:
            VehicleParams instance
        """
        with open(yaml_file, "r") as stream:
            data = safe_load(stream)
        veh_dict = data["vehicle"]

        # Filter to only include fields that VehicleParams accepts
        valid_fields = {f.name for f in VehicleParams.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in veh_dict.items() if k in valid_fields}

        return VehicleParams(**filtered_dict)


# =============================================================================
# Single-Track Model - Unified Implementation
# =============================================================================

class SingleTrackModel:
    """
    Single track vehicle model - unified best-of-both-worlds.

    Combines:
    - models-main: lateral weight transfer, clean structure
    - multimodel: road geometry (grade, bank)

    State vectors:
    - Dynamics:  [ux, uy, r, dfz_long, dfz_lat] (5 states)
    - Global:    [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi] (9 states)
    - Path:      [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi] (8 states)

    Inputs: [delta, fx_kn] (2 inputs)

    Road geometry: [theta, phi, k_psi] (grade, bank, curvature)
    """

    # State indices for convenience
    IDX_UX = 0
    IDX_UY = 1
    IDX_R = 2
    IDX_DFZ_LONG = 3
    IDX_DFZ_LAT = 4

    def __init__(
        self,
        params: VehicleParams,
        f_tire: 'FialaBrushTire',
        r_tire: 'FialaBrushTire',
        enable_weight_transfer: bool = True
    ):
        """
        Initialize the model.

        Args:
            params: Vehicle parameters
            f_tire: Front tire model
            r_tire: Rear tire model
            enable_weight_transfer: Enable dynamic weight transfer (default True)
        """
        self.params = params
        self.f_tire = f_tire
        self.r_tire = r_tire
        self.enable_weight_transfer = enable_weight_transfer

    # -------------------------------------------------------------------------
    # Helper functions (from models-main)
    # -------------------------------------------------------------------------

    def smooth_fx_distro_kn(self, fx_kn):
        """
        Calculate smooth Fx distribution between axles.
        From models-main: smooth transition between drive and brake distribution.
        """
        df = self.params.drive_f_frac
        dr = self.params.drive_r_frac
        bf = self.params.brake_f_frac
        br = self.params.brake_r_frac

        diff_front = df - bf
        diff_rear = dr - br
        sum_front = df + bf
        sum_rear = dr + br

        # Smooth transition using tanh
        f_frac = -0.5 * diff_front * ca.tanh(-2.0 * (fx_kn + 0.5)) + 0.5 * sum_front
        r_frac = 0.5 * diff_rear * ca.tanh(2.0 * (fx_kn + 0.5)) + 0.5 * sum_rear

        fxf_kn = f_frac * fx_kn
        fxr_kn = r_frac * fx_kn

        return fxf_kn, fxr_kn

    def slip_angles(self, ux_mps, uy_mps, r_radps, delta_rad):
        """
        Calculate front and rear slip angles.
        """
        alpha_f_rad = ca.arctan2(uy_mps + self.params.a_m * r_radps, ux_mps) - delta_rad
        alpha_r_rad = ca.arctan2(uy_mps - self.params.b_m * r_radps, ux_mps)
        return alpha_f_rad, alpha_r_rad

    def normal_force_weight_transfer(self, dfz_long_kn, dfz_lat_kn=0.0):
        """
        Calculate axle normal loads with weight transfer.

        Args:
            dfz_long_kn: Longitudinal weight transfer [kN] (positive = rear loaded)
            dfz_lat_kn: Lateral weight transfer [kN] (positive = right loaded)

        Returns:
            fzf_kn, fzr_kn: Front and rear axle normal loads [kN]

        Note: For a single-track model, lateral weight transfer affects the
        effective grip but doesn't split front/rear. We use it in tire calcs.
        """
        fzf_kn = (self.params.wf_n / 1000.0) - dfz_long_kn
        fzr_kn = (self.params.wr_n / 1000.0) + dfz_long_kn
        return fzf_kn, fzr_kn

    def saturate_fx(self, fx_kn, fz_kn, alpha_rad, mu, eps=0.002):
        """
        Saturate longitudinal force by friction limit: |Fx| <= mu*Fz*cos(alpha)
        Uses smooth min/max formulation from models-main.
        """
        fx_max = mu * fz_kn * ca.cos(alpha_rad)

        # Smooth saturation
        fx_sat = 0.5 * (
            ca.sqrt((fx_kn + fx_max)**2 + eps**2)
            - ca.sqrt((fx_kn - fx_max)**2 + eps**2)
        )
        return fx_sat

    def _smooth_braking_only(self, fx_kn):
        """
        Extract braking-only component of longitudinal force using smooth softplus.

        Returns min(fx, 0) smoothly: -1/beta * log(1 + exp(-beta * fx))

        This is used for the brake yaw moment calculation, which only applies
        during braking (negative Fx).
        """
        return -1.0 / BETA_SOFTPLUS * ca.log(1.0 + ca.exp(-BETA_SOFTPLUS * fx_kn))

    def calc_brake_yaw_moment_kn_m(
        self,
        fxf_kn: float,
        fxr_kn: float,
        fzf_kn: float,
        fzr_kn: float,
        ay_mps2: float
    ):
        """
        Calculate brake yaw moment due to differential braking.

        From Aggarwal & Gerdes 2025, Appendix B:
        During braking while cornering, lateral weight transfer causes different
        normal loads on left/right wheels. Braking force is partitioned according
        to wheel loads, creating a net yaw moment.

        Args:
            fxf_kn: Front axle longitudinal force [kN]
            fxr_kn: Rear axle longitudinal force [kN]
            fzf_kn: Front axle normal load [kN]
            fzr_kn: Rear axle normal load [kN]
            ay_mps2: Lateral acceleration [m/s^2]

        Returns:
            mz_brake_kn_m: Brake yaw moment [kN-m]
        """
        p = self.params

        # Extract braking-only forces (smooth approximation of min(fx, 0))
        fxf_brake_kn = self._smooth_braking_only(fxf_kn)
        fxr_brake_kn = self._smooth_braking_only(fxr_kn)

        # Lateral weight transfer sensitivity [kN per m/s^2]
        # dfz_lat = (m * h_com / track + m * g * h_l * R_phi / track) * ay
        h_l = p.h_roll_arm_m
        t_avg = p.track_avg_m
        dfz_lat_sens = (p.m_kg * p.h_com_m / t_avg +
                        p.m_kg * G_MPS2 * h_l * p.roll_rate_radpmps2 / t_avg) / 1000.0

        # Steady-state lateral weight transfer [kN]
        dfz_lat_kn = dfz_lat_sens * ay_mps2

        # Brake yaw moment from front and rear axles
        # M_z = Fx_brake * gamma * track * dfz_lat / Fz
        gamma = p.gamma_none

        # Protect against division by zero with small normal loads
        fzf_safe = ca.fmax(fzf_kn, 0.1)
        fzr_safe = ca.fmax(fzr_kn, 0.1)

        mz_f_kn_m = fxf_brake_kn * gamma * t_avg * dfz_lat_kn / fzf_safe
        mz_r_kn_m = fxr_brake_kn * (1.0 - gamma) * t_avg * dfz_lat_kn / fzr_safe

        mz_brake_kn_m = mz_f_kn_m + mz_r_kn_m

        return mz_brake_kn_m

    # -------------------------------------------------------------------------
    # Core dynamics - UNIFIED (models-main + multimodel road geometry)
    # -------------------------------------------------------------------------

    def temporal_dynamics(
        self,
        ux_mps: float,
        uy_mps: float,
        r_radps: float,
        dfz_long_kn: float,
        dfz_lat_kn: float,
        delta_rad: float,
        fx_kn: float,
        theta_rad: float = 0.0,
        phi_rad: float = 0.0
    ):
        """
        Calculate temporal velocity state derivatives (dx/dt).

        UNIFIED: Combines models-main dynamics with multimodel road geometry.

        Args:
            ux_mps: longitudinal velocity [m/s]
            uy_mps: lateral velocity [m/s]
            r_radps: yaw rate [rad/s]
            dfz_long_kn: longitudinal weight transfer [kN]
            dfz_lat_kn: lateral weight transfer [kN]
            delta_rad: steering angle [rad]
            fx_kn: longitudinal force command [kN]
            theta_rad: road grade [rad] (positive = uphill) - FROM MULTIMODEL
            phi_rad: road bank [rad] (positive = banked right) - FROM MULTIMODEL

        Returns:
            Tuple: (dux, duy, dr, ddfz_long, ddfz_lat)
        """
        p = self.params

        # Slip angles
        alpha_f_rad, alpha_r_rad = self.slip_angles(ux_mps, uy_mps, r_radps, delta_rad)

        # Normal loads (with longitudinal weight transfer)
        fzf_kn, fzr_kn = self.normal_force_weight_transfer(dfz_long_kn)

        # Force distribution
        fxf_kn, fxr_kn = self.smooth_fx_distro_kn(fx_kn)

        # Saturate Fx by wheel-lock limit
        mu_f = self.f_tire.mu_none
        mu_r = self.r_tire.mu_none
        eps_f, eps_r = 0.002, 0.002

        fxf_kn = self.saturate_fx(fxf_kn, fzf_kn, alpha_f_rad, mu_f, eps_f)
        fxr_kn = self.saturate_fx(fxr_kn, fzr_kn, alpha_r_rad, mu_r, eps_r)

        # Lateral forces from tire model
        fyf_kn = self.f_tire.calc_fy_kn(alpha_f_rad, fzf_kn, fxf_kn)
        fyr_kn = self.r_tire.calc_fy_kn(alpha_r_rad, fzr_kn, fxr_kn)

        # Convert to Newtons
        fxf_n = fxf_kn * 1000.0
        fxr_n = fxr_kn * 1000.0
        fyf_n = fyf_kn * 1000.0
        fyr_n = fyr_kn * 1000.0

        # =====================================================================
        # Drag and road geometry forces (UNIFIED)
        # =====================================================================

        # Rolling resistance and aerodynamic drag (from models-main)
        frr_n = -p.cd0_n
        faero_n = -(p.cd1_nspm * ux_mps + p.cd2_ns2pm2 * ux_mps**2)

        # Road grade force (FROM MULTIMODEL) - gravity component along x
        # Positive theta = uphill = negative force (resists motion)
        f_grade_n = -p.m_kg * G_MPS2 * ca.sin(theta_rad)

        # Total longitudinal drag
        fd_n = frr_n + faero_n + f_grade_n

        # Road bank force (FROM MULTIMODEL) - gravity component along y
        # Positive phi = banked right = force pushing left (negative y in SAE)
        f_bank_n = -p.m_kg * G_MPS2 * ca.cos(theta_rad) * ca.sin(phi_rad)

        # =====================================================================
        # Accelerations
        # =====================================================================

        # Body-frame accelerations (for brake yaw moment calculation)
        ax_mps2 = (1 / p.m_kg) * (
            fxf_n * ca.cos(delta_rad) - fyf_n * ca.sin(delta_rad) + fxr_n + fd_n
        )
        ay_mps2 = (1 / p.m_kg) * (
            fyf_n * ca.cos(delta_rad) + fxf_n * ca.sin(delta_rad) + fyr_n + f_bank_n
        )

        # Velocity derivatives (include kinematic coupling terms)
        dux_mps2 = ax_mps2 + r_radps * uy_mps
        duy_mps2 = ay_mps2 - r_radps * ux_mps

        # =====================================================================
        # Brake yaw moment (FROM MULTIMODEL - Aggarwal & Gerdes 2025)
        # =====================================================================
        # During braking while cornering, differential braking creates a yaw moment
        # due to lateral weight transfer affecting left/right wheel loads
        mz_brake_kn_m = self.calc_brake_yaw_moment_kn_m(
            fxf_kn, fxr_kn, fzf_kn, fzr_kn, ay_mps2
        )
        mz_brake_n_m = mz_brake_kn_m * 1000.0  # Convert to N-m

        # Yaw acceleration (includes brake yaw moment)
        dr_radps2 = (1 / p.iz_kgm2) * (
            p.a_m * fyf_n * ca.cos(delta_rad)
            + p.a_m * fxf_n * ca.sin(delta_rad)
            - p.b_m * fyr_n
            + mz_brake_n_m  # Brake yaw moment
        )

        # =====================================================================
        # Weight transfer dynamics (from models-main, optionally enabled)
        # =====================================================================

        if self.enable_weight_transfer:
            # Total forces for weight transfer calculation
            fx_total_n = (fxf_n * ca.cos(delta_rad) - fyf_n * ca.sin(delta_rad)
                         + fxr_n + fd_n)
            fy_total_n = (fyf_n * ca.cos(delta_rad) + fxf_n * ca.sin(delta_rad)
                         + fyr_n + f_bank_n)

            # Longitudinal weight transfer dynamics
            # dfz_long_ss = (h_com / L) * Fx_total
            dfz_long_ss_kn = (p.h_com_m / p.l_m) * fx_total_n / 1000.0
            ddfz_long_knps = (1.0 / p.tau_long_weight_transfer_s) * (
                dfz_long_ss_kn - dfz_long_kn
            )

            # Lateral weight transfer dynamics
            # dfz_lat_ss = (h_com / track + g * h_roll_arm * roll_rate) * Fy_total
            h_l = p.h_roll_arm_m
            t_avg = p.track_avg_m
            dfz_lat_ss_kn = ((p.h_com_m / t_avg) + G_MPS2 * h_l * p.roll_rate_radpmps2) * fy_total_n / 1000.0
            ddfz_lat_knps = (1.0 / p.tau_lat_weight_transfer_s) * (
                dfz_lat_ss_kn - dfz_lat_kn
            )
        else:
            ddfz_long_knps = 0.0
            ddfz_lat_knps = 0.0

        return dux_mps2, duy_mps2, dr_radps2, ddfz_long_knps, ddfz_lat_knps

    def temporal_global_dynamics(
        self,
        ux_mps, uy_mps, r_radps, dfz_long_kn, dfz_lat_kn,
        t_s, east_m, north_m, psi_rad,
        delta_rad, fx_kn,
        theta_rad=0.0, phi_rad=0.0
    ):
        """
        Calculate temporal global state derivatives.

        State: [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi] (9 states)
        """
        # Core dynamics
        dux, duy, dr, ddfz_long, ddfz_lat = self.temporal_dynamics(
            ux_mps, uy_mps, r_radps, dfz_long_kn, dfz_lat_kn,
            delta_rad, fx_kn, theta_rad, phi_rad
        )

        # Time derivative
        dt = 1.0

        # Global position derivatives (NED convention from models-main)
        deast_mps = -ux_mps * ca.sin(psi_rad) - uy_mps * ca.cos(psi_rad)
        dnorth_mps = ux_mps * ca.cos(psi_rad) - uy_mps * ca.sin(psi_rad)
        dpsi_radps = r_radps

        return (dux, duy, dr, ddfz_long, ddfz_lat,
                dt, deast_mps, dnorth_mps, dpsi_radps)

    def temporal_path_dynamics(
        self,
        ux_mps, uy_mps, r_radps, dfz_long_kn, dfz_lat_kn,
        t_s, e_m, dpsi_rad,
        delta_rad, fx_kn,
        k_psi_1pm, theta_rad=0.0, phi_rad=0.0
    ):
        """
        Calculate temporal path state derivatives.

        State: [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi] (8 states)
        Additional inputs: k_psi (curvature), theta (grade), phi (bank)
        """
        # Core dynamics
        dux, duy, dr, ddfz_long, ddfz_lat = self.temporal_dynamics(
            ux_mps, uy_mps, r_radps, dfz_long_kn, dfz_lat_kn,
            delta_rad, fx_kn, theta_rad, phi_rad
        )

        # Time derivative
        dt = 1.0

        # Path coordinate derivatives
        ds_mps = (ux_mps * ca.cos(dpsi_rad) - uy_mps * ca.sin(dpsi_rad)) / (1 - e_m * k_psi_1pm)
        de_mps = ux_mps * ca.sin(dpsi_rad) + uy_mps * ca.cos(dpsi_rad)
        ddpsi_radps = r_radps - k_psi_1pm * ds_mps

        return (dux, duy, dr, ddfz_long, ddfz_lat,
                dt, de_mps, ddpsi_radps, ds_mps)

    def power_limit(self, ux_mps, fx_kn):
        """
        Calculate power constraint: P = Fx * ux <= P_max

        Returns:
            power_ub_kw: Power upper bound violation (positive = violation)
        """
        power_kw = fx_kn * ux_mps
        power_limit_kw = self.params.p_eng_max_w / 1000.0
        return power_kw - power_limit_kw

    # -------------------------------------------------------------------------
    # Vectorized convenience methods for optimization
    # -------------------------------------------------------------------------

    def dynamics_dt_path_vec(self, x, u, k_psi, theta=0.0, phi=0.0):
        """
        Vectorized path dynamics: dx/dt

        Args:
            x: State vector [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi] (8,)
            u: Control vector [delta, fx_kn] (2,)
            k_psi: Path curvature [1/m]
            theta: Road grade [rad]
            phi: Road bank [rad]

        Returns:
            dx_dt: State derivative vector (8,)
            s_dot: Arc length rate [m/s] (for converting to spatial)
        """
        derivs = self.temporal_path_dynamics(
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
            u[0], u[1], k_psi, theta, phi
        )
        # derivs = (dux, duy, dr, ddfz_long, ddfz_lat, dt, de, ddpsi, ds)
        dx_dt = ca.vertcat(derivs[0], derivs[1], derivs[2], derivs[3], derivs[4],
                           derivs[5], derivs[6], derivs[7])
        s_dot = derivs[8]
        return dx_dt, s_dot

    def dynamics_ds_path_vec(self, x, u, k_psi, theta=0.0, phi=0.0):
        """
        Vectorized spatial path dynamics: dx/ds

        Converts temporal to spatial using ds/dt = s_dot

        Returns:
            dx_ds: Spatial state derivative vector (8,)
        """
        dx_dt, s_dot = self.dynamics_dt_path_vec(x, u, k_psi, theta, phi)

        # Convert to spatial: dx/ds = (dx/dt) / (ds/dt)
        dx_ds = dx_dt / s_dot

        # Time derivative in spatial domain: dt/ds = 1/s_dot
        # This is already handled by the division

        return dx_ds

    def dynamics_dt_global_vec(self, x, u, theta=0.0, phi=0.0):
        """
        Vectorized global dynamics: dx/dt

        Args:
            x: State vector [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi] (9,)
            u: Control vector [delta, fx_kn] (2,)
            theta: Road grade [rad]
            phi: Road bank [rad]

        Returns:
            dx_dt: State derivative vector (9,)
        """
        derivs = self.temporal_global_dynamics(
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8],
            u[0], u[1], theta, phi
        )
        return ca.vertcat(*derivs)

    def __repr__(self):
        wt_status = "enabled" if self.enable_weight_transfer else "disabled"
        return f"SingleTrackModel({self.params.name}, weight_transfer={wt_status})"


# =============================================================================
# State vector definitions (for reference)
# =============================================================================

class StateIndex:
    """Index definitions for state vectors."""

    # Dynamics state [ux, uy, r, dfz_long, dfz_lat]
    class Dynamics:
        UX = 0
        UY = 1
        R = 2
        DFZ_LONG = 3
        DFZ_LAT = 4
        SIZE = 5

    # Path state [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi]
    class Path:
        UX = 0
        UY = 1
        R = 2
        DFZ_LONG = 3
        DFZ_LAT = 4
        T = 5
        E = 6
        DPSI = 7
        SIZE = 8

    # Global state [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi]
    class Global:
        UX = 0
        UY = 1
        R = 2
        DFZ_LONG = 3
        DFZ_LAT = 4
        T = 5
        EAST = 6
        NORTH = 7
        PSI = 8
        SIZE = 9

    # Control [delta, fx_kn]
    class Control:
        DELTA = 0
        FX = 1
        SIZE = 2
