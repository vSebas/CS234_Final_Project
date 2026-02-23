"""
Fiala Brush Tire Model - Faithful adaptation from models-main

Removes casadi_tools dependency while preserving exact equations.

Reference: Subosits & Gerdes, "Impacts of Model Fidelity on Trajectory Optimization
for Autonomous Vehicles in Extreme Maneuvers", IEEE T-IV 2021.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import casadi as ca
from yaml import safe_load


def _clamp_val(x, lb, ub, eps=1e-6):
    """
    Smooth clamp function: clamp(x, lb, ub)

    Replaces casadi_tools.math.clamp_val
    Uses smooth min/max formulation.
    """
    # smooth_max(x, lb) then smooth_min(result, ub)
    # smooth_max(a, b) = 0.5 * (a + b + sqrt((a-b)^2 + eps))
    # smooth_min(a, b) = 0.5 * (a + b - sqrt((a-b)^2 + eps))

    # First clamp to lower bound
    x_lb = 0.5 * (x + lb + ca.sqrt((x - lb)**2 + eps**2))

    # Then clamp to upper bound
    x_clamped = 0.5 * (x_lb + ub - ca.sqrt((x_lb - ub)**2 + eps**2))

    return x_clamped


@dataclass
class FialaBrushTire:
    """
    Modified Fiala Brush Tire model.

    Faithful adaptation of models-main/models/fiala_brush_tire.py
    without casadi_tools dependency.

    The cornering stiffness is load-dependent:
        C_alpha = c0_alpha + c1_alpha * Fz

    Attributes:
        c0_alpha_nprad: Base cornering stiffness [N/rad]
        c1_alpha_1prad: Load-dependent cornering stiffness [1/rad] (slope)
        mu_none: Friction coefficient [-]
        fy_xi: Saturation parameter (default 0.95)
        max_allowed_fx_frac: Max Fx as fraction of friction circle (default 0.99)
    """

    c0_alpha_nprad: float   # Base cornering stiffness [N/rad]
    c1_alpha_1prad: float   # Load sensitivity [1/rad per kN]
    mu_none: float          # Friction coefficient
    fy_xi: float = 0.95     # Saturation parameter (paper: xi = 0.95)
    max_allowed_fx_frac: float = 0.99  # Fy_max scaling (paper: rho = 0.99)

    def calc_max_fy_kn(self, fz_kn, fx_kn):
        """
        Calculate maximum lateral force from friction circle.

        Fy_max = sqrt((mu * Fz)^2 - (0.99 * Fx)^2)

        Matches models-main/fiala_brush_tire.py:calc_max_fy_kn
        """
        return ca.sqrt(
            (self.mu_none * fz_kn)**2
            - (self.max_allowed_fx_frac * fx_kn)**2
        )

    def calc_alpha_sl_rad(self, fz_kn, fx_kn):
        """
        Calculate saturation (sliding) slip angle.

        alpha_slide = atan2(3 * Fy_max * xi, C_alpha)

        Matches models-main/fiala_brush_tire.py:calc_alpha_sl_rad
        """
        c_alpha_knprad = self.c0_alpha_nprad / 1000 + self.c1_alpha_1prad * fz_kn
        fy_max_kn = self.calc_max_fy_kn(fz_kn, fx_kn)
        return ca.atan2(3 * fy_max_kn * self.fy_xi, c_alpha_knprad)

    def calc_fy_kn(self, alpha_rad, fz_kn, fx_kn):
        """
        Calculate lateral tire force using Fiala brush model.

        Matches models-main/fiala_brush_tire.py:calc_fy_kn exactly.

        Args:
            alpha_rad: Slip angle [rad]
            fz_kn: Normal load [kN]
            fx_kn: Longitudinal force [kN]

        Returns:
            fy_kn: Lateral force [kN]
        """
        # Load-dependent cornering stiffness
        c_alpha_knprad = self.c0_alpha_nprad / 1000 + self.c1_alpha_1prad * fz_kn
        mu = self.mu_none

        # Clamp Fx to friction limit
        max_fx_abs_kn = mu * fz_kn * ca.cos(alpha_rad)
        fx_kn_clamped = _clamp_val(fx_kn, -max_fx_abs_kn, max_fx_abs_kn)

        # Maximum lateral force and saturation angle
        fy_max_kn = self.calc_max_fy_kn(fz_kn, fx_kn_clamped)
        alpha_slide_rad = self.calc_alpha_sl_rad(fz_kn, fx_kn_clamped)

        tan_alpha = ca.tan(alpha_rad)

        # Unsaturated region (cubic model)
        fy_unsat_kn = (
            -(c_alpha_knprad * tan_alpha)
            + ((c_alpha_knprad**2) / (3 * fy_max_kn) * tan_alpha * ca.fabs(tan_alpha))
            - ((c_alpha_knprad**3) / (27 * fy_max_kn**2) * tan_alpha**3)
        )

        # Saturated region
        xi = self.fy_xi
        fy_sat_kn = (
            -c_alpha_knprad * (1 - 2*xi + xi**2) * tan_alpha
            - fy_max_kn * (3*xi**2 - 2*xi**3) * ca.sign(alpha_rad)
        )

        # Switch based on slip angle
        return ca.if_else(ca.fabs(alpha_rad) < alpha_slide_rad, fy_unsat_kn, fy_sat_kn)

    def calc_max_fx_kn(self, alpha_rad, fz_kn):
        """
        Calculate maximum longitudinal force at given slip angle.

        Fx_max = mu * Fz * cos(alpha)

        Matches models-main/fiala_brush_tire.py:calc_max_fx_kn
        """
        return self.mu_none * fz_kn * ca.cos(alpha_rad)

    @staticmethod
    def load_from_yaml(yaml_file: Union[str, Path], select_tire: str) -> 'FialaBrushTire':
        """
        Load tire parameters from YAML file.

        Args:
            yaml_file: Path to YAML file
            select_tire: "front" or "rear"

        Returns:
            FialaBrushTire instance
        """
        with open(yaml_file, "r") as stream:
            data = safe_load(stream)

        # Support both "tire_front"/"tire_rear" and "front_tire"/"rear_tire" formats
        key = f"tire_{select_tire}"
        if key not in data:
            key = f"{select_tire}_tire"

        tire_dict = data[key]

        # Filter to only include fields that FialaBrushTire accepts
        valid_fields = {'c0_alpha_nprad', 'c1_alpha_1prad', 'mu_none', 'fy_xi', 'max_allowed_fx_frac'}
        filtered_dict = {k: v for k, v in tire_dict.items() if k in valid_fields}

        return FialaBrushTire(**filtered_dict)
