from __future__ import annotations

from pathlib import Path

import casadi as ca
from casadi_tools import math
from casadi_tools.nlp_utils import casadi_builder as cb
from yaml import safe_load


@cb.casadi_dataclass
class FialaBrushTire:
    """Class implementing a modified Fiala Brush Tire model."""

    c0_alpha_nprad: float
    c1_alpha_1prad: float
    mu_none: float

    fy_xi: float = 0.90
    max_allowed_fx_frac: float = 0.99

    @cb.casadi_method((1, 1))
    def calc_max_fy_kn(self, fz_kn, fx_kn):
        return ca.sqrt((self.mu_none * fz_kn) ** 2 - (self.max_allowed_fx_frac*fx_kn)**2)
    
    @cb.casadi_method((1, 1))
    def calc_alpha_sl_rad(self, fz_kn, fx_kn):
        c_alpha_knprad = self.c0_alpha_nprad/1000 + self.c1_alpha_1prad*fz_kn
        fy_max_kn = self.calc_max_fy_kn(fz_kn, fx_kn)
        return ca.atan2(3 * fy_max_kn * self.fy_xi, c_alpha_knprad)

    @cb.casadi_method((1, 1, 1))
    def calc_fy_kn(self, alpha_rad, fz_kn, fx_kn):
        c_alpha_knprad = self.c0_alpha_nprad/1000 + self.c1_alpha_1prad*fz_kn
        mu = self.mu_none

        max_fx_abs_kn = mu * fz_kn * ca.cos(alpha_rad)
        fx_kn = math.clamp_val(fx_kn, -max_fx_abs_kn, max_fx_abs_kn)

        fy_max_kn = self.calc_max_fy_kn(fz_kn, fx_kn)
        alpha_slide_rad = self.calc_alpha_sl_rad(fz_kn, fx_kn)
        tan_alpha = ca.tan(alpha_rad)

        fy_unsat_kn = (
            - (c_alpha_knprad * tan_alpha)
            + ((c_alpha_knprad**2) / (3 * fy_max_kn) * tan_alpha * ca.fabs(tan_alpha))
            - ((c_alpha_knprad**3) / (27 * fy_max_kn**2) * tan_alpha**3)
        )
        fy_sat_kn = (-c_alpha_knprad * (1 - 2*self.fy_xi + self.fy_xi**2)*tan_alpha
                     -fy_max_kn*(3*self.fy_xi**2 - 2*self.fy_xi**3)*ca.sign(alpha_rad))

        return ca.if_else(ca.fabs(alpha_rad) < alpha_slide_rad, fy_unsat_kn, fy_sat_kn)

    @cb.casadi_method((1, 1))
    def calc_max_fx_kn(self, alpha_rad, fz_kn):
        return self.mu_none * fz_kn * ca.cos(alpha_rad)

    
    @staticmethod
    def load_from_yaml(yaml_file: Path, select_tire: str) -> FialaBrushTire:
        tire_dict = FialaBrushTire._load_tires_dict_from_yaml(yaml_file, select_tire)
        tire = FialaBrushTire(**tire_dict)
        return tire

    @staticmethod
    def _load_tires_dict_from_yaml(yaml_file: Path, select_tire: str) -> dict:
        with open(yaml_file, "r") as stream:
            tire_dict = safe_load(stream)

        return tire_dict[select_tire + "_tire"]