from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import constants
from yaml import safe_load


@dataclass(frozen=True)
class VehicleParams:
    """
    Class with derived vehicle parameters.
    Instances of this class are immutable.
    """

    name: str

    # Mass and inertia
    m_kg: float
    iz_kgm2: float

    # Vehicle geometry
    a_m: float
    b_m: float
    track_f_m: float
    track_r_m: float
    h_com_m: float
    h_rc_f_m: float
    h_rc_r_m: float

    # Drag properties
    cd0_n: float
    cd1_nspm: float
    cd2_ns2pm2: float

    # Actuator properties
    drive_f_frac: float
    drive_r_frac: float
    brake_f_frac: float
    brake_r_frac: float

    p_eng_max_w: float
    max_delta_deg: float
    max_delta_dot_degps: float
    max_fx_kn: float
    min_fx_kn: float
    max_fx_dot_knps: float
    min_fx_dot_knps: float
    braking_n_to_bar_front: float # per caliper
    braking_n_to_bar_rear: float # per caliper

    # Load transfer properties
    tau_long_weight_transfer_s: float
    tau_lat_weight_transfer_s: float
    roll_rate_radpmps2: float
    gamma_none: float

    # Miscellaneous
    tire_radius_m: float
    J_kgm2: float # axle inertia

    @property
    def l_m(self):
        """Calculate wheelbase [m]."""
        return self.a_m + self.b_m

    @property
    def wf_n(self) -> float:
        """Calculate front axle static weight [N]."""
        return self.m_kg * constants.g * (self.b_m / self.l_m)

    @property
    def wr_n(self) -> float:
        """Calculate rear axle static weight [N]."""
        return self.m_kg * constants.g * (self.a_m / self.l_m)

    @property
    def max_delta_rad(self) -> float:
        return np.radians(self.max_delta_deg)

    @property
    def max_delta_dot_radps(self) -> float:
        return np.radians(self.max_ddelta_degps)

    @staticmethod
    def load_from_yaml(yaml_file: Path) -> VehicleParams:
        """
        Load vehicle parameters from yaml.
        """

        veh_dict = _load_params_dict_from_yaml(yaml_file)
        return VehicleParams(**veh_dict)


def _load_params_dict_from_yaml(yaml_file: Path) -> dict:
    with open(yaml_file, "r") as stream:
        veh_dict = safe_load(stream)
    return veh_dict["vehicle"]
