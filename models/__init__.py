"""
Vehicle dynamics models - faithful adaptation from models-main.

Reference: Aggarwal & Gerdes, "Friction-Robust Autonomous Racing Using Trajectory
Optimization Over Multiple Models", IEEE Open Journal of Control Systems, 2025.
"""

from pathlib import Path
from typing import Union

from .vehicle import VehicleParams, SingleTrackModel
from .tire import FialaBrushTire

__all__ = ['VehicleParams', 'SingleTrackModel', 'FialaBrushTire', 'load_vehicle_from_yaml']


def load_vehicle_from_yaml(
    yaml_file: Union[str, Path],
    enable_weight_transfer: bool = True
) -> SingleTrackModel:
    """
    Load complete vehicle model (vehicle + tires) from a YAML config file.

    Args:
        yaml_file: Path to YAML config file (e.g., models/config/vehicle_params_gti.yaml)
        enable_weight_transfer: Enable dynamic weight transfer (default True)

    Returns:
        SingleTrackModel: Complete vehicle model ready for simulation/optimization

    Example:
        >>> from models import load_vehicle_from_yaml
        >>> vehicle = load_vehicle_from_yaml("models/config/vehicle_params_gti.yaml")
        >>> print(vehicle)
        SingleTrackModel(VW_Golf_GTI, weight_transfer=enabled)
    """
    params = VehicleParams.load_from_yaml(yaml_file)
    f_tire = FialaBrushTire.load_from_yaml(yaml_file, "front")
    r_tire = FialaBrushTire.load_from_yaml(yaml_file, "rear")

    return SingleTrackModel(params, f_tire, r_tire, enable_weight_transfer)
