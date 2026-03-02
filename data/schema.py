"""
Dataset schema helpers for trajectory optimization episodes.

This module defines small utilities to build episode headers and compute
canonical per-step arrays (reward/rtg).
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def sha256_json(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def compute_rtg(reward: np.ndarray) -> np.ndarray:
    """Return-to-go from per-step rewards."""
    return np.flip(np.cumsum(np.flip(reward)))


@dataclass
class EpisodeHeader:
    episode_id: str
    episode_type: str
    map_id: str
    map_hash: str
    base_id: str
    solver_config: Dict[str, Any]
    solver_config_hash: str
    discretization: Dict[str, Any]
    obstacles: List[Dict[str, Any]]
    s_offset_m: float
    npz_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
