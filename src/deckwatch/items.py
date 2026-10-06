"""Detector output type and the fixed keypoint layout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEYPOINT_NAMES = (
    "top_far_left",
    "top_far_right",
    "top_near_right",
    "top_near_left",
    "base_near_left",
    "base_near_right",
)
TOP = (0, 1, 2, 3)
BASE_TO_TOP = ((4, 3), (5, 2))   # (base keypoint, top keypoint directly above it)
FLIP_IDX = (1, 0, 3, 2, 5, 4)


@dataclass(frozen=True, eq=False)
class Item:
    cls: str
    conf: float
    keypoints: np.ndarray   # (6, 2) raw image pixels, KEYPOINT_NAMES order
    kp_conf: np.ndarray     # (6,) keypoint visibility confidence, 0..1
