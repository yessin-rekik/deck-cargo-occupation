"""Single-view metrology on the calibrated deck: planes at height h, item heights, size fit."""
from __future__ import annotations

import numpy as np

from deckwatch.config import StandardSize


def plane_homography(P: np.ndarray, h: float) -> np.ndarray:
    """Homography mapping deck (X, Y) on the horizontal plane Z = h to image pixels."""
    return np.column_stack([P[:, 0], P[:, 1], h * P[:, 2] + P[:, 3]])


def _apply(H: np.ndarray, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def plane_to_image(P: np.ndarray, h: float, xy) -> np.ndarray:
    return _apply(plane_homography(P, h), xy)


def image_to_plane(P: np.ndarray, h: float, uv) -> np.ndarray:
    return _apply(np.linalg.inv(plane_homography(P, h)), uv)


def measure_height(P: np.ndarray, base_uv, top_uv) -> float:
    """Height of the point imaged at top_uv, which lies vertically above the deck point at base_uv.

    The base fixes (X, Y) on the deck; projecting (X, Y, h) must land on top_uv, which is linear
    in h for each image coordinate. Solved in closed form by least squares.
    """
    x, y = image_to_plane(P, 0.0, base_uv)[0]
    xb = np.array([x, y, 0.0, 1.0])
    u, v = np.asarray(top_uv, float)
    c = np.array([u * P[2, 2] - P[0, 2], v * P[2, 2] - P[1, 2]])
    d = np.array([P[0] @ xb - u * (P[2] @ xb), P[1] @ xb - v * (P[2] @ xb)])
    return float(c @ d / (c @ c))


def quad_dims(quad) -> tuple[float, float]:
    """(length, width) of a quadrilateral as the means of opposite sides, length >= width."""
    q = np.asarray(quad, float)
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    a = (sides[0] + sides[2]) / 2
    b = (sides[1] + sides[3]) / 2
    return float(max(a, b)), float(min(a, b))


def size_fit_height(P: np.ndarray, top_uv, sizes: tuple[StandardSize, ...], h_min: float, h_max: float,
                    tol: float, step: float = 0.05) -> float | None:
    """Height at which the projected top face best matches a standard footprint, or None."""
    if not sizes:
        return None
    best_h, best_err = None, np.inf
    for h in np.arange(h_min, h_max + step / 2, step):
        length, width = quad_dims(image_to_plane(P, h, top_uv))
        for s in sizes:
            err = max(abs(length - s.length_m) / s.length_m, abs(width - s.width_m) / s.width_m)
            if err < best_err:
                best_h, best_err = float(h), err
    return best_h if best_err <= tol else None
