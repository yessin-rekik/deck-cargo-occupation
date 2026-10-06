"""Camera calibration: solve the 3x4 projection matrix P from the 8 deck points."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon

from deckwatch.config import DeckConfig

FLOOR_POINTS = ("F1", "F2", "F3", "F4")
WALL_POINTS = ("W1", "W2", "W3", "W4")


class CalibrationError(Exception):
    """Calibration input is insufficient or inconsistent."""


def corner_world_points(deck: DeckConfig) -> dict[str, np.ndarray]:
    w, l, h, o = deck.width_m, deck.length_m, deck.wall_height_m, deck.wall_offset_m
    return {
        "F1": np.array([0.0, l, 0.0]),
        "F2": np.array([w, l, 0.0]),
        "F3": np.array([w, 0.0, 0.0]),
        "F4": np.array([0.0, 0.0, 0.0]),
        "W1": np.array([-o, l, h]),
        "W2": np.array([w + o, l, h]),
        "W3": np.array([w + o, 0.0, h]),
        "W4": np.array([-o, 0.0, h]),
    }


def project_points(P: np.ndarray, world) -> np.ndarray:
    world = np.atleast_2d(np.asarray(world, float))
    hom = np.hstack([world, np.ones((len(world), 1))]) @ P.T
    return hom[:, :2] / hom[:, 2:3]


def reprojection_rms(P: np.ndarray, world, image) -> float:
    err = project_points(P, world) - np.asarray(image, float)
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def _similarity(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centre on the mean, scale mean distance to sqrt(dim)."""
    n = pts.shape[1]
    c = pts.mean(axis=0)
    d = np.mean(np.linalg.norm(pts - c, axis=1))
    s = np.sqrt(n) / d
    T = np.eye(n + 1)
    T[:n, :n] *= s
    T[:n, n] = -s * c
    return T


def solve_projection(world, image) -> np.ndarray:
    """Normalised DLT followed by Levenberg-Marquardt refinement of the reprojection error."""
    world = np.asarray(world, float)
    image = np.asarray(image, float)
    if len(world) < 6:
        raise CalibrationError(f"need >= 6 points to solve the camera, got {len(world)}")
    if np.ptp(world[:, 2]) == 0:
        raise CalibrationError("points are coplanar; need wall-top points above the deck")

    Ti, Tw = _similarity(image), _similarity(world)
    wn = np.hstack([world, np.ones((len(world), 1))]) @ Tw.T
    im = np.hstack([image, np.ones((len(image), 1))]) @ Ti.T
    rows = []
    for X, (u, v, _) in zip(wn, im):
        rows.append(np.concatenate([X, np.zeros(4), -u * X]))
        rows.append(np.concatenate([np.zeros(4), X, -v * X]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    P = np.linalg.inv(Ti) @ vt[-1].reshape(3, 4) @ Tw
    P = P / P[2, 3]

    def residuals(p11):
        return (project_points(np.append(p11, 1.0).reshape(3, 4), world) - image).ravel()

    fit = least_squares(residuals, P.ravel()[:11], method="lm")
    return np.append(fit.x, 1.0).reshape(3, 4)


@dataclass(frozen=True)
class CalPoint:
    name: str
    deck: tuple[float, float, float]
    image: tuple[float, float]
    status: str                  # "clicked" | "derived"
    reason: str | None = None    # for derived points: "out_of_frame" | "occluded"


@dataclass(frozen=True, eq=False)
class Calibration:
    version: int
    P: np.ndarray
    rms_px: float
    points: dict[str, CalPoint]
    cargo_area: Polygon          # deck metres, Z = 0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "P": self.P.tolist(),
            "rms_px": self.rms_px,
            "points": {
                p.name: {"deck": list(p.deck), "image": list(p.image), "status": p.status, "reason": p.reason}
                for p in self.points.values()
            },
            "cargo_area_deck": [list(c) for c in self.cargo_area.exterior.coords[:-1]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        points = {
            name: CalPoint(name, tuple(float(v) for v in p["deck"]), tuple(float(v) for v in p["image"]),
                           p["status"], p.get("reason"))
            for name, p in d["points"].items()
        }
        return cls(int(d["version"]), np.asarray(d["P"], float), float(d["rms_px"]), points,
                   Polygon(d["cargo_area_deck"]))


def build_calibration(deck: DeckConfig, clicks: dict, substitutes=(), reasons: dict | None = None,
                      version: int = 1, reproj_max_px: float = 3.0) -> Calibration:
    """Solve P from clicked points (None = not visible) plus substitute floor points ((X, Y), (u, v))."""
    reasons = reasons or {}
    world_pts = corner_world_points(deck)
    unknown = set(clicks) - set(world_pts)
    if unknown:
        raise CalibrationError(f"unknown point names: {sorted(unknown)}")

    world, image = [], []
    for name, uv in clicks.items():
        if uv is not None:
            world.append(world_pts[name])
            image.append(uv)
    for (x, y), uv in substitutes:
        world.append(np.array([x, y, 0.0]))
        image.append(uv)
    world = np.asarray(world, float).reshape(-1, 3)
    image = np.asarray(image, float).reshape(-1, 2)

    n_floor = int(np.sum(world[:, 2] == 0.0))
    n_wall = len(world) - n_floor
    if n_floor < 4:
        raise CalibrationError(f"need >= 4 floor-plane points (corners or substitutes), got {n_floor}")
    if n_wall < 2:
        raise CalibrationError(f"need >= 2 visible wall-top points, got {n_wall}")

    P = solve_projection(world, image)
    rms = reprojection_rms(P, world, image)
    if rms > reproj_max_px:
        raise CalibrationError(
            f"reprojection error {rms:.2f}px exceeds {reproj_max_px}px; check clicks and deck measurements"
        )

    points = {}
    for name in FLOOR_POINTS + WALL_POINTS:
        uv = clicks.get(name)
        deck_xyz = tuple(float(v) for v in world_pts[name])
        if uv is None:
            proj = project_points(P, world_pts[name])[0]
            points[name] = CalPoint(name, deck_xyz, (float(proj[0]), float(proj[1])), "derived",
                                    reasons.get(name, "occluded"))
        else:
            points[name] = CalPoint(name, deck_xyz, (float(uv[0]), float(uv[1])), "clicked")

    cargo_area = Polygon([tuple(world_pts[n][:2]) for n in ("F4", "F3", "F2", "F1")])
    return Calibration(version, P, rms, points, cargo_area)
