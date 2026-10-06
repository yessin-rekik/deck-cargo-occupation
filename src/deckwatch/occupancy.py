"""Per-item 3D placement on the deck and frame occupancy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from deckwatch.config import GeometryConfig
from deckwatch.geometry import image_to_plane, measure_height, quad_dims, size_fit_height
from deckwatch.items import BASE_TO_TOP, TOP, Item

MIN_HEIGHT_M = 0.2
MIN_FOOTPRINT_M2 = 0.05


@dataclass(eq=False)
class PlacedItem:
    cls: str
    conf: float
    footprint: Polygon       # deck metres
    height_m: float
    height_source: str       # "measured" | "size_fit" | "assumed"
    stack_level: int
    length_m: float
    width_m: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "class": self.cls,
            "conf": round(self.conf, 3),
            "height_m": round(self.height_m, 2),
            "height_source": self.height_source,
            "stack_level": self.stack_level,
            "length_m": round(self.length_m, 2),
            "width_m": round(self.width_m, 2),
            "footprint_deck_m": [[round(x, 3), round(y, 3)] for x, y in self.footprint.exterior.coords[:-1]],
            "flags": list(self.flags),
        }


def estimate_height(item: Item, P: np.ndarray, geo: GeometryConfig, kp_conf_min: float):
    """Measured from a visible base point, else size-fit, else the class default."""
    flags: list[str] = []
    max_h = geo.max_stack * geo.unit_height_m
    visible = [(float(item.kp_conf[b]), b, t) for b, t in BASE_TO_TOP if item.kp_conf[b] >= kp_conf_min]
    if visible:
        _, b, t = max(visible)
        h = measure_height(P, item.keypoints[b], item.keypoints[t])
        if MIN_HEIGHT_M <= h <= max_h:
            return h, "measured", flags
        flags.append("height_outlier")
    h = size_fit_height(P, item.keypoints[list(TOP)], geo.standard_sizes, MIN_HEIGHT_M, max_h, geo.size_tol)
    if h is not None:
        return h, "size_fit", flags
    flags.append("height_assumed")
    return geo.default_height_m.get(item.cls, geo.unit_height_m), "assumed", flags


def place_item(item: Item, P: np.ndarray, geo: GeometryConfig, kp_conf_min: float) -> PlacedItem | None:
    h, source, flags = estimate_height(item, P, geo, kp_conf_min)
    corners = image_to_plane(P, h, item.keypoints[list(TOP)])
    if not np.all(np.isfinite(corners)):
        return None
    hull = MultiPoint([tuple(c) for c in corners]).convex_hull
    if not isinstance(hull, Polygon) or hull.area < MIN_FOOTPRINT_M2:
        return None
    rect = np.asarray(hull.minimum_rotated_rectangle.exterior.coords)[:4]
    length, width = quad_dims(rect)
    stack_level = max(1, int(round(h / geo.unit_height_m)))
    return PlacedItem(item.cls, float(item.conf), hull, float(h), source, stack_level, length, width, flags)


@dataclass(eq=False)
class Occupancy:
    items: list[PlacedItem]          # items whose footprint centroid lies in the cargo area
    union: BaseGeometry
    pct: float
    m2: float
    count_by_class: dict[str, int]
    count_by_stack_level: dict[str, int]


def compute_occupancy(placed: list[PlacedItem], cargo_area: Polygon) -> Occupancy:
    inside = [p for p in placed if cargo_area.contains(p.footprint.centroid)]
    union = unary_union([p.footprint for p in inside]).intersection(cargo_area) if inside else Polygon()
    by_class: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for p in inside:
        by_class[p.cls] = by_class.get(p.cls, 0) + 1
        by_level[str(p.stack_level)] = by_level.get(str(p.stack_level), 0) + 1
    m2 = float(union.area)
    return Occupancy(inside, union, 100.0 * m2 / cargo_area.area, m2, by_class, by_level)
