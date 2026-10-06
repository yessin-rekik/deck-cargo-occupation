import numpy as np
import pytest
from shapely.geometry import box

from deckwatch.config import GeometryConfig, StandardSize
from deckwatch.items import Item
from deckwatch.occupancy import compute_occupancy, place_item
from synth import DECK_L, DECK_W, box_keypoints, default_camera

GEO = GeometryConfig(standard_sizes=(StandardSize("10ft", 2.99, 2.44), StandardSize("20ft", 6.06, 2.44)))
CARGO = box(0, 0, DECK_W, DECK_L)
P = default_camera()


def make_item(x0, y0, lx=6.06, wy=2.44, h=2.6, cls="container", base_conf=0.9):
    kp = box_keypoints(P, x0, y0, lx, wy, h)
    return Item(cls, 0.9, kp, np.array([0.9] * 4 + [base_conf] * 2))


def test_measured_height_and_true_footprint():
    p = place_item(make_item(4.0, 10.0), P, GEO, 0.5)
    assert p.height_source == "measured"
    assert p.height_m == pytest.approx(2.6, abs=1e-6)
    assert p.stack_level == 1
    assert p.length_m == pytest.approx(6.06, abs=1e-3) and p.width_m == pytest.approx(2.44, abs=1e-3)
    assert p.footprint.area == pytest.approx(6.06 * 2.44, rel=1e-4)
    assert p.flags == []


def test_hidden_base_uses_size_fit():
    p = place_item(make_item(4.0, 30.0, base_conf=0.1), P, GEO, 0.5)
    assert p.height_source == "size_fit"
    assert p.height_m == pytest.approx(2.6, abs=0.051)


def test_hidden_base_non_standard_uses_class_default():
    p = place_item(make_item(4.0, 30.0, lx=4.0, wy=4.0, h=2.6, base_conf=0.1), P, GEO, 0.5)
    assert p.height_source == "assumed"
    assert p.height_m == 2.6
    assert "height_assumed" in p.flags


def test_stacked_container_is_level_two():
    p = place_item(make_item(4.0, 10.0, h=5.2), P, GEO, 0.5)
    assert p.stack_level == 2
    assert p.footprint.area == pytest.approx(6.06 * 2.44, rel=1e-4)


def test_outlier_height_falls_back():
    item = make_item(4.0, 10.0)
    kp = item.keypoints.copy()
    kp[4], kp[5] = kp[3], kp[2]           # base points collapsed onto top corners -> h = 0
    p = place_item(Item("container", 0.9, kp, item.kp_conf), P, GEO, 0.5)
    assert "height_outlier" in p.flags
    assert p.height_source == "size_fit"


def test_degenerate_item_dropped():
    kp = np.tile(np.array([[1000.0, 1000.0]]), (6, 1))
    assert place_item(Item("container", 0.9, kp, np.full(6, 0.1)), P, GEO, 0.5) is None


def test_crossed_keypoint_order_uses_hull():
    item = make_item(4.0, 10.0)
    kp = item.keypoints.copy()
    kp[[0, 1]] = kp[[1, 0]]               # far corners swapped -> self-intersecting order
    p = place_item(Item("container", 0.9, kp, item.kp_conf), P, GEO, 0.5)
    assert p.footprint.area == pytest.approx(6.06 * 2.44, rel=1e-3)


def test_stack_counts_once_and_clipping():
    a = place_item(make_item(2.0, 5.0), P, GEO, 0.5)
    stacked = place_item(make_item(2.0, 5.0, h=5.2), P, GEO, 0.5)
    edge = place_item(make_item(16.0, 40.0), P, GEO, 0.5)     # spans x 16..22.06, centroid inside
    occ = compute_occupancy([a, stacked, edge], CARGO)
    expected = 6.06 * 2.44 + 4.0 * 2.44
    assert occ.m2 == pytest.approx(expected, rel=1e-3)
    assert occ.pct == pytest.approx(100 * expected / 1200, rel=1e-3)
    assert occ.count_by_class == {"container": 3}
    assert occ.count_by_stack_level == {"1": 2, "2": 1}


def test_item_outside_cargo_area_not_counted():
    outside = place_item(make_item(25.0, 10.0), P, GEO, 0.5)
    occ = compute_occupancy([outside], CARGO)
    assert occ.items == [] and occ.pct == 0.0


def test_empty_occupancy():
    occ = compute_occupancy([], CARGO)
    assert occ.pct == 0.0 and occ.m2 == 0.0 and occ.union.is_empty
    assert occ.count_by_class == {} and occ.count_by_stack_level == {}


def test_to_dict_shape():
    d = place_item(make_item(4.0, 10.0), P, GEO, 0.5).to_dict()
    assert set(d) == {"class", "conf", "height_m", "height_source", "stack_level", "length_m",
                      "width_m", "footprint_deck_m", "flags"}
    assert len(d["footprint_deck_m"]) == 4
