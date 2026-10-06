import numpy as np
import pytest

from deckwatch.calibration import (
    Calibration,
    CalibrationError,
    build_calibration,
    corner_world_points,
    project_points,
    solve_projection,
)
from synth import DECK, DECK_L, DECK_W, default_camera, project


def exact_clicks(P):
    return {name: tuple(project(P, xyz)[0]) for name, xyz in corner_world_points(DECK).items()}


def test_corner_world_points_follow_deck_frame():
    pts = corner_world_points(DECK)
    np.testing.assert_allclose(pts["F4"], (0, 0, 0))
    np.testing.assert_allclose(pts["F1"], (0, DECK_L, 0))
    np.testing.assert_allclose(pts["F2"], (DECK_W, DECK_L, 0))
    np.testing.assert_allclose(pts["W3"], (DECK_W, 0, 3.0))


def test_solve_projection_recovers_camera():
    P_true = default_camera()
    rng = np.random.default_rng(0)
    world = rng.uniform([0, 0, 0], [DECK_W, DECK_L, 6], size=(20, 3))
    P = solve_projection(world, project(P_true, world))
    test_pts = rng.uniform([0, 0, 0], [DECK_W, DECK_L, 6], size=(10, 3))
    np.testing.assert_allclose(project_points(P, test_pts), project(P_true, test_pts), atol=0.5)


def test_solve_projection_rejects_coplanar_points():
    world = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0]], float)
    with pytest.raises(CalibrationError, match="coplanar"):
        solve_projection(world, world[:, :2] * 100)


def test_build_calibration_with_all_points_visible():
    P_true = default_camera()
    cal = build_calibration(DECK, exact_clicks(P_true))
    assert cal.rms_px < 0.5
    assert all(p.status == "clicked" for p in cal.points.values())
    assert cal.cargo_area.area == pytest.approx(DECK_W * DECK_L)


def test_hidden_floor_corner_is_derived_from_substitute():
    P_true = default_camera()
    clicks = exact_clicks(P_true)
    clicks["F1"] = None
    sub = ((10.0, 30.0), tuple(project(P_true, (10.0, 30.0, 0.0))[0]))
    cal = build_calibration(DECK, clicks, substitutes=[sub], reasons={"F1": "out_of_frame"})
    f1 = cal.points["F1"]
    assert f1.status == "derived" and f1.reason == "out_of_frame"
    np.testing.assert_allclose(f1.image, project(P_true, (0, DECK_L, 0))[0], atol=0.5)


def test_too_few_floor_points_refused():
    clicks = exact_clicks(default_camera())
    clicks["F1"] = None
    with pytest.raises(CalibrationError, match="floor"):
        build_calibration(DECK, clicks)


def test_too_few_wall_points_refused():
    clicks = exact_clicks(default_camera())
    clicks["W1"] = clicks["W2"] = clicks["W3"] = None
    with pytest.raises(CalibrationError, match="wall"):
        build_calibration(DECK, clicks)


def test_bad_click_refused_by_reprojection_error():
    clicks = exact_clicks(default_camera())
    u, v = clicks["F1"]
    clicks["F1"] = (u + 100.0, v)
    with pytest.raises(CalibrationError, match="reprojection"):
        build_calibration(DECK, clicks)


def test_unknown_point_name_refused():
    clicks = exact_clicks(default_camera())
    clicks["X9"] = (1.0, 1.0)
    with pytest.raises(CalibrationError, match="unknown"):
        build_calibration(DECK, clicks)


def test_dict_round_trip():
    cal = build_calibration(DECK, exact_clicks(default_camera()))
    back = Calibration.from_dict(cal.to_dict())
    np.testing.assert_allclose(back.P, cal.P)
    assert back.points["W2"] == cal.points["W2"]
    assert back.cargo_area.equals(cal.cargo_area)
    assert back.version == cal.version
