import numpy as np
import pytest

from deckwatch.config import StandardSize
from deckwatch.geometry import (
    image_to_plane,
    measure_height,
    plane_homography,
    plane_to_image,
    quad_dims,
    size_fit_height,
)
from synth import box_keypoints, default_camera, project

SIZES = (StandardSize("10ft", 2.99, 2.44), StandardSize("20ft", 6.06, 2.44))


def test_plane_homography_matches_full_projection():
    P = default_camera()
    xy = np.array([[3.0, 7.0], [15.0, 50.0]])
    for h in (0.0, 2.6, 5.2):
        expected = project(P, np.column_stack([xy, np.full(2, h)]))
        np.testing.assert_allclose(plane_to_image(P, h, xy), expected, atol=1e-6)
    assert plane_homography(P, 0.0).shape == (3, 3)


def test_image_to_plane_round_trip():
    P = default_camera()
    xy = np.array([[1.0, 2.0], [19.0, 58.0], [10.0, 30.0]])
    np.testing.assert_allclose(image_to_plane(P, 2.6, plane_to_image(P, 2.6, xy)), xy, atol=1e-6)


@pytest.mark.parametrize("h", [2.6, 5.2, 1.0])
def test_measure_height_exact(h):
    P = default_camera()
    kp = box_keypoints(P, 5.0, 20.0, 6.06, 2.44, h)
    assert measure_height(P, kp[4], kp[3]) == pytest.approx(h, abs=1e-6)
    assert measure_height(P, kp[5], kp[2]) == pytest.approx(h, abs=1e-6)


def test_measure_height_with_one_pixel_noise():
    P = default_camera()
    kp = box_keypoints(P, 5.0, 40.0, 6.06, 2.44, 2.6)
    assert measure_height(P, kp[4], kp[3] + np.array([1.0, -1.0])) == pytest.approx(2.6, abs=0.15)


def test_quad_dims_of_rectangle():
    quad = np.array([[0, 2.44], [6.06, 2.44], [6.06, 0], [0, 0]])
    length, width = quad_dims(quad)
    assert length == pytest.approx(6.06) and width == pytest.approx(2.44)


def test_size_fit_finds_height_of_standard_container():
    P = default_camera()
    kp = box_keypoints(P, 4.0, 25.0, 6.06, 2.44, 2.6)
    h = size_fit_height(P, kp[:4], SIZES, 0.2, 7.8, 0.15)
    assert h == pytest.approx(2.6, abs=0.051)


def test_size_fit_returns_none_for_non_standard_item():
    P = default_camera()
    kp = box_keypoints(P, 4.0, 25.0, 4.0, 4.0, 1.5)
    assert size_fit_height(P, kp[:4], SIZES, 0.2, 7.8, 0.15) is None


def test_size_fit_without_sizes_returns_none():
    P = default_camera()
    kp = box_keypoints(P, 4.0, 25.0, 6.06, 2.44, 2.6)
    assert size_fit_height(P, kp[:4], (), 0.2, 7.8, 0.15) is None
