import numpy as np

from synth import DECK_L, DECK_W, IMAGE_SIZE, WALL_H, default_camera, project


def test_target_projects_to_principal_point():
    P = default_camera()
    np.testing.assert_allclose(project(P, [(DECK_W / 2, DECK_L / 2, 0.0)])[0], (1920, 1080), atol=1e-6)


def test_deck_and_wall_corners_are_inside_image():
    P = default_camera()
    pts = [(x, y, z) for x in (0, DECK_W) for y in (0, DECK_L) for z in (0, WALL_H)]
    uv = project(P, pts)
    assert np.all(uv[:, 0] > 0) and np.all(uv[:, 0] < IMAGE_SIZE[0])
    assert np.all(uv[:, 1] > 0) and np.all(uv[:, 1] < IMAGE_SIZE[1])


def test_far_edge_is_higher_in_image_than_near_edge():
    P = default_camera()
    far, near = project(P, [(0, DECK_L, 0), (0, 0, 0)])
    assert far[1] < near[1]
