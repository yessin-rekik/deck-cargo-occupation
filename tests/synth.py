"""Synthetic camera and deck used across the test suite (no real frames needed)."""
import numpy as np

DECK_W = 20.0   # cargo area width (X, across)
DECK_L = 60.0   # cargo area length (Y, away from camera)
WALL_H = 3.0
IMAGE_SIZE = (3840, 2160)


def look_at_camera(center, target, f=3000.0, cx=1920.0, cy=1080.0):
    """Pinhole camera P = K [R | t] at `center` looking at `target`, image y pointing down."""
    center = np.asarray(center, float)
    target = np.asarray(target, float)
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ center
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return K @ np.hstack([R, t[:, None]])


def project(P, world):
    world = np.atleast_2d(np.asarray(world, float))
    hom = np.hstack([world, np.ones((len(world), 1))]) @ P.T
    return hom[:, :2] / hom[:, 2:3]


def default_camera():
    """Camera 40 m behind the near edge, 30 m up, aimed at the deck centre."""
    return look_at_camera((DECK_W / 2, -40.0, 30.0), (DECK_W / 2, DECK_L / 2, 0.0))


def box_keypoints(P, x0, y0, lx, wy, h):
    """Image keypoints K0..K5 of an axis-aligned box on [x0,x0+lx] x [y0,y0+wy], top at h."""
    world = [
        (x0, y0 + wy, h),        # K0 top_far_left
        (x0 + lx, y0 + wy, h),   # K1 top_far_right
        (x0 + lx, y0, h),        # K2 top_near_right
        (x0, y0, h),             # K3 top_near_left
        (x0, y0, 0.0),           # K4 base_near_left
        (x0 + lx, y0, 0.0),      # K5 base_near_right
    ]
    return project(P, world)


from deckwatch.config import DeckConfig  # noqa: E402

DECK = DeckConfig(length_m=DECK_L, width_m=DECK_W, wall_height_m=WALL_H, wall_offset_m=0.0)


def make_calibration(P=None):
    """Calibration built from exact clicks of all 8 points under camera P."""
    from deckwatch.calibration import build_calibration, corner_world_points

    P = default_camera() if P is None else P
    clicks = {name: tuple(project(P, xyz)[0]) for name, xyz in corner_world_points(DECK).items()}
    return build_calibration(DECK, clicks)


def write_config(tmp_path, calibration=None):
    import yaml

    data = {
        "image_size": list(IMAGE_SIZE),
        "filename_tz": "UTC",
        "db_path": "deckwatch.db",
        "deck": {"length_m": DECK_L, "width_m": DECK_W, "wall_height_m": WALL_H, "wall_offset_m": 0.0},
        "detector": {"model_path": "model.onnx"},
        "geometry": {"standard_sizes": [{"name": "10ft", "length_m": 2.99, "width_m": 2.44},
                                        {"name": "20ft", "length_m": 6.06, "width_m": 2.44}]},
    }
    if calibration is not None:
        data["calibration"] = calibration.to_dict()
    path = tmp_path / "deck.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class FakeDetector:
    model_version = "fake-v1"

    def __init__(self):
        self.items = []
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.items)


def jpeg_bytes(size=IMAGE_SIZE):
    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((size[1], size[0], 3), np.uint8))
    assert ok
    return buf.tobytes()
