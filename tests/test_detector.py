import json

import numpy as np
import pytest

from deckwatch.config import ConfigError, DetectorConfig
from deckwatch.detector import decode, letterbox, load_detector

NAMES = ["container", "other"]


def column(cx, cy, w, h, scores, kp_xy=(0.0, 0.0), vis=0.9):
    kps = []
    for k in range(6):
        kps += [kp_xy[0] + k, kp_xy[1] + k, vis]
    return [cx, cy, w, h, *scores, *kps]


def test_letterbox_4k_frame():
    img = np.zeros((2160, 3840, 3), np.uint8)
    out, r, pad = letterbox(img, 1280)
    assert out.shape == (1280, 1280, 3)
    assert r == pytest.approx(1 / 3) and pad == (0, 280)
    assert out[0, 0, 0] == 114 and out[640, 640, 0] == 0


def test_decode_filters_nms_and_unletterboxes():
    cols = [
        column(640, 640, 100, 100, (0.9, 0.05), kp_xy=(100, 300)),
        column(645, 640, 100, 100, (0.8, 0.05)),        # overlaps the first -> suppressed
        column(200, 200, 50, 50, (0.05, 0.2)),          # below conf_min -> dropped
    ]
    raw = np.array(cols, np.float32).T[None]            # (1, 24, 3)
    items = decode(raw, 0.5, (0, 100), NAMES, 0.4, 0.5)
    assert len(items) == 1
    it = items[0]
    assert it.cls == "container" and it.conf == pytest.approx(0.9)
    np.testing.assert_allclose(it.keypoints[0], (200.0, 400.0))
    np.testing.assert_allclose(it.keypoints[5], ((100 + 5) / 0.5, (300 + 5 - 100) / 0.5))
    np.testing.assert_allclose(it.kp_conf, np.full(6, 0.9), rtol=1e-6)


def test_decode_empty_and_wrong_width():
    assert decode(np.zeros((1, 24, 0), np.float32), 1.0, (0, 0), NAMES, 0.4, 0.5) == []
    with pytest.raises(ValueError, match="output width"):
        decode(np.zeros((1, 20, 3), np.float32), 1.0, (0, 0), NAMES, 0.4, 0.5)


def test_load_detector_missing_model(tmp_path):
    with pytest.raises(ConfigError, match="model not found"):
        load_detector(DetectorConfig(model_path=tmp_path / "m.onnx"))


def test_load_detector_rejects_wrong_keypoints(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"")
    (tmp_path / "m.json").write_text(json.dumps(
        {"model_version": "x", "class_names": NAMES, "keypoint_names": ["a"], "imgsz": 1280}), encoding="utf-8")
    with pytest.raises(ConfigError, match="keypoint"):
        load_detector(DetectorConfig(model_path=tmp_path / "m.onnx"))
