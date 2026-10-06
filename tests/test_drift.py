import numpy as np
import pytest

from deckwatch.config import DriftConfig
from deckwatch.drift import check_drift, extract_refs, load_refs, save_refs

CFG = DriftConfig(refs_dir=None)
CENTERS = {"W1": (800.0, 500.0), "W2": (3000.0, 500.0), "W3": (3200.0, 1700.0)}


@pytest.fixture(scope="module")
def frame():
    return np.random.default_rng(0).integers(0, 255, (2160, 3840), dtype=np.uint8)


def test_same_frame_is_ok(frame):
    status, shift = check_drift(frame, extract_refs(frame, CENTERS, 48), CFG)
    assert status == "ok" and shift == pytest.approx(0.0, abs=0.5)


def test_shifted_frame_is_suspected(frame):
    refs = extract_refs(frame, CENTERS, 48)
    status, shift = check_drift(np.roll(frame, 25, axis=1), refs, CFG)
    assert status == "suspected" and shift == pytest.approx(25.0, abs=1.0)


def test_occluded_refs_make_check_unverified(frame):
    refs = extract_refs(frame, CENTERS, 48)
    occluded = frame.copy()
    other = np.random.default_rng(1).integers(0, 255, (300, 300), dtype=np.uint8)
    for name in ("W1", "W2"):
        x, y = map(int, CENTERS[name])
        occluded[y - 150:y + 150, x - 150:x + 150] = other
    assert check_drift(occluded, refs, CFG) == ("unverified", None)


def test_no_refs_is_unverified(frame):
    assert check_drift(frame, [], CFG) == ("unverified", None)


def test_border_point_skipped(frame):
    assert [r.name for r in extract_refs(frame, {"W1": (5.0, 5.0)}, 48)] == []


def test_save_and_load_refs(frame, tmp_path):
    refs = extract_refs(frame, CENTERS, 48)
    save_refs(refs, tmp_path / "refs")
    loaded = load_refs(tmp_path / "refs", CENTERS)
    assert [r.name for r in loaded] == ["W1", "W2", "W3"]
    np.testing.assert_array_equal(loaded[0].patch, refs[0].patch)
    assert load_refs(tmp_path / "missing", CENTERS) == []
