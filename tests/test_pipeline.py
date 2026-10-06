from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from deckwatch.config import ConfigError, load_config
from deckwatch.items import Item
from deckwatch.pipeline import Pipeline, PipelineError
from deckwatch.store import Store
from synth import FakeDetector, box_keypoints, default_camera, jpeg_bytes, make_calibration, write_config

T0 = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
P = default_camera()


def container(x0, y0, h=2.6):
    return Item("container", 0.9, box_keypoints(P, x0, y0, 6.06, 2.44, h), np.full(6, 0.9))


@pytest.fixture
def setup(tmp_path):
    cfg = load_config(write_config(tmp_path, make_calibration()))
    det = FakeDetector()
    pipe = Pipeline(cfg, det, Store(cfg.db_path))
    return pipe, det


def at(minute):
    return T0 + timedelta(minutes=minute)


def test_empty_deck_frame(setup):
    pipe, _ = setup
    r = pipe.analyze(jpeg_bytes(), at(0))
    assert r["occupancy_pct"] == 0.0 and r["state"] == "IDLE" and r["operation"] is None
    assert r["count_container"] == 0 and r["flags"] == ["drift_unverified"]
    assert r["ts"] == "2026-09-20T08:00:00Z" and r["model_version"] == "fake-v1"


def test_one_container_with_items(setup):
    pipe, det = setup
    det.items = [container(4.0, 10.0)]
    r = pipe.analyze(jpeg_bytes(), at(0), include_items=True)
    assert r["occupancy_pct"] == pytest.approx(100 * 6.06 * 2.44 / 1200, abs=0.01)
    assert r["count_container"] == 1 and r["count_by_stack_level"] == {"1": 1}
    assert r["items"][0]["height_source"] == "measured"


def test_duplicate_ts_returns_stored_result_without_detecting(setup):
    pipe, det = setup
    first = pipe.analyze(jpeg_bytes(), at(0))
    again = pipe.analyze(jpeg_bytes(), at(0))
    assert again == first and det.calls == 1


def test_out_of_order_rejected(setup):
    pipe, _ = setup
    pipe.analyze(jpeg_bytes(), at(3))
    with pytest.raises(PipelineError) as e:
        pipe.analyze(jpeg_bytes(), at(0))
    assert e.value.code == "out_of_order" and e.value.http_status == 409


def test_unreadable_and_wrong_resolution_store_nothing(setup):
    pipe, _ = setup
    with pytest.raises(PipelineError) as e:
        pipe.analyze(b"not a jpeg", at(0))
    assert e.value.code == "unreadable_image" and e.value.http_status == 422
    with pytest.raises(PipelineError) as e:
        pipe.analyze(jpeg_bytes((640, 480)), at(0))
    assert e.value.code == "wrong_resolution"
    assert pipe.store.last_ts() is None


def test_operation_events_flow_through(setup):
    pipe, det = setup
    det.items = [container(2.0, 5.0)]
    pipe.analyze(jpeg_bytes(), at(0))
    pipe.analyze(jpeg_bytes(), at(3))
    det.items = [container(2.0, 5.0), container(2.0, 12.0)]
    pipe.analyze(jpeg_bytes(), at(6))
    r = pipe.analyze(jpeg_bytes(), at(9))
    assert r["state"] == "ACTIVE"
    assert r["operation"]["event"] == "T1" and r["operation"]["t1"] == "2026-09-20T08:06:00Z"
    assert pipe.status()["open_operation"]["id"] == 1
    assert len(pipe.operations()) == 1


def test_drift_suspected_frame_does_not_drive_operations(setup, monkeypatch):
    pipe, det = setup
    det.items = [container(2.0, 5.0)]
    pipe.analyze(jpeg_bytes(), at(0))
    monkeypatch.setattr("deckwatch.pipeline.check_drift", lambda *a: ("suspected", 30.0))
    det.items = [container(2.0, 5.0), container(2.0, 12.0)]
    r1 = pipe.analyze(jpeg_bytes(), at(3))
    r2 = pipe.analyze(jpeg_bytes(), at(6))
    assert "drift_suspected" in r1["flags"] and r1["change_pct"] is None
    assert r2["operation"] is None and r2["state"] == "IDLE"


def test_missing_calibration_raises(tmp_path):
    cfg = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="calibrat"):
        Pipeline(cfg, FakeDetector(), Store(cfg.db_path))


def test_frame_lookup(setup):
    pipe, _ = setup
    pipe.analyze(jpeg_bytes(), at(0))
    assert pipe.frame("2026-09-20T08:00:00Z")["ts"] == "2026-09-20T08:00:00Z"
    assert pipe.frame("2026-09-20T09:00:00Z") is None
