import threading
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


def test_malformed_calibration_raises_config_error(tmp_path):
    cal = make_calibration()
    bad = cal.to_dict()
    del bad["points"]["F1"]["deck"]
    cfg = load_config(write_config(tmp_path))
    cfg = type(cfg)(**{**cfg.__dict__, "calibration": bad})
    with pytest.raises(ConfigError, match="calibration section is invalid"):
        Pipeline(cfg, FakeDetector(), Store(cfg.db_path))


def test_frame_lookup(setup):
    pipe, _ = setup
    pipe.analyze(jpeg_bytes(), at(0))
    assert pipe.frame("2026-09-20T08:00:00Z")["ts"] == "2026-09-20T08:00:00Z"
    assert pipe.frame("2026-09-20T09:00:00Z") is None


@pytest.fixture
def two_pipelines(tmp_path):
    cfg = load_config(write_config(tmp_path, make_calibration()))
    det_a, det_b = FakeDetector(), FakeDetector()
    pipe_a = Pipeline(cfg, det_a, Store(cfg.db_path))
    pipe_b = Pipeline(cfg, det_b, Store(cfg.db_path))
    return pipe_a, det_a, pipe_b, det_b


def test_same_ts_from_two_processes_is_idempotent(two_pipelines):
    pipe_a, det_a, pipe_b, det_b = two_pipelines
    det_a.items = det_b.items = [container(2.0, 5.0)]
    r1 = pipe_a.analyze(jpeg_bytes(), at(0))
    r2 = pipe_b.analyze(jpeg_bytes(), at(0))
    assert r1 == r2
    assert pipe_a.store.conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 1


def test_two_writers_interleaved_keep_state_consistent(two_pipelines):
    pipe_a, det_a, pipe_b, det_b = two_pipelines
    det_a.items = det_b.items = [container(2.0, 5.0)]
    pipe_a.analyze(jpeg_bytes(), at(0))
    pipe_b.analyze(jpeg_bytes(), at(3))
    det_a.items = det_b.items = [container(2.0, 5.0), container(2.0, 12.0)]
    pipe_a.analyze(jpeg_bytes(), at(6))
    r = pipe_b.analyze(jpeg_bytes(), at(9))
    assert r["state"] == "ACTIVE"
    assert r["operation"]["event"] == "T1" and r["operation"]["t1"] == "2026-09-20T08:06:00Z"


def test_concurrent_writers_never_crash(tmp_path):
    cfg = load_config(write_config(tmp_path, make_calibration()))
    pipe_a = Pipeline(cfg, FakeDetector(), Store(cfg.db_path))
    pipe_b = Pipeline(cfg, FakeDetector(), Store(cfg.db_path))
    timestamps = [at(m) for m in range(0, 24, 3)]  # 8 distinct increasing timestamps
    errors = []
    err_lock = threading.Lock()

    def worker(pipe, ts_list):
        for ts in ts_list:
            try:
                pipe.analyze(jpeg_bytes(), ts)
            except BaseException as exc:  # capture anything; asserted below
                with err_lock:
                    errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(pipe_a, timestamps[0::2])),
        threading.Thread(target=worker, args=(pipe_b, timestamps[1::2])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for exc in errors:
        assert isinstance(exc, PipelineError) and exc.code == "out_of_order", repr(exc)

    store = Store(cfg.db_path)
    assert store.load_state()["last_ts"] == store.last_ts()


def test_op_ids_continue_after_state_loss(setup):
    pipe, det = setup
    det.items = [container(2.0, 5.0)]
    pipe.analyze(jpeg_bytes(), at(0))
    pipe.analyze(jpeg_bytes(), at(3))
    det.items = [container(2.0, 5.0), container(2.0, 12.0)]
    pipe.analyze(jpeg_bytes(), at(6))
    r = pipe.analyze(jpeg_bytes(), at(9))
    assert r["operation"]["id"] == 1

    pipe.store.conn.execute("DELETE FROM state")

    det.items = [container(2.0, 5.0), container(2.0, 12.0)]
    r2 = pipe.analyze(jpeg_bytes(), at(20))
    assert r2["operation"] is None

    det.items = [container(2.0, 5.0)]
    r3 = pipe.analyze(jpeg_bytes(), at(40))
    assert r3["operation"]["event"] == "T2"
    assert r3["operation"]["id"] == 2
