from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from deckwatch.api import create_app
from deckwatch.config import load_config
from deckwatch.pipeline import Pipeline
from deckwatch.store import Store
from synth import FakeDetector, jpeg_bytes, make_calibration, write_config


@pytest.fixture
def client(tmp_path):
    cfg = load_config(write_config(tmp_path, make_calibration()))
    return TestClient(create_app(Pipeline(cfg, FakeDetector(), Store(cfg.db_path))))


def post(client, ts, data=None, items=False):
    return client.post("/frames" + ("?items=true" if items else ""),
                       files={"file": ("f.jpg", data if data is not None else jpeg_bytes(), "image/jpeg")},
                       data={"ts": ts})


def test_post_frame_and_read_back(client):
    r = post(client, "2026-09-20T08:00:00Z", items=True)
    assert r.status_code == 200
    body = r.json()
    assert body["ts"] == "2026-09-20T08:00:00Z" and body["items"] == []
    assert client.get("/frames/2026-09-20T08:00:00Z").json()["ts"] == "2026-09-20T08:00:00Z"
    assert client.get("/status").json()["latest_frame"]["ts"] == "2026-09-20T08:00:00Z"
    assert client.get("/operations").json() == []


def test_naive_ts_uses_filename_tz(client):
    assert post(client, "2026-09-20T08:00:00").json()["ts"] == "2026-09-20T08:00:00Z"


def test_errors_map_to_http_status(client):
    assert post(client, "2026-09-20T08:00:00Z", data=b"garbage").status_code == 422
    assert post(client, "not-a-time").status_code == 422
    post(client, "2026-09-20T08:03:00Z")
    r = post(client, "2026-09-20T08:00:00Z")
    assert r.status_code == 409 and r.json()["error"] == "out_of_order"
    r = client.get("/frames/2026-09-21T00:00:00Z")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_concurrent_posts_never_500(client):
    stamps = [f"2026-09-20T08:{m:02d}:00Z" for m in range(0, 24, 3)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(lambda t: post(client, t).status_code, stamps))
    assert set(codes) <= {200, 409}
    assert 200 in codes


def test_frame_lookup_normalises_ts(client):
    post(client, "2026-09-20T08:00:00Z")
    assert client.get("/frames/2026-09-20T11:30:00+03:30").json()["ts"] == "2026-09-20T08:00:00Z"
    assert client.get("/frames/2026-09-20T08:00:00").status_code == 200
    assert client.get("/frames/not-a-time").status_code == 422


def test_unexpected_exception_returns_json_500(tmp_path):
    cfg = load_config(write_config(tmp_path, make_calibration()))

    class Boom:
        config = cfg

        def analyze(self, *a, **k):
            raise RuntimeError("boom")

    client = TestClient(create_app(Boom()), raise_server_exceptions=False)
    r = post(client, "2026-09-20T08:00:00Z")
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "message": "boom"}


def test_oversized_upload_returns_413(client, monkeypatch):
    import deckwatch.api as api

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    r = post(client, "2026-09-20T08:00:00Z", data=b"x" * 20)
    assert r.status_code == 413
    assert r.json()["error"] == "payload_too_large"
