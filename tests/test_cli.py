import json
import os
from datetime import timezone

import pytest

import deckwatch.cli as cli
from deckwatch.config import load_config
from deckwatch.pipeline import Pipeline, PipelineError
from deckwatch.store import Store
from deckwatch.timeutil import to_iso
from synth import FakeDetector, jpeg_bytes, make_calibration, write_config


def test_parse_ts_from_filename_is_utc(tmp_path):
    f = tmp_path / "after_20260918_152541.jpg"
    f.write_bytes(b"x")
    assert to_iso(cli.parse_ts(f, None, "UTC")) == "2026-09-18T15:25:41Z"
    assert to_iso(cli.parse_ts(f, None, "Asia/Tehran")) == "2026-09-18T11:55:41Z"


def test_parse_ts_explicit_wins(tmp_path):
    f = tmp_path / "after_20260918_152541.jpg"
    f.write_bytes(b"x")
    assert to_iso(cli.parse_ts(f, "2026-09-18T18:55:37+03:30", "UTC")) == "2026-09-18T15:25:37Z"
    with pytest.raises(PipelineError) as e:
        cli.parse_ts(f, "yesterday", "UTC")
    assert e.value.code == "invalid_timestamp"


def test_parse_ts_falls_back_to_mtime(tmp_path):
    f = tmp_path / "frame.jpg"
    f.write_bytes(b"x")
    os.utime(f, (1_790_000_000, 1_790_000_000))
    dt = cli.parse_ts(f, None, "UTC")
    assert dt.tzinfo is not None and dt.timestamp() == 1_790_000_000
    assert dt.astimezone(timezone.utc).year == 2026


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = write_config(tmp_path, make_calibration())

    def fake_build(p):
        cfg = load_config(p)
        return Pipeline(cfg, FakeDetector(), Store(cfg.db_path))

    monkeypatch.setattr(cli, "build_pipeline", fake_build)
    return path


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_analyze_then_status_and_operations(tmp_path, config_path, capsys):
    frame = tmp_path / "periodic_20260920_080000.jpg"
    frame.write_bytes(jpeg_bytes())
    code, out = run(capsys, "--config", str(config_path), "analyze", str(frame))
    assert code == 0 and out["ts"] == "2026-09-20T08:00:00Z" and out["occupancy_pct"] == 0.0
    code, out = run(capsys, "--config", str(config_path), "status")
    assert code == 0 and out["state"] == "IDLE" and out["latest_frame"]["ts"] == "2026-09-20T08:00:00Z"
    code, out = run(capsys, "--config", str(config_path), "operations")
    assert code == 0 and out == []


def test_corrupt_frame_exits_2(tmp_path, config_path, capsys):
    frame = tmp_path / "periodic_20260920_080000.jpg"
    frame.write_bytes(b"garbage")
    code, out = run(capsys, "--config", str(config_path), "analyze", str(frame))
    assert code == 2 and out["error"] == "unreadable_image"


def test_missing_frame_exits_2(tmp_path, config_path, capsys):
    code, out = run(capsys, "--config", str(config_path), "analyze", str(tmp_path / "nope.jpg"))
    assert code == 2 and out["error"] == "unreadable_image"


def test_missing_config_exits_3(tmp_path, capsys):
    code, out = run(capsys, "--config", str(tmp_path / "nope.yaml"), "status")
    assert code == 3 and out["error"] == "config_error"


def test_invalid_filename_tz_exits_3(tmp_path, capsys):
    path = write_config(tmp_path, make_calibration())
    path.write_text(path.read_text(encoding="utf-8") + "filename_tz: Mars/Olympus\n", encoding="utf-8")
    code, out = run(capsys, "--config", str(path), "status")
    assert code == 3 and out["error"] == "config_error"


def test_invalid_filename_date_exits_2(tmp_path, config_path, capsys):
    frame = tmp_path / "periodic_20261399_999999.jpg"
    frame.write_bytes(jpeg_bytes())
    code, out = run(capsys, "--config", str(config_path), "analyze", str(frame))
    assert code == 2 and out["error"] == "invalid_timestamp"


def test_unexpected_exception_exits_1_with_json(tmp_path, config_path, capsys, monkeypatch):
    frame = tmp_path / "periodic_20260920_080000.jpg"
    frame.write_bytes(jpeg_bytes())

    class Boom:
        config = type("C", (), {"filename_tz": "UTC"})()

        def analyze(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "build_pipeline", lambda p: Boom())
    code = cli.main(["--config", str(config_path), "analyze", str(frame)])
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert code == 1
    assert out == {"error": "internal_error", "message": "boom"}
    assert "RuntimeError" in captured.err
