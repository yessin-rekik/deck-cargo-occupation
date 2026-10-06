from deckwatch.store import Store


def result(ts, occ=1.0):
    return {"ts": ts, "occupancy_pct": occ, "change_pct": 0.0, "flags": [], "calibration_version": 1,
            "model_version": "m1", "state": "IDLE", "operation": None}


def test_empty_store(tmp_path):
    s = Store(tmp_path / "d.db")
    assert s.last_ts() is None and s.load_state() is None and s.latest_frame() is None
    assert s.status() == {"state": "IDLE", "latest_frame": None, "open_operation": None}


def test_commit_and_read_back(tmp_path):
    s = Store(tmp_path / "d.db")
    s.commit_frame(result("2026-09-20T08:00:00Z"), [{"class": "container"}], {"mode": "IDLE"}, None, "a.jpg")
    assert s.last_ts() == "2026-09-20T08:00:00Z"
    assert s.get_frame("2026-09-20T08:00:00Z")["occupancy_pct"] == 1.0
    assert "items" not in s.get_frame("2026-09-20T08:00:00Z")
    assert s.get_frame("2026-09-20T08:00:00Z", include_items=True)["items"] == [{"class": "container"}]
    assert s.get_frame("2026-09-20T09:00:00Z") is None


def test_state_and_operations_persist_across_instances(tmp_path):
    path = tmp_path / "d.db"
    s = Store(path)
    op = {"id": 1, "status": "open", "t1": "2026-09-20T08:03:00Z", "t2": None}
    s.commit_frame(result("2026-09-20T08:03:00Z"), [], {"mode": "ACTIVE"}, op)
    s.commit_frame(result("2026-09-20T08:06:00Z"), [], None, {**op, "status": "closed", "t2": "2026-09-20T08:03:00Z"})
    s.close()
    s2 = Store(path)
    assert s2.load_state() == {"mode": "ACTIVE"}
    ops = s2.list_operations()
    assert len(ops) == 1 and ops[0]["status"] == "closed"
    assert s2.list_operations(since="2026-09-21T00:00:00Z") == []


def test_status_reports_open_operation(tmp_path):
    s = Store(tmp_path / "d.db")
    op = {"id": 1, "status": "open", "t1": "2026-09-20T08:03:00Z", "t2": None}
    s.commit_frame(result("2026-09-20T08:03:00Z"), [], {"mode": "ACTIVE"}, op)
    st = s.status()
    assert st["state"] == "ACTIVE" and st["open_operation"]["id"] == 1
    assert st["latest_frame"]["ts"] == "2026-09-20T08:03:00Z"
