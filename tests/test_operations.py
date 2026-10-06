import json
from datetime import datetime, timedelta, timezone

import pytest
from shapely.geometry import box

from deckwatch.config import OperationsConfig
from deckwatch.operations import FrameObs, OperationTracker, match_items
from deckwatch.timeutil import from_iso, parse_timestamp, to_iso

T0 = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
AREA = 1200.0
A, B, C, D = (2, 5), (2, 10), (10, 10), (10, 20)


def item(pos, h=2.6):
    x, y = pos
    return (box(x, y, x + 6.0, y + 2.5), h)     # 15 m² each


def obs(minute, *items):
    return FrameObs(T0 + timedelta(minutes=minute), 100.0 * sum(fp.area for fp, _ in items) / AREA, list(items))


def run(frames):
    """Step through frames, JSON round-tripping state each time (simulates a fresh process)."""
    tracker = OperationTracker(OperationsConfig(), AREA, height_tol=1.3)
    state, updates = None, []
    for f in frames:
        state, upd = tracker.step(state, f)
        state = json.loads(json.dumps(state))
        updates.append(upd)
    return updates


def ts(minute):
    return to_iso(T0 + timedelta(minutes=minute))


def events(updates):
    return [(i, u.event) for i, u in enumerate(updates) if u.event]


def test_timeutil_round_trip_and_naive_rejected():
    assert to_iso(T0) == "2026-09-20T08:00:00Z"
    assert from_iso("2026-09-20T08:00:00Z") == T0
    assert parse_timestamp("2026-09-20T11:30:00+03:30") == T0
    assert parse_timestamp("2026-09-20T08:00:00") == T0
    with pytest.raises(ValueError):
        to_iso(datetime(2026, 9, 20))
    with pytest.raises(ValueError):
        parse_timestamp("not a time")


def test_match_items_ignores_corner_noise_but_not_height():
    before = [item(A)]
    noisy = [(box(2.2, 5.1, 8.1, 7.4), 2.6)]
    assert match_items(before, noisy, 0.3, 1.3) == ([], [])
    appeared, disappeared = match_items(before, [item(A, h=5.2)], 0.3, 1.3)
    assert len(appeared) == 1 and len(disappeared) == 1


def test_first_frame_is_baseline():
    ups = run([obs(0, item(A), item(B), item(C))])
    assert ups[0].state == "IDLE" and ups[0].event is None and ups[0].change_pct == 0.0


def test_stable_deck_has_no_events():
    assert events(run([obs(m, item(A)) for m in range(0, 60, 3)])) == []


def test_single_change_opens_and_closes_operation():
    frames = [obs(0, item(A)), obs(3, item(A))] + [obs(m, item(A), item(B)) for m in range(6, 39, 3)]
    ups = run(frames)
    assert events(ups) == [(3, "T1"), (12, "T2")]       # T1 confirmed at minute 9, T2 at minute 36
    t1 = ups[3].operation
    assert t1["status"] == "open" and t1["t1"] == ts(6) and t1["t1_lower_bound"] == ts(3)
    assert t1["occupancy_before"] == 1.25
    op = ups[12].operation
    assert op["status"] == "closed" and op["t2"] == ts(6)
    assert op["occupancy_after"] == 2.5 and op["net_pp"] == 1.25
    assert op["items_appeared"] == 1 and op["items_disappeared"] == 0
    assert ups[12].state == "IDLE"


def test_one_frame_flicker_is_ignored():
    frames = [obs(0, item(A)), obs(3, item(A), item(B)), obs(6, item(A)), obs(9, item(A), item(B)), obs(12, item(A))]
    ups = run(frames)
    assert events(ups) == []
    assert all(u.state == "IDLE" for u in ups)


def test_swap_with_equal_area_is_an_operation():
    frames = [obs(0, item(A)), obs(3, item(A))] + [obs(m, item(B)) for m in range(6, 39, 3)]
    ups = run(frames)
    op = ups[12].operation
    assert ups[12].event == "T2"
    assert op["net_pp"] == 0.0 and op["items_appeared"] == 1 and op["items_disappeared"] == 1


def test_stacking_on_same_footprint_is_an_operation():
    frames = [obs(0, item(A)), obs(3, item(A))] + [obs(m, item(A, h=5.2)) for m in range(6, 39, 3)]
    ups = run(frames)
    assert events(ups) == [(3, "T1"), (12, "T2")]
    assert ups[12].operation["net_pp"] == 0.0


def test_long_operation_ends_after_quiet_period():
    frames = [obs(0, item(A)), obs(3, item(A)), obs(6, item(A), item(B)), obs(9, item(A), item(B)),
              obs(12, item(A), item(B), item(C))]
    frames += [obs(m, item(A), item(B), item(C), item(D)) for m in range(15, 48, 3)]
    ups = run(frames)
    assert events(ups) == [(3, "T1"), (15, "T2")]       # T2 event at minute 45 (index 15)
    op = ups[15].operation
    assert op["t1"] == ts(6) and op["t2"] == ts(15) and op["items_appeared"] == 3
    assert all(u.state == "ACTIVE" for u in ups[3:15])


def test_gap_while_idle_creates_uncertain_operation():
    ups = run([obs(0, item(A)), obs(3, item(A)), obs(63, item(A), item(B))])
    assert ups[2].event == "T2"
    op = ups[2].operation
    assert op["t1"] == op["t2"] == ts(63) and op["t1_lower_bound"] == ts(3)
    assert op["flags"] == ["t1_t2_uncertain"]
    assert ups[2].state == "IDLE"


def test_gap_while_active_is_flagged():
    frames = [obs(0, item(A)), obs(3, item(A)), obs(6, item(A), item(B)), obs(9, item(A), item(B)),
              obs(30, item(A), item(B), item(C))]
    frames += [obs(m, item(A), item(B), item(C)) for m in range(33, 63, 3)]
    ups = run(frames)
    assert "gap_during_operation" in ups[4].operation["flags"]
    assert ups[-1].event == "T2" and ups[-1].operation["t2"] == ts(30)


def test_change_in_confirming_frame_extends_t2():
    frames = [obs(0, item(A)), obs(3, item(A)), obs(6, item(A), item(B))]
    frames += [obs(m, item(A), item(B), item(C)) for m in range(9, 42, 3)]
    ups = run(frames)
    assert events(ups) == [(3, "T1"), (13, "T2")]       # T1 confirmed at minute 9, T2 at minute 39
    op = ups[13].operation
    assert op["t1"] == ts(6) and op["t2"] == ts(9) and op["items_appeared"] == 2


def test_two_different_one_frame_artifacts_are_ignored():
    frames = [obs(0, item(A)), obs(3, item(A)), obs(6, item(A), item(C)), obs(9, item(A), item(D)),
              obs(12, item(A)), obs(15, item(A))]
    ups = run(frames)
    assert events(ups) == []
    assert all(u.state == "IDLE" for u in ups)


def test_one_frame_dropout_while_active_does_not_move_t2():
    frames = [obs(0, item(A)), obs(3, item(A))] + [obs(m, item(A), item(B)) for m in range(6, 30, 3)]
    frames += [obs(30, item(A)), obs(33, item(A), item(B)), obs(36, item(A), item(B))]
    ups = run(frames)
    assert events(ups) == [(3, "T1"), (12, "T2")]       # closes at minute 36 (36 - 6 >= 30)
    op = ups[12].operation
    assert op["t2"] == ts(6) and op["items_appeared"] == 1 and op["occupancy_after"] == 2.5
