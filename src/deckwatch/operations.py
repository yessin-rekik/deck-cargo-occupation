"""Session-level operation detection (T1/T2) from per-frame item footprints."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from shapely import wkt
from shapely.geometry.base import BaseGeometry

from deckwatch.config import OperationsConfig
from deckwatch.timeutil import from_iso, to_iso

Footprint = tuple[BaseGeometry, float]   # (deck footprint, height_m)

PUBLIC_OP_KEYS = ("id", "status", "t1", "t1_lower_bound", "t2", "occupancy_before", "occupancy_after",
                  "net_pp", "items_appeared", "items_disappeared", "flags")


@dataclass(eq=False)
class FrameObs:
    ts: datetime
    occupancy_pct: float
    items: list[Footprint]


@dataclass
class OpUpdate:
    state: str               # "IDLE" | "ACTIVE"
    change_pct: float
    event: str | None        # "T1" | "T2" | None
    operation: dict | None


def match_items(before: list[Footprint], after: list[Footprint], iou_min: float, height_tol: float):
    """Greedy one-to-one matching by footprint IoU; heights must agree within height_tol."""
    pairs = []
    for i, (fa, ha) in enumerate(before):
        for j, (fb, hb) in enumerate(after):
            if abs(ha - hb) > height_tol:
                continue
            inter = fa.intersection(fb).area
            if inter > 0:
                iou = inter / fa.union(fb).area
                if iou >= iou_min:
                    pairs.append((iou, i, j))
    used_before, used_after = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_before and j not in used_after:
            used_before.add(i)
            used_after.add(j)
    appeared = [after[j] for j in range(len(after)) if j not in used_after]
    disappeared = [before[i] for i in range(len(before)) if i not in used_before]
    return appeared, disappeared


def _enc(items: list[Footprint]) -> list:
    return [[fp.wkt, h] for fp, h in items]


def _dec(items: list) -> list[Footprint]:
    return [(wkt.loads(s), h) for s, h in items]


def _public(op: dict) -> dict:
    return {k: op[k] for k in PUBLIC_OP_KEYS}


class OperationTracker:
    def __init__(self, cfg: OperationsConfig, cargo_area_m2: float, height_tol: float):
        self.cfg = cfg
        self.area = cargo_area_m2
        self.height_tol = height_tol
        self.quiet = timedelta(minutes=cfg.quiet_period_min)
        self.gap = timedelta(minutes=cfg.gap_max_min)

    def step(self, state: dict | None, obs: FrameObs) -> tuple[dict, OpUpdate]:
        if state is None:
            return self._baseline(obs, next_op_id=1), OpUpdate("IDLE", 0.0, None, None)
        s = dict(state)
        gap = obs.ts - from_iso(s["last_ts"]) > self.gap
        s["last_ts"] = to_iso(obs.ts)
        if s["mode"] == "ACTIVE":
            return self._step_active(s, obs, gap)
        return self._step_idle(s, obs, gap)

    def _changed_m2(self, before: list[Footprint], after: list[Footprint]) -> float:
        appeared, disappeared = match_items(before, after, self.cfg.match_iou, self.height_tol)
        return sum(fp.area for fp, _ in appeared) + sum(fp.area for fp, _ in disappeared)

    def _baseline(self, obs: FrameObs, next_op_id: int) -> dict:
        return {
            "mode": "IDLE",
            "next_op_id": next_op_id,
            "last_ts": to_iso(obs.ts),
            "stable_items": _enc(obs.items),
            "stable_occ": [obs.occupancy_pct],
            "last_stable_ts": to_iso(obs.ts),
            "pending": None,
            "active": None,
        }

    def _step_idle(self, s: dict, obs: FrameObs, gap: bool):
        d_S = self._changed_m2(_dec(s["stable_items"]), obs.items)
        pct = 100.0 * d_S / self.area
        changed = d_S > self.cfg.change_min_m2

        if changed and gap:
            op = self._open(s, t1=obs.ts, flags=["t1_t2_uncertain"])
            op["last_change_ts"] = to_iso(obs.ts)
            return s, OpUpdate("IDLE", pct, "T2", self._close(s, op, [obs.occupancy_pct], obs))

        if not changed:
            s["pending"] = None
            s["stable_occ"] = (s["stable_occ"] + [obs.occupancy_pct])[-3:]
            s["last_stable_ts"] = to_iso(obs.ts)
            return s, OpUpdate("IDLE", pct, None, None)

        # changed and no gap
        if s["pending"] is None:
            s["pending"] = {
                "first_ts": to_iso(obs.ts),
                "last_ts": to_iso(obs.ts),
                "items": _enc(obs.items),
            }
            return s, OpUpdate("IDLE", pct, None, None)

        # pending exists: compare against pending items
        d_P = self._changed_m2(_dec(s["pending"]["items"]), obs.items)
        if d_P < d_S:
            # CONFIRM: current is closer to pending than to baseline
            op = self._open(s, t1=from_iso(s["pending"]["first_ts"]), flags=[])
            last_change_ts = to_iso(obs.ts) if d_P > self.cfg.change_min_m2 else s["pending"]["last_ts"]
            op.update(last_change_ts=last_change_ts, running=_enc(obs.items), quiet_occ=[obs.occupancy_pct])
            s.update(mode="ACTIVE", active=op, pending=None)
            return s, OpUpdate("ACTIVE", pct, "T1", _public(op))

        # re-arm pending with current
        s["pending"] = {
            "first_ts": to_iso(obs.ts),
            "last_ts": to_iso(obs.ts),
            "items": _enc(obs.items),
        }
        return s, OpUpdate("IDLE", pct, None, None)

    def _step_active(self, s: dict, obs: FrameObs, gap: bool):
        op = dict(s["active"])
        d_R = self._changed_m2(_dec(op["running"]), obs.items)
        pct = 100.0 * d_R / self.area

        if gap and "gap_during_operation" not in op["flags"]:
            op["flags"] = op["flags"] + ["gap_during_operation"]

        if op.get("pending") is None:
            if d_R <= self.cfg.change_min_m2:
                # Quiet frame: append to quiet_occ, keep running
                op["quiet_occ"] = op["quiet_occ"] + [obs.occupancy_pct]
            else:
                # Set pending
                op["pending"] = {
                    "ts": to_iso(obs.ts),
                    "items": _enc(obs.items),
                }
        else:
            # pending exists
            d_P = self._changed_m2(_dec(op["pending"]["items"]), obs.items)
            if d_R <= self.cfg.change_min_m2:
                # Flicker (reverted)
                op["pending"] = None
                op["quiet_occ"] = op["quiet_occ"] + [obs.occupancy_pct]
            elif d_P < d_R:
                # CONFIRM
                last_change_ts = to_iso(obs.ts) if d_P > self.cfg.change_min_m2 else op["pending"]["ts"]
                op.update(last_change_ts=last_change_ts, running=_enc(obs.items), quiet_occ=[obs.occupancy_pct], pending=None)
            else:
                # re-arm pending
                op["pending"] = {
                    "ts": to_iso(obs.ts),
                    "items": _enc(obs.items),
                }

        # Check for close only if pending is None
        if op.get("pending") is None and obs.ts - from_iso(op["last_change_ts"]) >= self.quiet:
            return s, OpUpdate("IDLE", pct, "T2", self._close(s, op, op["quiet_occ"], obs))

        s["active"] = op
        return s, OpUpdate("ACTIVE", pct, None, _public(op))

    def _open(self, s: dict, t1: datetime, flags: list[str]) -> dict:
        op = {
            "id": s["next_op_id"],
            "status": "open",
            "t1": to_iso(t1),
            "t1_lower_bound": s["last_stable_ts"],
            "t2": None,
            "occupancy_before": round(statistics.median(s["stable_occ"]), 2),
            "occupancy_after": None,
            "net_pp": None,
            "items_appeared": None,
            "items_disappeared": None,
            "flags": list(flags),
        }
        s["next_op_id"] += 1
        return op

    def _close(self, s: dict, op: dict, after_occ: list[float], obs: FrameObs) -> dict:
        appeared, disappeared = match_items(_dec(s["stable_items"]), obs.items, self.cfg.match_iou, self.height_tol)
        after = round(statistics.median(after_occ), 2)
        op.update(status="closed", t2=op["last_change_ts"], occupancy_after=after,
                  net_pp=round(after - op["occupancy_before"], 2),
                  items_appeared=len(appeared), items_disappeared=len(disappeared))
        next_id = s["next_op_id"]
        s.clear()
        s.update(self._baseline(obs, next_id))
        s["stable_occ"] = list(after_occ[-3:])
        return _public(op)
