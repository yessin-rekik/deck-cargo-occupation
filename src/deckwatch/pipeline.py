"""analyze(frame, ts): decode -> drift check -> detect -> place -> occupancy -> T1/T2 -> store."""
from __future__ import annotations

import threading
from datetime import datetime

import cv2
import numpy as np

from deckwatch.calibration import Calibration
from deckwatch.config import Config, ConfigError
from deckwatch.drift import DriftRef, check_drift, load_refs
from deckwatch.occupancy import compute_occupancy, place_item
from deckwatch.operations import FrameObs, OperationTracker
from deckwatch.store import Store
from deckwatch.timeutil import parse_timestamp, to_iso


class PipelineError(Exception):
    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message}


class Pipeline:
    def __init__(self, config: Config, detector, store: Store, refs: list[DriftRef] | None = None):
        if not config.calibration:
            raise ConfigError("deck.yaml has no calibration section; run `deckwatch calibrate` first")
        self.config = config
        self.detector = detector
        self.store = store
        self.cal = Calibration.from_dict(config.calibration)
        if refs is None:
            centers = {n: p.image for n, p in self.cal.points.items() if n.startswith("W") and p.status == "clicked"}
            refs = load_refs(config.drift.refs_dir, centers)
        self.refs = refs
        self.tracker = OperationTracker(config.operations, self.cal.cargo_area.area,
                                        height_tol=config.geometry.unit_height_m / 2)
        self._lock = threading.Lock()

    def analyze(self, image_bytes: bytes, ts: datetime, path: str | None = None,
                include_items: bool = False) -> dict:
        with self._lock:
            return self._analyze(image_bytes, ts, path, include_items)

    def status(self) -> dict:
        with self._lock:
            return self.store.status()

    def operations(self, since: str | None = None) -> list[dict]:
        with self._lock:
            since_iso = to_iso(parse_timestamp(since, self.config.filename_tz)) if since else None
            return self.store.list_operations(since_iso)

    def frame(self, ts: str, include_items: bool = False) -> dict | None:
        with self._lock:
            return self.store.get_frame(ts, include_items=include_items)

    def _analyze(self, image_bytes: bytes, ts: datetime, path: str | None, include_items: bool) -> dict:
        ts_iso = to_iso(ts)
        existing = self.store.get_frame(ts_iso, include_items=include_items)
        if existing is not None:
            return existing
        last = self.store.last_ts()
        if last is not None and ts_iso < last:
            raise PipelineError("out_of_order", f"frame {ts_iso} is older than the last stored frame {last}", 409)

        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) if image_bytes else None
        if img is None:
            raise PipelineError("unreadable_image", "could not decode image", 422)
        h, w = img.shape[:2]
        if (w, h) != tuple(self.config.image_size):
            ew, eh = self.config.image_size
            raise PipelineError("wrong_resolution", f"expected {ew}x{eh}, got {w}x{h}", 422)

        flags = []
        drift_status, _ = check_drift(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.refs, self.config.drift)
        if drift_status == "suspected":
            flags.append("drift_suspected")
        elif drift_status == "unverified":
            flags.append("drift_unverified")

        kp_min = self.config.detector.kp_conf_min
        placed = [p for p in (place_item(it, self.cal.P, self.config.geometry, kp_min)
                              for it in self.detector.detect(img)) if p is not None]
        occ = compute_occupancy(placed, self.cal.cargo_area)

        state = self.store.load_state()
        if drift_status == "suspected":
            mode, change_pct, event, operation = (state or {}).get("mode", "IDLE"), None, None, None
            state = None                     # leave the stored tracker state untouched
        else:
            obs = FrameObs(ts, occ.pct, [(p.footprint, p.height_m) for p in occ.items])
            state, upd = self.tracker.step(state, obs)
            mode, change_pct, event, operation = upd.state, round(upd.change_pct, 2), upd.event, upd.operation

        result = {
            "ts": ts_iso,
            "occupancy_pct": round(occ.pct, 2),
            "occupied_m2": round(occ.m2, 2),
            "count_container": occ.count_by_class.get("container", 0),
            "count_other": occ.count_by_class.get("other", 0),
            "count_by_stack_level": occ.count_by_stack_level,
            "change_pct": change_pct,
            "state": mode,
            "operation": {"event": event, **operation} if operation else None,
            "flags": flags,
            "calibration_version": self.cal.version,
            "model_version": getattr(self.detector, "model_version", "unknown"),
        }
        items = [p.to_dict() for p in occ.items]
        self.store.commit_frame(result, items, state, operation, path)
        return {**result, "items": items} if include_items else result
