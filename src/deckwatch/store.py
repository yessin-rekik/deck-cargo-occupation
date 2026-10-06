"""SQLite persistence for frames, items, operations and tracker state."""
from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

from deckwatch.config import ConfigError

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    ts TEXT PRIMARY KEY,
    path TEXT,
    occupancy_pct REAL NOT NULL,
    change_pct REAL,
    flags TEXT NOT NULL,
    calibration_version INTEGER,
    model_version TEXT,
    result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    frame_ts TEXT NOT NULL REFERENCES frames(ts),
    idx INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (frame_ts, idx)
);
CREATE TABLE IF NOT EXISTS operations (
    id INTEGER PRIMARY KEY,
    t1 TEXT NOT NULL,
    t2 TEXT,
    status TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path):
        self._in_transaction = False
        try:
            self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30, isolation_level=None)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.executescript(SCHEMA)
        except sqlite3.OperationalError as exc:
            raise ConfigError(f"cannot open database {path}: {exc}") from exc

    def close(self) -> None:
        self.conn.close()

    @contextlib.contextmanager
    def write_transaction(self):
        """BEGIN IMMEDIATE ... COMMIT/ROLLBACK. Re-entrant: nested calls join the outer transaction."""
        if self._in_transaction:
            yield
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._in_transaction = True
        try:
            yield
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")
        finally:
            self._in_transaction = False

    def last_ts(self) -> str | None:
        return self.conn.execute("SELECT MAX(ts) FROM frames").fetchone()[0]

    def get_frame(self, ts: str, include_items: bool = False) -> dict | None:
        row = self.conn.execute("SELECT result FROM frames WHERE ts = ?", (ts,)).fetchone()
        if row is None:
            return None
        result = json.loads(row[0])
        if include_items:
            rows = self.conn.execute("SELECT data FROM items WHERE frame_ts = ? ORDER BY idx", (ts,))
            result["items"] = [json.loads(r[0]) for r in rows]
        return result

    def latest_frame(self) -> dict | None:
        ts = self.last_ts()
        return self.get_frame(ts) if ts else None

    def load_state(self) -> dict | None:
        row = self.conn.execute("SELECT data FROM state WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else None

    def commit_frame(self, result: dict, items: list[dict], state: dict | None, operation: dict | None,
                     path: str | None = None) -> None:
        with self.write_transaction():
            self.conn.execute(
                "INSERT INTO frames (ts, path, occupancy_pct, change_pct, flags, calibration_version,"
                " model_version, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (result["ts"], path, result["occupancy_pct"], result["change_pct"], json.dumps(result["flags"]),
                 result["calibration_version"], result["model_version"], json.dumps(result)),
            )
            self.conn.executemany(
                "INSERT INTO items (frame_ts, idx, data) VALUES (?, ?, ?)",
                [(result["ts"], i, json.dumps(it)) for i, it in enumerate(items)],
            )
            if state is not None:
                self.conn.execute(
                    "INSERT INTO state (id, data) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                    (json.dumps(state),),
                )
            if operation is not None:
                self.conn.execute(
                    "INSERT INTO operations (id, t1, t2, status, data) VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(id) DO UPDATE SET t1 = excluded.t1, t2 = excluded.t2,"
                    " status = excluded.status, data = excluded.data",
                    (operation["id"], operation["t1"], operation["t2"], operation["status"], json.dumps(operation)),
                )

    def list_operations(self, since: str | None = None) -> list[dict]:
        if since is None:
            rows = self.conn.execute("SELECT data FROM operations ORDER BY id")
        else:
            rows = self.conn.execute(
                "SELECT data FROM operations WHERE t1 >= ? OR t2 >= ? OR status = 'open' ORDER BY id",
                (since, since),
            )
        return [json.loads(r[0]) for r in rows]

    def max_operation_id(self) -> int:
        row = self.conn.execute("SELECT MAX(id) FROM operations").fetchone()
        return row[0] or 0

    def status(self) -> dict:
        state = self.load_state() or {}
        row = self.conn.execute(
            "SELECT data FROM operations WHERE status = 'open' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "state": state.get("mode", "IDLE"),
            "latest_frame": self.latest_frame(),
            "open_operation": json.loads(row[0]) if row else None,
        }
