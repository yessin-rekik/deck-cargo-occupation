# DeckWatch

DeckWatch watches a cargo-deck camera feed and turns each frame into deck
occupancy (%), per-item 3D geometry (position, footprint, height, stack
level) and session-level loading/unloading events (T1/T2 "operations"). It
persists everything in SQLite and exposes it through a CLI and an HTTP API
for integration with other systems (e.g. a Laravel application).

**Status:** the runtime core (calibration math, detector interface,
geometry/occupancy, drift check, operation state machine, store, CLI, API)
is done and covered by the test suite. The pose-detection model and the
`deckwatch calibrate` tool ship in Plan B. Until drift reference patches
exist for a camera (written by `calibrate`), every frame DeckWatch analyzes
carries the flag `drift_unverified` instead of a real drift check.

## Install

```bash
python -m pip install -e ".[cpu,api]"
# or, with a GPU-enabled onnxruntime:
python -m pip install -e ".[gpu,api]"
```

Run the test suite:

```bash
venv/Scripts/python -m pytest -q
```

## Configuration

Copy the example config and fill in the vessel's measurements:

```bash
cp config/deck.example.yaml config/deck.yaml
```

Every `deck.*` value (`length_m`, `width_m`, `wall_height_m`, `wall_offset_m`)
is **required** and has no default — DeckWatch never assumes a deck's
dimensions. They must be measured on the vessel itself. Loading a config
that leaves any of them empty (as the shipped example does) fails fast with
a `ConfigError` naming the missing value.

Relative paths in `deck.yaml` (`detector.model_path`, `drift.refs_dir`,
`db_path`) resolve against the directory containing the config file, not the
current working directory.

`filename_tz` is the IANA time zone used to interpret a naive timestamp
(one with no UTC offset) parsed either from a frame's filename or from a
command's `--ts`/`ts` argument. It defaults to `UTC`.

The `calibration` section (the solved camera projection matrix, the 8
deck/wall reference points and the cargo-area polygon) is written by
`deckwatch calibrate` (Plan B), not edited by hand. `deckwatch analyze` and
`deckwatch serve` refuse to start if it is missing or malformed.

## Laravel integration — CLI

```
deckwatch [--config config/deck.yaml] <command> ...
```

Commands:

| Command | Description |
|---|---|
| `analyze <frame.jpg> [--ts ISO8601] [--items]` | Analyze one frame; prints `FrameResult` JSON. `--items` adds the per-item list. |
| `status` | Current state, the latest frame, and the open operation (if any). |
| `operations [--since ISO8601]` | List operations started, ended, or still open since the given time. |
| `serve [--host 127.0.0.1] [--port 8000]` | Run the HTTP API (see below). |

`--config` defaults to `config/deck.yaml`, resolved relative to the current
working directory — pass an absolute path if you invoke `deckwatch` from
somewhere else (e.g. a Laravel job running from a different cwd).

Every command prints exactly one JSON document to stdout and nothing else on
success; a fresh CLI process picks up exactly where the previous one left
off (all state lives in the SQLite database).

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | `internal_error` — an unexpected exception; the JSON error is on stdout, the Python traceback is on stderr |
| 2 | A frame-level error (`PipelineError`), or an argparse usage error (bad arguments) |
| 3 | A configuration error (`ConfigError`) |

**Error code strings** (the `"error"` field of the JSON document on a
non-zero exit, or of a non-2xx HTTP response):

| Code | Meaning |
|---|---|
| `unreadable_image` | The frame file is missing, unreadable, or not a decodable image |
| `wrong_resolution` | The decoded image's size doesn't match `image_size` in `deck.yaml` |
| `invalid_timestamp` | `--ts`/`ts` isn't a valid ISO 8601 timestamp, or a filename timestamp isn't a real date |
| `out_of_order` | The frame's timestamp is older than the last stored frame |
| `config_error` | `deck.yaml` (or a file it references) is missing or invalid |
| `internal_error` | An unexpected exception (CLI exit 1 / HTTP 500) |

**Timestamp rules:** passing an explicit `--ts` is recommended for
Laravel-driven uploads (no filename parsing ambiguity). If omitted, DeckWatch
parses a timestamp from the filename against the pattern `..._YYYYMMDD_HHMMSS...`
(e.g. `camera1_20261006_153000.jpg`); if the filename doesn't match, it falls
back to the file's modification time (mtime). A naive timestamp (`--ts` or a
filename match with no UTC offset) is interpreted in `filename_tz`. Prefer
`--ts` or a well-formed filename over the mtime fallback: mtime changes when
files are copied, re-synced or restored, so it can silently drift away from
when the frame was actually captured.

## Laravel integration — HTTP

```
deckwatch serve [--host 127.0.0.1] [--port 8000]
```

`deckwatch serve` binds `127.0.0.1:8000` by default. It has **no
authentication** — keep it reachable only from localhost (e.g. have Laravel
call it over loopback, or put it behind a reverse proxy that adds auth).

**Endpoints:**

| Method & path | Description |
|---|---|
| `POST /frames` | Multipart upload: `file` (the image) + `ts` (form field, ISO 8601) + optional `?items=true` query param. Returns the same `FrameResult` JSON as `deckwatch analyze`. |
| `GET /status` | Current state, latest frame, open operation. |
| `GET /operations?since=ISO8601` | List operations started, ended, or still open since the given time. |
| `GET /frames/{ts}?items=true` | Look up a previously analyzed frame by timestamp. |

**Status codes:** `200` success, `404` frame not found, `409` out-of-order
frame, `413` upload too large, `422` invalid image/resolution/timestamp,
`500` unexpected server error (JSON body `{"error": "internal_error", ...}`).

Timestamps may include a `+HH:MM` UTC offset (e.g. `2026-10-06T15:30:00+03:30`).
**URL-encode the `+`** as `%2B` in the query string or path segment — an
unencoded `+` is decoded as a space by HTTP, which breaks the timestamp.

Uploads are capped at 30 MiB (`MAX_UPLOAD_BYTES` in `deckwatch.api`); larger
bodies get `413 {"error": "payload_too_large", ...}`.

## Operations semantics

DeckWatch tracks loading/unloading as session-level operations, not
per-item events:

- **T1** marks the first frame of a confirmed change versus the last stable
  item set. "Confirmed" means the very next frame must also differ from the
  stable set by more than `change_min_m2` — this filters one-frame detector
  flicker before anything is recorded.
- While the operation is **ACTIVE**, DeckWatch keeps updating the running
  item set every frame.
- **T2** marks the last changed frame once the running item set has gone
  unchanged for `quiet_period_min` (default 30 minutes, time-based rather
  than frame-count-based, so it works at any camera cadence).
- **Gap operations:** if more than `gap_max_min` elapses between frames and
  the camera was idle, and the first frame after the gap already differs
  from the stable set, DeckWatch can't tell when the change actually
  happened. It emits a single `"T2"` event with `t1 == t2` (both set to that
  frame's timestamp) and the flag `t1_t2_uncertain`.

Example `FrameResult` (`--items` / `?items=true` adds `"items"`):

```json
{
  "ts": "2026-10-06T12:00:00Z",
  "occupancy_pct": 42.7,
  "occupied_m2": 318.4,
  "count_container": 21,
  "count_other": 3,
  "count_by_stack_level": {"1": 17, "2": 4},
  "change_pct": 0.4,
  "state": "IDLE",
  "operation": null,
  "flags": [],
  "calibration_version": 1,
  "model_version": "pose-v1",
  "items": [
    {"class": "container", "conf": 0.93, "height_m": 2.58, "height_source": "measured",
     "stack_level": 1, "length_m": 6.04, "width_m": 2.45,
     "footprint_deck_m": [[1.2, 3.4], [7.2, 3.4], [7.2, 5.9], [1.2, 5.9]]}
  ]
}
```

`operation` is non-null when the frame opens or closes an operation
(`"event": "T1"` or `"T2"`) or belongs to one that's currently open.

**`since` semantics:** both `deckwatch operations --since` and
`GET /operations?since=` return operations that were started on or after
`since`, OR ended on or after `since`, OR are still open — i.e. anything
that could have changed since `since`, not just operations that started
after it. This way polling with the last poll's timestamp never misses an
update to an operation that was already open at that time.

## Operational notes

- Use one SQLite database per camera (`db_path` in `deck.yaml`).
- Concurrent writers (multiple CLI runs, or the CLI and `deckwatch serve`
  together) are safe: writes are serialized with SQLite `BEGIN IMMEDIATE`
  transactions, and a write for the same timestamp from two writers is
  idempotent. Frames must still arrive in non-decreasing timestamp order per
  camera — an out-of-order frame is rejected (`409`/exit 2), it is not
  queued or reordered.
- The database grows with every frame (roughly ~5 MB/day at a 3-minute
  capture cadence); there is no pruning or rotation yet.
