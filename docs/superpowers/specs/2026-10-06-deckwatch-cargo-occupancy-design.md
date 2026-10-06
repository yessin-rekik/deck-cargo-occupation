# DeckWatch — Cargo Deck Occupancy & Operation Detection — Design

- **Date:** 2026-10-06
- **Status:** Draft, awaiting review
- **Owner:** SOFONN

## 1. Goal

Given a raw camera frame of the vessel's cargo deck every 3 minutes, report:

1. **Occupancy %**: the share of the cargo area covered by cargo, measured from the
   top-face footprint of each cargo item (4 corners). A stack counts once.
2. **Operations**: the start (**T1**) and end (**T2**) of each loading/unloading session,
   with occupancy before and after.

It is built as a standalone Python project. The existing Laravel app integrates with it later
through a CLI (JSON on stdout) or an HTTP API.

## 2. Requirements and decisions

| # | Decision |
|---|----------|
| R1 | Input: **raw** frames from fixed camera `CAMO1`, 3840×2160 JPEG, one every 3 min. |
| R2 | 100% = the **brown cargo area inside the yellow border**. The green strip and the yellow/black hatched areas are excluded. |
| R3 | Cargo is mostly containers, occasionally other objects. Classes: `container`, `other`. |
| R4 | Footprint = top face of each item, as a 4-corner rotated box. Overlaps (stacks) are merged and counted once. |
| R5 | Operations are **session-level**: T1 = first changed frame; T2 = last changed frame, followed by a 30-min quiet period. One operation can span hours. |
| R6 | Detection uses a trained model: **YOLO11-OBB** (Ultralytics, AGPL-3.0, accepted for now; needs a commercial licence before it ships in a closed product). |
| R7 | Training runs on the dev PC (RTX 4060 8 GB, CUDA via PyTorch wheels). Inference runs on a server whose hardware isn't known yet, so the model ships as **ONNX** and runs on ONNX Runtime (CUDA if available, else CPU). |
| R8 | The view is always clear during operations. Bad weather or darkness means no operation is happening, so there's no image-quality gating. Floodlit night frames are normal and must be supported. |
| R9 | Labels: an auto pre-labeling pipeline plus a visual QA pass by Claude, then the user corrects in **CVAT** (Docker). |

## 3. Dataset findings

- 1,542 JPEGs in `dataset/`, from 2026-09-17 to 2026-10-05:
  - `periodic_*`: 1,221 frames, every 20 min.
  - `before_*` / `after_*`: 161 / 160 frames, from an earlier trigger.
- **Usable for training: about 460 raw 3840×2160 frames** (335 periodic and about 125 before/after).
  The 886 `*_view` images are crops warped with unknown settings that changed over time
  (3 output sizes), so they can't be mapped back to the camera image. They're excluded.
- `dataset/empty_deck/after_20260918_152541.jpg`: a raw, floodlit night frame, 0% reference,
  and the calibration source.
- `dataset/deck_with_cargo/after_20260930_020840.jpg`: a warped `_view` image, used only as a
  visual reference.
- The existing before/after trigger produces false positives (e.g. `before_20260919_092529` /
  `after_20260919_092601` show no cargo change).

## 4. Architecture

```
deckwatch/
├── src/deckwatch/            # runtime package (deployed to server)
│   ├── calibration.py        # 8-point deck model, homography, drift check
│   ├── detector.py           # ONNX Runtime wrapper: frame -> [OBB]
│   ├── occupancy.py          # project, union, clip, % area
│   ├── operations.py         # T1/T2 state machine
│   ├── store.py              # SQLite persistence
│   ├── pipeline.py           # analyze(frame, ts) -> FrameResult
│   ├── cli.py                # deckwatch analyze|status|operations|calibrate
│   └── api.py                # FastAPI: POST /frames, GET /status, GET /operations
├── config/deck.yaml          # versioned calibration + thresholds
├── models/                   # deckwatch-obb.onnx (+ metadata)
├── training/                 # dev-only
│   ├── select_frames.py
│   ├── prelabel.py
│   ├── review_overlays.py
│   ├── cvat/                 # docker-compose + import/export helpers
│   ├── train.py
│   ├── export.py
│   └── evaluate.py
└── tests/
```

The server needs only `src/deckwatch`, `config/deck.yaml` and the `.onnx` model. Runtime
dependencies: `numpy`, `opencv-python-headless`, `shapely`, `onnxruntime[-gpu]`, `pyyaml`,
`fastapi` + `uvicorn` (API only). Training dependencies (`torch` CUDA build, `ultralytics`,
Grounding DINO / SAM) live in a separate `training` extra.

## 5. Components

### 5.1 Calibration (`calibration.py`, `deckwatch calibrate`)

An 8-point deck model, clicked once on an empty-deck raw frame:

- **4 floor points (F1–F4):** corners of the brown cargo area on the deck surface.
- **4 wall-top points (W1–W4):** tops of the side walls (bulwarks/crash rails) directly above
  F1–F4.
- Wall height (metres) and cargo-area dimensions (metres), entered by the user. These enable
  m² output and the height model.

**"Not visible" option:** each point can be marked `not_visible` with reason `out_of_frame` or
`occluded`.

- **Floor:** a homography needs ≥ 4 points known to lie on the deck surface. A hidden corner is
  replaced by user-clicked substitute points on the deck surface (yellow grid-line crossings,
  bay-marker corners) at known deck coordinates. The hidden corner is then computed through the
  homography and stored as `derived`.
- **Wall tops:** a hidden point is derived from the floor point below it, the wall height, and the
  visible wall tops. **At least 2 wall tops must be truly visible**, or saving is refused.
- Before saving, the tool renders all 8 points (clicked vs derived styled differently) and the
  deck outline over the frame for confirmation.

**Output:** `config/deck.yaml` with `calibration_version`, point coordinates and status
(`clicked`/`derived`/`not_visible`), the image→deck homography, the cargo-area polygon (image
and deck coordinates), the vertical model, and reference image patches around the visible wall
tops.

**Drift check (every frame):** template-match each reference wall-top patch inside a small
search window. Only points that are visible in this frame are used (cranes and people can hide
some). If the median shift is greater than `drift_px` (default 15 px), the frame is flagged
`drift_suspected`. With fewer than 2 points matched, it's flagged `drift_check: unverified`.

### 5.2 Detector (`detector.py`)

- Loads `models/deckwatch-obb.onnx`. Picks `CUDAExecutionProvider` when available, else CPU.
- Input: a raw BGR frame. Letterboxed to the model size (default 1280).
- Output: a list of `OBB(class, confidence, corners[4×2] in raw image pixels)`. A rotated
  box covers occluded corners too, so no special handling is needed.
- Confidence threshold `conf_min` (default 0.4) and NMS IoU are set in `deck.yaml`.

### 5.3 Occupancy (`occupancy.py`)

1. **Footprint correction:** a single view can't measure each item's height, so v1 assumes a
   height per class from `deck.yaml` (defaults: `container: 2.6 m`, `other: 1.0 m`).
   - Floor points give the plane homography at height 0. Wall-top points give it at the wall
     height. The homography for a plane at height *h* is built from the point pairs interpolated
     between the two.
   - Each top face is projected with its class's plane homography. This fixes tall items
     appearing shifted and enlarged under the oblique view.
   - Stacks taller than the assumed height stay partly uncorrected. The size of that error is
     measured on the test split (§8).
   - Setting `footprint_correction: false` projects with the floor homography only. This is
     useful as an A/B comparison during evaluation.
2. Union all footprint polygons (`shapely.unary_union`), so stacks and overlaps count once.
3. Clip the union to the cargo-area polygon.
4. `occupancy_pct = area(clipped_union) / area(cargo_area) × 100`.
5. Also output: `occupied_m2`, `count_container`, `count_other`, and the union polygon in deck
   coordinates, which feeds change detection.

### 5.4 Operations (`operations.py`)

**Change signal** between frame *t* and the last stable footprint *S*:
`change = area(U_t Δ S) / area(cargo_area)` (symmetric difference). This catches swaps that leave
the % almost the same. A frame is *changed* if `change > change_min` (default 1.5%).

**Confirmation:** a change only counts once the next frame also differs from *S* by more than
`change_min`. This filters one-frame detector flicker.

**State machine:**

```
IDLE ──(confirmed change)──▶ ACTIVE ──(no change vs. running footprint for quiet_period)──▶ IDLE
      T1 = ts of first changed frame          T2 = ts of last changed frame
```

- While ACTIVE, the running footprint is updated every frame. "No change" means consecutive
  frames differ by `≤ change_min`.
- `quiet_period` defaults to 30 min. It's time-based, not a frame count, so it works at 3-min and
  20-min cadence alike.
- Frames flagged `drift_suspected` are stored but **excluded** from state decisions.

**Operation record:**
- `id`, `t1`, `t1_lower_bound` (last stable frame before T1), `t2`.
- `occupancy_before` (median of the last ≤ 3 stable frames before T1) and `occupancy_after`
  (median of the stable frames after T2).
- `net_pp`, `items_appeared`, `items_disappeared` (by IoU matching of footprints, threshold 0.3).
- `flags`.

**Gaps:** if there's more than `gap_max` (default 15 min) between frames:
- While ACTIVE: the operation stays open and gets the flag `gap_during_operation`.
- While IDLE: if the first frame after the gap differs from *S*, an operation is created with
  T1 = T2 = that frame and the flag `t1_t2_uncertain`.

### 5.5 Store (`store.py`)

SQLite (`deckwatch.db`), tables:
- `frames`: ts, path, occupancy, counts, flags, change value, calibration version, model version.
- `detections`: per frame; class, conf, image corners, deck footprint.
- `operations`
- `state`: a single row holding the state-machine snapshot.

All state lives here, so a fresh CLI process every 3 min continues exactly where the last one
stopped. Writes for a frame happen in a single transaction.

### 5.6 Interfaces

- **CLI**
  - `deckwatch analyze <frame.jpg> [--ts ISO8601]`: `ts` defaults to the timestamp parsed from the
    filename, otherwise the file's mtime. Prints `FrameResult` JSON. Exit code 0 means success.
  - `deckwatch status`: current state and the latest frame.
  - `deckwatch operations [--since]`
  - `deckwatch replay <dir>`: processes the frames in time order and writes an HTML report of the
    operations with thumbnails.
  - `deckwatch calibrate <empty_frame.jpg>`: the 8-point tool, an OpenCV window.
- **API (FastAPI):** `POST /frames` (multipart image + ts), `GET /status`, `GET /operations`,
  `GET /frames/{ts}`. It uses the same `pipeline.analyze()` and returns the same JSON as the CLI.

**`FrameResult` JSON (shape):**

```json
{
  "ts": "2026-10-06T12:00:00Z",
  "occupancy_pct": 42.7,
  "occupied_m2": 318.4,
  "count_container": 21,
  "count_other": 3,
  "change_pct": 0.4,
  "state": "IDLE",
  "operation": null,
  "flags": [],
  "calibration_version": 1,
  "model_version": "obb-v1"
}
```

`operation` is filled when the frame opens or closes an operation (`event: "T1" | "T2"`) or
belongs to one.

## 6. Error handling

| Case | Behaviour |
|------|-----------|
| Unreadable or corrupt image, wrong resolution | Error JSON on stdout, exit ≠ 0 (HTTP 422). Nothing stored. |
| `ts` older than the last stored frame | Rejected as `out_of_order` (exit ≠ 0, HTTP 409). |
| `ts` already stored | Returns the stored result, so retries are safe. |
| Camera drift | Result returned with flag `drift_suspected`; excluded from T1/T2. |
| Model or `deck.yaml` missing or invalid | Fails at startup with a clear message. |

## 7. Training pipeline (dev PC)

1. **`select_frames.py`**
   - Picks about 300 raw frames from the ~460 available, spread by time of day (day/night) and
     occupancy level.
   - Drops near-duplicates by perceptual hash.
   - Forces in the `empty_deck` frame.
2. **`prelabel.py`**
   - Runs Grounding DINO with prompts like "shipping container", "cargo box", "crate", "pallet",
     "pipe bundle", then SAM masks.
   - Fits a minimum-area rotated rectangle to each mask, keeping only those whose centre is
     inside the cargo area.
   - Assigns `container` or `other` from the prompt and the rectangle's aspect ratio and size.
   - Writes YOLO-OBB `.txt` files.
3. **`review_overlays.py`**
   - Renders the pre-labels over the frames.
   - Claude reviews them visually, fixing or deleting obvious misses, duplicates and junk, and
     writes a list of uncertain frames.
4. **CVAT**
   - `training/cvat/docker-compose.yml`. An import helper loads the frames and pre-labels as a
     task.
   - The user corrects them using rotated boxes (SAM assist available), then exports in the
     "Ultralytics YOLO Oriented Bounding Boxes" format.
5. **`train.py`**
   - YOLO11-OBB, starting from pretrained weights, `imgsz=1280`, on CUDA, with a batch size that
     fits in 8 GB.
   - Augmentation: brightness/contrast (to cover floodlit nights), small rotations, flips.
   - **Split by date** (about 70/15/15) to avoid near-duplicate leakage.
6. **`export.py`:** ONNX export, plus a metadata JSON (model version, class names, imgsz,
   training metrics).
7. **`evaluate.py`:** the acceptance metrics in §8 on the held-out test split.

## 8. Testing and acceptance

**Unit tests (pytest, no model):**
- Geometry: union of overlapping boxes, clipping, stacked duplicates, the homography round trip,
  derived "not visible" points, footprint correction with a synthetic vertical model.
- State machine: scripted footprint sequences covering a single change, flicker (no T1), a swap,
  a long operation, the quiet-period close, gaps during IDLE/ACTIVE, drift frames ignored, and
  restart from SQLite.
- Store: idempotent re-submit, out-of-order rejection.

**Model acceptance (test split):**
- `container` mAP50 (OBB) ≥ 0.85.
- Occupancy error ≤ 3 percentage points (mean absolute error), compared with the occupancy
  computed from the ground-truth labels through the same geometry.
- The empty-deck frame: occupancy ≤ 1%.

**Replay check:**
- `deckwatch replay dataset/` over all raw frames produces an operations report for the user to
  review.
- The known false-trigger pair `before_20260919_092529` / `after_20260919_092601` must not create
  an operation.

## 9. Out of scope (v1)

- Connecting to Laravel (the interfaces are ready for it; wiring it up comes later).
- Frame acquisition from the camera (RTSP or folder watcher). Frames are pushed in through the CLI
  or API.
- Per-bay occupancy, container ID/OCR, people or crane detection, multiple cameras or decks.
- Using the `_view` images.

## 10. Risks

| Risk | Mitigation |
|------|------------|
| ~460 raw frames may be too few | Pretrained weights plus augmentation. Add more raw frames from the live feed later (`replay` makes active learning easy). |
| Assumed per-class heights are wrong for tall stacks | Measure the error on the test split with correction on and off. If needed, v2 adds stack-level estimation (e.g. a `stack_2` class). |
| AGPL licence | Get an Ultralytics enterprise licence, or swap to an Apache-licensed detector behind the same `detector.py` interface before a commercial release. |
| Recalibration after the camera is moved | `drift_suspected` flag plus a versioned `deck.yaml`. Re-run `deckwatch calibrate`. |
