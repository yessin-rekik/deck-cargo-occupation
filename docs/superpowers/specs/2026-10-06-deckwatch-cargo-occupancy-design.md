# DeckWatch — Cargo Deck Occupancy & Operation Detection — Design

- **Date:** 2026-10-06
- **Status:** Draft, awaiting review
- **Owner:** SOFONN

## 1. Goal

Given a raw camera frame of the vessel's cargo deck every 3 minutes, report:

1. **Occupancy %**: the share of the cargo area covered by cargo, measured from the true deck
   footprint of each item's top face (4 corners). A stack counts once.
2. **Operations**: the start (**T1**) and end (**T2**) of each loading/unloading session,
   with occupancy before and after.

Calibration of the fixed camera (§5.1) turns the image into real deck geometry. It gives
per-item height, stack level and real dimensions, not just a percentage.

It is built as a standalone Python project. The existing Laravel app integrates with it later
through a CLI (JSON on stdout) or an HTTP API.

## 2. Requirements and decisions

| # | Decision |
|---|----------|
| R1 | Input: **raw** frames from fixed camera `CAMO1`, 3840×2160 JPEG, one every 3 min. |
| R2 | 100% = the **brown cargo area inside the yellow border**. The green strip and the yellow/black hatched areas are excluded. |
| R3 | Cargo is mostly containers, occasionally other objects. Classes: `container`, `other`. |
| R4 | Footprint = the item's top face (4 corners) projected to the deck at the item's **measured height**. Overlaps (stacks) are merged and counted once. |
| R5 | Operations are **session-level**: T1 = first changed frame; T2 = last changed frame, followed by a 30-min quiet period. One operation can span hours. |
| R6 | Detection uses a trained **YOLO11-pose** model with 6 keypoints per item: 4 top corners + 2 base points (§5.2). Ultralytics, AGPL-3.0, accepted for now; needs a commercial licence before it ships in a closed product. |
| R7 | Training runs on the dev PC (RTX 4060 8 GB, CUDA via PyTorch wheels). Inference runs on a server whose hardware isn't known yet, so the model ships as **ONNX** and runs on ONNX Runtime (CUDA if available, else CPU). |
| R8 | The view is always clear during operations. Bad weather or darkness means no operation is happening, so there's no image-quality gating. Floodlit night frames are normal and must be supported. |
| R9 | Labels: an auto pre-labeling pipeline plus a visual QA pass by Claude, then the user corrects in **CVAT** (Docker). |
| R10 | The camera is fully calibrated (a 3×4 projection matrix **P**) from 8 deck points with real 3D coordinates. All metric quantities (m², height, stack level, item dimensions) come from **P**. |

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
│   ├── calibration.py        # 8-point deck model, projection matrix P, drift check
│   ├── geometry.py           # plane homographies, height measurement, footprint projection
│   ├── detector.py           # ONNX Runtime wrapper: frame -> [Item]
│   ├── occupancy.py          # footprints, union, clip, % area
│   ├── operations.py         # T1/T2 state machine
│   ├── store.py              # SQLite persistence
│   ├── pipeline.py           # analyze(frame, ts) -> FrameResult
│   ├── cli.py                # deckwatch analyze|status|operations|replay|calibrate
│   └── api.py                # FastAPI: POST /frames, GET /status, GET /operations
├── config/deck.yaml          # versioned calibration + thresholds
├── models/                   # deckwatch-pose.onnx (+ metadata)
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

**Deck coordinate frame (right-handed, metres):** the origin is at F4 (near-left corner of the
cargo area, as seen by the camera). X runs across the deck to the right, Y runs along it away
from the camera, Z is up. So F1 = (0, L, 0), F2 = (W, L, 0), F3 = (W, 0, 0), F4 = (0, 0, 0).

**User-supplied measurements** (stored in `deck.yaml`):
- Cargo-area length and width.
- Wall height above the deck.
- The wall-top points' horizontal offset from the floor corners (default 0, i.e. directly above).
- The spacing of the yellow grid lines, used for substitute points.

**8 points, clicked once on an empty-deck raw frame:**
- **F1–F4 (floor):** corners of the brown cargo area, at Z = 0.
- **W1–W4 (wall tops):** tops of the side walls (bulwarks/crash rails) above F1–F4, at Z = wall
  height.

**"Not visible" option:** each point can be marked `not_visible` with reason `out_of_frame` or
`occluded`. The rules:
- **≥ 4 floor-plane points.** A hidden floor corner is replaced by substitute points: yellow
  grid-line crossings or bay-marker corners at known deck coordinates.
- **≥ 2 truly visible wall tops.** Otherwise saving is refused, because **P** needs points that
  aren't all on the same plane.

**Solving:**
1. **P** is computed by DLT (direct linear transform) from all clicked points, then refined by
   minimising the reprojection error.
2. Saving is refused if the RMS reprojection error is greater than `reproj_max_px` (default 3 px).
3. Hidden points are recomputed through **P** and stored as `derived`.
4. Before saving, the tool renders all 8 points (clicked vs derived styled differently), the deck
   outline, and a wireframe 3D box of the cargo space over the frame for confirmation.

**Output:** `config/deck.yaml` containing:
- `calibration_version`
- Each point's image coordinates, deck coordinates, and status (`clicked` / `derived` /
  `not_visible`).
- **P** and the RMS reprojection error.
- The cargo-area polygon, in deck and image coordinates.
- Reference image patches around the visible wall tops.

**Drift check (every frame):** template-match each reference wall-top patch inside a small
search window. Only points that are visible in this frame are used (cranes and people can hide
some). If the median shift is greater than `drift_px` (default 15 px), the frame is flagged
`drift_suspected`. With fewer than 2 points matched, it's flagged `drift_check: unverified`.

### 5.2 Detector (`detector.py`)

- Loads `models/deckwatch-pose.onnx`. Picks `CUDAExecutionProvider` when available, else CPU.
- Input: a raw BGR frame. Letterboxed to the model size (default 1280).
- Output: a list of `Item(class, confidence, bbox, keypoints[6×2], kp_conf[6])` in raw image
  pixels.

**Keypoints:** fixed meaning, ordered as seen in the image. "Far" means further from the camera,
i.e. higher in the image.

| Idx | Name | Meaning |
|-----|------|---------|
| K0 | `top_far_left` | Top-face corner |
| K1 | `top_far_right` | Top-face corner |
| K2 | `top_near_right` | Top-face corner |
| K3 | `top_near_left` | Top-face corner |
| K4 | `base_near_left` | Deck-contact point directly below K3 |
| K5 | `base_near_right` | Deck-contact point directly below K2 |

- A keypoint is treated as **not visible** when its `kp_conf` is below `kp_conf_min` (default 0.5).
- Base points K4/K5 are often hidden by neighbours. That's expected and handled in §5.3.
- Horizontal-flip augmentation uses `flip_idx = [1, 0, 3, 2, 5, 4]`.
- Confidence threshold `conf_min` (default 0.4) and NMS IoU are set in `deck.yaml`.

### 5.3 Geometry and occupancy (`geometry.py`, `occupancy.py`)

From **P**:
- **Plane homography at height h:**
  `H_h = [p1 | p2 | h·p3 + p4]`, where `pi` are the columns of **P**. It maps deck (X, Y) at
  height h to the image. `H_0` is the floor homography.
- **Height measurement:**
  1. Take a visible base point b (K4 or K5; when both are visible, use the one with higher
     confidence).
  2. Project b through `H_0⁻¹` to deck coordinates (X, Y).
  3. Solve for the h that minimises the distance between `P·[X, Y, h, 1]` and the matching top
     corner (K3 or K2). This is a linear least-squares problem in h with a closed form.

**For each item:**
1. **Height:**
   - Measured as above whenever a base point is visible.
   - Otherwise, **size-fit fallback**: search for the h at which the top face, projected through
     `H_h⁻¹`, best matches one of the standard footprints in `deck.yaml`
     (`standard_sizes`, e.g. 10 ft, 20 ft offshore containers and common baskets).
   - If no size fits within `size_tol` (default 15%), use the class default height
     (`container: 2.6 m`, `other: 1.0 m`) and flag the item `height_assumed`.
2. **Footprint:** project K0–K3 through `H_h⁻¹`, giving a 4-sided shape on the deck in metres.
   The top face of a stack sits directly over the stack, so this is the stack's footprint.
3. **Derived attributes:**
   - `height_m`
   - `stack_level = round(h / unit_height)`, with `unit_height` in `deck.yaml` (default 2.6 m)
   - `length_m` and `width_m` of the footprint
   - `height_source` ∈ {`measured`, `size_fit`, `assumed`}

**Per frame:**
1. Union all footprints (`shapely.unary_union`), so overlaps count once.
2. Clip the union to the cargo-area polygon.
3. `occupancy_pct = area(clipped_union) / area(cargo_area) × 100`. Also report `occupied_m2`.
4. Report counts per class and per `stack_level`, plus the union polygon in deck coordinates,
   which feeds change detection.

### 5.4 Operations (`operations.py`)

**Change signal** between frame *t* and the last stable item set *S*:
- Match the items one-to-one: footprint IoU ≥ `match_iou` (0.3) and height difference
  ≤ `unit_height / 2`.
- `changed_m2` = area of unmatched items (appeared + disappeared).
- A frame is *changed* if `changed_m2 > change_min_m2` (default 2.0 m², below the footprint of
  a 10 ft container). `change_pct` = `changed_m2` as a % of the cargo area.
- Why items rather than the union's symmetric difference: corner noise on every item would add
  up over a full deck and exceed one container's area. Matching ignores that noise but still
  catches swaps and re-stacks.

**Confirmation:** a change only counts once the next frame also differs from *S* by more than
`change_min_m2`. This filters one-frame detector flicker.

**State machine:**

```
IDLE ──(confirmed change)──▶ ACTIVE ──(no change vs. running footprint for quiet_period)──▶ IDLE
      T1 = ts of first changed frame          T2 = ts of last changed frame
```

- While ACTIVE, the running item set is updated every frame. "No change" means consecutive
  frames differ by `≤ change_min_m2`.
- `quiet_period` defaults to 30 min. It's time-based, not a frame count, so it works at 3-min and
  20-min cadence alike.
- Frames flagged `drift_suspected` are stored but **excluded** from state decisions.

**Operation record:**
- `id`, `t1`, `t1_lower_bound` (last stable frame before T1), `t2`.
- `occupancy_before` (median of the last ≤ 3 stable frames before T1) and `occupancy_after`
  (median of the stable frames after T2).
- `net_pp`, `items_appeared`, `items_disappeared` (by IoU matching of footprints, threshold 0.3).
- `flags`.

Stack-level changes on an unchanged footprint (a container placed on top of another) don't
change the occupancy %. They are still reported in `items_appeared`, because the top face's
measured height changes.

**Gaps:** if there's more than `gap_max` (default 15 min) between frames:
- While ACTIVE: the operation stays open and gets the flag `gap_during_operation`.
- While IDLE: if the first frame after the gap differs from *S*, an operation is created with
  T1 = T2 = that frame and the flag `t1_t2_uncertain`.

### 5.5 Store (`store.py`)

SQLite (`deckwatch.db`), tables:
- `frames`: ts, path, occupancy, counts, flags, change value, calibration version, model version.
- `items`: per frame; class, conf, image keypoints + visibility, deck footprint, height,
  height source, stack level, dimensions.
- `operations`
- `state`: a single row holding the state-machine snapshot.

All state lives here, so a fresh CLI process every 3 min continues exactly where the last one
stopped. Writes for a frame happen in a single transaction.

### 5.6 Interfaces

- **CLI**
  - `deckwatch analyze <frame.jpg> [--ts ISO8601] [--items]`: `ts` defaults to the timestamp parsed
    from the filename, otherwise the file's mtime. Prints `FrameResult` JSON; `--items` adds the
    per-item list. Exit code 0 means success.
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

`items` is only included with `--items` (CLI) or `?items=1` (API). `operation` is filled when
the frame opens or closes an operation (`event: "T1" | "T2"`) or belongs to one.

## 6. Error handling

| Case | Behaviour |
|------|-----------|
| Unreadable or corrupt image, wrong resolution | Error JSON on stdout, exit ≠ 0 (HTTP 422). Nothing stored. |
| `ts` older than the last stored frame | Rejected as `out_of_order` (exit ≠ 0, HTTP 409). |
| `ts` already stored | Returns the stored result, so retries are safe. |
| Camera drift | Result returned with flag `drift_suspected`; excluded from T1/T2. |
| Measured height out of range (< 0.2 m or > `max_stack × unit_height`) | Discard the measurement, fall back to size-fit or assumed height, flag `height_outlier`. |
| Model or `deck.yaml` missing or invalid | Fails at startup with a clear message. |

## 7. Training pipeline (dev PC)

**Order matters:** calibration (§5.1) is done first, because pre-labeling uses **P**.

1. **`select_frames.py`**
   - Picks about 300 raw frames from the ~460 available, spread by time of day (day/night) and
     occupancy level.
   - Drops near-duplicates by perceptual hash.
   - Forces in the `empty_deck` frame.
2. **`prelabel.py`**
   - Runs Grounding DINO with prompts like "shipping container", "cargo box", "crate", "pallet",
     "pipe bundle", then SAM masks. Only masks whose centre is inside the cargo area are kept.
   - Fits a 4-sided shape to each mask. Its far corners become K0/K1, and its near bottom corners
     become K4/K5.
   - K3/K2 are placed on the vertical line through K4/K5, at the class default height (via **P**).
   - Assigns `container` or `other` from the prompt and the projected size.
   - Writes YOLO-pose `.txt` files with visibility flags.
3. **`review_overlays.py`**
   - Renders the pre-labels (top face, base points, vertical edges) over the frames.
   - Claude reviews them visually, fixing or deleting obvious misses, duplicates and junk, and
     writes a list of uncertain frames.
4. **CVAT**
   - `training/cvat/docker-compose.yml`. An import helper loads the frames and pre-labels as a
     task with a 6-point **skeleton** (K0–K5).
   - The user corrects them: drag the points, and mark hidden points `outside` (not visible) or
     `occluded` (an estimated position).
   - Export in the "Ultralytics YOLO Pose" format.
5. **`train.py`**
   - YOLO11-pose, starting from pretrained weights, `imgsz=1280`, on CUDA, with a batch size that
     fits in 8 GB.
   - Augmentation: brightness/contrast (to cover floodlit nights), small rotations, flips
     (with `flip_idx`).
   - **Split by date** (about 70/15/15) to avoid near-duplicate leakage.
6. **`export.py`:** ONNX export, plus a metadata JSON (model version, class names, keypoint names,
   imgsz, training metrics).
7. **`evaluate.py`:** the acceptance metrics in §8 on the held-out test split.

## 8. Testing and acceptance

**Unit tests (pytest, no model):**
- Calibration: DLT on a synthetic camera recovers **P** (reprojection < 0.5 px), derived
  "not visible" points, refusal when < 2 wall tops or reprojection error too high.
- Geometry: `H_h` round trip, height measurement on synthetic boxes (error < 1 cm without
  noise), size-fit fallback, union of overlapping footprints, clipping, stacked duplicates.
- State machine: scripted footprint sequences covering a single change, flicker (no T1), a swap,
  a stack-level change, a long operation, the quiet-period close, gaps during IDLE/ACTIVE, drift
  frames ignored, and restart from SQLite.
- Store: idempotent re-submit, out-of-order rejection.

**Model acceptance (test split):**
- `container` box mAP50 ≥ 0.85.
- Top-corner error on the deck (K0–K3 projected at ground-truth height): mean ≤ 0.25 m.
- Height error (`measured` items vs ground truth from the labels through **P**): mean ≤ 0.4 m.
  Stack-level accuracy ≥ 90%.
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
| Base points hidden in dense stowage, so many items fall back to size-fit or assumed height | `height_source` is tracked per item. The share of each source is reported in evaluation and replay. Extend `standard_sizes` as new cargo types appear. |
| The deck measurements the user supplies are inaccurate | A high reprojection error blocks saving. Grid-line substitute points cross-check the scale. |
| Small calibration errors amplify into height errors for items far from the camera | Evaluation reports height error by distance band. A second calibration point set (mid-deck grid points) can be added if needed. |
| AGPL licence | Get an Ultralytics enterprise licence, or swap to an Apache-licensed detector behind the same `detector.py` interface before a commercial release. |
| Recalibration after the camera is moved | `drift_suspected` flag plus a versioned `deck.yaml`. Re-run `deckwatch calibrate`. |
