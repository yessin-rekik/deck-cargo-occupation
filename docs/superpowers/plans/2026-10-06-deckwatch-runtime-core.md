# DeckWatch Runtime Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deployable `deckwatch` Python package. It takes a raw deck frame and a
timestamp, and returns occupancy %, per-item 3D geometry, and T1/T2 operation events. Results
are persisted in SQLite and exposed through a CLI and an HTTP API.

**Architecture:**
- Pure, separately tested modules, in dependency order: config → calibration (camera matrix
  **P** via DLT) → geometry (plane homographies, height measurement) → occupancy (per-item
  footprints, union) → operations (T1/T2 state machine, JSON-serialisable state) → store
  (SQLite) → drift → detector (ONNX pose decoding) → pipeline (glue) → CLI / API.
- Everything is tested with a **synthetic pinhole camera** looking at a 20 × 60 m deck. No model
  file or real frames are needed for this plan.

**Tech Stack:** Python 3.12, numpy, scipy, shapely 2, opencv-python-headless, PyYAML, tzdata,
onnxruntime (CPU or GPU extra), FastAPI + uvicorn + python-multipart, pytest + httpx.

**Spec:** `docs/superpowers/specs/2026-10-06-deckwatch-cargo-occupancy-design.md`

**Out of this plan (follow-up "Plan B"):**
- The interactive `deckwatch calibrate` click tool and drift-reference capture.
- The training pipeline: frame selection, pre-labeling, CVAT, train, export, evaluate.
- `deckwatch replay` and its HTML report.
- Plan B consumes the interfaces built here: `build_calibration`, `Calibration.to_dict`,
  `extract_refs` / `save_refs`, and the model metadata JSON format of `load_detector`.

## Global Constraints

- Python interpreter: `venv/Scripts/python` (Windows venv, Python 3.12). All commands run from
  the repo root `E:\SOFONN-DECK\deckwatch` in Git Bash.
- Package layout: `src/deckwatch/`, tests in `tests/`, pytest `pythonpath = ["tests"]`, so tests
  import helpers with `from synth import ...`.
- Deck frame (right-handed, metres):
  - Origin at F4 (near-left corner of the cargo area). X across to the right, Y away from the
    camera, Z up.
  - F1 = (0, L, 0), F2 = (W, L, 0), F3 = (W, 0, 0), F4 = (0, 0, 0).
  - W1..W4 are F1..F4 raised to Z = wall height, offset outward by `wall_offset_m`.
- Keypoint order (fixed):
  - K0 `top_far_left`, K1 `top_far_right`, K2 `top_near_right`, K3 `top_near_left`,
    K4 `base_near_left` (below K3), K5 `base_near_right` (below K2).
  - `flip_idx = [1, 0, 3, 2, 5, 4]`.
- Classes: `container`, `other`.
- Defaults (from spec):

  | Setting | Value |
  |---|---|
  | `conf_min` | 0.4 |
  | `kp_conf_min` | 0.5 |
  | `imgsz` | 1280 |
  | `unit_height_m` | 2.6 |
  | `max_stack` | 3 |
  | `default_height_m` | container 2.6, other 1.0 |
  | `size_tol` | 0.15 |
  | `change_min_m2` | 2.0 |
  | `quiet_period_min` | 30 |
  | `gap_max_min` | 15 |
  | `match_iou` | 0.3 |
  | `drift_px` | 15 |
  | `reproj_max_px` | 3.0 |
  | height outlier | < 0.2 m or > `max_stack × unit_height_m` |

- Timestamps:
  - Always timezone-aware internally.
  - Stored and emitted as UTC `YYYY-MM-DDTHH:MM:SSZ`.
  - Filename timestamps (`*_YYYYMMDD_HHMMSS*.jpg`) are interpreted in `filename_tz` (default
    `UTC`). The CAMO1 overlay shows local time, UTC+03:30.
- **No deck dimensions are assumed anywhere in product code or shipped config.** `deck.length_m`,
  `deck.width_m`, `deck.wall_height_m` and `deck.wall_offset_m` are required in `deck.yaml` and
  have no defaults. `config/deck.example.yaml` ships them empty (`null`), so loading it fails with
  a `ConfigError` telling the user to fill them in. Tests use their own synthetic deck (`tests/synth.py`).
- Exit codes: 0 ok, 2 frame error (`PipelineError`), 3 config error (`ConfigError`).
- HTTP statuses: 422 unreadable image / wrong resolution / invalid ts; 409 out of order; 404 frame
  not found.
- Every commit message ends with the line
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. It's omitted from the
  commands below for brevity; add it as a second `-m`.

## Review Focus

1. **A frame with zero detections (empty deck)** must give 0.0 %, state IDLE, no exception. It's
   pinned in Task 5 (`test_empty_occupancy`) and Task 10 (`test_empty_deck_frame`).
2. **The model returning degenerate or self-intersecting top-corner quads** must never crash. A
   degenerate quad drops the item; a crossed order is fixed by the convex hull. Pinned in Task 5
   (`test_degenerate_item_dropped`, `test_crossed_keypoint_order_uses_hull`).
3. **Filename timestamps are UTC, while the overlay is local (+03:30).**
   `after_20260918_152541.jpg` must be stored as `2026-09-18T15:25:41Z`, not shifted. Pinned in
   Task 11 (`test_parse_ts_from_filename_is_utc`).
4. **The first frame ever processed shows an already-loaded deck.** It must become the baseline;
   no operation is created. Pinned in Task 6 (`test_first_frame_is_baseline`).
5. **Concurrent HTTP posts from Laravel** must not produce a 500 or a SQLite threading error.
   Requests are serialised by the pipeline lock. Pinned in Task 12
   (`test_concurrent_posts_never_500`).

---

### Task 1: Project scaffold and synthetic camera

**Files:**
- Create: `pyproject.toml`
- Create: `src/deckwatch/__init__.py`
- Create: `tests/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Produces, in `tests/synth.py`:
  - Constants `DECK_W = 20.0`, `DECK_L = 60.0`, `WALL_H = 3.0`, `IMAGE_SIZE = (3840, 2160)`.
  - `look_at_camera(center, target, f=3000.0, cx=1920.0, cy=1080.0) -> np.ndarray (3,4)`
  - `project(P, world) -> np.ndarray (N,2)`
  - `default_camera() -> np.ndarray (3,4)`
  - `box_keypoints(P, x0, y0, lx, wy, h) -> np.ndarray (6,2)`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "deckwatch"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.0",
  "scipy>=1.13",
  "shapely>=2.0",
  "opencv-python-headless>=4.10",
  "pyyaml>=6.0",
  "tzdata>=2024.1",
]

[project.optional-dependencies]
cpu = ["onnxruntime>=1.19"]
gpu = ["onnxruntime-gpu>=1.19"]
api = ["fastapi>=0.115", "uvicorn>=0.30", "python-multipart>=0.0.9"]
dev = ["pytest>=8", "httpx>=0.27"]

[project.scripts]
deckwatch = "deckwatch.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
```

- [ ] **Step 2: Create `src/deckwatch/__init__.py`**

```python
"""DeckWatch: cargo deck occupancy and operation detection."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Install the package and its dev dependencies**

Run: `venv/Scripts/python -m pip install -e ".[cpu,api,dev]"`
Expected: `Successfully installed ... deckwatch-0.1.0 ...`

- [ ] **Step 4: Create `tests/synth.py`**

```python
"""Synthetic camera and deck used across the test suite (no real frames needed)."""
import numpy as np

DECK_W = 20.0   # cargo area width (X, across)
DECK_L = 60.0   # cargo area length (Y, away from camera)
WALL_H = 3.0
IMAGE_SIZE = (3840, 2160)


def look_at_camera(center, target, f=3000.0, cx=1920.0, cy=1080.0):
    """Pinhole camera P = K [R | t] at `center` looking at `target`, image y pointing down."""
    center = np.asarray(center, float)
    target = np.asarray(target, float)
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ center
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return K @ np.hstack([R, t[:, None]])


def project(P, world):
    world = np.atleast_2d(np.asarray(world, float))
    hom = np.hstack([world, np.ones((len(world), 1))]) @ P.T
    return hom[:, :2] / hom[:, 2:3]


def default_camera():
    """Camera 40 m behind the near edge, 30 m up, aimed at the deck centre."""
    return look_at_camera((DECK_W / 2, -40.0, 30.0), (DECK_W / 2, DECK_L / 2, 0.0))


def box_keypoints(P, x0, y0, lx, wy, h):
    """Image keypoints K0..K5 of an axis-aligned box on [x0,x0+lx] x [y0,y0+wy], top at h."""
    world = [
        (x0, y0 + wy, h),        # K0 top_far_left
        (x0 + lx, y0 + wy, h),   # K1 top_far_right
        (x0 + lx, y0, h),        # K2 top_near_right
        (x0, y0, h),             # K3 top_near_left
        (x0, y0, 0.0),           # K4 base_near_left
        (x0 + lx, y0, 0.0),      # K5 base_near_right
    ]
    return project(P, world)
```

- [ ] **Step 5: Write `tests/test_synth.py`**

```python
import numpy as np

from synth import DECK_L, DECK_W, IMAGE_SIZE, WALL_H, default_camera, project


def test_target_projects_to_principal_point():
    P = default_camera()
    np.testing.assert_allclose(project(P, [(DECK_W / 2, DECK_L / 2, 0.0)])[0], (1920, 1080), atol=1e-6)


def test_deck_and_wall_corners_are_inside_image():
    P = default_camera()
    pts = [(x, y, z) for x in (0, DECK_W) for y in (0, DECK_L) for z in (0, WALL_H)]
    uv = project(P, pts)
    assert np.all(uv[:, 0] > 0) and np.all(uv[:, 0] < IMAGE_SIZE[0])
    assert np.all(uv[:, 1] > 0) and np.all(uv[:, 1] < IMAGE_SIZE[1])


def test_far_edge_is_higher_in_image_than_near_edge():
    P = default_camera()
    far, near = project(P, [(0, DECK_L, 0), (0, 0, 0)])
    assert far[1] < near[1]
```

- [ ] **Step 6: Run the tests**

Run: `venv/Scripts/python -m pytest tests/test_synth.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/deckwatch/__init__.py tests/synth.py tests/test_synth.py
git commit -m "chore: scaffold deckwatch package and synthetic test camera"
```

---

### Task 2: Configuration loading

**Files:**
- Create: `src/deckwatch/config.py`
- Create: `config/deck.example.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces, in `deckwatch.config`:
  - `ConfigError(Exception)`
  - `StandardSize(name, length_m, width_m)`
  - `DeckConfig(length_m, width_m, wall_height_m, wall_offset_m)`. All four are required, with no
    defaults: `length_m`, `width_m` and `wall_height_m` must be > 0, and `wall_offset_m` must be >= 0.
  - `DetectorConfig(model_path: Path, imgsz=1280, conf_min=0.4, nms_iou=0.5, kp_conf_min=0.5)`
  - `GeometryConfig(unit_height_m=2.6, max_stack=3, default_height_m={"container": 2.6, "other": 1.0}, size_tol=0.15, standard_sizes=())`
  - `OperationsConfig(change_min_m2=2.0, quiet_period_min=30.0, gap_max_min=15.0, match_iou=0.3)`
  - `DriftConfig(refs_dir: Path, patch_px=48, search_px=60, match_min=0.6, drift_px=15.0, min_points=2)`
  - `Config(path, image_size, filename_tz, deck, detector, geometry, operations, drift, db_path, calibration: dict | None)`
  - `load_config(path) -> Config`
  - Relative paths (`detector.model_path`, `db_path`, `drift.refs_dir`) are resolved against the
    config file's directory.

- [ ] **Step 1: Write the failing tests `tests/test_config.py`**

```python
from pathlib import Path

import pytest

from deckwatch.config import ConfigError, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "deck.example.yaml"

FULL = """
image_size: [3840, 2160]
deck: {length_m: 61.5, width_m: 19.2, wall_height_m: 2.9, wall_offset_m: 0.4}
detector: {model_path: ../models/m.onnx}
geometry:
  standard_sizes:
    - {name: 10ft, length_m: 2.99, width_m: 2.44}
"""


def write(tmp_path, text):
    p = tmp_path / "deck.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_full_config_loads_with_defaults(tmp_path):
    cfg = load_config(write(tmp_path, FULL))
    assert cfg.image_size == (3840, 2160)
    assert cfg.filename_tz == "UTC"
    assert (cfg.deck.length_m, cfg.deck.width_m, cfg.deck.wall_height_m, cfg.deck.wall_offset_m) == (61.5, 19.2, 2.9, 0.4)
    assert cfg.detector.conf_min == 0.4
    assert cfg.geometry.default_height_m == {"container": 2.6, "other": 1.0}
    assert [s.name for s in cfg.geometry.standard_sizes] == ["10ft"]
    assert cfg.operations.change_min_m2 == 2.0
    assert cfg.calibration is None


def test_relative_paths_resolve_against_config_dir(tmp_path):
    cfg = load_config(write(tmp_path, FULL))
    assert cfg.detector.model_path == tmp_path / "../models/m.onnx"
    assert cfg.db_path == tmp_path / "../deckwatch.db"
    assert cfg.drift.refs_dir == tmp_path / "drift_refs"


def test_shipped_example_has_no_deck_dimensions():
    with pytest.raises(ConfigError, match="deck.length_m must be provided"):
        load_config(EXAMPLE)


@pytest.mark.parametrize("entry", ["length_m: 61.5", "width_m: 19.2", "wall_height_m: 2.9", "wall_offset_m: 0.4"])
def test_every_deck_dimension_is_required(tmp_path, entry):
    key = entry.split(":")[0]
    text = FULL.replace(entry, f"unused_{entry}")
    with pytest.raises(ConfigError, match=f"deck.{key}"):
        load_config(write(tmp_path, text))


def test_null_deck_dimension_raises(tmp_path):
    with pytest.raises(ConfigError, match="deck.width_m must be provided"):
        load_config(write(tmp_path, FULL.replace("width_m: 19.2", "width_m: null")))


def test_non_positive_deck_dimension_raises(tmp_path):
    with pytest.raises(ConfigError, match="deck.length_m must be > 0"):
        load_config(write(tmp_path, FULL.replace("length_m: 61.5", "length_m: 0")))
    with pytest.raises(ConfigError, match="deck.wall_offset_m must be >= 0"):
        load_config(write(tmp_path, FULL.replace("wall_offset_m: 0.4", "wall_offset_m: -1")))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.config'`.

- [ ] **Step 3: Create `config/deck.example.yaml`**

```yaml
# DeckWatch configuration. Relative paths are resolved against this file's directory.
# Copy to config/deck.yaml and fill in every `deck` value with measurements taken on the vessel.
# DeckWatch refuses to start while any deck value is empty.
image_size: [3840, 2160]
filename_tz: UTC          # dataset filenames are UTC; the CAMO1 overlay shows local time (UTC+03:30)
db_path: ../deckwatch.db

deck:                     # metres, measured on the vessel - REQUIRED, no defaults
  length_m:               # brown cargo area along the deck (near edge -> far edge)
  width_m:                # brown cargo area across the deck
  wall_height_m:          # wall top above the deck surface
  wall_offset_m:          # wall tops' horizontal offset outward from the floor corners (0 if directly above)

detector:
  model_path: ../models/deckwatch-pose.onnx
  imgsz: 1280
  conf_min: 0.4
  nms_iou: 0.5
  kp_conf_min: 0.5

geometry:
  unit_height_m: 2.6
  max_stack: 3
  default_height_m: {container: 2.6, other: 1.0}
  size_tol: 0.15
  standard_sizes:
    - {name: 10ft, length_m: 2.99, width_m: 2.44}
    - {name: 20ft, length_m: 6.06, width_m: 2.44}

operations:
  change_min_m2: 2.0
  quiet_period_min: 30
  gap_max_min: 15
  match_iou: 0.3

drift:
  refs_dir: drift_refs
  patch_px: 48
  search_px: 60
  match_min: 0.6
  drift_px: 15
  min_points: 2

# calibration: written by `deckwatch calibrate`
```

- [ ] **Step 4: Implement `src/deckwatch/config.py`**

```python
"""Load and validate deck.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    """deck.yaml (or a file it points to) is missing or invalid."""


@dataclass(frozen=True)
class StandardSize:
    name: str
    length_m: float
    width_m: float


@dataclass(frozen=True)
class DeckConfig:
    """Cargo-area measurements from the vessel. All required - DeckWatch never assumes them."""
    length_m: float
    width_m: float
    wall_height_m: float
    wall_offset_m: float


@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path
    imgsz: int = 1280
    conf_min: float = 0.4
    nms_iou: float = 0.5
    kp_conf_min: float = 0.5


@dataclass(frozen=True)
class GeometryConfig:
    unit_height_m: float = 2.6
    max_stack: int = 3
    default_height_m: dict[str, float] = field(default_factory=lambda: {"container": 2.6, "other": 1.0})
    size_tol: float = 0.15
    standard_sizes: tuple[StandardSize, ...] = ()


@dataclass(frozen=True)
class OperationsConfig:
    change_min_m2: float = 2.0
    quiet_period_min: float = 30.0
    gap_max_min: float = 15.0
    match_iou: float = 0.3


@dataclass(frozen=True)
class DriftConfig:
    refs_dir: Path
    patch_px: int = 48
    search_px: int = 60
    match_min: float = 0.6
    drift_px: float = 15.0
    min_points: int = 2


@dataclass(frozen=True)
class Config:
    path: Path
    image_size: tuple[int, int]
    filename_tz: str
    deck: DeckConfig
    detector: DetectorConfig
    geometry: GeometryConfig
    operations: OperationsConfig
    drift: DriftConfig
    db_path: Path
    calibration: dict | None


def _req(raw: dict, dotted: str):
    cur = raw
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"deck.yaml: missing required key '{dotted}'")
        cur = cur[part]
    return cur


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    base = path.parent

    width_px, height_px = _req(raw, "image_size")
    dims = {}
    for name in ("length_m", "width_m", "wall_height_m", "wall_offset_m"):
        value = _req(raw, f"deck.{name}")
        if value is None:
            raise ConfigError(f"deck.{name} must be provided in deck.yaml (measured on the vessel)")
        try:
            dims[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"deck.{name} must be a number, got {value!r}") from exc
    for name in ("length_m", "width_m", "wall_height_m"):
        if dims[name] <= 0:
            raise ConfigError(f"deck.{name} must be > 0")
    if dims["wall_offset_m"] < 0:
        raise ConfigError("deck.wall_offset_m must be >= 0")
    deck = DeckConfig(**dims)

    det = raw.get("detector") or {}
    detector = DetectorConfig(
        model_path=base / _req(raw, "detector.model_path"),
        imgsz=int(det.get("imgsz", 1280)),
        conf_min=float(det.get("conf_min", 0.4)),
        nms_iou=float(det.get("nms_iou", 0.5)),
        kp_conf_min=float(det.get("kp_conf_min", 0.5)),
    )

    geo = raw.get("geometry") or {}
    geometry = GeometryConfig(
        unit_height_m=float(geo.get("unit_height_m", 2.6)),
        max_stack=int(geo.get("max_stack", 3)),
        default_height_m={"container": 2.6, "other": 1.0, **(geo.get("default_height_m") or {})},
        size_tol=float(geo.get("size_tol", 0.15)),
        standard_sizes=tuple(
            StandardSize(str(s["name"]), float(s["length_m"]), float(s["width_m"]))
            for s in geo.get("standard_sizes") or []
        ),
    )

    ops = raw.get("operations") or {}
    operations = OperationsConfig(
        change_min_m2=float(ops.get("change_min_m2", 2.0)),
        quiet_period_min=float(ops.get("quiet_period_min", 30.0)),
        gap_max_min=float(ops.get("gap_max_min", 15.0)),
        match_iou=float(ops.get("match_iou", 0.3)),
    )

    dr = raw.get("drift") or {}
    drift = DriftConfig(
        refs_dir=base / dr.get("refs_dir", "drift_refs"),
        patch_px=int(dr.get("patch_px", 48)),
        search_px=int(dr.get("search_px", 60)),
        match_min=float(dr.get("match_min", 0.6)),
        drift_px=float(dr.get("drift_px", 15.0)),
        min_points=int(dr.get("min_points", 2)),
    )

    return Config(
        path=path,
        image_size=(int(width_px), int(height_px)),
        filename_tz=str(raw.get("filename_tz", "UTC")),
        deck=deck,
        detector=detector,
        geometry=geometry,
        operations=operations,
        drift=drift,
        db_path=base / raw.get("db_path", "../deckwatch.db"),
        calibration=raw.get("calibration"),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/deckwatch/config.py config/deck.example.yaml tests/test_config.py
git commit -m "feat: load and validate deck.yaml configuration"
```

---

### Task 3: Camera calibration (P from 8 points, "not visible" support)

**Files:**
- Create: `src/deckwatch/calibration.py`
- Modify: `tests/synth.py` (append `DECK` and `make_calibration`)
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `DeckConfig` (Task 2).
- Produces, in `deckwatch.calibration`:
  - `FLOOR_POINTS = ("F1","F2","F3","F4")`, `WALL_POINTS = ("W1","W2","W3","W4")`
  - `CalibrationError(Exception)`
  - `corner_world_points(deck: DeckConfig) -> dict[str, np.ndarray(3,)]`
  - `project_points(P, world) -> np.ndarray (N,2)`
  - `reprojection_rms(P, world, image) -> float`
  - `solve_projection(world (N,3), image (N,2)) -> np.ndarray (3,4)`, normalised so `P[2,3] == 1`
  - `CalPoint(name, deck: tuple[3], image: tuple[2], status: "clicked"|"derived", reason: str|None)`
  - `Calibration(version, P, rms_px, points: dict[str, CalPoint], cargo_area: shapely Polygon)`
    with `.to_dict()` and `Calibration.from_dict(d)`
  - `build_calibration(deck, clicks: dict[str, (u,v)|None], substitutes=(), reasons=None, version=1, reproj_max_px=3.0) -> Calibration`.
    Each substitute is `((X, Y), (u, v))`, a point on the deck surface.
- Produces, in `tests/synth.py`: `DECK: DeckConfig` and `make_calibration(P=None) -> Calibration`.

- [ ] **Step 1: Write the failing tests `tests/test_calibration.py`**

```python
import numpy as np
import pytest

from deckwatch.calibration import (
    Calibration,
    CalibrationError,
    build_calibration,
    corner_world_points,
    project_points,
    solve_projection,
)
from synth import DECK, DECK_L, DECK_W, default_camera, project


def exact_clicks(P):
    return {name: tuple(project(P, xyz)[0]) for name, xyz in corner_world_points(DECK).items()}


def test_corner_world_points_follow_deck_frame():
    pts = corner_world_points(DECK)
    np.testing.assert_allclose(pts["F4"], (0, 0, 0))
    np.testing.assert_allclose(pts["F1"], (0, DECK_L, 0))
    np.testing.assert_allclose(pts["F2"], (DECK_W, DECK_L, 0))
    np.testing.assert_allclose(pts["W3"], (DECK_W, 0, 3.0))


def test_solve_projection_recovers_camera():
    P_true = default_camera()
    rng = np.random.default_rng(0)
    world = rng.uniform([0, 0, 0], [DECK_W, DECK_L, 6], size=(20, 3))
    P = solve_projection(world, project(P_true, world))
    test_pts = rng.uniform([0, 0, 0], [DECK_W, DECK_L, 6], size=(10, 3))
    np.testing.assert_allclose(project_points(P, test_pts), project(P_true, test_pts), atol=0.5)


def test_solve_projection_rejects_coplanar_points():
    world = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0]], float)
    with pytest.raises(CalibrationError, match="coplanar"):
        solve_projection(world, world[:, :2] * 100)


def test_build_calibration_with_all_points_visible():
    P_true = default_camera()
    cal = build_calibration(DECK, exact_clicks(P_true))
    assert cal.rms_px < 0.5
    assert all(p.status == "clicked" for p in cal.points.values())
    assert cal.cargo_area.area == pytest.approx(DECK_W * DECK_L)


def test_hidden_floor_corner_is_derived_from_substitute():
    P_true = default_camera()
    clicks = exact_clicks(P_true)
    clicks["F1"] = None
    sub = ((10.0, 30.0), tuple(project(P_true, (10.0, 30.0, 0.0))[0]))
    cal = build_calibration(DECK, clicks, substitutes=[sub], reasons={"F1": "out_of_frame"})
    f1 = cal.points["F1"]
    assert f1.status == "derived" and f1.reason == "out_of_frame"
    np.testing.assert_allclose(f1.image, project(P_true, (0, DECK_L, 0))[0], atol=0.5)


def test_too_few_floor_points_refused():
    clicks = exact_clicks(default_camera())
    clicks["F1"] = None
    with pytest.raises(CalibrationError, match="floor"):
        build_calibration(DECK, clicks)


def test_too_few_wall_points_refused():
    clicks = exact_clicks(default_camera())
    clicks["W1"] = clicks["W2"] = clicks["W3"] = None
    with pytest.raises(CalibrationError, match="wall"):
        build_calibration(DECK, clicks)


def test_bad_click_refused_by_reprojection_error():
    clicks = exact_clicks(default_camera())
    u, v = clicks["F1"]
    clicks["F1"] = (u + 100.0, v)
    with pytest.raises(CalibrationError, match="reprojection"):
        build_calibration(DECK, clicks)


def test_unknown_point_name_refused():
    clicks = exact_clicks(default_camera())
    clicks["X9"] = (1.0, 1.0)
    with pytest.raises(CalibrationError, match="unknown"):
        build_calibration(DECK, clicks)


def test_dict_round_trip():
    cal = build_calibration(DECK, exact_clicks(default_camera()))
    back = Calibration.from_dict(cal.to_dict())
    np.testing.assert_allclose(back.P, cal.P)
    assert back.points["W2"] == cal.points["W2"]
    assert back.cargo_area.equals(cal.cargo_area)
    assert back.version == cal.version
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_calibration.py -v`
Expected: FAIL with `ImportError` (`DECK` / `deckwatch.calibration` missing).

- [ ] **Step 3: Append to `tests/synth.py`**

```python


from deckwatch.config import DeckConfig  # noqa: E402

DECK = DeckConfig(length_m=DECK_L, width_m=DECK_W, wall_height_m=WALL_H, wall_offset_m=0.0)


def make_calibration(P=None):
    """Calibration built from exact clicks of all 8 points under camera P."""
    from deckwatch.calibration import build_calibration, corner_world_points

    P = default_camera() if P is None else P
    clicks = {name: tuple(project(P, xyz)[0]) for name, xyz in corner_world_points(DECK).items()}
    return build_calibration(DECK, clicks)
```

- [ ] **Step 4: Implement `src/deckwatch/calibration.py`**

```python
"""Camera calibration: solve the 3x4 projection matrix P from the 8 deck points."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon

from deckwatch.config import DeckConfig

FLOOR_POINTS = ("F1", "F2", "F3", "F4")
WALL_POINTS = ("W1", "W2", "W3", "W4")


class CalibrationError(Exception):
    """Calibration input is insufficient or inconsistent."""


def corner_world_points(deck: DeckConfig) -> dict[str, np.ndarray]:
    w, l, h, o = deck.width_m, deck.length_m, deck.wall_height_m, deck.wall_offset_m
    return {
        "F1": np.array([0.0, l, 0.0]),
        "F2": np.array([w, l, 0.0]),
        "F3": np.array([w, 0.0, 0.0]),
        "F4": np.array([0.0, 0.0, 0.0]),
        "W1": np.array([-o, l, h]),
        "W2": np.array([w + o, l, h]),
        "W3": np.array([w + o, 0.0, h]),
        "W4": np.array([-o, 0.0, h]),
    }


def project_points(P: np.ndarray, world) -> np.ndarray:
    world = np.atleast_2d(np.asarray(world, float))
    hom = np.hstack([world, np.ones((len(world), 1))]) @ P.T
    return hom[:, :2] / hom[:, 2:3]


def reprojection_rms(P: np.ndarray, world, image) -> float:
    err = project_points(P, world) - np.asarray(image, float)
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def _similarity(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centre on the mean, scale mean distance to sqrt(dim)."""
    n = pts.shape[1]
    c = pts.mean(axis=0)
    d = np.mean(np.linalg.norm(pts - c, axis=1))
    s = np.sqrt(n) / d
    T = np.eye(n + 1)
    T[:n, :n] *= s
    T[:n, n] = -s * c
    return T


def solve_projection(world, image) -> np.ndarray:
    """Normalised DLT followed by Levenberg-Marquardt refinement of the reprojection error."""
    world = np.asarray(world, float)
    image = np.asarray(image, float)
    if len(world) < 6:
        raise CalibrationError(f"need >= 6 points to solve the camera, got {len(world)}")
    if np.ptp(world[:, 2]) == 0:
        raise CalibrationError("points are coplanar; need wall-top points above the deck")

    Ti, Tw = _similarity(image), _similarity(world)
    wn = np.hstack([world, np.ones((len(world), 1))]) @ Tw.T
    im = np.hstack([image, np.ones((len(image), 1))]) @ Ti.T
    rows = []
    for X, (u, v, _) in zip(wn, im):
        rows.append(np.concatenate([X, np.zeros(4), -u * X]))
        rows.append(np.concatenate([np.zeros(4), X, -v * X]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    P = np.linalg.inv(Ti) @ vt[-1].reshape(3, 4) @ Tw
    P = P / P[2, 3]

    def residuals(p11):
        return (project_points(np.append(p11, 1.0).reshape(3, 4), world) - image).ravel()

    fit = least_squares(residuals, P.ravel()[:11], method="lm")
    return np.append(fit.x, 1.0).reshape(3, 4)


@dataclass(frozen=True)
class CalPoint:
    name: str
    deck: tuple[float, float, float]
    image: tuple[float, float]
    status: str                  # "clicked" | "derived"
    reason: str | None = None    # for derived points: "out_of_frame" | "occluded"


@dataclass(frozen=True, eq=False)
class Calibration:
    version: int
    P: np.ndarray
    rms_px: float
    points: dict[str, CalPoint]
    cargo_area: Polygon          # deck metres, Z = 0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "P": self.P.tolist(),
            "rms_px": self.rms_px,
            "points": {
                p.name: {"deck": list(p.deck), "image": list(p.image), "status": p.status, "reason": p.reason}
                for p in self.points.values()
            },
            "cargo_area_deck": [list(c) for c in self.cargo_area.exterior.coords[:-1]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        points = {
            name: CalPoint(name, tuple(float(v) for v in p["deck"]), tuple(float(v) for v in p["image"]),
                           p["status"], p.get("reason"))
            for name, p in d["points"].items()
        }
        return cls(int(d["version"]), np.asarray(d["P"], float), float(d["rms_px"]), points,
                   Polygon(d["cargo_area_deck"]))


def build_calibration(deck: DeckConfig, clicks: dict, substitutes=(), reasons: dict | None = None,
                      version: int = 1, reproj_max_px: float = 3.0) -> Calibration:
    """Solve P from clicked points (None = not visible) plus substitute floor points ((X, Y), (u, v))."""
    reasons = reasons or {}
    world_pts = corner_world_points(deck)
    unknown = set(clicks) - set(world_pts)
    if unknown:
        raise CalibrationError(f"unknown point names: {sorted(unknown)}")

    world, image = [], []
    for name, uv in clicks.items():
        if uv is not None:
            world.append(world_pts[name])
            image.append(uv)
    for (x, y), uv in substitutes:
        world.append(np.array([x, y, 0.0]))
        image.append(uv)
    world = np.asarray(world, float).reshape(-1, 3)
    image = np.asarray(image, float).reshape(-1, 2)

    n_floor = int(np.sum(world[:, 2] == 0.0))
    n_wall = len(world) - n_floor
    if n_floor < 4:
        raise CalibrationError(f"need >= 4 floor-plane points (corners or substitutes), got {n_floor}")
    if n_wall < 2:
        raise CalibrationError(f"need >= 2 visible wall-top points, got {n_wall}")

    P = solve_projection(world, image)
    rms = reprojection_rms(P, world, image)
    if rms > reproj_max_px:
        raise CalibrationError(
            f"reprojection error {rms:.2f}px exceeds {reproj_max_px}px; check clicks and deck measurements"
        )

    points = {}
    for name in FLOOR_POINTS + WALL_POINTS:
        uv = clicks.get(name)
        deck_xyz = tuple(float(v) for v in world_pts[name])
        if uv is None:
            proj = project_points(P, world_pts[name])[0]
            points[name] = CalPoint(name, deck_xyz, (float(proj[0]), float(proj[1])), "derived",
                                    reasons.get(name, "occluded"))
        else:
            points[name] = CalPoint(name, deck_xyz, (float(uv[0]), float(uv[1])), "clicked")

    cargo_area = Polygon([tuple(world_pts[n][:2]) for n in ("F4", "F3", "F2", "F1")])
    return Calibration(version, P, rms, points, cargo_area)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_calibration.py -v`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add src/deckwatch/calibration.py tests/synth.py tests/test_calibration.py
git commit -m "feat: solve camera projection from 8 deck points with not-visible support"
```

---

### Task 4: Geometry (plane homographies, height measurement, size fit)

**Files:**
- Create: `src/deckwatch/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `StandardSize` (Task 2); a `P` matrix (3,4) from Task 3 or `synth.default_camera()`.
- Produces, in `deckwatch.geometry`:
  - `plane_homography(P, h) -> np.ndarray (3,3)`, mapping deck (X, Y) at height h to the image
  - `plane_to_image(P, h, xy) -> np.ndarray (N,2)`
  - `image_to_plane(P, h, uv) -> np.ndarray (N,2)`
  - `measure_height(P, base_uv, top_uv) -> float`
  - `quad_dims(quad (4,2)) -> (length, width)`, where length ≥ width
  - `size_fit_height(P, top_uv (4,2), sizes, h_min, h_max, tol, step=0.05) -> float | None`

- [ ] **Step 1: Write the failing tests `tests/test_geometry.py`**

```python
import numpy as np
import pytest

from deckwatch.config import StandardSize
from deckwatch.geometry import (
    image_to_plane,
    measure_height,
    plane_homography,
    plane_to_image,
    quad_dims,
    size_fit_height,
)
from synth import box_keypoints, default_camera, project

SIZES = (StandardSize("10ft", 2.99, 2.44), StandardSize("20ft", 6.06, 2.44))


def test_plane_homography_matches_full_projection():
    P = default_camera()
    xy = np.array([[3.0, 7.0], [15.0, 50.0]])
    for h in (0.0, 2.6, 5.2):
        expected = project(P, np.column_stack([xy, np.full(2, h)]))
        np.testing.assert_allclose(plane_to_image(P, h, xy), expected, atol=1e-6)
    assert plane_homography(P, 0.0).shape == (3, 3)


def test_image_to_plane_round_trip():
    P = default_camera()
    xy = np.array([[1.0, 2.0], [19.0, 58.0], [10.0, 30.0]])
    np.testing.assert_allclose(image_to_plane(P, 2.6, plane_to_image(P, 2.6, xy)), xy, atol=1e-6)


@pytest.mark.parametrize("h", [2.6, 5.2, 1.0])
def test_measure_height_exact(h):
    P = default_camera()
    kp = box_keypoints(P, 5.0, 20.0, 6.06, 2.44, h)
    assert measure_height(P, kp[4], kp[3]) == pytest.approx(h, abs=1e-6)
    assert measure_height(P, kp[5], kp[2]) == pytest.approx(h, abs=1e-6)


def test_measure_height_with_one_pixel_noise():
    P = default_camera()
    kp = box_keypoints(P, 5.0, 40.0, 6.06, 2.44, 2.6)
    assert measure_height(P, kp[4], kp[3] + np.array([1.0, -1.0])) == pytest.approx(2.6, abs=0.15)


def test_quad_dims_of_rectangle():
    quad = np.array([[0, 2.44], [6.06, 2.44], [6.06, 0], [0, 0]])
    length, width = quad_dims(quad)
    assert length == pytest.approx(6.06) and width == pytest.approx(2.44)


def test_size_fit_finds_height_of_standard_container():
    P = default_camera()
    kp = box_keypoints(P, 4.0, 25.0, 6.06, 2.44, 2.6)
    h = size_fit_height(P, kp[:4], SIZES, 0.2, 7.8, 0.15)
    assert h == pytest.approx(2.6, abs=0.051)


def test_size_fit_returns_none_for_non_standard_item():
    P = default_camera()
    kp = box_keypoints(P, 4.0, 25.0, 4.0, 4.0, 1.5)
    assert size_fit_height(P, kp[:4], SIZES, 0.2, 7.8, 0.15) is None


def test_size_fit_without_sizes_returns_none():
    P = default_camera()
    kp = box_keypoints(P, 4.0, 25.0, 6.06, 2.44, 2.6)
    assert size_fit_height(P, kp[:4], (), 0.2, 7.8, 0.15) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.geometry'`.

- [ ] **Step 3: Implement `src/deckwatch/geometry.py`**

```python
"""Single-view metrology on the calibrated deck: planes at height h, item heights, size fit."""
from __future__ import annotations

import numpy as np

from deckwatch.config import StandardSize


def plane_homography(P: np.ndarray, h: float) -> np.ndarray:
    """Homography mapping deck (X, Y) on the horizontal plane Z = h to image pixels."""
    return np.column_stack([P[:, 0], P[:, 1], h * P[:, 2] + P[:, 3]])


def _apply(H: np.ndarray, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def plane_to_image(P: np.ndarray, h: float, xy) -> np.ndarray:
    return _apply(plane_homography(P, h), xy)


def image_to_plane(P: np.ndarray, h: float, uv) -> np.ndarray:
    return _apply(np.linalg.inv(plane_homography(P, h)), uv)


def measure_height(P: np.ndarray, base_uv, top_uv) -> float:
    """Height of the point imaged at top_uv, which lies vertically above the deck point at base_uv.

    The base fixes (X, Y) on the deck; projecting (X, Y, h) must land on top_uv, which is linear
    in h for each image coordinate. Solved in closed form by least squares.
    """
    x, y = image_to_plane(P, 0.0, base_uv)[0]
    xb = np.array([x, y, 0.0, 1.0])
    u, v = np.asarray(top_uv, float)
    c = np.array([u * P[2, 2] - P[0, 2], v * P[2, 2] - P[1, 2]])
    d = np.array([P[0] @ xb - u * (P[2] @ xb), P[1] @ xb - v * (P[2] @ xb)])
    return float(c @ d / (c @ c))


def quad_dims(quad) -> tuple[float, float]:
    """(length, width) of a quadrilateral as the means of opposite sides, length >= width."""
    q = np.asarray(quad, float)
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    a = (sides[0] + sides[2]) / 2
    b = (sides[1] + sides[3]) / 2
    return float(max(a, b)), float(min(a, b))


def size_fit_height(P: np.ndarray, top_uv, sizes: tuple[StandardSize, ...], h_min: float, h_max: float,
                    tol: float, step: float = 0.05) -> float | None:
    """Height at which the projected top face best matches a standard footprint, or None."""
    if not sizes:
        return None
    best_h, best_err = None, np.inf
    for h in np.arange(h_min, h_max + step / 2, step):
        length, width = quad_dims(image_to_plane(P, h, top_uv))
        for s in sizes:
            err = max(abs(length - s.length_m) / s.length_m, abs(width - s.width_m) / s.width_m)
            if err < best_err:
                best_h, best_err = float(h), err
    return best_h if best_err <= tol else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_geometry.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deckwatch/geometry.py tests/test_geometry.py
git commit -m "feat: plane homographies, height measurement and size-fit from P"
```

---

### Task 5: Items, placement and occupancy

**Files:**
- Create: `src/deckwatch/items.py`
- Create: `src/deckwatch/occupancy.py`
- Test: `tests/test_occupancy.py`

**Interfaces:**
- Consumes: `GeometryConfig`, `StandardSize` (Task 2); `image_to_plane`, `measure_height`,
  `quad_dims`, `size_fit_height` (Task 4).
- Produces, in `deckwatch.items`:
  - `KEYPOINT_NAMES` (6-tuple, see Global Constraints), `TOP = (0, 1, 2, 3)`,
    `BASE_TO_TOP = ((4, 3), (5, 2))`, `FLIP_IDX = (1, 0, 3, 2, 5, 4)`
  - `Item(cls: str, conf: float, keypoints: np.ndarray (6,2), kp_conf: np.ndarray (6,))`
- Produces, in `deckwatch.occupancy`:
  - `PlacedItem(cls, conf, footprint: Polygon, height_m, height_source, stack_level, length_m, width_m, flags: list[str])`
    with `.to_dict()`, whose keys are `class, conf, height_m, height_source, stack_level, length_m, width_m, footprint_deck_m, flags`
  - `estimate_height(item, P, geo, kp_conf_min) -> (h, source, flags)`
  - `place_item(item, P, geo, kp_conf_min) -> PlacedItem | None`
  - `Occupancy(items, union, pct, m2, count_by_class: dict[str,int], count_by_stack_level: dict[str,int])`
  - `compute_occupancy(placed, cargo_area) -> Occupancy`. Only items whose footprint centroid is
    inside the cargo area are counted.

- [ ] **Step 1: Write the failing tests `tests/test_occupancy.py`**

```python
import numpy as np
import pytest
from shapely.geometry import box

from deckwatch.config import GeometryConfig, StandardSize
from deckwatch.items import Item
from deckwatch.occupancy import compute_occupancy, place_item
from synth import DECK_L, DECK_W, box_keypoints, default_camera

GEO = GeometryConfig(standard_sizes=(StandardSize("10ft", 2.99, 2.44), StandardSize("20ft", 6.06, 2.44)))
CARGO = box(0, 0, DECK_W, DECK_L)
P = default_camera()


def make_item(x0, y0, lx=6.06, wy=2.44, h=2.6, cls="container", base_conf=0.9):
    kp = box_keypoints(P, x0, y0, lx, wy, h)
    return Item(cls, 0.9, kp, np.array([0.9] * 4 + [base_conf] * 2))


def test_measured_height_and_true_footprint():
    p = place_item(make_item(4.0, 10.0), P, GEO, 0.5)
    assert p.height_source == "measured"
    assert p.height_m == pytest.approx(2.6, abs=1e-6)
    assert p.stack_level == 1
    assert p.length_m == pytest.approx(6.06, abs=1e-3) and p.width_m == pytest.approx(2.44, abs=1e-3)
    assert p.footprint.area == pytest.approx(6.06 * 2.44, rel=1e-4)
    assert p.flags == []


def test_hidden_base_uses_size_fit():
    p = place_item(make_item(4.0, 30.0, base_conf=0.1), P, GEO, 0.5)
    assert p.height_source == "size_fit"
    assert p.height_m == pytest.approx(2.6, abs=0.051)


def test_hidden_base_non_standard_uses_class_default():
    p = place_item(make_item(4.0, 30.0, lx=4.0, wy=4.0, h=2.6, base_conf=0.1), P, GEO, 0.5)
    assert p.height_source == "assumed"
    assert p.height_m == 2.6
    assert "height_assumed" in p.flags


def test_stacked_container_is_level_two():
    p = place_item(make_item(4.0, 10.0, h=5.2), P, GEO, 0.5)
    assert p.stack_level == 2
    assert p.footprint.area == pytest.approx(6.06 * 2.44, rel=1e-4)


def test_outlier_height_falls_back():
    item = make_item(4.0, 10.0)
    kp = item.keypoints.copy()
    kp[4], kp[5] = kp[3], kp[2]           # base points collapsed onto top corners -> h = 0
    p = place_item(Item("container", 0.9, kp, item.kp_conf), P, GEO, 0.5)
    assert "height_outlier" in p.flags
    assert p.height_source == "size_fit"


def test_degenerate_item_dropped():
    kp = np.tile(np.array([[1000.0, 1000.0]]), (6, 1))
    assert place_item(Item("container", 0.9, kp, np.full(6, 0.1)), P, GEO, 0.5) is None


def test_crossed_keypoint_order_uses_hull():
    item = make_item(4.0, 10.0)
    kp = item.keypoints.copy()
    kp[[0, 1]] = kp[[1, 0]]               # far corners swapped -> self-intersecting order
    p = place_item(Item("container", 0.9, kp, item.kp_conf), P, GEO, 0.5)
    assert p.footprint.area == pytest.approx(6.06 * 2.44, rel=1e-3)


def test_stack_counts_once_and_clipping():
    a = place_item(make_item(2.0, 5.0), P, GEO, 0.5)
    stacked = place_item(make_item(2.0, 5.0, h=5.2), P, GEO, 0.5)
    edge = place_item(make_item(16.0, 40.0), P, GEO, 0.5)     # spans x 16..22.06, centroid inside
    occ = compute_occupancy([a, stacked, edge], CARGO)
    expected = 6.06 * 2.44 + 4.0 * 2.44
    assert occ.m2 == pytest.approx(expected, rel=1e-3)
    assert occ.pct == pytest.approx(100 * expected / 1200, rel=1e-3)
    assert occ.count_by_class == {"container": 3}
    assert occ.count_by_stack_level == {"1": 2, "2": 1}


def test_item_outside_cargo_area_not_counted():
    outside = place_item(make_item(25.0, 10.0), P, GEO, 0.5)
    occ = compute_occupancy([outside], CARGO)
    assert occ.items == [] and occ.pct == 0.0


def test_empty_occupancy():
    occ = compute_occupancy([], CARGO)
    assert occ.pct == 0.0 and occ.m2 == 0.0 and occ.union.is_empty
    assert occ.count_by_class == {} and occ.count_by_stack_level == {}


def test_to_dict_shape():
    d = place_item(make_item(4.0, 10.0), P, GEO, 0.5).to_dict()
    assert set(d) == {"class", "conf", "height_m", "height_source", "stack_level", "length_m",
                      "width_m", "footprint_deck_m", "flags"}
    assert len(d["footprint_deck_m"]) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_occupancy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.items'`.

- [ ] **Step 3: Implement `src/deckwatch/items.py`**

```python
"""Detector output type and the fixed keypoint layout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEYPOINT_NAMES = (
    "top_far_left",
    "top_far_right",
    "top_near_right",
    "top_near_left",
    "base_near_left",
    "base_near_right",
)
TOP = (0, 1, 2, 3)
BASE_TO_TOP = ((4, 3), (5, 2))   # (base keypoint, top keypoint directly above it)
FLIP_IDX = (1, 0, 3, 2, 5, 4)


@dataclass(frozen=True, eq=False)
class Item:
    cls: str
    conf: float
    keypoints: np.ndarray   # (6, 2) raw image pixels, KEYPOINT_NAMES order
    kp_conf: np.ndarray     # (6,) keypoint visibility confidence, 0..1
```

- [ ] **Step 4: Implement `src/deckwatch/occupancy.py`**

```python
"""Per-item 3D placement on the deck and frame occupancy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from deckwatch.config import GeometryConfig
from deckwatch.geometry import image_to_plane, measure_height, quad_dims, size_fit_height
from deckwatch.items import BASE_TO_TOP, TOP, Item

MIN_HEIGHT_M = 0.2
MIN_FOOTPRINT_M2 = 0.05


@dataclass(eq=False)
class PlacedItem:
    cls: str
    conf: float
    footprint: Polygon       # deck metres
    height_m: float
    height_source: str       # "measured" | "size_fit" | "assumed"
    stack_level: int
    length_m: float
    width_m: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "class": self.cls,
            "conf": round(self.conf, 3),
            "height_m": round(self.height_m, 2),
            "height_source": self.height_source,
            "stack_level": self.stack_level,
            "length_m": round(self.length_m, 2),
            "width_m": round(self.width_m, 2),
            "footprint_deck_m": [[round(x, 3), round(y, 3)] for x, y in self.footprint.exterior.coords[:-1]],
            "flags": list(self.flags),
        }


def estimate_height(item: Item, P: np.ndarray, geo: GeometryConfig, kp_conf_min: float):
    """Measured from a visible base point, else size-fit, else the class default."""
    flags: list[str] = []
    max_h = geo.max_stack * geo.unit_height_m
    visible = [(float(item.kp_conf[b]), b, t) for b, t in BASE_TO_TOP if item.kp_conf[b] >= kp_conf_min]
    if visible:
        _, b, t = max(visible)
        h = measure_height(P, item.keypoints[b], item.keypoints[t])
        if MIN_HEIGHT_M <= h <= max_h:
            return h, "measured", flags
        flags.append("height_outlier")
    h = size_fit_height(P, item.keypoints[list(TOP)], geo.standard_sizes, MIN_HEIGHT_M, max_h, geo.size_tol)
    if h is not None:
        return h, "size_fit", flags
    flags.append("height_assumed")
    return geo.default_height_m.get(item.cls, geo.unit_height_m), "assumed", flags


def place_item(item: Item, P: np.ndarray, geo: GeometryConfig, kp_conf_min: float) -> PlacedItem | None:
    h, source, flags = estimate_height(item, P, geo, kp_conf_min)
    corners = image_to_plane(P, h, item.keypoints[list(TOP)])
    if not np.all(np.isfinite(corners)):
        return None
    hull = MultiPoint([tuple(c) for c in corners]).convex_hull
    if not isinstance(hull, Polygon) or hull.area < MIN_FOOTPRINT_M2:
        return None
    rect = np.asarray(hull.minimum_rotated_rectangle.exterior.coords)[:4]
    length, width = quad_dims(rect)
    stack_level = max(1, int(round(h / geo.unit_height_m)))
    return PlacedItem(item.cls, float(item.conf), hull, float(h), source, stack_level, length, width, flags)


@dataclass(eq=False)
class Occupancy:
    items: list[PlacedItem]          # items whose footprint centroid lies in the cargo area
    union: BaseGeometry
    pct: float
    m2: float
    count_by_class: dict[str, int]
    count_by_stack_level: dict[str, int]


def compute_occupancy(placed: list[PlacedItem], cargo_area: Polygon) -> Occupancy:
    inside = [p for p in placed if cargo_area.contains(p.footprint.centroid)]
    union = unary_union([p.footprint for p in inside]).intersection(cargo_area) if inside else Polygon()
    by_class: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for p in inside:
        by_class[p.cls] = by_class.get(p.cls, 0) + 1
        by_level[str(p.stack_level)] = by_level.get(str(p.stack_level), 0) + 1
    m2 = float(union.area)
    return Occupancy(inside, union, 100.0 * m2 / cargo_area.area, m2, by_class, by_level)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_occupancy.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/deckwatch/items.py src/deckwatch/occupancy.py tests/test_occupancy.py
git commit -m "feat: place items in 3D and compute deck occupancy"
```

---

### Task 6: Timestamps and the T1/T2 operation tracker

**Files:**
- Create: `src/deckwatch/timeutil.py`
- Create: `src/deckwatch/operations.py`
- Test: `tests/test_operations.py`

**Interfaces:**
- Consumes: `OperationsConfig` (Task 2).
- Produces, in `deckwatch.timeutil`:
  - `to_iso(dt) -> "YYYY-MM-DDTHH:MM:SSZ"`. Raises `ValueError` for a naive dt.
  - `from_iso(s) -> aware datetime`
  - `parse_timestamp(s, default_tz="UTC") -> aware datetime`. A naive string gets
    `default_tz`; an invalid string raises `ValueError`.
- Produces, in `deckwatch.operations`:
  - `FrameObs(ts: aware datetime, occupancy_pct: float, items: list[tuple[Polygon, float]])`,
    where each item is `(footprint, height_m)`
  - `OpUpdate(state: "IDLE"|"ACTIVE", change_pct: float, event: "T1"|"T2"|None, operation: dict|None)`
  - `match_items(before, after, iou_min, height_tol) -> (appeared, disappeared)`
  - `OperationTracker(cfg: OperationsConfig, cargo_area_m2: float, height_tol: float)` with
    `.step(state: dict | None, obs: FrameObs) -> (new_state: dict, OpUpdate)`.
    The state is JSON-serialisable.
  - Public operation dict keys: `id, status ("open"|"closed"), t1, t1_lower_bound, t2,
    occupancy_before, occupancy_after, net_pp, items_appeared, items_disappeared, flags`.

- [ ] **Step 1: Write the failing tests `tests/test_operations.py`**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_operations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.operations'`.

- [ ] **Step 3: Implement `src/deckwatch/timeutil.py`**

```python
"""Timestamp helpers: everything internal is timezone-aware, everything stored is UTC ISO."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_timestamp(s: str, default_tz: str = "UTC") -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(default_tz))
```

- [ ] **Step 4: Implement `src/deckwatch/operations.py`**

```python
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
            "pending_ts": None,
            "active": None,
        }

    def _step_idle(self, s: dict, obs: FrameObs, gap: bool):
        m2 = self._changed_m2(_dec(s["stable_items"]), obs.items)
        pct = 100.0 * m2 / self.area
        changed = m2 > self.cfg.change_min_m2
        if changed and gap:
            op = self._open(s, t1=obs.ts, flags=["t1_t2_uncertain"])
            op["last_change_ts"] = to_iso(obs.ts)
            return s, OpUpdate("IDLE", pct, "T2", self._close(s, op, [obs.occupancy_pct], obs))
        if changed and s["pending_ts"] is not None:
            op = self._open(s, t1=from_iso(s["pending_ts"]), flags=[])
            op.update(last_change_ts=s["pending_ts"], running=_enc(obs.items), quiet_occ=[obs.occupancy_pct])
            s.update(mode="ACTIVE", active=op, pending_ts=None)
            return s, OpUpdate("ACTIVE", pct, "T1", _public(op))
        if changed:
            s["pending_ts"] = to_iso(obs.ts)
            return s, OpUpdate("IDLE", pct, None, None)
        s.update(pending_ts=None, stable_occ=(s["stable_occ"] + [obs.occupancy_pct])[-3:],
                 last_stable_ts=to_iso(obs.ts))
        return s, OpUpdate("IDLE", pct, None, None)

    def _step_active(self, s: dict, obs: FrameObs, gap: bool):
        op = dict(s["active"])
        m2 = self._changed_m2(_dec(op["running"]), obs.items)
        pct = 100.0 * m2 / self.area
        if gap and "gap_during_operation" not in op["flags"]:
            op["flags"] = op["flags"] + ["gap_during_operation"]
        if m2 > self.cfg.change_min_m2:
            op.update(last_change_ts=to_iso(obs.ts), quiet_occ=[obs.occupancy_pct])
        else:
            op["quiet_occ"] = op["quiet_occ"] + [obs.occupancy_pct]
        op["running"] = _enc(obs.items)
        if obs.ts - from_iso(op["last_change_ts"]) >= self.quiet:
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_operations.py -v`
Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/deckwatch/timeutil.py src/deckwatch/operations.py tests/test_operations.py
git commit -m "feat: session-level T1/T2 operation tracker with item matching"
```

---

### Task 7: SQLite store

**Files:**
- Create: `src/deckwatch/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces `deckwatch.store.Store(path)` with these methods:
  - `last_ts() -> str | None`
  - `get_frame(ts, include_items=False) -> dict | None`
  - `latest_frame() -> dict | None`
  - `load_state() -> dict | None`
  - `commit_frame(result: dict, items: list[dict], state: dict | None, operation: dict | None, path: str | None = None)`
    - Atomic. `result` must hold `ts, occupancy_pct, change_pct, flags, calibration_version, model_version`.
    - The operation is upserted by `id`. A `None` state leaves the stored state unchanged.
  - `list_operations(since: str | None = None) -> list[dict]`, filtering on `t1 >= since`, ordered by id
  - `status() -> {"state": "IDLE"|"ACTIVE", "latest_frame": dict|None, "open_operation": dict|None}`
  - `close()`
- The connection uses `check_same_thread=False`. Callers serialise access; the pipeline lock does
  this in Task 10.

- [ ] **Step 1: Write the failing tests `tests/test_store.py`**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.store'`.

- [ ] **Step 3: Implement `src/deckwatch/store.py`**

```python
"""SQLite persistence for frames, items, operations and tracker state."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

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
        with self.conn:
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
            rows = self.conn.execute("SELECT data FROM operations WHERE t1 >= ? ORDER BY id", (since,))
        return [json.loads(r[0]) for r in rows]

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_store.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deckwatch/store.py tests/test_store.py
git commit -m "feat: SQLite store for frames, items, operations and state"
```

---

### Task 8: Camera drift check

**Files:**
- Create: `src/deckwatch/drift.py`
- Test: `tests/test_drift.py`

**Interfaces:**
- Consumes: `DriftConfig` (Task 2).
- Produces, in `deckwatch.drift`:
  - `DriftRef(name: str, center: (x, y), patch: np.ndarray gray)`
  - `extract_refs(gray, centers: dict[str, (x, y)], patch_px) -> list[DriftRef]`. Points too close
    to the border are skipped.
  - `save_refs(refs, directory)`, which writes `<name>.png`
  - `load_refs(directory, centers) -> list[DriftRef]`. Missing files are skipped, and a missing
    directory gives `[]`.
  - `check_drift(gray, refs, cfg) -> (status: "ok"|"suspected"|"unverified", median_shift_px: float|None)`

- [ ] **Step 1: Write the failing tests `tests/test_drift.py`**

```python
import numpy as np
import pytest

from deckwatch.config import DriftConfig
from deckwatch.drift import check_drift, extract_refs, load_refs, save_refs

CFG = DriftConfig(refs_dir=None)
CENTERS = {"W1": (800.0, 500.0), "W2": (3000.0, 500.0), "W3": (3200.0, 1700.0)}


@pytest.fixture(scope="module")
def frame():
    return np.random.default_rng(0).integers(0, 255, (2160, 3840), dtype=np.uint8)


def test_same_frame_is_ok(frame):
    status, shift = check_drift(frame, extract_refs(frame, CENTERS, 48), CFG)
    assert status == "ok" and shift == pytest.approx(0.0, abs=0.5)


def test_shifted_frame_is_suspected(frame):
    refs = extract_refs(frame, CENTERS, 48)
    status, shift = check_drift(np.roll(frame, 25, axis=1), refs, CFG)
    assert status == "suspected" and shift == pytest.approx(25.0, abs=1.0)


def test_occluded_refs_make_check_unverified(frame):
    refs = extract_refs(frame, CENTERS, 48)
    occluded = frame.copy()
    other = np.random.default_rng(1).integers(0, 255, (300, 300), dtype=np.uint8)
    for name in ("W1", "W2"):
        x, y = map(int, CENTERS[name])
        occluded[y - 150:y + 150, x - 150:x + 150] = other
    assert check_drift(occluded, refs, CFG) == ("unverified", None)


def test_no_refs_is_unverified(frame):
    assert check_drift(frame, [], CFG) == ("unverified", None)


def test_border_point_skipped(frame):
    assert [r.name for r in extract_refs(frame, {"W1": (5.0, 5.0)}, 48)] == []


def test_save_and_load_refs(frame, tmp_path):
    refs = extract_refs(frame, CENTERS, 48)
    save_refs(refs, tmp_path / "refs")
    loaded = load_refs(tmp_path / "refs", CENTERS)
    assert [r.name for r in loaded] == ["W1", "W2", "W3"]
    np.testing.assert_array_equal(loaded[0].patch, refs[0].patch)
    assert load_refs(tmp_path / "missing", CENTERS) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_drift.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.drift'`.

- [ ] **Step 3: Implement `src/deckwatch/drift.py`**

```python
"""Detect camera movement by re-finding reference patches around the wall-top points."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from deckwatch.config import DriftConfig


@dataclass(eq=False)
class DriftRef:
    name: str
    center: tuple[float, float]
    patch: np.ndarray            # grayscale, patch_px x patch_px


def extract_refs(gray: np.ndarray, centers: dict, patch_px: int) -> list[DriftRef]:
    half = patch_px // 2
    refs = []
    for name, (x, y) in centers.items():
        x, y = int(round(x)), int(round(y))
        if x - half < 0 or y - half < 0 or x + half > gray.shape[1] or y + half > gray.shape[0]:
            continue
        refs.append(DriftRef(name, (float(x), float(y)), gray[y - half:y + half, x - half:x + half].copy()))
    return refs


def save_refs(refs: list[DriftRef], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for r in refs:
        cv2.imwrite(str(directory / f"{r.name}.png"), r.patch)


def load_refs(directory: str | Path, centers: dict) -> list[DriftRef]:
    directory = Path(directory)
    refs = []
    for name, (x, y) in centers.items():
        f = directory / f"{name}.png"
        patch = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.is_file() else None
        if patch is not None:
            refs.append(DriftRef(name, (float(x), float(y)), patch))
    return refs


def check_drift(gray: np.ndarray, refs: list[DriftRef], cfg: DriftConfig) -> tuple[str, float | None]:
    shifts = []
    for r in refs:
        ph, pw = r.patch.shape
        cx, cy = r.center
        x0 = max(int(round(cx - pw / 2 - cfg.search_px)), 0)
        y0 = max(int(round(cy - ph / 2 - cfg.search_px)), 0)
        x1 = min(int(round(cx + pw / 2 + cfg.search_px)), gray.shape[1])
        y1 = min(int(round(cy + ph / 2 + cfg.search_px)), gray.shape[0])
        window = gray[y0:y1, x0:x1]
        if window.shape[0] < ph or window.shape[1] < pw:
            continue
        res = cv2.matchTemplate(window, r.patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if not np.isfinite(score) or score < cfg.match_min:
            continue          # occluded (crane, person, cargo) - ignore this point
        mx, my = x0 + loc[0] + pw / 2, y0 + loc[1] + ph / 2
        shifts.append(float(np.hypot(mx - cx, my - cy)))
    if len(shifts) < cfg.min_points:
        return "unverified", None
    median = float(np.median(shifts))
    return ("suspected" if median > cfg.drift_px else "ok"), median
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_drift.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deckwatch/drift.py tests/test_drift.py
git commit -m "feat: camera drift check against wall-top reference patches"
```

---

### Task 9: ONNX pose detector

**Files:**
- Create: `src/deckwatch/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `ConfigError`, `DetectorConfig` (Task 2); `Item`, `KEYPOINT_NAMES` (Task 5).
- Produces, in `deckwatch.detector`:
  - `letterbox(img, size) -> (padded (size,size,3) uint8, ratio: float, (pad_x, pad_y))`
  - `decode(raw, ratio, pad, class_names, conf_min, nms_iou) -> list[Item]`, where `raw` is the
    Ultralytics pose ONNX output `(1, 4 + nc + 6*3, N)` (box `cx, cy, w, h`, class scores, then
    `x, y, vis` per keypoint, already sigmoided)
  - `OnnxPoseDetector(model_path, class_names, model_version, imgsz, conf_min, nms_iou)` with
    attribute `model_version` and `detect(frame_bgr) -> list[Item]`
  - `load_detector(cfg: DetectorConfig) -> OnnxPoseDetector`. This reads `<model>.json` metadata
    `{"model_version", "class_names", "keypoint_names", "imgsz"}`, and raises `ConfigError` if the
    model or metadata is missing or the keypoints don't match `KEYPOINT_NAMES`.

- [ ] **Step 1: Write the failing tests `tests/test_detector.py`**

```python
import json

import numpy as np
import pytest

from deckwatch.config import ConfigError, DetectorConfig
from deckwatch.detector import decode, letterbox, load_detector

NAMES = ["container", "other"]


def column(cx, cy, w, h, scores, kp_xy=(0.0, 0.0), vis=0.9):
    kps = []
    for k in range(6):
        kps += [kp_xy[0] + k, kp_xy[1] + k, vis]
    return [cx, cy, w, h, *scores, *kps]


def test_letterbox_4k_frame():
    img = np.zeros((2160, 3840, 3), np.uint8)
    out, r, pad = letterbox(img, 1280)
    assert out.shape == (1280, 1280, 3)
    assert r == pytest.approx(1 / 3) and pad == (0, 280)
    assert out[0, 0, 0] == 114 and out[640, 640, 0] == 0


def test_decode_filters_nms_and_unletterboxes():
    cols = [
        column(640, 640, 100, 100, (0.9, 0.05), kp_xy=(100, 300)),
        column(645, 640, 100, 100, (0.8, 0.05)),        # overlaps the first -> suppressed
        column(200, 200, 50, 50, (0.05, 0.2)),          # below conf_min -> dropped
    ]
    raw = np.array(cols, np.float32).T[None]            # (1, 24, 3)
    items = decode(raw, 0.5, (0, 100), NAMES, 0.4, 0.5)
    assert len(items) == 1
    it = items[0]
    assert it.cls == "container" and it.conf == pytest.approx(0.9)
    np.testing.assert_allclose(it.keypoints[0], (200.0, 400.0))
    np.testing.assert_allclose(it.keypoints[5], ((100 + 5) / 0.5, (300 + 5 - 100) / 0.5))
    np.testing.assert_allclose(it.kp_conf, np.full(6, 0.9), rtol=1e-6)


def test_decode_empty_and_wrong_width():
    assert decode(np.zeros((1, 24, 0), np.float32), 1.0, (0, 0), NAMES, 0.4, 0.5) == []
    with pytest.raises(ValueError, match="output width"):
        decode(np.zeros((1, 20, 3), np.float32), 1.0, (0, 0), NAMES, 0.4, 0.5)


def test_load_detector_missing_model(tmp_path):
    with pytest.raises(ConfigError, match="model not found"):
        load_detector(DetectorConfig(model_path=tmp_path / "m.onnx"))


def test_load_detector_rejects_wrong_keypoints(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"")
    (tmp_path / "m.json").write_text(json.dumps(
        {"model_version": "x", "class_names": NAMES, "keypoint_names": ["a"], "imgsz": 1280}), encoding="utf-8")
    with pytest.raises(ConfigError, match="keypoint"):
        load_detector(DetectorConfig(model_path=tmp_path / "m.onnx"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.detector'`.

- [ ] **Step 3: Implement `src/deckwatch/detector.py`**

```python
"""YOLO11-pose ONNX inference: raw frame -> list[Item] in raw image pixels."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from deckwatch.config import ConfigError, DetectorConfig
from deckwatch.items import KEYPOINT_NAMES, Item

NUM_KPTS = len(KEYPOINT_NAMES)


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[py:py + nh, px:px + nw] = resized
    return out, r, (px, py)


def decode(raw, ratio: float, pad: tuple[int, int], class_names: list[str], conf_min: float,
           nms_iou: float) -> list[Item]:
    x = np.asarray(raw, np.float32)
    if x.ndim == 3:
        x = x[0]
    x = x.T                                            # (N, C)
    nc = len(class_names)
    if x.shape[1] != 4 + nc + NUM_KPTS * 3:
        raise ValueError(f"unexpected model output width {x.shape[1]}, expected {4 + nc + NUM_KPTS * 3}")
    scores = x[:, 4:4 + nc]
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf >= conf_min
    x, cls, conf = x[keep], cls[keep], conf[keep]
    if len(x) == 0:
        return []
    tl = x[:, 0:2] - x[:, 2:4] / 2
    boxes = [[float(a), float(b), float(c), float(d)] for (a, b), (c, d) in zip(tl, x[:, 2:4])]
    idx = np.asarray(cv2.dnn.NMSBoxes(boxes, conf.tolist(), conf_min, nms_iou)).reshape(-1)
    kpts = x[:, 4 + nc:].reshape(-1, NUM_KPTS, 3)
    offset = np.array(pad, np.float32)
    items = []
    for i in idx:
        xy = (kpts[i, :, :2] - offset) / ratio
        items.append(Item(class_names[int(cls[i])], float(conf[i]), xy.astype(np.float64),
                          kpts[i, :, 2].astype(np.float64)))
    return items


class OnnxPoseDetector:
    def __init__(self, model_path: Path, class_names: list[str], model_version: str, imgsz: int,
                 conf_min: float, nms_iou: float):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.model_version = model_version
        self.imgsz, self.conf_min, self.nms_iou = imgsz, conf_min, nms_iou

    def detect(self, frame_bgr: np.ndarray) -> list[Item]:
        img, r, pad = letterbox(frame_bgr, self.imgsz)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: blob})[0]
        return decode(raw, r, pad, self.class_names, self.conf_min, self.nms_iou)


def load_detector(cfg: DetectorConfig) -> OnnxPoseDetector:
    model = Path(cfg.model_path)
    meta_path = model.with_suffix(".json")
    if not model.is_file():
        raise ConfigError(f"model not found: {model}")
    if not meta_path.is_file():
        raise ConfigError(f"model metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if list(meta.get("keypoint_names", [])) != list(KEYPOINT_NAMES):
        raise ConfigError(f"model keypoint_names {meta.get('keypoint_names')} do not match {list(KEYPOINT_NAMES)}")
    return OnnxPoseDetector(model, meta["class_names"], meta["model_version"], int(meta.get("imgsz", cfg.imgsz)),
                            cfg.conf_min, cfg.nms_iou)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_detector.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deckwatch/detector.py tests/test_detector.py
git commit -m "feat: ONNX pose detector with letterbox and output decoding"
```

---

### Task 10: Pipeline

**Files:**
- Create: `src/deckwatch/pipeline.py`
- Modify: `tests/synth.py` (append `write_config`, `FakeDetector`, `jpeg_bytes`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above. Specifically `Config`, `ConfigError`, `Calibration.from_dict`,
  `place_item`, `compute_occupancy`, `OperationTracker`, `FrameObs`, `Store`, `check_drift`,
  `load_refs`, `to_iso`, `parse_timestamp`.
- Produces, in `deckwatch.pipeline`:
  - `PipelineError(code, message, http_status)` with `.to_dict() -> {"error", "message"}`
  - `Pipeline(config, detector, store, refs=None)`. With `refs=None`, the drift refs are loaded
    from `config.drift.refs_dir` for the clicked W points. It raises `ConfigError` when
    `config.calibration` is missing.
  - Methods, all serialised by one lock:
    - `analyze(image_bytes, ts: aware datetime, path=None, include_items=False) -> dict`
    - `status() -> dict`
    - `operations(since: str | None = None) -> list[dict]`, where `since` is ISO and a naive
      value is read in `filename_tz`
    - `frame(ts: str, include_items=False) -> dict | None`
  - FrameResult keys: `ts, occupancy_pct, occupied_m2, count_container, count_other,
    count_by_stack_level, change_pct, state, operation, flags, calibration_version, model_version`,
    plus `items` when requested. `operation` is `{"event": "T1"|"T2"|None, **public op}` or `None`.
- Produces, in `tests/synth.py`: `write_config(tmp_path, calibration=None) -> Path`,
  `FakeDetector` (`.items`, `.calls`, `model_version = "fake-v1"`), and
  `jpeg_bytes(size=IMAGE_SIZE) -> bytes`.

- [ ] **Step 1: Append to `tests/synth.py`**

```python


def write_config(tmp_path, calibration=None):
    import yaml

    data = {
        "image_size": list(IMAGE_SIZE),
        "filename_tz": "UTC",
        "db_path": "deckwatch.db",
        "deck": {"length_m": DECK_L, "width_m": DECK_W, "wall_height_m": WALL_H, "wall_offset_m": 0.0},
        "detector": {"model_path": "model.onnx"},
        "geometry": {"standard_sizes": [{"name": "10ft", "length_m": 2.99, "width_m": 2.44},
                                        {"name": "20ft", "length_m": 6.06, "width_m": 2.44}]},
    }
    if calibration is not None:
        data["calibration"] = calibration.to_dict()
    path = tmp_path / "deck.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class FakeDetector:
    model_version = "fake-v1"

    def __init__(self):
        self.items = []
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.items)


def jpeg_bytes(size=IMAGE_SIZE):
    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((size[1], size[0], 3), np.uint8))
    assert ok
    return buf.tobytes()
```

- [ ] **Step 2: Write the failing tests `tests/test_pipeline.py`**

```python
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from deckwatch.config import ConfigError, load_config
from deckwatch.items import Item
from deckwatch.pipeline import Pipeline, PipelineError
from deckwatch.store import Store
from synth import FakeDetector, box_keypoints, default_camera, jpeg_bytes, make_calibration, write_config

T0 = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
P = default_camera()


def container(x0, y0, h=2.6):
    return Item("container", 0.9, box_keypoints(P, x0, y0, 6.06, 2.44, h), np.full(6, 0.9))


@pytest.fixture
def setup(tmp_path):
    cfg = load_config(write_config(tmp_path, make_calibration()))
    det = FakeDetector()
    pipe = Pipeline(cfg, det, Store(cfg.db_path))
    return pipe, det


def at(minute):
    return T0 + timedelta(minutes=minute)


def test_empty_deck_frame(setup):
    pipe, _ = setup
    r = pipe.analyze(jpeg_bytes(), at(0))
    assert r["occupancy_pct"] == 0.0 and r["state"] == "IDLE" and r["operation"] is None
    assert r["count_container"] == 0 and r["flags"] == ["drift_unverified"]
    assert r["ts"] == "2026-09-20T08:00:00Z" and r["model_version"] == "fake-v1"


def test_one_container_with_items(setup):
    pipe, det = setup
    det.items = [container(4.0, 10.0)]
    r = pipe.analyze(jpeg_bytes(), at(0), include_items=True)
    assert r["occupancy_pct"] == pytest.approx(100 * 6.06 * 2.44 / 1200, abs=0.01)
    assert r["count_container"] == 1 and r["count_by_stack_level"] == {"1": 1}
    assert r["items"][0]["height_source"] == "measured"


def test_duplicate_ts_returns_stored_result_without_detecting(setup):
    pipe, det = setup
    first = pipe.analyze(jpeg_bytes(), at(0))
    again = pipe.analyze(jpeg_bytes(), at(0))
    assert again == first and det.calls == 1


def test_out_of_order_rejected(setup):
    pipe, _ = setup
    pipe.analyze(jpeg_bytes(), at(3))
    with pytest.raises(PipelineError) as e:
        pipe.analyze(jpeg_bytes(), at(0))
    assert e.value.code == "out_of_order" and e.value.http_status == 409


def test_unreadable_and_wrong_resolution_store_nothing(setup):
    pipe, _ = setup
    with pytest.raises(PipelineError) as e:
        pipe.analyze(b"not a jpeg", at(0))
    assert e.value.code == "unreadable_image" and e.value.http_status == 422
    with pytest.raises(PipelineError) as e:
        pipe.analyze(jpeg_bytes((640, 480)), at(0))
    assert e.value.code == "wrong_resolution"
    assert pipe.store.last_ts() is None


def test_operation_events_flow_through(setup):
    pipe, det = setup
    det.items = [container(2.0, 5.0)]
    pipe.analyze(jpeg_bytes(), at(0))
    pipe.analyze(jpeg_bytes(), at(3))
    det.items = [container(2.0, 5.0), container(2.0, 12.0)]
    pipe.analyze(jpeg_bytes(), at(6))
    r = pipe.analyze(jpeg_bytes(), at(9))
    assert r["state"] == "ACTIVE"
    assert r["operation"]["event"] == "T1" and r["operation"]["t1"] == "2026-09-20T08:06:00Z"
    assert pipe.status()["open_operation"]["id"] == 1
    assert len(pipe.operations()) == 1


def test_drift_suspected_frame_does_not_drive_operations(setup, monkeypatch):
    pipe, det = setup
    det.items = [container(2.0, 5.0)]
    pipe.analyze(jpeg_bytes(), at(0))
    monkeypatch.setattr("deckwatch.pipeline.check_drift", lambda *a: ("suspected", 30.0))
    det.items = [container(2.0, 5.0), container(2.0, 12.0)]
    r1 = pipe.analyze(jpeg_bytes(), at(3))
    r2 = pipe.analyze(jpeg_bytes(), at(6))
    assert "drift_suspected" in r1["flags"] and r1["change_pct"] is None
    assert r2["operation"] is None and r2["state"] == "IDLE"


def test_missing_calibration_raises(tmp_path):
    cfg = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="calibrat"):
        Pipeline(cfg, FakeDetector(), Store(cfg.db_path))


def test_frame_lookup(setup):
    pipe, _ = setup
    pipe.analyze(jpeg_bytes(), at(0))
    assert pipe.frame("2026-09-20T08:00:00Z")["ts"] == "2026-09-20T08:00:00Z"
    assert pipe.frame("2026-09-20T09:00:00Z") is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.pipeline'`.

- [ ] **Step 4: Implement `src/deckwatch/pipeline.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: 9 passed.

- [ ] **Step 6: Run the full suite**

Run: `venv/Scripts/python -m pytest -q`
Expected: all tests pass (Tasks 1–10).

- [ ] **Step 7: Commit**

```bash
git add src/deckwatch/pipeline.py tests/synth.py tests/test_pipeline.py
git commit -m "feat: analysis pipeline gluing drift, detection, occupancy, T1/T2 and storage"
```

---

### Task 11: CLI

**Files:**
- Create: `src/deckwatch/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config`, `ConfigError` (Task 2); `load_detector` (Task 9); `Store` (Task 7);
  `Pipeline`, `PipelineError` (Task 10); `parse_timestamp`, `to_iso` (Task 6).
- Produces, in `deckwatch.cli`:
  - `parse_ts(path, explicit: str | None, filename_tz: str) -> aware datetime`. Order: explicit
    `--ts`, then filename `YYYYMMDD_HHMMSS` in `filename_tz`, then file mtime (UTC). An invalid
    `--ts` raises `PipelineError("invalid_timestamp", ..., 422)`.
  - `build_pipeline(config_path) -> Pipeline`
  - `main(argv=None) -> int`, with subcommands:
    - `analyze <frame> [--ts] [--items]`
    - `status`
    - `operations [--since]`
    - `serve [--host 127.0.0.1] [--port 8000]`
    - Global option `--config` (default `config/deck.yaml`).
    - Output: one JSON document on stdout. Exit codes 0 / 2 / 3.

- [ ] **Step 1: Write the failing tests `tests/test_cli.py`**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.cli'`.

- [ ] **Step 3: Implement `src/deckwatch/cli.py`**

```python
"""deckwatch command line: every command prints one JSON document to stdout."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from deckwatch.config import ConfigError, load_config
from deckwatch.pipeline import Pipeline, PipelineError
from deckwatch.store import Store
from deckwatch.timeutil import parse_timestamp, to_iso

TS_PATTERN = re.compile(r"(\d{8})_(\d{6})")
EXIT_OK, EXIT_FRAME_ERROR, EXIT_CONFIG_ERROR = 0, 2, 3


def parse_ts(path: str | Path, explicit: str | None, filename_tz: str) -> datetime:
    if explicit:
        try:
            return parse_timestamp(explicit, filename_tz)
        except ValueError as exc:
            raise PipelineError("invalid_timestamp", f"invalid --ts value: {explicit}", 422) from exc
    m = TS_PATTERN.search(Path(path).name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo(filename_tz))
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)


def build_pipeline(config_path: str | Path) -> Pipeline:
    from deckwatch.detector import load_detector

    cfg = load_config(config_path)
    return Pipeline(cfg, load_detector(cfg.detector), Store(cfg.db_path))


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _analyze(args) -> int:
    frame = Path(args.frame)
    if not frame.is_file():
        raise PipelineError("unreadable_image", f"file not found: {frame}", 422)
    pipeline = build_pipeline(args.config)
    ts = parse_ts(frame, args.ts, pipeline.config.filename_tz)
    _print(pipeline.analyze(frame.read_bytes(), ts, path=str(frame), include_items=args.items))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deckwatch")
    parser.add_argument("--config", default="config/deck.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="analyze one frame and print the FrameResult JSON")
    a.add_argument("frame")
    a.add_argument("--ts", help="ISO 8601 timestamp; default: from filename, else file mtime")
    a.add_argument("--items", action="store_true", help="include per-item details")
    sub.add_parser("status", help="current state, latest frame and open operation")
    o = sub.add_parser("operations", help="list operations")
    o.add_argument("--since", help="ISO 8601; only operations with t1 >= since")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    try:
        if args.command == "analyze":
            return _analyze(args)
        cfg = load_config(args.config)
        if args.command == "status":
            _print(Store(cfg.db_path).status())
        elif args.command == "operations":
            since = to_iso(parse_ts("", args.since, cfg.filename_tz)) if args.since else None
            _print(Store(cfg.db_path).list_operations(since))
        elif args.command == "serve":
            from deckwatch.api import serve

            serve(args.config, args.host, args.port)
        return EXIT_OK
    except ConfigError as exc:
        _print({"error": "config_error", "message": str(exc)})
        return EXIT_CONFIG_ERROR
    except PipelineError as exc:
        _print(exc.to_dict())
        return EXIT_FRAME_ERROR
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: 7 passed.

- [ ] **Step 5: Smoke-test the installed entry point**

Run: `venv/Scripts/deckwatch --config config/deck.example.yaml status; echo "exit=$?"`
Expected:
`{"error": "config_error", "message": "deck.length_m must be provided in deck.yaml (measured on the vessel)"}`
followed by `exit=3`. The unfilled example config is refused.

- [ ] **Step 6: Commit**

```bash
git add src/deckwatch/cli.py tests/test_cli.py
git commit -m "feat: deckwatch CLI with analyze, status, operations and serve"
```

---

### Task 12: HTTP API

**Files:**
- Create: `src/deckwatch/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Pipeline`, `PipelineError` (Task 10); `parse_timestamp` (Task 6);
  `build_pipeline` (Task 11).
- Produces, in `deckwatch.api`:
  - `create_app(pipeline: Pipeline) -> FastAPI`, with routes:
    - `POST /frames` (multipart `file` + form `ts`, query `items`)
    - `GET /status`
    - `GET /operations?since=`
    - `GET /frames/{ts}?items=`
  - `serve(config_path, host, port)`
  - Errors return the `PipelineError.to_dict()` body with its HTTP status. An unknown frame gives
    404 `{"error": "not_found"}`.

- [ ] **Step 1: Write the failing tests `tests/test_api.py`**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deckwatch.api'`.

- [ ] **Step 3: Implement `src/deckwatch/api.py`**

```python
"""HTTP API over the same Pipeline the CLI uses (for the Laravel integration)."""
from __future__ import annotations

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from deckwatch.pipeline import Pipeline, PipelineError
from deckwatch.timeutil import parse_timestamp


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="DeckWatch")

    @app.exception_handler(PipelineError)
    def _pipeline_error(request: Request, exc: PipelineError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.post("/frames")
    def post_frame(file: UploadFile = File(...), ts: str = Form(...), items: bool = Query(False)):
        try:
            dt = parse_timestamp(ts, pipeline.config.filename_tz)
        except ValueError as exc:
            raise PipelineError("invalid_timestamp", f"invalid ts: {ts}", 422) from exc
        return pipeline.analyze(file.file.read(), dt, path=file.filename, include_items=items)

    @app.get("/status")
    def status():
        return pipeline.status()

    @app.get("/operations")
    def operations(since: str | None = None):
        try:
            return pipeline.operations(since)
        except ValueError as exc:
            raise PipelineError("invalid_timestamp", f"invalid since: {since}", 422) from exc

    @app.get("/frames/{ts}")
    def frame(ts: str, items: bool = Query(False)):
        result = pipeline.frame(ts, include_items=items)
        if result is None:
            return JSONResponse(status_code=404, content={"error": "not_found", "message": f"no frame at {ts}"})
        return result

    return app


def serve(config_path: str, host: str, port: int) -> None:
    import uvicorn

    from deckwatch.cli import build_pipeline

    uvicorn.run(create_app(build_pipeline(config_path)), host=host, port=port)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full suite**

Run: `venv/Scripts/python -m pytest -q`
Expected: all tests pass, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/deckwatch/api.py tests/test_api.py
git commit -m "feat: FastAPI HTTP interface over the analysis pipeline"
```
