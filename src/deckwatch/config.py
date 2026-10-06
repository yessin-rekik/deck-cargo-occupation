"""Load and validate deck.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


def _num(dotted: str, value, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{dotted} must be a number, got {value!r}") from exc


def _parse(raw: dict, path: Path, base: Path) -> Config:
    raw_img = _req(raw, "image_size")
    try:
        width_px, height_px = raw_img
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"image_size must be exactly two positive integers, got {raw_img!r}") from exc
    width_px = _num("image_size", width_px, int)
    height_px = _num("image_size", height_px, int)
    if width_px <= 0 or height_px <= 0:
        raise ConfigError(f"image_size must be exactly two positive integers, got {raw_img!r}")

    dims = {}
    for name in ("length_m", "width_m", "wall_height_m", "wall_offset_m"):
        value = _req(raw, f"deck.{name}")
        if value is None:
            raise ConfigError(f"deck.{name} must be provided in deck.yaml (measured on the vessel)")
        dims[name] = _num(f"deck.{name}", value)
    for name in ("length_m", "width_m", "wall_height_m"):
        if dims[name] <= 0:
            raise ConfigError(f"deck.{name} must be > 0")
    if dims["wall_offset_m"] < 0:
        raise ConfigError("deck.wall_offset_m must be >= 0")
    deck = DeckConfig(**dims)

    det = raw.get("detector") or {}
    imgsz = _num("detector.imgsz", det.get("imgsz", 1280), int)
    conf_min = _num("detector.conf_min", det.get("conf_min", 0.4))
    nms_iou = _num("detector.nms_iou", det.get("nms_iou", 0.5))
    kp_conf_min = _num("detector.kp_conf_min", det.get("kp_conf_min", 0.5))
    if imgsz <= 0:
        raise ConfigError("detector.imgsz must be > 0")
    if not (0 < conf_min <= 1):
        raise ConfigError("detector.conf_min must be in (0, 1]")
    if not (0 < nms_iou <= 1):
        raise ConfigError("detector.nms_iou must be in (0, 1]")
    if not (0 <= kp_conf_min <= 1):
        raise ConfigError("detector.kp_conf_min must be in [0, 1]")
    detector = DetectorConfig(
        model_path=base / _req(raw, "detector.model_path"),
        imgsz=imgsz,
        conf_min=conf_min,
        nms_iou=nms_iou,
        kp_conf_min=kp_conf_min,
    )

    geo = raw.get("geometry") or {}
    unit_height_m = _num("geometry.unit_height_m", geo.get("unit_height_m", 2.6))
    max_stack = _num("geometry.max_stack", geo.get("max_stack", 3), int)
    size_tol = _num("geometry.size_tol", geo.get("size_tol", 0.15))
    if unit_height_m <= 0:
        raise ConfigError("geometry.unit_height_m must be > 0")
    if max_stack < 1:
        raise ConfigError("geometry.max_stack must be >= 1")
    if size_tol <= 0:
        raise ConfigError("geometry.size_tol must be > 0")

    default_height_m = {"container": 2.6, "other": 1.0}
    raw_dh = geo.get("default_height_m") or {}
    if not isinstance(raw_dh, dict):
        raise ConfigError(f"geometry.default_height_m must be a mapping, got {raw_dh!r}")
    for key, value in raw_dh.items():
        height = _num(f"geometry.default_height_m.{key}", value)
        if height <= 0:
            raise ConfigError(f"geometry.default_height_m.{key} must be > 0")
        default_height_m[str(key)] = height

    standard_sizes = []
    for entry in geo.get("standard_sizes") or []:
        try:
            name = str(entry["name"])
            length_m = float(entry["length_m"])
            width_m = float(entry["width_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"geometry.standard_sizes entry is malformed: {entry!r} ({exc})") from exc
        if length_m <= 0 or width_m <= 0:
            raise ConfigError(f"geometry.standard_sizes '{name}' must have length_m > 0 and width_m > 0")
        standard_sizes.append(StandardSize(name, length_m, width_m))

    geometry = GeometryConfig(
        unit_height_m=unit_height_m,
        max_stack=max_stack,
        default_height_m=default_height_m,
        size_tol=size_tol,
        standard_sizes=tuple(standard_sizes),
    )

    ops = raw.get("operations") or {}
    change_min_m2 = _num("operations.change_min_m2", ops.get("change_min_m2", 2.0))
    quiet_period_min = _num("operations.quiet_period_min", ops.get("quiet_period_min", 30.0))
    gap_max_min = _num("operations.gap_max_min", ops.get("gap_max_min", 15.0))
    match_iou = _num("operations.match_iou", ops.get("match_iou", 0.3))
    if change_min_m2 < 0:
        raise ConfigError("operations.change_min_m2 must be >= 0")
    if quiet_period_min <= 0:
        raise ConfigError("operations.quiet_period_min must be > 0")
    if gap_max_min <= 0:
        raise ConfigError("operations.gap_max_min must be > 0")
    if not (0 < match_iou <= 1):
        raise ConfigError("operations.match_iou must be in (0, 1]")
    operations = OperationsConfig(
        change_min_m2=change_min_m2,
        quiet_period_min=quiet_period_min,
        gap_max_min=gap_max_min,
        match_iou=match_iou,
    )

    dr = raw.get("drift") or {}
    patch_px = _num("drift.patch_px", dr.get("patch_px", 48), int)
    search_px = _num("drift.search_px", dr.get("search_px", 60), int)
    match_min = _num("drift.match_min", dr.get("match_min", 0.6))
    drift_px = _num("drift.drift_px", dr.get("drift_px", 15.0))
    min_points = _num("drift.min_points", dr.get("min_points", 2), int)
    if patch_px <= 0:
        raise ConfigError("drift.patch_px must be > 0")
    if search_px < 0:
        raise ConfigError("drift.search_px must be >= 0")
    if not (0 < match_min <= 1):
        raise ConfigError("drift.match_min must be in (0, 1]")
    if drift_px <= 0:
        raise ConfigError("drift.drift_px must be > 0")
    if min_points < 1:
        raise ConfigError("drift.min_points must be >= 1")
    drift = DriftConfig(
        refs_dir=base / dr.get("refs_dir", "drift_refs"),
        patch_px=patch_px,
        search_px=search_px,
        match_min=match_min,
        drift_px=drift_px,
        min_points=min_points,
    )

    filename_tz = str(raw.get("filename_tz", "UTC"))
    try:
        ZoneInfo(filename_tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"filename_tz '{filename_tz}' is not a valid IANA time zone") from exc

    return Config(
        path=path,
        image_size=(int(width_px), int(height_px)),
        filename_tz=filename_tz,
        deck=deck,
        detector=detector,
        geometry=geometry,
        operations=operations,
        drift=drift,
        db_path=base / raw.get("db_path", "../deckwatch.db"),
        calibration=raw.get("calibration"),
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    base = path.parent

    try:
        return _parse(raw, path, base)
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"deck.yaml: {exc}") from exc
