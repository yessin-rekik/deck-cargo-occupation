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
