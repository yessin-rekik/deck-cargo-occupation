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
