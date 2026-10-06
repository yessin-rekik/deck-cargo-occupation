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
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid model metadata JSON in {meta_path}: {exc}") from exc
    for key in ("class_names", "model_version"):
        if key not in meta:
            raise ConfigError(f"model metadata {meta_path} missing key '{key}'")
    if list(meta.get("keypoint_names", [])) != list(KEYPOINT_NAMES):
        raise ConfigError(f"model keypoint_names {meta.get('keypoint_names')} do not match {list(KEYPOINT_NAMES)}")
    try:
        return OnnxPoseDetector(model, meta["class_names"], meta["model_version"], int(meta.get("imgsz", cfg.imgsz)),
                                cfg.conf_min, cfg.nms_iou)
    except Exception as exc:
        raise ConfigError(f"cannot load model {model}: {exc}") from exc
