"""Dog bounding box provider with a safe full-frame fallback."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


COCO_DOG_LABEL = 18


@lru_cache(maxsize=1)
def _load_torchvision_detector():
    try:
        import torch
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        from torchvision.transforms.functional import to_tensor
    except Exception:
        return None

    try:
        weights = None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None)
        model.eval()
        return torch, to_tensor, model
    except Exception:
        return None


def detect_dog_bbox(frame_bgr: np.ndarray, score_threshold: float = 0.6) -> tuple[int, int, int, int]:
    """Return ``x1, y1, x2, y2`` for the most likely dog.

    If no local detector weights are available, it returns the full frame, which
    keeps the top-down pose pipeline operational.
    """

    h, w = frame_bgr.shape[:2]
    fallback = (0, 0, w, h)
    detector = _load_torchvision_detector()
    if detector is None:
        return fallback

    torch, to_tensor, model = detector
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        pred = model([to_tensor(rgb)])[0]

    best = None
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"], strict=False):
        if int(label) == COCO_DOG_LABEL and float(score) >= score_threshold:
            if best is None or float(score) > best[0]:
                best = (float(score), box.detach().cpu().numpy())
    if best is None:
        return fallback
    x1, y1, x2, y2 = best[1].astype(int).tolist()
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
