"""Fallback side-view quadruped pose for runs without a trained checkpoint."""

from __future__ import annotations

import math

import cv2
import numpy as np

from dog_gait.pose.schema import BODYPARTS


RIGHT_FACING_TEMPLATE = {
    "nose": (0.93, 0.30),
    "upper_jaw": (0.90, 0.27),
    "lower_jaw": (0.90, 0.34),
    "mouth_end_right": (0.87, 0.32),
    "mouth_end_left": (0.87, 0.32),
    "right_eye": (0.82, 0.23),
    "left_eye": (0.84, 0.23),
    "right_earbase": (0.76, 0.18),
    "left_earbase": (0.78, 0.18),
    "right_earend": (0.73, 0.08),
    "left_earend": (0.79, 0.08),
    "neck_base": (0.69, 0.34),
    "neck_end": (0.72, 0.26),
    "throat_base": (0.73, 0.40),
    "throat_end": (0.80, 0.36),
    "back_base": (0.63, 0.31),
    "back_middle": (0.45, 0.33),
    "back_end": (0.27, 0.34),
    "tail_base": (0.15, 0.33),
    "tail_end": (0.04, 0.21),
    "front_left_thigh": (0.66, 0.53),
    "front_left_knee": (0.68, 0.72),
    "front_left_paw": (0.73, 0.93),
    "front_right_thigh": (0.73, 0.52),
    "front_right_knee": (0.75, 0.72),
    "front_right_paw": (0.78, 0.92),
    "back_left_thigh": (0.31, 0.54),
    "back_left_knee": (0.29, 0.72),
    "back_left_paw": (0.24, 0.93),
    "back_right_thigh": (0.40, 0.53),
    "back_right_knee": (0.39, 0.72),
    "back_right_paw": (0.36, 0.92),
    "belly_bottom": (0.50, 0.61),
    "body_middle_right": (0.43, 0.46),
    "body_middle_left": (0.53, 0.46),
}


def estimate_subject_bbox(frame_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Estimate a dog-sized central bounding box.

    GrabCut gives a decent foreground proposal for many side-view phone videos.
    If it fails, the fallback still avoids placing points on the extreme frame
    corners, which is the main failure mode of an untrained heatmap model.
    """

    h, w = frame_bgr.shape[:2]
    fallback = (int(0.07 * w), int(0.10 * h), int(0.93 * w), int(0.88 * h))
    if h < 32 or w < 32:
        return 0, 0, w, h

    rect = (int(0.06 * w), int(0.08 * h), int(0.88 * w), int(0.82 * h))
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        small = frame_bgr
        scale = 1.0
        if max(h, w) > 900:
            scale = 900 / max(h, w)
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            rect_small = (int(rect[0] * scale), int(rect[1] * scale), int(rect[2] * scale), int(rect[3] * scale))
            mask = np.zeros((sh, sw), np.uint8)
        else:
            rect_small = rect
        cv2.grabCut(small, mask, rect_small, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback
        contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(contour)
        if scale != 1.0:
            x, y, bw, bh = [int(v / scale) for v in (x, y, bw, bh)]
        area_ratio = (bw * bh) / max(1, w * h)
        aspect = bw / max(1, bh)
        if area_ratio < 0.05 or area_ratio > 0.85 or aspect < 0.8 or aspect > 5.5:
            return fallback
        pad_x = int(0.06 * bw)
        pad_y = int(0.12 * bh)
        return max(0, x - pad_x), max(0, y - pad_y), min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    except cv2.error:
        return fallback


def template_keypoints(
    frame_bgr: np.ndarray,
    frame_index: int,
    facing: str = "right",
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return a coherent ``(bodyparts, 3)`` quadruped pose template."""

    if bbox is None:
        bbox = estimate_subject_bbox(frame_bgr)
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    phase = frame_index * 0.28
    gait_offsets = {
        "front_left_paw": math.sin(phase),
        "back_right_paw": math.sin(phase),
        "front_right_paw": math.sin(phase + math.pi),
        "back_left_paw": math.sin(phase + math.pi),
    }
    knee_offsets = {
        "front_left_knee": math.sin(phase + 0.5),
        "back_right_knee": math.sin(phase + 0.5),
        "front_right_knee": math.sin(phase + math.pi + 0.5),
        "back_left_knee": math.sin(phase + math.pi + 0.5),
    }

    keypoints = []
    for part in BODYPARTS:
        nx, ny = RIGHT_FACING_TEMPLATE.get(part, (0.5, 0.5))
        if facing == "left":
            nx = 1.0 - nx
        if part in gait_offsets:
            sign = -1 if facing == "left" else 1
            nx += sign * 0.035 * gait_offsets[part]
            ny += 0.025 * abs(gait_offsets[part])
        if part in knee_offsets:
            nx += 0.018 * knee_offsets[part]
        x = x1 + np.clip(nx, 0.02, 0.98) * width
        y = y1 + np.clip(ny, 0.02, 0.98) * height
        keypoints.append((float(x), float(y), 0.95))
    return np.array(keypoints, dtype=np.float32)
