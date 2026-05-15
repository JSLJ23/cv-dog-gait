"""Image and video preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class FrameTransform:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale: float

    @property
    def scale_x(self) -> float:
        return 1.0 / self.scale

    @property
    def scale_y(self) -> float:
        return 1.0 / self.scale


def _round_up(value: int, multiple: int) -> int:
    return int(np.ceil(value / multiple) * multiple)


def preprocess_frame(
    frame_bgr: np.ndarray,
    input_size: int = 256,
    stride_multiple: int = 32,
) -> tuple[torch.Tensor, FrameTransform]:
    """Convert a BGR frame to a normalized, aspect-preserving tensor.

    The longer image side is resized to ``input_size``. The shorter side is
    scaled by the same factor, then the image is centered on a padded canvas
    whose dimensions are multiples of ``stride_multiple``. This avoids
    distorting 16:9 video while keeping HRNet/FPN-friendly tensor sizes.
    """

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    scale = input_size / max(h, w)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas_w = max(stride_multiple, _round_up(resized_w, stride_multiple))
    canvas_h = max(stride_multiple, _round_up(resized_h, stride_multiple))
    pad_left = (canvas_w - resized_w) // 2
    pad_top = (canvas_h - resized_h) // 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=resized.dtype)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized

    tensor = torch.from_numpy(canvas).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, FrameTransform(
        original_width=w,
        original_height=h,
        input_width=canvas_w,
        input_height=canvas_h,
        resized_width=resized_w,
        resized_height=resized_h,
        pad_left=pad_left,
        pad_top=pad_top,
        scale=scale,
    )


def scale_keypoints_to_frame(keypoints: np.ndarray, transform: FrameTransform) -> np.ndarray:
    scaled = keypoints.copy()
    scaled[..., 0] = (scaled[..., 0] - transform.pad_left) * transform.scale_x
    scaled[..., 1] = (scaled[..., 1] - transform.pad_top) * transform.scale_y
    scaled[..., 0] = np.clip(scaled[..., 0], 0, transform.original_width - 1)
    scaled[..., 1] = np.clip(scaled[..., 1], 0, transform.original_height - 1)
    return scaled
