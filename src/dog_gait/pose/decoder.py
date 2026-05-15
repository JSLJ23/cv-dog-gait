"""Heatmap decoding utilities."""

from __future__ import annotations

import torch


def decode_heatmaps(heatmaps: torch.Tensor, stride: float, apply_sigmoid: bool = False) -> torch.Tensor:
    """Decode heatmaps to ``(batch, joints, 3)`` x/y/confidence tensors."""

    if heatmaps.dim() != 4:
        raise ValueError(f"Expected heatmaps shaped (B, K, H, W), got {tuple(heatmaps.shape)}")
    if apply_sigmoid:
        heatmaps = torch.sigmoid(heatmaps)

    batch_size, num_joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch_size, num_joints, height * width)
    scores, indexes = flat.max(dim=2)
    y = torch.div(indexes, width, rounding_mode="floor").float()
    x = (indexes % width).float()
    coords = torch.stack(
        [
            x * stride + 0.5 * stride,
            y * stride + 0.5 * stride,
            scores.clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    return coords
