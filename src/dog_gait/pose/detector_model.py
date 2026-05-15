"""Faster R-CNN detector model for checkpoint tests and future inference."""

from __future__ import annotations

import torch
import torchvision.models.detection as detection
from torch import nn


class FasterRCNNDetector(nn.Module):
    """Faster R-CNN detector wrapper for quadruped checkpoints."""

    def __init__(self, box_score_thresh: float = 0.01) -> None:
        super().__init__()
        self.model = detection.fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            box_score_thresh=box_score_thresh,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    def forward(
        self,
        x: list[torch.Tensor] | torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        if isinstance(x, torch.Tensor):
            x = list(x)
        return self.model(x, targets)
