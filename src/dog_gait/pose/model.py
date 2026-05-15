"""PyTorch HRNet-W32 heatmap pose model.

The module preserves common HRNet-W32 top-down pose checkpoint key names so
compatible state dictionaries can be loaded directly:

``backbone.model.*`` and ``heads.bodypart.heatmap_head.*``.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import nn

from dog_gait.pose.schema import BODYPARTS, CHECKPOINT_BODYPARTS, CHECKPOINT_TO_DOG_INDICES


def _pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h))


class HRNetBackbone(nn.Module):
    """HRNet backbone used by the top-down pose model."""

    def __init__(
        self,
        stride: int = 4,
        model_name: str = "hrnet_w32",
        pretrained: bool = False,
        interpolate_branches: bool = False,
        increased_channel_count: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            feature_location="incre" if increased_channel_count else "",
            out_indices=(1, 2, 3, 4),
        )
        self.interpolate_branches = interpolate_branches
        self.out_channels = 32 if model_name == "hrnet_w32" and not interpolate_branches else self.model.feature_info.channels()[0]

    def prepare_output(self, y_list: list[torch.Tensor]) -> torch.Tensor:
        if not self.interpolate_branches:
            return y_list[0]

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        return torch.cat(
            [
                y_list[0],
                F.interpolate(y_list[1], size=(x0_h, x0_w), mode="bilinear"),
                F.interpolate(y_list[2], size=(x0_h, x0_w), mode="bilinear"),
                F.interpolate(y_list[3], size=(x0_h, x0_w), mode="bilinear"),
            ],
            1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prepare_output(self.model(x))


class DeconvModule(nn.Module):
    """Deconvolutional prediction module for heatmap outputs."""

    def __init__(
        self,
        channels: list[int],
        kernel_size: list[int],
        strides: list[int],
        final_conv: dict | None = None,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(kernel_size) + 1 == len(strides) + 1):
            raise ValueError("There must be one more channel entry than kernel sizes and strides.")
        in_channels = channels[0]
        head_stride = 1
        self.deconv_layers = nn.Identity()
        if len(kernel_size) > 0:
            layers: list[nn.Module] = []
            for out_channels, kernel, stride in zip(channels[1:], kernel_size, strides, strict=False):
                layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride))
                layers.append(nn.ReLU())
                in_channels = out_channels
                head_stride *= stride
            self.deconv_layers = nn.Sequential(*layers[:-1])
        self.stride = head_stride
        self.final_conv = nn.Identity()
        if final_conv is not None:
            self.final_conv = nn.Conv2d(
                in_channels=channels[-1],
                out_channels=final_conv["out_channels"],
                kernel_size=final_conv["kernel_size"],
                stride=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_conv(self.deconv_layers(x))


class HeatmapHead(nn.Module):
    """Heatmap head configured for quadruped HRNet-W32 checkpoints."""

    def __init__(self, num_keypoints: int, backbone_output_channels: int = 32) -> None:
        super().__init__()
        self.heatmap_head = DeconvModule(
            channels=[backbone_output_channels, num_keypoints],
            kernel_size=[1],
            strides=[1],
        )
        self.stride = self.heatmap_head.stride
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"heatmap": self.heatmap_head(x)}


class HeatmapPoseModel(nn.Module):
    """Top-down HRNet-W32 pose model for dog bodypart heatmaps."""

    def __init__(self, num_keypoints: int, width: int = 32) -> None:
        super().__init__()
        if width != 32:
            raise ValueError("The quadruped HRNet-W32 pose model uses width=32.")
        self.backbone = HRNetBackbone(
            model_name="hrnet_w32",
            pretrained=False,
            interpolate_branches=False,
            increased_channel_count=False,
        )
        self.heads = nn.ModuleDict({"bodypart": HeatmapHead(num_keypoints, backbone_output_channels=32)})
        self.neck = None
        self.output_features = False
        self._strides = {"bodypart": self.backbone.stride / self.heads["bodypart"].stride}

    @property
    def stride(self) -> int:
        return int(self._strides["bodypart"])

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if x.dim() == 3:
            x = x[None, :]
        x = _pad_to_multiple(x, 32)
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path | None, device: torch.device) -> bool:
    """Load a checkpoint if present.

    Returns True when compatible weights were loaded. The loader accepts common
    snapshot layouts: raw state dict, {"model": state}, or {"state_dict": state}.
    Incompatible keys are ignored so fine-tuned heads can still load a backbone.
    """

    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if not path.exists():
        return False

    snapshot = torch.load(path, map_location=device)
    state = snapshot.get("model") if isinstance(snapshot, dict) else snapshot
    if isinstance(snapshot, dict) and state is None:
        state = snapshot.get("state_dict", snapshot)
    if not isinstance(state, dict):
        return False

    target_state = model.state_dict()
    cleaned = OrderedDict()
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key in target_state and value.shape != target_state[key].shape:
            value = _adapt_checkpoint_tensor(value, target_state[key])
            if value is None:
                continue
        cleaned[key] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    loaded_keys = set(cleaned) - set(incompatible.unexpected_keys)
    return len(loaded_keys) > 0


def _adapt_checkpoint_tensor(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Adapt quadruped checkpoint heads to the dog-only bodypart set."""

    checkpoint_count = len(CHECKPOINT_BODYPARTS)
    dog_count = len(BODYPARTS)
    if source.ndim == 1 and source.shape[0] == checkpoint_count and target.shape[0] == dog_count:
        return source[list(CHECKPOINT_TO_DOG_INDICES)]
    if source.ndim >= 2:
        if source.shape[0] == checkpoint_count and target.shape[0] == dog_count and source.shape[1:] == target.shape[1:]:
            return source[list(CHECKPOINT_TO_DOG_INDICES), ...]
        if source.shape[1] == checkpoint_count and target.shape[1] == dog_count and source.shape[0] == target.shape[0]:
            return source[:, list(CHECKPOINT_TO_DOG_INDICES), ...]
    return None
