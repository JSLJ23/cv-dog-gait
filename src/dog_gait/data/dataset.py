"""Tiny supervised pose dataset for fine-tuning."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dog_gait.pose.preprocess import preprocess_frame
from dog_gait.pose.schema import BODY_INDEX, BODYPARTS


class AnnotationDataset(Dataset):
    def __init__(self, annotation_csv: str | Path, image_root: str | Path, input_size: int = 256):
        self.df = pd.read_csv(annotation_csv)
        self.df = self.df[self.df["bodypart"].isin(BODY_INDEX)].reset_index(drop=True)
        self.image_root = Path(image_root)
        self.input_size = input_size
        self.frames = sorted(self.df[["image", "frame"]].drop_duplicates().itertuples(index=False, name=None))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_name, frame = self.frames[index]
        image = cv2.imread(str(self.image_root / image_name))
        if image is None:
            raise FileNotFoundError(self.image_root / image_name)
        tensor, transform = preprocess_frame(image, self.input_size)
        heatmap_h = tensor.shape[1] // 4
        heatmap_w = tensor.shape[2] // 4
        heatmaps = torch.zeros((len(BODYPARTS), heatmap_h, heatmap_w), dtype=torch.float32)
        rows = self.df[(self.df["image"] == image_name) & (self.df["frame"] == frame)]
        for row in rows.itertuples(index=False):
            idx = BODY_INDEX.get(row.bodypart)
            if idx is None:
                continue
            canvas_x = float(row.x) * transform.scale + transform.pad_left
            canvas_y = float(row.y) * transform.scale + transform.pad_top
            x = int(np.clip(canvas_x / 4, 0, heatmap_w - 1))
            y = int(np.clip(canvas_y / 4, 0, heatmap_h - 1))
            heatmaps[idx, y, x] = 1.0
        return {"image": tensor, "heatmap": heatmaps}
