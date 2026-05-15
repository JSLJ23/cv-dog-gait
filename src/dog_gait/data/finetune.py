"""Fine-tuning entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_gait.data.dataset import AnnotationDataset
from dog_gait.pose.model import HeatmapPoseModel, load_checkpoint
from dog_gait.pose.schema import BODYPARTS


@dataclass
class TrainResult:
    checkpoint_path: str
    history: pd.DataFrame


def safe_checkpoint_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    safe = safe.strip("_")
    if not safe:
        raise ValueError("Fine-tuned checkpoint name cannot be empty.")
    return safe


def fine_tuned_checkpoint_path(output_dir: str | Path, base_checkpoint: str | Path | None, name: str) -> Path:
    base_stem = Path(base_checkpoint).stem if base_checkpoint else "hrnet_w32"
    return Path(output_dir) / f"{base_stem}_{safe_checkpoint_name(name)}.pt"


class FineTuner:
    def __init__(self, output_dir: str | Path = "data/checkpoints", device: str = "auto", input_size: int = 256) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_size = input_size

    def train(
        self,
        annotation_dir: str | Path,
        base_checkpoint: str | Path | None = None,
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 1e-3,
        fine_tuned_name: str = "fine_tuned",
    ) -> TrainResult:
        annotation_dir = Path(annotation_dir)
        annotation_csv = annotation_dir / "annotations.csv"
        image_root = annotation_dir / "frames"
        if not annotation_csv.exists():
            raise FileNotFoundError(f"Expected annotation CSV at {annotation_csv}")

        dataset = AnnotationDataset(annotation_csv, image_root, input_size=self.input_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = HeatmapPoseModel(num_keypoints=len(BODYPARTS)).to(self.device)
        load_checkpoint(model, base_checkpoint, self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        history = []
        model.train()
        for epoch in range(1, epochs + 1):
            losses = []
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["heatmap"].to(self.device)
                pred = model(images)["bodypart"]["heatmap"]
                if pred.shape[-2:] != targets.shape[-2:]:
                    targets = torch.nn.functional.interpolate(targets, size=pred.shape[-2:], mode="nearest")
                loss = criterion(torch.sigmoid(pred), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch, "loss": sum(losses) / max(1, len(losses))})

        ckpt = fine_tuned_checkpoint_path(self.output_dir, base_checkpoint, fine_tuned_name)
        torch.save({"model": model.state_dict(), "history": history}, ckpt)
        return TrainResult(checkpoint_path=str(ckpt), history=pd.DataFrame(history))
