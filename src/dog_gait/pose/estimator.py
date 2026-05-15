"""Public video inference API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd
import torch

from dog_gait.pose.decoder import decode_heatmaps
from dog_gait.pose.detector import detect_dog_bbox
from dog_gait.pose.heuristic import template_keypoints
from dog_gait.pose.model import HeatmapPoseModel, load_checkpoint
from dog_gait.pose.preprocess import FrameTransform, preprocess_frame, scale_keypoints_to_frame
from dog_gait.pose.schema import BODYPARTS


ProgressCallback = Callable[[int, int | None], None]


@dataclass
class _FrameBatchItem:
    frame_index: int
    tensor: torch.Tensor
    transform: FrameTransform
    offset_x: int
    offset_y: int


@dataclass
class PoseResult:
    keypoints: pd.DataFrame
    fps: float
    frame_count: int
    video_path: str
    checkpoint_loaded: bool
    inference_source: str
    device: str


class PoseEstimator:
    def __init__(self, device: str = "auto", input_size: int = 256) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = HeatmapPoseModel(num_keypoints=len(BODYPARTS)).to(self.device)
        self.model.eval()

    def predict_video(
        self,
        video_path: str | Path,
        checkpoint_path: str | Path | None = None,
        confidence: float = 0.2,
        crop_mode: str = "full-frame",
        max_frames: int | None = None,
        dog_facing: str = "right",
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PoseResult:
        """Run pose inference and return a tidy keypoint table."""

        checkpoint_loaded = load_checkpoint(self.model, checkpoint_path, self.device)
        self.model.eval()
        batch_size = _default_batch_size(self.device) if batch_size is None else max(1, int(batch_size))

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        progress_total = _progress_total(expected_frames, max_frames)

        rows: list[dict] = []
        frame_index = 0
        processed_frames = 0
        batch: list[_FrameBatchItem] = []

        def append_keypoints(frame_number: int, keypoints: np.ndarray) -> None:
            for bodypart, (x, y, score) in zip(BODYPARTS, keypoints, strict=True):
                rows.append(
                    {
                        "frame": frame_number,
                        "bodypart": bodypart,
                        "x": float(x),
                        "y": float(y),
                        "likelihood": float(score),
                        "visible": bool(score >= confidence),
                    }
                )

        def flush_batch() -> None:
            nonlocal processed_frames
            if not batch:
                return

            tensors = torch.stack([item.tensor for item in batch]).to(
                self.device,
                non_blocking=_device_type(self.device) == "cuda",
            )
            with _autocast_context(self.device):
                output = self.model(tensors)
            poses = decode_heatmaps(output["bodypart"]["heatmap"], stride=self.model.stride, apply_sigmoid=False)
            poses_np = poses.detach().cpu().numpy()
            for item, keypoints in zip(batch, poses_np, strict=True):
                keypoints = scale_keypoints_to_frame(keypoints, item.transform)
                keypoints[:, 0] += item.offset_x
                keypoints[:, 1] += item.offset_y
                append_keypoints(item.frame_index, keypoints)
            processed_frames += len(batch)
            batch.clear()
            if progress_callback is not None:
                progress_callback(processed_frames, progress_total)

        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break

                if checkpoint_loaded:
                    work_frame = frame
                    offset_x = 0
                    offset_y = 0
                    if crop_mode == "dog-detector":
                        x1, y1, x2, y2 = detect_dog_bbox(frame)
                        work_frame = frame[y1:y2, x1:x2]
                        offset_x, offset_y = x1, y1
                        if work_frame.size == 0:
                            work_frame = frame
                            offset_x = offset_y = 0

                    tensor, transform = preprocess_frame(work_frame, self.input_size)
                    if batch and tensor.shape != batch[-1].tensor.shape:
                        flush_batch()
                    batch.append(
                        _FrameBatchItem(
                            frame_index=frame_index,
                            tensor=tensor,
                            transform=transform,
                            offset_x=offset_x,
                            offset_y=offset_y,
                        )
                    )
                    if len(batch) >= batch_size:
                        flush_batch()
                else:
                    keypoints = template_keypoints(frame, frame_index=frame_index, facing=dog_facing)
                    append_keypoints(frame_index, keypoints)
                    processed_frames += 1
                    if progress_callback is not None:
                        progress_callback(processed_frames, progress_total)
                frame_index += 1
            flush_batch()

        cap.release()
        return PoseResult(
            keypoints=pd.DataFrame(rows),
            fps=fps,
            frame_count=frame_index or expected_frames,
            video_path=video_path,
            checkpoint_loaded=checkpoint_loaded,
            inference_source="checkpoint" if checkpoint_loaded else "side-view template fallback",
            device=str(self.device),
        )


def _progress_total(expected_frames: int, max_frames: int | None) -> int | None:
    if max_frames is None:
        return expected_frames if expected_frames > 0 else None
    if expected_frames <= 0:
        return max_frames
    return min(expected_frames, max_frames)


def _device_type(device: torch.device | str) -> str:
    return device.type if isinstance(device, torch.device) else str(device).split(":", maxsplit=1)[0]


def _default_batch_size(device: torch.device | str) -> int:
    return 16 if _device_type(device) == "cuda" else 4


def _autocast_context(device: torch.device | str):
    device_type = _device_type(device)
    return torch.autocast(device_type=device_type, enabled=device_type == "cuda")
