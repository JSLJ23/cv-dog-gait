from pathlib import Path

import cv2
import numpy as np
import torch

from dog_gait.pose.estimator import PoseEstimator
from dog_gait.pose.schema import BODYPARTS


class DummyModel:
    def eval(self) -> None:
        return None


def _write_source_video(path: Path, frame_count: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (96, 64))
    assert writer.isOpened()
    for frame_index in range(frame_count):
        frame = np.full((64, 96, 3), 30 + frame_index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_predict_video_reports_progress_for_processed_frames(tmp_path):
    source = tmp_path / "source.mp4"
    _write_source_video(source, frame_count=5)
    estimator = PoseEstimator.__new__(PoseEstimator)
    estimator.device = "cpu"
    estimator.input_size = 256
    estimator.model = DummyModel()
    progress = []

    result = estimator.predict_video(source, max_frames=3, progress_callback=lambda done, total: progress.append((done, total)))

    assert result.frame_count == 3
    assert progress == [(1, 3), (2, 3), (3, 3)]


class BatchRecordingModel:
    stride = 4

    def __init__(self) -> None:
        self.batch_sizes = []

    def eval(self) -> None:
        return None

    def __call__(self, batch: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        self.batch_sizes.append(batch.shape[0])
        heatmap = torch.zeros((batch.shape[0], len(BODYPARTS), 8, 8), dtype=batch.dtype, device=batch.device)
        heatmap[:, :, 2, 3] = 1.0
        return {"bodypart": {"heatmap": heatmap}}


def test_predict_video_batches_checkpoint_inference(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    _write_source_video(source, frame_count=5)
    model = BatchRecordingModel()
    estimator = PoseEstimator.__new__(PoseEstimator)
    estimator.device = torch.device("cpu")
    estimator.input_size = 64
    estimator.model = model
    progress = []
    monkeypatch.setattr("dog_gait.pose.estimator.load_checkpoint", lambda model, path, device: True)

    result = estimator.predict_video(
        source,
        checkpoint_path="weights.pt",
        max_frames=5,
        batch_size=2,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert result.frame_count == 5
    assert result.inference_source == "checkpoint"
    assert model.batch_sizes == [2, 2, 1]
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert len(result.keypoints) == 5 * len(BODYPARTS)
