from pathlib import Path

import pytest
import torch

from dog_gait.pose.detector_model import FasterRCNNDetector
from dog_gait.pose.model import HeatmapPoseModel, load_checkpoint
from dog_gait.pose.schema import BODYPARTS


POSE_CHECKPOINT = Path("data/default_checkpoints/superanimal_quadruped_hrnet_w32.pt")
DOG_DETECTOR_CHECKPOINT = Path("data/default_checkpoints/superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt")


@pytest.mark.skipif(not POSE_CHECKPOINT.exists(), reason="Quadruped pose checkpoint is not downloaded")
def test_pose_model_loads_checkpoint_and_runs_dummy_tensor():
    model = HeatmapPoseModel(num_keypoints=len(BODYPARTS))
    load_result = load_checkpoint(model, POSE_CHECKPOINT, torch.device("cpu"))
    model.eval()

    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 144, 256)))

    heatmap = output["bodypart"]["heatmap"]
    assert load_result
    assert heatmap.shape == (1, len(BODYPARTS), 40, 64)
    assert torch.isfinite(heatmap).all()


@pytest.mark.skipif(not DOG_DETECTOR_CHECKPOINT.exists(), reason="Quadruped dog detector checkpoint is not downloaded")
def test_dog_detector_loads_checkpoint_and_runs_dummy_tensor():
    model = FasterRCNNDetector()
    checkpoint = torch.load(DOG_DETECTOR_CHECKPOINT, map_location="cpu")
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    with torch.inference_mode():
        detections = model([torch.zeros((3, 144, 256))])

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert isinstance(detections, list)
    assert {"boxes", "labels", "scores"} <= set(detections[0])
