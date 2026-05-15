import cv2
import numpy as np
import pandas as pd

from dog_gait.data.dataset import AnnotationDataset
from dog_gait.pose.schema import BODYPARTS


def test_annotation_dataset_filters_non_dog_bodyparts(tmp_path):
    image_root = tmp_path / "frames"
    image_root.mkdir()
    cv2.imwrite(str(image_root / "frame_000001.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {"image": "frame_000001.jpg", "frame": 1, "bodypart": "nose", "x": 8, "y": 8, "likelihood": 1.0},
            {
                "image": "frame_000001.jpg",
                "frame": 1,
                "bodypart": "right_antler_base",
                "x": 16,
                "y": 16,
                "likelihood": 1.0,
            },
        ]
    ).to_csv(annotations, index=False)

    dataset = AnnotationDataset(annotations, image_root, input_size=32)
    sample = dataset[0]

    assert sample["heatmap"].shape[0] == len(BODYPARTS)
    assert sample["heatmap"].sum().item() == 1.0
