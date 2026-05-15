import pandas as pd

from dog_gait.analysis.angles import compute_limb_angles


def test_compute_limb_angles_handles_confidence():
    keypoints = pd.DataFrame(
        [
            {"frame": 0, "bodypart": "front_left_thigh", "x": 0, "y": 0, "likelihood": 1.0},
            {"frame": 0, "bodypart": "front_left_knee", "x": 1, "y": 0, "likelihood": 1.0},
            {"frame": 0, "bodypart": "front_left_paw", "x": 1, "y": 1, "likelihood": 1.0},
            {"frame": 0, "bodypart": "front_right_thigh", "x": 0, "y": 0, "likelihood": 0.1},
            {"frame": 0, "bodypart": "front_right_knee", "x": 1, "y": 0, "likelihood": 1.0},
            {"frame": 0, "bodypart": "front_right_paw", "x": 1, "y": 1, "likelihood": 1.0},
        ]
    )

    angles = compute_limb_angles(keypoints, confidence_threshold=0.2)
    front_left = angles[angles["limb"] == "front_left"].iloc[0]
    front_right = angles[angles["limb"] == "front_right"].iloc[0]

    assert round(front_left.angle_degrees) == 90
    assert bool(front_left.confident)
    assert not bool(front_right.confident)
