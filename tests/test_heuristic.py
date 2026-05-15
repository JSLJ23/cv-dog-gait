import numpy as np

from dog_gait.pose.heuristic import template_keypoints
from dog_gait.pose.schema import BODYPARTS


def test_template_keypoints_are_inside_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    keypoints = template_keypoints(frame, frame_index=0, bbox=(20, 10, 180, 90))

    assert keypoints.shape == (len(BODYPARTS), 3)
    assert keypoints[:, 0].min() >= 0
    assert keypoints[:, 0].max() <= 200
    assert keypoints[:, 1].min() >= 0
    assert keypoints[:, 1].max() <= 100
    assert keypoints[:, 2].min() > 0.9


def test_template_keypoints_use_thigh_positions():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    keypoints = template_keypoints(frame, frame_index=0, bbox=(20, 10, 180, 90))
    front_left_thigh = keypoints[BODYPARTS.index("front_left_thigh")]

    assert np.isclose(front_left_thigh[0], 125.6)
    assert np.isclose(front_left_thigh[1], 52.4)
