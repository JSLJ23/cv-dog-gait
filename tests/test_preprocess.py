import numpy as np

from dog_gait.pose.preprocess import preprocess_frame, scale_keypoints_to_frame


def test_preprocess_preserves_16_9_aspect_ratio_with_letterbox_mapping():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tensor, transform = preprocess_frame(frame, input_size=256)

    assert tensor.shape == (3, 160, 256)
    assert transform.resized_width == 256
    assert transform.resized_height == 144
    assert transform.pad_left == 0
    assert transform.pad_top == 8

    center_on_canvas = np.array([[128.0, 80.0, 0.9]], dtype=np.float32)
    center_on_frame = scale_keypoints_to_frame(center_on_canvas, transform)

    assert center_on_frame[0, 0] == 640
    assert center_on_frame[0, 1] == 360
