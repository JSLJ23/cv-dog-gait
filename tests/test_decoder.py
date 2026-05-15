import torch
from pytest import approx

from dog_gait.pose.decoder import decode_heatmaps


def test_decode_heatmaps_returns_centered_coordinates():
    heatmaps = torch.zeros((1, 2, 4, 4))
    heatmaps[0, 0, 2, 3] = 0.9
    heatmaps[0, 1, 1, 0] = 0.5

    decoded = decode_heatmaps(heatmaps, stride=4)

    assert decoded.shape == (1, 2, 3)
    assert decoded[0, 0, 0].item() == 14
    assert decoded[0, 0, 1].item() == 10
    assert decoded[0, 0, 2].item() == approx(0.9)
