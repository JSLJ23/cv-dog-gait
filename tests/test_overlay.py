from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dog_gait.analysis.overlay import render_overlay_video, render_overlay_videos
from dog_gait.pose.schema import BODYPARTS


def _write_source_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (96, 64))
    assert writer.isOpened()
    for frame_index in range(3):
        frame = np.full((64, 96, 3), 40 + frame_index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _keypoints() -> pd.DataFrame:
    rows = []
    for frame in range(3):
        for index, bodypart in enumerate(BODYPARTS):
            rows.append(
                {
                    "frame": frame,
                    "bodypart": bodypart,
                    "x": 10 + index,
                    "y": 20 + index,
                    "likelihood": 0.95,
                }
            )
    return pd.DataFrame(rows)


def _assert_readable_video(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame.shape[:2] == (64, 96)


def test_render_overlay_video_writes_browser_webm_and_download_mp4(tmp_path):
    source = tmp_path / "source.mp4"
    browser_output = tmp_path / "browser" / "overlay.webm"
    download_output = tmp_path / "downloads" / "overlay.mp4"
    _write_source_video(source)
    keypoints = _keypoints()

    browser_rendered = render_overlay_video(source, keypoints, browser_output, confidence=0.2)
    download_rendered = render_overlay_video(source, keypoints, download_output, confidence=0.2)

    assert browser_rendered == browser_output
    assert download_rendered == download_output
    _assert_readable_video(browser_output)
    _assert_readable_video(download_output)


def test_render_overlay_videos_reports_progress_and_writes_outputs(tmp_path):
    source = tmp_path / "source.mp4"
    browser_output = tmp_path / "browser" / "overlay.webm"
    download_output = tmp_path / "downloads" / "overlay.mp4"
    _write_source_video(source)
    keypoints = _keypoints()
    progress = []
    stages = []

    rendered = render_overlay_videos(
        source,
        keypoints,
        [browser_output, download_output],
        confidence=0.2,
        progress_callback=lambda done, total: progress.append((done, total)),
        stage_callback=stages.append,
    )

    assert rendered == [browser_output, download_output]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert stages == ["opening-writers", "rendering", "finalizing", "complete"]
    _assert_readable_video(browser_output)
    _assert_readable_video(download_output)
