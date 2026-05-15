"""Frame extraction utilities."""

from __future__ import annotations

from pathlib import Path

import cv2


def extract_frames(video_path: str | Path, output_dir: str | Path, every_n: int = 30, max_frames: int = 24) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    saved: list[Path] = []
    frame_index = 0
    while len(saved) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, every_n) == 0:
            path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)
        frame_index += 1
    cap.release()
    return saved
