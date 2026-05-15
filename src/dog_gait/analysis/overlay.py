"""Frame overlay rendering."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from queue import Queue
from threading import Thread
import subprocess

import numpy as np
import pandas as pd
import cv2

from dog_gait.pose.schema import SKELETON


ProgressCallback = Callable[[int, int | None], None]
StageCallback = Callable[[str], None]
_WRITER_SENTINEL = object()


COLORS = {
    "front_left": (36, 160, 237),
    "front_right": (58, 204, 95),
    "back_left": (210, 95, 218),
    "back_right": (255, 174, 66),
    "body": (245, 245, 245),
}


def _limb_color(part: str) -> tuple[int, int, int]:
    if part.startswith("front_left"):
        return COLORS["front_left"]
    if part.startswith("front_right"):
        return COLORS["front_right"]
    if part.startswith("back_left"):
        return COLORS["back_left"]
    if part.startswith("back_right"):
        return COLORS["back_right"]
    return COLORS["body"]


class H264MovWriter:
    def __init__(self, output_path: Path, fps: float, size: tuple[int, int]) -> None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as exc:
            raise RuntimeError("H.264 MOV export requires the imageio-ffmpeg package.") from exc

        width, height = size
        command = [
            get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("H.264 MOV encoder stdin is closed.")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr is not None else ""
        returncode = self.process.wait()
        if returncode != 0:
            raise RuntimeError(f"H.264 MOV export failed: {stderr.strip() or f'ffmpeg exited with {returncode}'}")


def _video_writer_fourccs(output_path: Path) -> tuple[str, ...]:
    if output_path.suffix.lower() == ".webm":
        return ("VP80", "VP90")
    return ("avc1", "mp4v")


def _open_video_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | H264MovWriter:
    if output_path.suffix.lower() == ".mov":
        return H264MovWriter(output_path, fps, size)
    for fourcc in _video_writer_fourccs(output_path):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not create overlay video writer for {output_path}")


def _writer_worker(
    writer: cv2.VideoWriter | H264MovWriter,
    frame_queue: Queue[np.ndarray | object],
    errors: list[BaseException],
) -> None:
    try:
        while True:
            item = frame_queue.get()
            try:
                if item is _WRITER_SENTINEL:
                    return
                if not errors:
                    writer.write(item)
            except BaseException as exc:
                errors.append(exc)
            finally:
                frame_queue.task_done()
    finally:
        try:
            writer.release()
        except BaseException as exc:
            errors.append(exc)


def _start_writer_threads(
    writers: list[cv2.VideoWriter | H264MovWriter],
    errors: list[BaseException],
    queue_size: int = 8,
) -> tuple[list[Queue[np.ndarray | object]], list[Thread]]:
    queues: list[Queue[np.ndarray | object]] = []
    threads: list[Thread] = []
    for index, writer in enumerate(writers):
        frame_queue: Queue[np.ndarray | object] = Queue(maxsize=queue_size)
        thread = Thread(target=_writer_worker, args=(writer, frame_queue, errors), name=f"overlay-writer-{index}")
        thread.start()
        queues.append(frame_queue)
        threads.append(thread)
    return queues, threads


def _finish_writer_threads(queues: list[Queue[np.ndarray | object]], threads: list[Thread]) -> None:
    for frame_queue in queues:
        frame_queue.put(_WRITER_SENTINEL)
    for frame_queue in queues:
        frame_queue.join()
    for thread in threads:
        thread.join()


def draw_skeleton(frame_bgr: np.ndarray, frame_keypoints: pd.DataFrame, confidence: float = 0.2) -> np.ndarray:
    output = frame_bgr.copy()
    if frame_keypoints.empty:
        return output
    points = {
        row.bodypart: (int(row.x), int(row.y), float(row.likelihood))
        for row in frame_keypoints.itertuples(index=False)
        if float(row.likelihood) >= confidence
    }

    for a, b in SKELETON:
        if a in points and b in points:
            cv2.line(output, points[a][:2], points[b][:2], _limb_color(a), 4, cv2.LINE_AA)

    for bodypart, (x, y, _) in points.items():
        cv2.circle(output, (x, y), 7, (10, 10, 10), -1, cv2.LINE_AA)
        cv2.circle(output, (x, y), 5, _limb_color(bodypart), -1, cv2.LINE_AA)
    return output


def render_overlay_video(
    source_video: str | Path,
    keypoints: pd.DataFrame,
    output_path: str | Path,
    confidence: float = 0.2,
    progress_callback: ProgressCallback | None = None,
    stage_callback: StageCallback | None = None,
) -> Path:
    return render_overlay_videos(source_video, keypoints, [output_path], confidence, progress_callback, stage_callback)[0]


def render_overlay_videos(
    source_video: str | Path,
    keypoints: pd.DataFrame,
    output_paths: Iterable[str | Path],
    confidence: float = 0.2,
    progress_callback: ProgressCallback | None = None,
    stage_callback: StageCallback | None = None,
) -> list[Path]:
    source_video = str(source_video)
    output_paths = [Path(path) for path in output_paths]
    if not output_paths:
        raise ValueError("At least one overlay output path is required.")
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_total = total_frames if total_frames > 0 else None
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Could not determine source video dimensions: {source_video}")

    writers: list[cv2.VideoWriter | H264MovWriter] = []
    queues: list[Queue[np.ndarray | object]] = []
    threads: list[Thread] = []
    writer_errors: list[BaseException] = []
    try:
        if stage_callback is not None:
            stage_callback("opening-writers")
        writers = [_open_video_writer(output_path, fps, (width, height)) for output_path in output_paths]
        queues, threads = _start_writer_threads(writers, writer_errors)
        by_frame = dict(tuple(keypoints.groupby("frame")))

        if stage_callback is not None:
            stage_callback("rendering")
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            overlay = draw_skeleton(frame, by_frame.get(frame_index, pd.DataFrame()), confidence)
            for frame_queue in queues:
                frame_queue.put(overlay)
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, progress_total)
        if stage_callback is not None:
            stage_callback("finalizing")
    finally:
        cap.release()
        if queues:
            _finish_writer_threads(queues, threads)
        else:
            for writer in writers:
                writer.release()
    if writer_errors:
        raise RuntimeError("Overlay video writer failed.") from writer_errors[0]
    if stage_callback is not None:
        stage_callback("complete")
    return output_paths


def first_overlay_frame(source_video: str | Path, keypoints: pd.DataFrame, confidence: float = 0.2) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(source_video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    group = keypoints[keypoints["frame"] == int(keypoints["frame"].min())]
    overlay = draw_skeleton(frame, group, confidence)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
