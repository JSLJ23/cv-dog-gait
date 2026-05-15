"""Annotation persistence for the Streamlit application."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from dog_gait.pose.schema import BODYPARTS


ANNOTATION_COLUMNS = ["image", "frame", "bodypart", "x", "y", "likelihood"]
DOG_BODYPART_SET = set(BODYPARTS)


class AnnotationStore:
    def __init__(self, root: str | Path = "data/annotations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, video_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in video_id)
        return self.root / f"{safe}.jsonl"

    def save_frame_labels(self, video_id: str, frame_index: int, labels: dict[str, dict[str, float]]) -> None:
        record = {"video_id": video_id, "frame": int(frame_index), "labels": labels}
        with self._path(video_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def load(self, video_id: str) -> pd.DataFrame:
        path = self._path(video_id)
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return pd.DataFrame(columns=["video_id", "frame", "bodypart", "x", "y", "likelihood"])
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                for bodypart, point in record["labels"].items():
                    if bodypart not in DOG_BODYPART_SET:
                        continue
                    rows.append(
                        {
                            "video_id": record["video_id"],
                            "frame": int(record["frame"]),
                            "bodypart": bodypart,
                            "x": float(point["x"]),
                            "y": float(point["y"]),
                            "likelihood": float(point.get("likelihood", 1.0)),
                        }
                    )
        return pd.DataFrame(rows)

    def list_videos(self) -> list[str]:
        return [p.stem for p in sorted(self.root.glob("*.jsonl"))]


def safe_dataset_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    safe = safe.strip("_")
    if not safe:
        raise ValueError("Dataset name cannot be empty.")
    return safe


class FineTuneDatasetStore:
    """Named fine-tuning dataset layout.

    Each dataset is stored as:
      data/annotations/<dataset_name>/frames/*.jpg
      data/annotations/<dataset_name>/annotations.csv
    """

    def __init__(self, root: str | Path = "data/annotations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, name: str) -> Path:
        return self.root / safe_dataset_name(name)

    def frames_dir(self, name: str) -> Path:
        path = self.dataset_dir(name) / "frames"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def annotations_path(self, name: str) -> Path:
        return self.dataset_dir(name) / "annotations.csv"

    def list_datasets(self) -> list[str]:
        datasets = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and (path / "annotations.csv").exists() and (path / "frames").exists():
                datasets.append(path.name)
        return datasets

    def add_frames(self, name: str, frame_paths: list[str | Path]) -> list[Path]:
        target_dir = self.frames_dir(name)
        copied = []
        for frame_path in frame_paths:
            frame_path = Path(frame_path)
            target = target_dir / frame_path.name
            if frame_path.resolve() != target.resolve():
                shutil.copy2(frame_path, target)
            copied.append(target)
        return copied

    def load_frame_labels(self, name: str, image_path: str | Path, frame_index: int) -> dict[str, dict[str, float]]:
        csv_path = self.annotations_path(name)
        if not csv_path.exists():
            return {}
        image_name = Path(image_path).name
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}
        frame_rows = df[(df["image"] == image_name) & (df["frame"] == int(frame_index))]
        labels: dict[str, dict[str, float]] = {}
        for row in frame_rows.itertuples(index=False):
            if str(row.bodypart) not in DOG_BODYPART_SET:
                continue
            labels[str(row.bodypart)] = {
                "x": float(row.x),
                "y": float(row.y),
                "likelihood": float(getattr(row, "likelihood", 1.0)),
            }
        return labels

    def reset_frame_labels(self, name: str, image_path: str | Path, frame_index: int) -> int:
        csv_path = self.annotations_path(name)
        if not csv_path.exists():
            return 0
        image_name = Path(image_path).name
        df = pd.read_csv(csv_path)
        keep = ~((df["image"] == image_name) & (df["frame"] == int(frame_index)))
        removed = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
        if df.empty:
            df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        df.to_csv(csv_path, index=False)
        return removed

    def save_frame_labels(
        self,
        name: str,
        image_path: str | Path,
        frame_index: int,
        labels: dict[str, dict[str, float]],
    ) -> int:
        dataset_dir = self.dataset_dir(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        image_name = Path(image_path).name
        rows = []
        for bodypart, point in labels.items():
            if bodypart not in DOG_BODYPART_SET:
                continue
            rows.append(
                {
                    "image": image_name,
                    "frame": int(frame_index),
                    "bodypart": bodypart,
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "likelihood": float(point.get("likelihood", 1.0)),
                }
            )
        if not rows:
            return 0

        csv_path = self.annotations_path(name)
        new_df = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            keep = ~((df["image"] == image_name) & (df["frame"] == int(frame_index)) & (df["bodypart"].isin(new_df["bodypart"])))
            df = pd.concat([df[keep], new_df], ignore_index=True)
        else:
            df = new_df
        df = df.sort_values(["image", "frame", "bodypart"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        return len(rows)

    def count_labels(self, name: str) -> int:
        csv_path = self.annotations_path(name)
        if not csv_path.exists():
            return 0
        df = pd.read_csv(csv_path)
        return int(df["bodypart"].isin(DOG_BODYPART_SET).sum())
