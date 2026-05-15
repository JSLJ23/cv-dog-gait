"""Gait-angle computations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dog_gait.pose.schema import LIMB_TRIPLETS


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def compute_limb_angles(keypoints: pd.DataFrame, confidence_threshold: float = 0.2) -> pd.DataFrame:
    """Compute per-frame limb joint angles from tidy keypoint rows."""

    required = {"frame", "bodypart", "x", "y", "likelihood"}
    missing = required - set(keypoints.columns)
    if missing:
        raise ValueError(f"Missing keypoint columns: {sorted(missing)}")

    rows = []
    for frame, group in keypoints.groupby("frame", sort=True):
        by_part = {row.bodypart: row for row in group.itertuples(index=False)}
        for limb, (proximal, joint, distal) in LIMB_TRIPLETS.items():
            parts = [by_part.get(proximal), by_part.get(joint), by_part.get(distal)]
            confident = all(part is not None and part.likelihood >= confidence_threshold for part in parts)
            if confident:
                angle = _angle_degrees(
                    np.array([parts[0].x, parts[0].y], dtype=float),
                    np.array([parts[1].x, parts[1].y], dtype=float),
                    np.array([parts[2].x, parts[2].y], dtype=float),
                )
            else:
                angle = float("nan")
            rows.append({"frame": int(frame), "limb": limb, "angle_degrees": angle, "confident": confident})
    return pd.DataFrame(rows)


def smooth_angles(angles: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add a rolling-smoothed angle column."""

    if angles.empty:
        return angles.assign(angle_smooth=pd.Series(dtype=float))
    result = angles.copy()
    result["angle_smooth"] = (
        result.groupby("limb")["angle_degrees"]
        .transform(lambda s: s.rolling(window=window, min_periods=1, center=True).mean())
    )
    return result


def summarize_stride_proxy(angles: pd.DataFrame) -> pd.DataFrame:
    """Produce simple gait summary statistics from angle traces."""

    if angles.empty:
        return pd.DataFrame(columns=["limb", "min_angle", "max_angle", "range_degrees", "valid_frames"])
    grouped = angles.groupby("limb")["angle_degrees"]
    return pd.DataFrame(
        {
            "min_angle": grouped.min(),
            "max_angle": grouped.max(),
            "range_degrees": grouped.max() - grouped.min(),
            "valid_frames": grouped.count(),
        }
    ).reset_index()
