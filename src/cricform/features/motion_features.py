from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cricform.features.joint_angles import (
    absolute_angle_difference_degrees,
    angle_degrees,
    line_angle_degrees,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/features")

MOVEMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "video_id",
    "frame_index",
    "timestamp_ms",
    "phase",
    "usable_for_motion",
    "head_x",
    "head_y",
    "head_displacement_proxy",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "front_knee_bend_proxy_deg",
    "shoulder_hip_separation_proxy_deg",
    "wrist_center_x",
    "wrist_center_y",
    "wrist_speed_proxy",
    "wrist_acceleration_proxy",
    "wrist_path_smoothness_proxy",
    "backlift_height_proxy",
    "follow_through_height_proxy",
)

REQUIRED_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_wrist",
    "right_wrist",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class MovementFeatureSummary:
    input_landmark_parquet_path: str
    input_phase_timeline_csv_path: str | None
    output_features_csv_path: str
    output_summary_json_path: str
    rows: int
    usable_frames: int
    max_backlift_height_proxy: float | None
    follow_through_height_proxy: float | None
    mean_head_displacement_proxy: float | None
    head_motion_std_proxy: float | None
    min_front_knee_bend_proxy_deg: float | None
    mean_shoulder_hip_separation_proxy_deg: float | None
    mean_wrist_speed_proxy: float | None
    mean_wrist_path_smoothness_proxy: float | None
    limitation: str


def build_movement_feature_table(
    landmark_dataframe: pd.DataFrame,
    phase_timeline_dataframe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build frame-level interpretable movement-feature proxies."""

    if landmark_dataframe.empty:
        return _empty_movement_feature_table()

    image_df = landmark_dataframe[
        (landmark_dataframe["coordinate_space"] == "image")
        & (landmark_dataframe["pose_index"] == 0)
    ].copy()

    if image_df.empty:
        return _empty_movement_feature_table()

    pivot = image_df.pivot_table(
        index=["video_id", "frame_index", "timestamp_ms"],
        columns="landmark_name",
        values=["x", "y"],
        aggfunc="first",
    )

    pivot.columns = [f"{axis}_{landmark_name}" for axis, landmark_name in pivot.columns]
    pivot = pivot.reset_index().sort_values("frame_index").reset_index(drop=True)

    _ensure_landmark_columns(pivot)

    features = pivot[["video_id", "frame_index", "timestamp_ms"]].copy()
    features["phase"] = _phase_series(features, phase_timeline_dataframe)

    features["head_x"] = pivot["x_nose"]
    features["head_y"] = pivot["y_nose"]

    features["left_knee_angle_deg"] = pivot.apply(
        lambda row: _joint_angle_from_row(row, "left_hip", "left_knee", "left_ankle"),
        axis=1,
    )
    features["right_knee_angle_deg"] = pivot.apply(
        lambda row: _joint_angle_from_row(row, "right_hip", "right_knee", "right_ankle"),
        axis=1,
    )
    features["front_knee_bend_proxy_deg"] = features[
        ["left_knee_angle_deg", "right_knee_angle_deg"]
    ].min(axis=1, skipna=True)

    features["shoulder_hip_separation_proxy_deg"] = pivot.apply(
        _shoulder_hip_separation_from_row,
        axis=1,
    )

    features["wrist_center_x"] = pivot[["x_left_wrist", "x_right_wrist"]].mean(
        axis=1,
        skipna=True,
    )
    features["wrist_center_y"] = pivot[["y_left_wrist", "y_right_wrist"]].mean(
        axis=1,
        skipna=True,
    )

    shoulder_center_y = pivot[["y_left_shoulder", "y_right_shoulder"]].mean(
        axis=1,
        skipna=True,
    )
    features["backlift_height_proxy"] = shoulder_center_y - features["wrist_center_y"]

    _add_head_displacement(features)
    _add_wrist_motion_features(features)

    features["follow_through_height_proxy"] = np.where(
        features["phase"] == "follow_through",
        features["backlift_height_proxy"],
        np.nan,
    )

    core_columns = [
        "head_x",
        "head_y",
        "wrist_center_x",
        "wrist_center_y",
        "backlift_height_proxy",
    ]
    features["usable_for_motion"] = features[core_columns].notna().all(axis=1)

    return features[list(MOVEMENT_FEATURE_COLUMNS)]


def write_movement_feature_outputs(
    landmark_parquet_path: Path,
    phase_timeline_csv_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> MovementFeatureSummary:
    """Write frame-level movement features and a video-level summary."""

    if not landmark_parquet_path.exists():
        raise FileNotFoundError(f"Landmark Parquet file does not exist: {landmark_parquet_path}")

    if phase_timeline_csv_path is not None and not phase_timeline_csv_path.exists():
        raise FileNotFoundError(f"Phase timeline CSV does not exist: {phase_timeline_csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = _video_stem_from_landmark_parquet_path(landmark_parquet_path)
    output_features_csv_path = output_dir / f"{video_stem}.movement_features.csv"
    output_summary_json_path = output_dir / f"{video_stem}.movement_summary.json"

    landmark_dataframe = pd.read_parquet(landmark_parquet_path)
    phase_timeline_dataframe = (
        pd.read_csv(phase_timeline_csv_path)
        if phase_timeline_csv_path is not None and phase_timeline_csv_path.exists()
        else None
    )

    features = build_movement_feature_table(
        landmark_dataframe=landmark_dataframe,
        phase_timeline_dataframe=phase_timeline_dataframe,
    )
    features.to_csv(output_features_csv_path, index=False)

    summary_values = summarize_movement_features(features)
    summary_values["limitation"] = (
        "Movement features are 2D pose-derived proxies. They are not validated "
        "biomechanics, coaching-grade metrics, or injury-risk assessments."
    )

    output_summary_json_path.write_text(json.dumps(summary_values, indent=2) + "\n")

    return MovementFeatureSummary(
        input_landmark_parquet_path=str(landmark_parquet_path),
        input_phase_timeline_csv_path=(
            str(phase_timeline_csv_path) if phase_timeline_csv_path is not None else None
        ),
        output_features_csv_path=str(output_features_csv_path),
        output_summary_json_path=str(output_summary_json_path),
        rows=int(summary_values["rows"]),
        usable_frames=int(summary_values["usable_frames"]),
        max_backlift_height_proxy=summary_values["max_backlift_height_proxy"],
        follow_through_height_proxy=summary_values["follow_through_height_proxy"],
        mean_head_displacement_proxy=summary_values["mean_head_displacement_proxy"],
        head_motion_std_proxy=summary_values["head_motion_std_proxy"],
        min_front_knee_bend_proxy_deg=summary_values["min_front_knee_bend_proxy_deg"],
        mean_shoulder_hip_separation_proxy_deg=summary_values[
            "mean_shoulder_hip_separation_proxy_deg"
        ],
        mean_wrist_speed_proxy=summary_values["mean_wrist_speed_proxy"],
        mean_wrist_path_smoothness_proxy=summary_values["mean_wrist_path_smoothness_proxy"],
        limitation=str(summary_values["limitation"]),
    )


def summarize_movement_features(features: pd.DataFrame) -> dict[str, object]:
    """Summarize frame-level movement features into video-level proxies."""

    if features.empty:
        return {
            "rows": 0,
            "usable_frames": 0,
            "max_backlift_height_proxy": None,
            "follow_through_height_proxy": None,
            "mean_head_displacement_proxy": None,
            "head_motion_std_proxy": None,
            "min_front_knee_bend_proxy_deg": None,
            "mean_shoulder_hip_separation_proxy_deg": None,
            "mean_wrist_speed_proxy": None,
            "mean_wrist_path_smoothness_proxy": None,
        }

    return {
        "rows": int(len(features)),
        "usable_frames": int(features["usable_for_motion"].sum()),
        "max_backlift_height_proxy": _max_or_none(features["backlift_height_proxy"]),
        "follow_through_height_proxy": _max_or_none(features["follow_through_height_proxy"]),
        "mean_head_displacement_proxy": _mean_or_none(features["head_displacement_proxy"]),
        "head_motion_std_proxy": _head_motion_std_or_none(features),
        "min_front_knee_bend_proxy_deg": _min_or_none(features["front_knee_bend_proxy_deg"]),
        "mean_shoulder_hip_separation_proxy_deg": _mean_or_none(
            features["shoulder_hip_separation_proxy_deg"]
        ),
        "mean_wrist_speed_proxy": _mean_or_none(features["wrist_speed_proxy"]),
        "mean_wrist_path_smoothness_proxy": _mean_or_none(
            features["wrist_path_smoothness_proxy"]
        ),
    }



def _empty_movement_feature_table() -> pd.DataFrame:
    return pd.DataFrame(columns=list(MOVEMENT_FEATURE_COLUMNS))

def _ensure_landmark_columns(pivot: pd.DataFrame) -> None:
    for landmark_name in REQUIRED_LANDMARK_NAMES:
        for axis in ("x", "y"):
            column = f"{axis}_{landmark_name}"
            if column not in pivot.columns:
                pivot[column] = np.nan


def _phase_series(
    features: pd.DataFrame,
    phase_timeline_dataframe: pd.DataFrame | None,
) -> pd.Series:
    if phase_timeline_dataframe is None or phase_timeline_dataframe.empty:
        return pd.Series([pd.NA] * len(features), index=features.index, dtype="object")

    if "frame_index" not in phase_timeline_dataframe.columns:
        return pd.Series([pd.NA] * len(features), index=features.index, dtype="object")

    if "phase" not in phase_timeline_dataframe.columns:
        return pd.Series([pd.NA] * len(features), index=features.index, dtype="object")

    phase_map = dict(
        zip(
            phase_timeline_dataframe["frame_index"].astype(int),
            phase_timeline_dataframe["phase"],
            strict=False,
        )
    )

    return features["frame_index"].map(phase_map)


def _joint_angle_from_row(
    row: pd.Series,
    point_a: str,
    point_b: str,
    point_c: str,
) -> float:
    a = _point_from_row(row, point_a)
    b = _point_from_row(row, point_b)
    c = _point_from_row(row, point_c)

    if a is None or b is None or c is None:
        return math.nan

    return angle_degrees(a, b, c)


def _shoulder_hip_separation_from_row(row: pd.Series) -> float:
    left_shoulder = _point_from_row(row, "left_shoulder")
    right_shoulder = _point_from_row(row, "right_shoulder")
    left_hip = _point_from_row(row, "left_hip")
    right_hip = _point_from_row(row, "right_hip")

    if (
        left_shoulder is None
        or right_shoulder is None
        or left_hip is None
        or right_hip is None
    ):
        return math.nan

    shoulder_angle = line_angle_degrees(left_shoulder, right_shoulder)
    hip_angle = line_angle_degrees(left_hip, right_hip)

    return absolute_angle_difference_degrees(shoulder_angle, hip_angle)


def _point_from_row(row: pd.Series, landmark_name: str) -> tuple[float, float] | None:
    x = row.get(f"x_{landmark_name}")
    y = row.get(f"y_{landmark_name}")

    if pd.isna(x) or pd.isna(y):
        return None

    return (float(x), float(y))


def _add_head_displacement(features: pd.DataFrame) -> None:
    valid_head = features[["head_x", "head_y"]].dropna()

    if valid_head.empty:
        features["head_displacement_proxy"] = np.nan
        return

    start_x = float(valid_head.iloc[0]["head_x"])
    start_y = float(valid_head.iloc[0]["head_y"])

    dx = features["head_x"] - start_x
    dy = features["head_y"] - start_y

    features["head_displacement_proxy"] = np.sqrt(dx**2 + dy**2)


def _add_wrist_motion_features(features: pd.DataFrame) -> None:
    delta_time_sec = features["timestamp_ms"].diff() / 1000.0
    frame_delta = features["frame_index"].diff()

    fallback_delta_sec = frame_delta.where(frame_delta > 0, 1).fillna(1) / 24.0
    delta_time_sec = delta_time_sec.where(delta_time_sec > 0, fallback_delta_sec)

    velocity_x = features["wrist_center_x"].diff() / delta_time_sec
    velocity_y = features["wrist_center_y"].diff() / delta_time_sec

    features["wrist_speed_proxy"] = np.sqrt(velocity_x**2 + velocity_y**2).fillna(0.0)

    features["wrist_acceleration_proxy"] = (
        features["wrist_speed_proxy"].diff().abs() / delta_time_sec
    ).fillna(0.0)

    features["wrist_path_smoothness_proxy"] = 1.0 / (
        1.0 + features["wrist_acceleration_proxy"]
    )


def _video_stem_from_landmark_parquet_path(landmark_parquet_path: Path) -> str:
    name = landmark_parquet_path.name

    if name.endswith(".landmarks.parquet"):
        return name.removesuffix(".landmarks.parquet")

    return landmark_parquet_path.stem


def _mean_or_none(series: pd.Series) -> float | None:
    mean_value = series.mean(skipna=True)

    if pd.isna(mean_value):
        return None

    return float(mean_value)


def _max_or_none(series: pd.Series) -> float | None:
    max_value = series.max(skipna=True)

    if pd.isna(max_value):
        return None

    return float(max_value)


def _min_or_none(series: pd.Series) -> float | None:
    min_value = series.min(skipna=True)

    if pd.isna(min_value):
        return None

    return float(min_value)


def _head_motion_std_or_none(features: pd.DataFrame) -> float | None:
    if features[["head_x", "head_y"]].dropna().empty:
        return None

    std_x = features["head_x"].std(skipna=True)
    std_y = features["head_y"].std(skipna=True)

    if pd.isna(std_x) or pd.isna(std_y):
        return None

    return float(math.sqrt(std_x**2 + std_y**2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute movement features from landmark Parquet.")
    parser.add_argument("landmark_parquet_path", type=Path)
    parser.add_argument("--phase-timeline-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = write_movement_feature_outputs(
        landmark_parquet_path=args.landmark_parquet_path,
        phase_timeline_csv_path=args.phase_timeline_csv,
        output_dir=args.output_dir,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
