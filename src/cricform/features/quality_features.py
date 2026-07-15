from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data/processed/features")
EXPECTED_POSE_LANDMARK_COUNT = 33

FRAME_QUALITY_COLUMNS: tuple[str, ...] = (
    "video_id",
    "source_video_path",
    "frame_index",
    "timestamp_ms",
    "pose_count",
    "has_pose",
    "image_landmark_count",
    "world_landmark_count",
    "landmark_coverage",
    "mean_visibility",
    "mean_presence",
    "coordinate_valid_ratio",
    "frame_quality_score",
)


@dataclass(frozen=True)
class PoseQualitySummary:
    input_jsonl_path: str
    frame_quality_csv_path: str
    summary_json_path: str
    total_frames_processed: int
    frames_with_pose: int
    pose_detection_rate: float
    mean_frame_quality_score: float
    mean_visibility: float | None
    mean_presence: float | None
    low_quality_frame_rate: float


def compute_frame_pose_quality(record: dict[str, Any]) -> dict[str, Any]:
    """Compute quality metrics for one MediaPipe pose JSONL frame record."""

    source_video_path = str(record.get("video_path", ""))
    video_id = Path(source_video_path).stem if source_video_path else "unknown_video"

    image_landmarks = _first_pose_landmarks(record, "pose_landmarks")
    world_landmarks = _first_pose_landmarks(record, "pose_world_landmarks")

    image_landmark_count = len(image_landmarks)
    world_landmark_count = len(world_landmarks)

    mean_visibility = _mean_landmark_value(image_landmarks, "visibility")
    mean_presence = _mean_landmark_value(image_landmarks, "presence")
    coordinate_valid_ratio = _image_coordinate_valid_ratio(image_landmarks)

    landmark_coverage = _clamp01(image_landmark_count / EXPECTED_POSE_LANDMARK_COUNT)
    has_pose = image_landmark_count > 0

    if has_pose:
        frame_quality_score = _clamp01(
            0.40 * landmark_coverage
            + 0.25 * (mean_visibility or 0.0)
            + 0.25 * (mean_presence or 0.0)
            + 0.10 * coordinate_valid_ratio
        )
    else:
        frame_quality_score = 0.0

    return {
        "video_id": video_id,
        "source_video_path": source_video_path,
        "frame_index": int(record["frame_index"]),
        "timestamp_ms": int(record["timestamp_ms"]),
        "pose_count": int(record.get("pose_count", 0)),
        "has_pose": has_pose,
        "image_landmark_count": image_landmark_count,
        "world_landmark_count": world_landmark_count,
        "landmark_coverage": landmark_coverage,
        "mean_visibility": mean_visibility,
        "mean_presence": mean_presence,
        "coordinate_valid_ratio": coordinate_valid_ratio,
        "frame_quality_score": frame_quality_score,
    }


def pose_quality_dataframe(pose_jsonl_path: Path) -> pd.DataFrame:
    """Load pose JSONL and return one quality row per processed frame."""

    if not pose_jsonl_path.exists():
        raise FileNotFoundError(f"Pose JSONL file does not exist: {pose_jsonl_path}")

    rows: list[dict[str, Any]] = []

    with pose_jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {pose_jsonl_path}"
                ) from error

            rows.append(compute_frame_pose_quality(record))

    return pd.DataFrame(rows, columns=list(FRAME_QUALITY_COLUMNS))


def summarize_pose_quality(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Summarize frame-level pose quality into video-level metrics."""

    total_frames_processed = int(len(dataframe))

    if dataframe.empty:
        return {
            "total_frames_processed": 0,
            "frames_with_pose": 0,
            "pose_detection_rate": 0.0,
            "mean_frame_quality_score": 0.0,
            "mean_visibility": None,
            "mean_presence": None,
            "low_quality_frame_rate": 0.0,
        }

    frames_with_pose = int(dataframe["has_pose"].sum())
    pose_detection_rate = frames_with_pose / total_frames_processed

    low_quality_frames = int((dataframe["frame_quality_score"] < 0.5).sum())
    low_quality_frame_rate = low_quality_frames / total_frames_processed

    return {
        "total_frames_processed": total_frames_processed,
        "frames_with_pose": frames_with_pose,
        "pose_detection_rate": float(pose_detection_rate),
        "mean_frame_quality_score": _series_mean_or_zero(dataframe["frame_quality_score"]),
        "mean_visibility": _series_mean_or_none(dataframe["mean_visibility"]),
        "mean_presence": _series_mean_or_none(dataframe["mean_presence"]),
        "low_quality_frame_rate": float(low_quality_frame_rate),
    }


def write_pose_quality_outputs(
    pose_jsonl_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> PoseQualitySummary:
    """Write frame-level pose quality CSV and video-level summary JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = _video_stem_from_pose_jsonl_path(pose_jsonl_path)
    frame_quality_csv_path = output_dir / f"{video_stem}.pose_quality.csv"
    summary_json_path = output_dir / f"{video_stem}.pose_quality_summary.json"

    dataframe = pose_quality_dataframe(pose_jsonl_path)
    dataframe.to_csv(frame_quality_csv_path, index=False)

    summary_values = summarize_pose_quality(dataframe)
    summary_json_path.write_text(json.dumps(summary_values, indent=2) + "\n")

    return PoseQualitySummary(
        input_jsonl_path=str(pose_jsonl_path),
        frame_quality_csv_path=str(frame_quality_csv_path),
        summary_json_path=str(summary_json_path),
        total_frames_processed=int(summary_values["total_frames_processed"]),
        frames_with_pose=int(summary_values["frames_with_pose"]),
        pose_detection_rate=float(summary_values["pose_detection_rate"]),
        mean_frame_quality_score=float(summary_values["mean_frame_quality_score"]),
        mean_visibility=summary_values["mean_visibility"],
        mean_presence=summary_values["mean_presence"],
        low_quality_frame_rate=float(summary_values["low_quality_frame_rate"]),
    )


def _video_stem_from_pose_jsonl_path(pose_jsonl_path: Path) -> str:
    name = pose_jsonl_path.name

    if name.endswith(".pose.jsonl"):
        return name.removesuffix(".pose.jsonl")

    return pose_jsonl_path.stem


def _first_pose_landmarks(record: dict[str, Any], key: str) -> list[dict[str, Any]]:
    poses = record.get(key) or []

    if not poses:
        return []

    first_pose = poses[0]

    if not isinstance(first_pose, list):
        return []

    return first_pose


def _mean_landmark_value(landmarks: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for landmark in landmarks
        if (value := _optional_float(landmark.get(key))) is not None
    ]

    if not values:
        return None

    return float(sum(values) / len(values))


def _image_coordinate_valid_ratio(landmarks: list[dict[str, Any]]) -> float:
    if not landmarks:
        return 0.0

    valid_count = 0

    for landmark in landmarks:
        x = _optional_float(landmark.get("x"))
        y = _optional_float(landmark.get("y"))

        if x is None or y is None:
            continue

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            valid_count += 1

    return float(valid_count / len(landmarks))


def _series_mean_or_none(series: pd.Series) -> float | None:
    mean_value = series.mean(skipna=True)

    if pd.isna(mean_value):
        return None

    return float(mean_value)


def _series_mean_or_zero(series: pd.Series) -> float:
    mean_value = series.mean(skipna=True)

    if pd.isna(mean_value):
        return 0.0

    return float(mean_value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pose quality features from pose JSONL.")
    parser.add_argument("pose_jsonl_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = write_pose_quality_outputs(
        pose_jsonl_path=args.pose_jsonl_path,
        output_dir=args.output_dir,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
