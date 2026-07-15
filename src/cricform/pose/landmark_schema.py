from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_POSE_JSONL_DIR = Path("data/interim/pose_landmarks")

POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

LANDMARK_TABLE_COLUMNS: tuple[str, ...] = (
    "video_id",
    "source_video_path",
    "frame_index",
    "timestamp_ms",
    "pose_index",
    "coordinate_space",
    "landmark_index",
    "landmark_name",
    "x",
    "y",
    "z",
    "visibility",
    "presence",
)


@dataclass(frozen=True)
class LandmarkTableSummary:
    input_jsonl_path: str
    output_parquet_path: str
    rows: int
    unique_frames: int
    unique_pose_instances: int
    coordinate_spaces: list[str]


def landmark_name(landmark_index: int) -> str:
    """Return a readable MediaPipe pose landmark name for an index."""

    if 0 <= landmark_index < len(POSE_LANDMARK_NAMES):
        return POSE_LANDMARK_NAMES[landmark_index]

    return f"unknown_{landmark_index}"


def default_landmark_parquet_path(
    pose_jsonl_path: Path,
    output_dir: Path = DEFAULT_POSE_JSONL_DIR,
) -> Path:
    """Return a default tidy landmark Parquet path for a Pose JSONL file."""

    name = pose_jsonl_path.name

    if name.endswith(".pose.jsonl"):
        video_stem = name.removesuffix(".pose.jsonl")
    else:
        video_stem = pose_jsonl_path.stem

    return output_dir / f"{video_stem}.landmarks.parquet"


def pose_jsonl_to_landmark_dataframe(pose_jsonl_path: Path) -> pd.DataFrame:
    """Convert nested pose JSONL output into a tidy long-form landmark table."""

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

            rows.extend(_record_to_rows(record))

    return pd.DataFrame(rows, columns=list(LANDMARK_TABLE_COLUMNS))


def write_landmark_parquet(
    pose_jsonl_path: Path,
    output_parquet_path: Path | None = None,
) -> LandmarkTableSummary:
    """Convert pose JSONL to tidy Parquet and return a small summary."""

    if output_parquet_path is None:
        output_parquet_path = default_landmark_parquet_path(pose_jsonl_path)

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pose_jsonl_to_landmark_dataframe(pose_jsonl_path)
    dataframe.to_parquet(output_parquet_path, index=False)

    if dataframe.empty:
        unique_frames = 0
        unique_pose_instances = 0
        coordinate_spaces: list[str] = []
    else:
        unique_frames = int(dataframe["frame_index"].nunique())
        unique_pose_instances = int(
            dataframe[["frame_index", "pose_index", "coordinate_space"]]
            .drop_duplicates()
            .shape[0]
        )
        coordinate_spaces = sorted(dataframe["coordinate_space"].dropna().unique().tolist())

    return LandmarkTableSummary(
        input_jsonl_path=str(pose_jsonl_path),
        output_parquet_path=str(output_parquet_path),
        rows=int(len(dataframe)),
        unique_frames=unique_frames,
        unique_pose_instances=unique_pose_instances,
        coordinate_spaces=coordinate_spaces,
    )


def _record_to_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    source_video_path = str(record.get("video_path", ""))
    video_id = Path(source_video_path).stem if source_video_path else "unknown_video"
    frame_index = int(record["frame_index"])
    timestamp_ms = int(record["timestamp_ms"])

    rows: list[dict[str, Any]] = []

    coordinate_specs = (
        ("image", "pose_landmarks"),
        ("world", "pose_world_landmarks"),
    )

    for coordinate_space, key in coordinate_specs:
        poses = record.get(key) or []

        for pose_index, pose_landmarks in enumerate(poses):
            for landmark in pose_landmarks:
                landmark_index = int(landmark["landmark_index"])

                rows.append(
                    {
                        "video_id": video_id,
                        "source_video_path": source_video_path,
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "pose_index": pose_index,
                        "coordinate_space": coordinate_space,
                        "landmark_index": landmark_index,
                        "landmark_name": landmark_name(landmark_index),
                        "x": _optional_float(landmark.get("x")),
                        "y": _optional_float(landmark.get("y")),
                        "z": _optional_float(landmark.get("z")),
                        "visibility": _optional_float(landmark.get("visibility")),
                        "presence": _optional_float(landmark.get("presence")),
                    }
                )

    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pose JSONL to tidy Parquet.")
    parser.add_argument("pose_jsonl_path", type=Path)
    parser.add_argument("--output-parquet", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = write_landmark_parquet(
        pose_jsonl_path=args.pose_jsonl_path,
        output_parquet_path=args.output_parquet,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
