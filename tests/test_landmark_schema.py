from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cricform.pose.landmark_schema import (
    LANDMARK_TABLE_COLUMNS,
    POSE_LANDMARK_NAMES,
    default_landmark_parquet_path,
    landmark_name,
    pose_jsonl_to_landmark_dataframe,
    write_landmark_parquet,
)


def test_pose_landmark_names_match_mediapipe_pose_count() -> None:
    assert len(POSE_LANDMARK_NAMES) == 33
    assert landmark_name(0) == "nose"
    assert landmark_name(11) == "left_shoulder"
    assert landmark_name(12) == "right_shoulder"
    assert landmark_name(15) == "left_wrist"
    assert landmark_name(16) == "right_wrist"
    assert landmark_name(32) == "right_foot_index"
    assert landmark_name(99) == "unknown_99"


def test_default_landmark_parquet_path() -> None:
    output_path = default_landmark_parquet_path(
        pose_jsonl_path=Path("data/interim/pose_landmarks/sample.pose.jsonl"),
        output_dir=Path("data/interim/pose_landmarks"),
    )

    assert output_path == Path("data/interim/pose_landmarks/sample.landmarks.parquet")


def test_pose_jsonl_to_landmark_dataframe(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sample.pose.jsonl"

    record = {
        "video_path": "data/raw/videos/sample.mp4",
        "frame_index": 3,
        "timestamp_ms": 125,
        "pose_count": 1,
        "pose_landmarks": [
            [
                {
                    "landmark_index": 0,
                    "x": 0.5,
                    "y": 0.2,
                    "z": -0.1,
                    "visibility": 0.9,
                    "presence": 0.8,
                },
                {
                    "landmark_index": 15,
                    "x": 0.3,
                    "y": 0.6,
                    "z": -0.2,
                    "visibility": 0.7,
                    "presence": 0.6,
                },
            ]
        ],
        "pose_world_landmarks": [
            [
                {
                    "landmark_index": 0,
                    "x": 0.01,
                    "y": 0.02,
                    "z": 0.03,
                    "visibility": 0.9,
                    "presence": 0.8,
                }
            ]
        ],
    }

    jsonl_path.write_text(json.dumps(record) + "\n")

    dataframe = pose_jsonl_to_landmark_dataframe(jsonl_path)

    assert list(dataframe.columns) == list(LANDMARK_TABLE_COLUMNS)
    assert len(dataframe) == 3

    first_row = dataframe.iloc[0].to_dict()

    assert first_row["video_id"] == "sample"
    assert first_row["source_video_path"] == "data/raw/videos/sample.mp4"
    assert first_row["frame_index"] == 3
    assert first_row["timestamp_ms"] == 125
    assert first_row["pose_index"] == 0
    assert first_row["coordinate_space"] == "image"
    assert first_row["landmark_index"] == 0
    assert first_row["landmark_name"] == "nose"
    assert first_row["x"] == 0.5


def test_write_landmark_parquet(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sample.pose.jsonl"
    parquet_path = tmp_path / "sample.landmarks.parquet"

    records = [
        {
            "video_path": "data/raw/videos/sample.mp4",
            "frame_index": 0,
            "timestamp_ms": 0,
            "pose_count": 0,
            "pose_landmarks": [],
            "pose_world_landmarks": [],
        },
        {
            "video_path": "data/raw/videos/sample.mp4",
            "frame_index": 1,
            "timestamp_ms": 42,
            "pose_count": 1,
            "pose_landmarks": [
                [
                    {
                        "landmark_index": 0,
                        "x": 0.5,
                        "y": 0.2,
                        "z": -0.1,
                        "visibility": 0.9,
                        "presence": 0.8,
                    }
                ]
            ],
            "pose_world_landmarks": [],
        },
    ]

    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    summary = write_landmark_parquet(
        pose_jsonl_path=jsonl_path,
        output_parquet_path=parquet_path,
    )

    assert parquet_path.exists()
    assert summary.rows == 1
    assert summary.unique_frames == 1
    assert summary.unique_pose_instances == 1
    assert summary.coordinate_spaces == ["image"]

    dataframe = pd.read_parquet(parquet_path)
    assert list(dataframe.columns) == list(LANDMARK_TABLE_COLUMNS)
    assert len(dataframe) == 1
    assert dataframe.iloc[0]["landmark_name"] == "nose"
