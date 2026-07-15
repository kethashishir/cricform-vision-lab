from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cricform.features.quality_features import (
    FRAME_QUALITY_COLUMNS,
    compute_frame_pose_quality,
    pose_quality_dataframe,
    summarize_pose_quality,
    write_pose_quality_outputs,
)


def _landmark(
    landmark_index: int,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    visibility: float = 0.8,
    presence: float = 0.6,
) -> dict[str, float | int]:
    return {
        "landmark_index": landmark_index,
        "x": x,
        "y": y,
        "z": z,
        "visibility": visibility,
        "presence": presence,
    }


def _record_without_pose() -> dict:
    return {
        "video_path": "data/raw/videos/sample.mp4",
        "frame_index": 0,
        "timestamp_ms": 0,
        "pose_count": 0,
        "pose_landmarks": [],
        "pose_world_landmarks": [],
    }


def _record_with_full_pose() -> dict:
    landmarks = [_landmark(index) for index in range(33)]

    return {
        "video_path": "data/raw/videos/sample.mp4",
        "frame_index": 1,
        "timestamp_ms": 42,
        "pose_count": 1,
        "pose_landmarks": [landmarks],
        "pose_world_landmarks": [landmarks],
    }


def test_compute_frame_pose_quality_without_pose() -> None:
    quality = compute_frame_pose_quality(_record_without_pose())

    assert quality["video_id"] == "sample"
    assert quality["has_pose"] is False
    assert quality["image_landmark_count"] == 0
    assert quality["world_landmark_count"] == 0
    assert quality["landmark_coverage"] == 0.0
    assert quality["mean_visibility"] is None
    assert quality["mean_presence"] is None
    assert quality["coordinate_valid_ratio"] == 0.0
    assert quality["frame_quality_score"] == 0.0


def test_compute_frame_pose_quality_with_full_pose() -> None:
    quality = compute_frame_pose_quality(_record_with_full_pose())

    assert quality["has_pose"] is True
    assert quality["image_landmark_count"] == 33
    assert quality["world_landmark_count"] == 33
    assert quality["landmark_coverage"] == 1.0
    assert quality["mean_visibility"] == pytest.approx(0.8)
    assert quality["mean_presence"] == pytest.approx(0.6)
    assert quality["coordinate_valid_ratio"] == 1.0
    assert quality["frame_quality_score"] == pytest.approx(0.85)


def test_pose_quality_dataframe(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sample.pose.jsonl"

    records = [
        _record_without_pose(),
        _record_with_full_pose(),
    ]

    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    dataframe = pose_quality_dataframe(jsonl_path)

    assert list(dataframe.columns) == list(FRAME_QUALITY_COLUMNS)
    assert len(dataframe) == 2
    assert dataframe.iloc[0]["has_pose"] == False  # noqa: E712
    assert dataframe.iloc[1]["has_pose"] == True  # noqa: E712


def test_summarize_pose_quality() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "has_pose": False,
                "frame_quality_score": 0.0,
                "mean_visibility": None,
                "mean_presence": None,
            },
            {
                "has_pose": True,
                "frame_quality_score": 0.85,
                "mean_visibility": 0.8,
                "mean_presence": 0.6,
            },
        ]
    )

    summary = summarize_pose_quality(dataframe)

    assert summary["total_frames_processed"] == 2
    assert summary["frames_with_pose"] == 1
    assert summary["pose_detection_rate"] == 0.5
    assert summary["mean_frame_quality_score"] == pytest.approx(0.425)
    assert summary["mean_visibility"] == pytest.approx(0.8)
    assert summary["mean_presence"] == pytest.approx(0.6)
    assert summary["low_quality_frame_rate"] == 0.5


def test_write_pose_quality_outputs(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sample.pose.jsonl"
    output_dir = tmp_path / "features"

    records = [
        _record_without_pose(),
        _record_with_full_pose(),
    ]

    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    summary = write_pose_quality_outputs(
        pose_jsonl_path=jsonl_path,
        output_dir=output_dir,
    )

    assert Path(summary.frame_quality_csv_path).exists()
    assert Path(summary.summary_json_path).exists()
    assert summary.total_frames_processed == 2
    assert summary.frames_with_pose == 1
    assert summary.pose_detection_rate == 0.5

    summary_json = json.loads(Path(summary.summary_json_path).read_text())
    assert summary_json["total_frames_processed"] == 2
    assert summary_json["frames_with_pose"] == 1
