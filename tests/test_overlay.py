from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cricform.ingest.create_sample_video import create_synthetic_batting_video
from cricform.video.overlay import (
    default_overlay_output_path,
    draw_pose_overlay,
    load_pose_records_by_frame,
    render_pose_overlay_video,
)


def _pose_record(frame_index: int = 0, pose_count: int = 1) -> dict:
    landmarks = []

    for landmark_index in range(33):
        landmarks.append(
            {
                "landmark_index": landmark_index,
                "x": 0.30 + 0.01 * (landmark_index % 5),
                "y": 0.25 + 0.015 * (landmark_index % 7),
                "z": 0.0,
                "visibility": 0.9,
                "presence": 0.8,
            }
        )

    return {
        "video_path": "data/raw/videos/sample.mp4",
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 42,
        "pose_count": pose_count,
        "pose_landmarks": [landmarks] if pose_count > 0 else [],
        "pose_world_landmarks": [landmarks] if pose_count > 0 else [],
    }


def test_default_overlay_output_path() -> None:
    output_path = default_overlay_output_path(
        video_path=Path("data/raw/videos/sample.mp4"),
        output_dir=Path("outputs/sample_overlays"),
    )

    assert output_path == Path("outputs/sample_overlays/sample_pose_overlay.mp4")


def test_load_pose_records_by_frame(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "sample.pose.jsonl"
    records = [_pose_record(frame_index=0), _pose_record(frame_index=6)]

    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    loaded = load_pose_records_by_frame(jsonl_path)

    assert sorted(loaded.keys()) == [0, 6]
    assert loaded[6]["timestamp_ms"] == 252


def test_draw_pose_overlay_changes_frame_pixels() -> None:
    frame = np.full((180, 320, 3), 245, dtype=np.uint8)
    annotated = draw_pose_overlay(frame, _pose_record(frame_index=0), frame_index=0)

    assert annotated.shape == frame.shape
    assert not np.array_equal(frame, annotated)


def test_draw_pose_overlay_handles_missing_pose_record() -> None:
    frame = np.full((180, 320, 3), 245, dtype=np.uint8)
    annotated = draw_pose_overlay(frame, pose_record=None, frame_index=0)

    assert annotated.shape == frame.shape
    assert not np.array_equal(frame, annotated)


def test_render_pose_overlay_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    pose_jsonl_path = tmp_path / "sample.pose.jsonl"
    output_video_path = tmp_path / "sample_overlay.mp4"

    create_synthetic_batting_video(
        output_path=video_path,
        width=320,
        height=180,
        fps=12,
        num_frames=12,
    )

    records = [_pose_record(frame_index=0), _pose_record(frame_index=6)]
    pose_jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    summary = render_pose_overlay_video(
        video_path=video_path,
        pose_jsonl_path=pose_jsonl_path,
        output_video_path=output_video_path,
    )

    assert output_video_path.exists()
    assert output_video_path.stat().st_size > 0
    assert summary.frames_seen == 12
    assert summary.frames_written == 12
    assert summary.frames_with_pose_overlay == 2

    capture = cv2.VideoCapture(str(output_video_path))

    try:
        assert capture.isOpened()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == 12

        success, frame = capture.read()
        assert success
        assert frame is not None
    finally:
        capture.release()


def test_render_pose_overlay_rejects_invalid_max_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    pose_jsonl_path = tmp_path / "sample.pose.jsonl"

    create_synthetic_batting_video(
        output_path=video_path,
        width=320,
        height=180,
        fps=12,
        num_frames=12,
    )
    pose_jsonl_path.write_text(json.dumps(_pose_record(frame_index=0)) + "\n")

    with pytest.raises(ValueError):
        render_pose_overlay_video(
            video_path=video_path,
            pose_jsonl_path=pose_jsonl_path,
            output_video_path=tmp_path / "bad.mp4",
            max_frames=0,
        )
