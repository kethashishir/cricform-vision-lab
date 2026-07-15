from pathlib import Path

import pytest

from cricform.pose.mediapipe_pose import default_pose_output_path, frame_timestamp_ms


def test_frame_timestamp_ms() -> None:
    assert frame_timestamp_ms(frame_index=0, fps=24.0) == 0
    assert frame_timestamp_ms(frame_index=12, fps=24.0) == 500
    assert frame_timestamp_ms(frame_index=24, fps=24.0) == 1000


def test_frame_timestamp_ms_falls_back_when_fps_invalid() -> None:
    assert frame_timestamp_ms(frame_index=7, fps=0.0) == 7


def test_frame_timestamp_ms_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError):
        frame_timestamp_ms(frame_index=-1, fps=24.0)


def test_default_pose_output_path() -> None:
    output_path = default_pose_output_path(
        video_path=Path("data/raw/videos/example.mp4"),
        output_dir=Path("data/interim/pose_landmarks"),
    )

    assert output_path == Path("data/interim/pose_landmarks/example.pose.jsonl")
