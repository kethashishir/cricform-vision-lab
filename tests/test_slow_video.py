from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cricform.video.slow_video import create_slow_video


def _write_test_video(path: Path, fps: float = 24.0, frame_count: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )

    assert writer.isOpened()

    for index in range(frame_count):
        frame = np.full((48, 64, 3), index * 10, dtype=np.uint8)
        writer.write(frame)

    writer.release()


def test_create_slow_video(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output_slow.mp4"

    _write_test_video(input_path, fps=24.0, frame_count=8)

    summary = create_slow_video(
        input_video_path=input_path,
        output_video_path=output_path,
        slow_factor=3.0,
    )

    assert output_path.exists()
    assert summary["frames_written"] == 8
    assert summary["input_fps"] == 24.0
    assert summary["output_fps"] == 8.0
    assert summary["slow_factor"] == 3.0


def test_create_slow_video_rejects_invalid_factor(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    _write_test_video(input_path)

    with pytest.raises(ValueError, match="slow_factor"):
        create_slow_video(input_path, tmp_path / "bad.mp4", slow_factor=0)
