from pathlib import Path

import cv2
import pytest

from cricform.ingest.create_sample_video import create_synthetic_batting_video
from cricform.video.frame_extract import extract_frames


def test_extract_frames_from_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    frame_output_dir = tmp_path / "frames"

    create_synthetic_batting_video(
        output_path=video_path,
        width=320,
        height=180,
        fps=12,
        num_frames=24,
    )

    saved_paths = extract_frames(
        video_path=video_path,
        output_dir=frame_output_dir,
        every_n_frames=6,
        overwrite=False,
    )

    assert len(saved_paths) == 4

    expected_names = [
        "frame_000000.jpg",
        "frame_000006.jpg",
        "frame_000012.jpg",
        "frame_000018.jpg",
    ]

    assert [path.name for path in saved_paths] == expected_names

    for frame_path in saved_paths:
        assert frame_path.exists()
        assert frame_path.stat().st_size > 0

        frame = cv2.imread(str(frame_path))
        assert frame is not None
        assert frame.shape[0] == 180
        assert frame.shape[1] == 320


def test_extract_frames_refuses_to_overwrite_existing_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    frame_output_dir = tmp_path / "frames"

    create_synthetic_batting_video(
        output_path=video_path,
        width=320,
        height=180,
        fps=12,
        num_frames=12,
    )

    extract_frames(
        video_path=video_path,
        output_dir=frame_output_dir,
        every_n_frames=6,
    )

    with pytest.raises(FileExistsError):
        extract_frames(
            video_path=video_path,
            output_dir=frame_output_dir,
            every_n_frames=6,
        )


def test_extract_frames_rejects_invalid_sampling_rate(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"

    create_synthetic_batting_video(
        output_path=video_path,
        width=320,
        height=180,
        fps=12,
        num_frames=12,
    )

    with pytest.raises(ValueError):
        extract_frames(
            video_path=video_path,
            output_dir=tmp_path / "frames",
            every_n_frames=0,
        )
