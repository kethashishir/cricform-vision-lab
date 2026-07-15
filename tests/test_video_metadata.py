from pathlib import Path

import pytest

from cricform.ingest.create_sample_video import create_synthetic_batting_video
from cricform.video.metadata import read_video_metadata


def test_read_video_metadata_for_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"

    create_synthetic_batting_video(
        output_path=video_path,
        width=320,
        height=180,
        fps=12,
        num_frames=24,
    )

    metadata = read_video_metadata(video_path)

    assert metadata.frame_count == 24
    assert round(metadata.fps) == 12
    assert metadata.width == 320
    assert metadata.height == 180
    assert metadata.duration_sec == pytest.approx(2.0, abs=0.1)
    assert metadata.path.endswith("sample.mp4")
