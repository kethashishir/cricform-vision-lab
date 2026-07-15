from pathlib import Path

import cv2

from cricform.ingest.create_sample_video import create_synthetic_batting_video


def test_create_synthetic_batting_video(tmp_path: Path) -> None:
    output_path = tmp_path / "synthetic_batting_sample.mp4"

    created_path = create_synthetic_batting_video(
        output_path=output_path,
        width=320,
        height=180,
        fps=12,
        num_frames=12,
    )

    assert created_path.exists()
    assert created_path.stat().st_size > 0

    capture = cv2.VideoCapture(str(created_path))

    try:
        assert capture.isOpened()

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(capture.get(cv2.CAP_PROP_FPS)))

        success, frame = capture.read()

        assert frame_count == 12
        assert width == 320
        assert height == 180
        assert fps == 12
        assert success
        assert frame is not None
    finally:
        capture.release()
