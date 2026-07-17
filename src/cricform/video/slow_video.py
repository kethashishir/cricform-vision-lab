from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def create_slow_video(
    input_video_path: Path,
    output_video_path: Path,
    slow_factor: float = 3.0,
) -> dict[str, object]:
    """Create a slower presentation copy by reducing output FPS."""

    if slow_factor <= 0:
        raise ValueError("slow_factor must be greater than 0")

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video_path}")

    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {input_video_path}")

    input_fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if input_fps <= 0:
        input_fps = 24.0

    output_fps = max(input_fps / slow_factor, 1.0)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not open output video writer: {output_video_path}")

    frames_written = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        writer.write(frame)
        frames_written += 1

    capture.release()
    writer.release()

    return {
        "input_video_path": str(input_video_path),
        "output_video_path": str(output_video_path),
        "input_fps": round(float(input_fps), 4),
        "output_fps": round(float(output_fps), 4),
        "slow_factor": slow_factor,
        "frames_written": frames_written,
        "limitation": (
            "Slow video output is for presentation only. It does not change pose "
            "timestamps, movement metrics, phase detection, or analysis results."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a slow presentation video.")
    parser.add_argument("input_video_path", type=Path)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--slow-factor", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = create_slow_video(
        input_video_path=args.input_video_path,
        output_video_path=args.output_video,
        slow_factor=args.slow_factor,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
