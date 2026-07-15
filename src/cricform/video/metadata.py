from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float
    fourcc: str


def read_video_metadata(video_path: Path) -> VideoMetadata:
    """Read basic metadata from a video file using OpenCV."""

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))

        duration_sec = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=str(video_path),
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_sec=duration_sec,
            fourcc=_decode_fourcc(fourcc_int),
        )
    finally:
        capture.release()


def _decode_fourcc(fourcc_int: int) -> str:
    chars = [
        chr((fourcc_int >> 8 * index) & 0xFF)
        for index in range(4)
    ]
    decoded = "".join(chars).strip()
    return decoded if decoded else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read video metadata.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--json", action="store_true", help="Print metadata as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = read_video_metadata(args.video_path)

    if args.json:
        print(json.dumps(asdict(metadata), indent=2))
        return

    print(f"Video: {metadata.path}")
    print(f"Frames: {metadata.frame_count}")
    print(f"FPS: {metadata.fps:.2f}")
    print(f"Size: {metadata.width}x{metadata.height}")
    print(f"Duration: {metadata.duration_sec:.2f}s")
    print(f"FOURCC: {metadata.fourcc}")


if __name__ == "__main__":
    main()
