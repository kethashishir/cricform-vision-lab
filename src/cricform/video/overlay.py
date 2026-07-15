from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_OVERLAY_OUTPUT_DIR = Path("outputs/sample_overlays")

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


@dataclass(frozen=True)
class OverlaySummary:
    video_path: str
    pose_jsonl_path: str
    output_video_path: str
    frames_seen: int
    frames_written: int
    frames_with_pose_overlay: int


def default_overlay_output_path(
    video_path: Path,
    output_dir: Path = DEFAULT_OVERLAY_OUTPUT_DIR,
) -> Path:
    """Return a default overlay MP4 path for a source video."""

    return output_dir / f"{video_path.stem}_pose_overlay.mp4"


def load_pose_records_by_frame(pose_jsonl_path: Path) -> dict[int, dict[str, Any]]:
    """Load pose JSONL records keyed by source frame index."""

    if not pose_jsonl_path.exists():
        raise FileNotFoundError(f"Pose JSONL file does not exist: {pose_jsonl_path}")

    records: dict[int, dict[str, Any]] = {}

    with pose_jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {pose_jsonl_path}"
                ) from error

            records[int(record["frame_index"])] = record

    return records


def render_pose_overlay_video(
    video_path: Path,
    pose_jsonl_path: Path,
    output_video_path: Path | None = None,
    max_frames: int | None = None,
) -> OverlaySummary:
    """Render a video with pose landmarks and frame status labels."""

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than 0 when provided")

    if output_video_path is None:
        output_video_path = default_overlay_output_path(video_path)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    pose_records = load_pose_records_by_frame(pose_jsonl_path)

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 24.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_video_path}")

        frames_seen = 0
        frames_written = 0
        frames_with_pose_overlay = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                pose_record = pose_records.get(frames_seen)
                annotated_frame = draw_pose_overlay(frame, pose_record, frames_seen)

                if pose_record is not None and int(pose_record.get("pose_count", 0)) > 0:
                    frames_with_pose_overlay += 1

                writer.write(annotated_frame)

                frames_seen += 1
                frames_written += 1

                if max_frames is not None and frames_written >= max_frames:
                    break

        finally:
            writer.release()

    finally:
        capture.release()

    return OverlaySummary(
        video_path=str(video_path),
        pose_jsonl_path=str(pose_jsonl_path),
        output_video_path=str(output_video_path),
        frames_seen=frames_seen,
        frames_written=frames_written,
        frames_with_pose_overlay=frames_with_pose_overlay,
    )


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    pose_record: dict[str, Any] | None,
    frame_index: int,
) -> np.ndarray:
    """Draw pose landmarks and status text onto a video frame."""

    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]

    if pose_record is None:
        _draw_status_panel(
            annotated,
            title="CricForm Vision Lab",
            lines=[
                f"frame: {frame_index}",
                "pose status: not sampled",
            ],
            status_color=(90, 90, 90),
        )
        return annotated

    pose_count = int(pose_record.get("pose_count", 0))
    timestamp_ms = int(pose_record.get("timestamp_ms", 0))

    image_landmarks = _first_pose_landmarks(pose_record)

    if pose_count <= 0 or not image_landmarks:
        _draw_status_panel(
            annotated,
            title="CricForm Vision Lab",
            lines=[
                f"frame: {frame_index}",
                f"timestamp_ms: {timestamp_ms}",
                "pose status: no pose detected",
            ],
            status_color=(40, 40, 220),
        )
        return annotated

    landmark_points = _landmark_pixel_points(image_landmarks, width, height)

    _draw_connections(annotated, landmark_points)
    _draw_landmark_points(annotated, landmark_points)

    _draw_status_panel(
        annotated,
        title="CricForm Vision Lab",
        lines=[
            f"frame: {frame_index}",
            f"timestamp_ms: {timestamp_ms}",
            f"pose status: detected ({pose_count})",
        ],
        status_color=(40, 160, 60),
    )

    return annotated


def _first_pose_landmarks(pose_record: dict[str, Any]) -> list[dict[str, Any]]:
    poses = pose_record.get("pose_landmarks") or []

    if not poses:
        return []

    first_pose = poses[0]

    if not isinstance(first_pose, list):
        return []

    return first_pose


def _landmark_pixel_points(
    landmarks: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[int, tuple[int, int]]:
    points: dict[int, tuple[int, int]] = {}

    for landmark in landmarks:
        landmark_index = int(landmark["landmark_index"])
        x = _optional_float(landmark.get("x"))
        y = _optional_float(landmark.get("y"))

        if x is None or y is None:
            continue

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue

        points[landmark_index] = (int(round(x * width)), int(round(y * height)))

    return points


def _draw_connections(
    frame_bgr: np.ndarray,
    landmark_points: dict[int, tuple[int, int]],
) -> None:
    for start_index, end_index in POSE_CONNECTIONS:
        start = landmark_points.get(start_index)
        end = landmark_points.get(end_index)

        if start is None or end is None:
            continue

        cv2.line(frame_bgr, start, end, (30, 220, 220), 2, cv2.LINE_AA)


def _draw_landmark_points(
    frame_bgr: np.ndarray,
    landmark_points: dict[int, tuple[int, int]],
) -> None:
    for point in landmark_points.values():
        cv2.circle(frame_bgr, point, 4, (40, 80, 255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, point, 5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_status_panel(
    frame_bgr: np.ndarray,
    title: str,
    lines: list[str],
    status_color: tuple[int, int, int],
) -> None:
    panel = frame_bgr.copy()

    x0 = 12
    y0 = 12
    x1 = min(frame_bgr.shape[1] - 12, 360)
    y1 = 36 + 26 * (len(lines) + 1)

    cv2.rectangle(panel, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.68, frame_bgr, 0.32, 0, frame_bgr)

    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), status_color, 2)

    cv2.putText(
        frame_bgr,
        title,
        (x0 + 12, y0 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for line_index, line in enumerate(lines):
        y = y0 + 56 + 24 * line_index
        cv2.putText(
            frame_bgr,
            line,
            (x0 + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pose landmarks onto a video.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("pose_jsonl_path", type=Path)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = render_pose_overlay_video(
        video_path=args.video_path,
        pose_jsonl_path=args.pose_jsonl_path,
        output_video_path=args.output_video,
        max_frames=args.max_frames,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
