from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

DEFAULT_MODEL_PATH = Path("models/pose_landmarker/pose_landmarker_lite.task")
DEFAULT_OUTPUT_DIR = Path("data/interim/pose_landmarks")


@dataclass(frozen=True)
class PoseExtractionSummary:
    video_path: str
    model_path: str
    output_jsonl_path: str
    frames_seen: int
    frames_processed: int
    frames_with_pose: int


def frame_timestamp_ms(frame_index: int, fps: float) -> int:
    """Convert a frame index to a monotonically increasing video timestamp."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    if fps <= 0:
        return frame_index

    return int(round((frame_index / fps) * 1000))


def default_pose_output_path(video_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Return the default JSONL output path for a video's pose extraction output."""

    return output_dir / f"{video_path.stem}.pose.jsonl"


def extract_pose_landmarks_to_jsonl(
    video_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_jsonl_path: Path | None = None,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> PoseExtractionSummary:
    """Run MediaPipe Pose Landmarker on a video and write frame-level JSONL.

    This writes lightweight, inspectable JSONL for Phase 4 only. Phase 5 will
    replace this with a stricter landmark schema and Parquet storage.
    """

    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be greater than 0")

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than 0 when provided")

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pose model does not exist: {model_path}. Run `make download-pose-model` first."
        )

    if output_jsonl_path is None:
        output_jsonl_path = default_pose_output_path(video_path)

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))

        options = _build_pose_landmarker_options(
            model_path=model_path,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        frames_seen = 0
        frames_processed = 0
        frames_with_pose = 0

        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            with output_jsonl_path.open("w", encoding="utf-8") as output_file:
                while True:
                    success, frame_bgr = capture.read()

                    if not success:
                        break

                    should_process = frames_seen % every_n_frames == 0

                    if should_process:
                        timestamp_ms = frame_timestamp_ms(frames_seen, fps)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frame_rgb = np.ascontiguousarray(frame_rgb)

                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=frame_rgb,
                        )

                        result = landmarker.detect_for_video(mp_image, timestamp_ms)
                        pose_count = len(result.pose_landmarks)

                        if pose_count > 0:
                            frames_with_pose += 1

                        record = {
                            "video_path": str(video_path),
                            "frame_index": frames_seen,
                            "timestamp_ms": timestamp_ms,
                            "pose_count": pose_count,
                            "pose_landmarks": [
                                _landmarks_to_records(pose_landmarks)
                                for pose_landmarks in result.pose_landmarks
                            ],
                            "pose_world_landmarks": [
                                _landmarks_to_records(pose_world_landmarks)
                                for pose_world_landmarks in result.pose_world_landmarks
                            ],
                        }

                        output_file.write(json.dumps(record) + "\n")
                        frames_processed += 1

                        if max_frames is not None and frames_processed >= max_frames:
                            break

                    frames_seen += 1

    finally:
        capture.release()

    return PoseExtractionSummary(
        video_path=str(video_path),
        model_path=str(model_path),
        output_jsonl_path=str(output_jsonl_path),
        frames_seen=frames_seen,
        frames_processed=frames_processed,
        frames_with_pose=frames_with_pose,
    )


def _build_pose_landmarker_options(
    model_path: Path,
    min_pose_detection_confidence: float,
    min_pose_presence_confidence: float,
    min_tracking_confidence: float,
) -> Any:
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))

    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )


def _landmarks_to_records(landmarks: list[Any]) -> list[dict[str, float | None]]:
    records: list[dict[str, float | None]] = []

    for landmark_index, landmark in enumerate(landmarks):
        records.append(
            {
                "landmark_index": float(landmark_index),
                "x": _as_optional_float(getattr(landmark, "x", None)),
                "y": _as_optional_float(getattr(landmark, "y", None)),
                "z": _as_optional_float(getattr(landmark, "z", None)),
                "visibility": _as_optional_float(getattr(landmark, "visibility", None)),
                "presence": _as_optional_float(getattr(landmark, "presence", None)),
            }
        )

    return records


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MediaPipe pose extraction on a video.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--every-n-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = extract_pose_landmarks_to_jsonl(
        video_path=args.video_path,
        model_path=args.model_path,
        output_jsonl_path=args.output_jsonl,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
