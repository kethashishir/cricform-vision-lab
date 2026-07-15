from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OUTPUT_PATH = Path("data/raw/videos/synthetic_batting_sample.mp4")


def create_synthetic_batting_video(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    width: int = 640,
    height: int = 360,
    fps: int = 24,
    num_frames: int = 72,
    overwrite: bool = False,
) -> Path:
    """Create a tiny synthetic batting-style video for pipeline smoke tests.

    This fixture is intentionally synthetic. It is not real cricket footage and
    should not be used for model quality, coaching, or biomechanics claims.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True or use --overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for frame_idx in range(num_frames):
            frame = _draw_frame(frame_idx, num_frames, width, height)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def _draw_frame(frame_idx: int, num_frames: int, width: int, height: int) -> np.ndarray:
    progress = frame_idx / max(num_frames - 1, 1)

    frame = np.full((height, width, 3), 245, dtype=np.uint8)

    ground_y = int(height * 0.78)
    crease_x = int(width * 0.36)

    cv2.line(frame, (0, ground_y), (width, ground_y), (110, 110, 110), 2)
    cv2.line(frame, (crease_x, ground_y - 45), (crease_x, ground_y + 20), (160, 160, 160), 2)

    hip = np.array([int(width * 0.36), int(height * 0.58)])
    shoulder = np.array([int(width * 0.35), int(height * 0.42)])
    head = np.array([int(width * 0.35), int(height * 0.32)])

    front_foot = np.array([int(width * 0.44), ground_y])
    back_foot = np.array([int(width * 0.30), ground_y])

    left_hand = shoulder + np.array([15, 26])
    right_hand = shoulder + np.array([25, 34])
    hands = ((left_hand + right_hand) / 2).astype(int)

    bat_length = int(height * 0.34)
    bat_angle_deg = _bat_angle_for_progress(progress)
    bat_angle_rad = np.deg2rad(bat_angle_deg)

    bat_tip = hands + np.array(
        [
            int(bat_length * np.cos(bat_angle_rad)),
            int(bat_length * np.sin(bat_angle_rad)),
        ]
    )

    ball_x = int(width * (0.88 - 0.42 * progress))
    ball_y = int(ground_y - 28 - 18 * np.sin(progress * np.pi))

    _draw_stick_batter(frame, head, shoulder, hip, front_foot, back_foot, left_hand, right_hand)
    cv2.line(frame, tuple(hands), tuple(bat_tip), (40, 40, 40), 7)
    cv2.circle(frame, (ball_x, ball_y), 7, (30, 30, 210), -1)

    label = f"synthetic fixture | frame {frame_idx + 1:02d}/{num_frames}"
    cv2.putText(
        frame,
        label,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )

    phase_label = _phase_label_for_progress(progress)
    cv2.putText(
        frame,
        phase_label,
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )

    return frame


def _draw_stick_batter(
    frame: np.ndarray,
    head: np.ndarray,
    shoulder: np.ndarray,
    hip: np.ndarray,
    front_foot: np.ndarray,
    back_foot: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> None:
    cv2.circle(frame, tuple(head), 18, (70, 70, 70), 3)
    cv2.line(frame, tuple(shoulder), tuple(hip), (70, 70, 70), 4)

    cv2.line(frame, tuple(shoulder), tuple(left_hand), (70, 70, 70), 4)
    cv2.line(frame, tuple(shoulder), tuple(right_hand), (70, 70, 70), 4)

    cv2.line(frame, tuple(hip), tuple(front_foot), (70, 70, 70), 4)
    cv2.line(frame, tuple(hip), tuple(back_foot), (70, 70, 70), 4)

    cv2.circle(frame, tuple(left_hand), 5, (70, 70, 70), -1)
    cv2.circle(frame, tuple(right_hand), 5, (70, 70, 70), -1)


def _bat_angle_for_progress(progress: float) -> float:
    if progress < 0.25:
        return -130 + 25 * (progress / 0.25)

    if progress < 0.55:
        local_progress = (progress - 0.25) / 0.30
        return -105 + 150 * local_progress

    if progress < 0.80:
        local_progress = (progress - 0.55) / 0.25
        return 45 + 45 * local_progress

    local_progress = (progress - 0.80) / 0.20
    return 90 - 18 * local_progress


def _phase_label_for_progress(progress: float) -> str:
    if progress < 0.18:
        return "phase proxy: stance"
    if progress < 0.35:
        return "phase proxy: backlift"
    if progress < 0.55:
        return "phase proxy: downswing"
    if progress < 0.62:
        return "phase proxy: contact zone"
    if progress < 0.85:
        return "phase proxy: follow-through"
    return "phase proxy: recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic batting video fixture.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output MP4 path.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=72)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_synthetic_batting_video(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=args.num_frames,
        overwrite=args.overwrite,
    )
    print(f"Created synthetic sample video: {output_path}")


if __name__ == "__main__":
    main()
