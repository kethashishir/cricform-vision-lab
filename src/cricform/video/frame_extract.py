from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    overwrite: bool = False,
    image_ext: str = ".jpg",
) -> list[Path]:
    """Extract frames from a video into a deterministic folder.

    Frames are saved as:

        output_dir / video_stem / frame_000000.jpg

    Args:
        video_path: Input video file.
        output_dir: Parent directory for extracted frames.
        every_n_frames: Save one frame every N frames.
        max_frames: Optional maximum number of frames to save.
        overwrite: Whether to delete existing extracted frames for this video.
        image_ext: Image extension, usually .jpg or .png.
    """

    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be greater than 0")

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than 0 when provided")

    if image_ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("image_ext must be one of: .jpg, .jpeg, .png")

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(video_output_dir.glob(f"frame_*{image_ext}"))

    if existing_frames and not overwrite:
        raise FileExistsError(
            f"Found existing frames in {video_output_dir}. "
            "Pass overwrite=True or use --overwrite."
        )

    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    saved_paths: list[Path] = []

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_idx = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            should_save = frame_idx % every_n_frames == 0

            if should_save:
                output_path = video_output_dir / f"frame_{frame_idx:06d}{image_ext}"
                write_success = cv2.imwrite(str(output_path), frame)

                if not write_success:
                    raise RuntimeError(f"Could not write frame: {output_path}")

                saved_paths.append(output_path)

                if max_frames is not None and len(saved_paths) >= max_frames:
                    break

            frame_idx += 1

    finally:
        capture.release()

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/frames"),
    )
    parser.add_argument("--every-n-frames", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-ext", choices=[".jpg", ".jpeg", ".png"], default=".jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    saved_paths = extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        image_ext=args.image_ext,
    )

    print(f"Extracted {len(saved_paths)} frames")
    if saved_paths:
        print(f"First frame: {saved_paths[0]}")
        print(f"Last frame: {saved_paths[-1]}")


if __name__ == "__main__":
    main()
