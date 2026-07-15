from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
)

PoseRunner = Callable[[Path, Path, Path, int, int], dict[str, Any]]


@dataclass(frozen=True)
class PoseAuditRow:
    video_id: str
    split: str
    shot_type: str
    video_path: str
    output_jsonl_path: str
    frames_seen: int
    frames_processed: int
    frames_with_pose: int
    pose_detection_rate: float
    status: str
    error: str


@dataclass(frozen=True)
class PoseAuditSummary:
    sample_dir: str
    output_csv_path: str
    output_summary_path: str
    total_videos: int
    attempted_videos: int
    failed_videos: int
    videos_with_pose: int
    mean_pose_detection_rate: float
    best_video: dict[str, Any] | None
    limitation: str


def discover_sample_videos(sample_dir: Path) -> list[Path]:
    """Find real sampled video files under a sample directory."""

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory does not exist: {sample_dir}")

    return sorted(
        path
        for path in sample_dir.rglob("*")
        if path.is_file() and _is_real_video_file(path)
    )


def infer_sample_video_labels(video_path: Path, sample_dir: Path) -> tuple[str, str]:
    """Infer split and shot type from sampled video path."""

    try:
        relative_path = video_path.relative_to(sample_dir)
    except ValueError:
        return "unknown", "unknown"

    parts = relative_path.parts

    if len(parts) >= 3:
        return parts[0], parts[1]

    if len(parts) >= 2:
        return "unknown", parts[-2]

    return "unknown", "unknown"


def make_video_id(split: str, shot_type: str, video_path: Path) -> str:
    """Create a filesystem-safe video identifier."""

    raw_video_id = f"{split}_{shot_type}_{video_path.stem}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw_video_id)


def run_batch_pose_audit(
    sample_dir: Path,
    model_path: Path,
    output_dir: Path,
    pose_jsonl_dir: Path,
    every_n_frames: int = 5,
    max_frames: int = 30,
    limit: int | None = None,
    pose_runner: PoseRunner | None = None,
) -> PoseAuditSummary:
    """Run pose extraction on sampled real videos and summarize detection quality."""

    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be greater than 0")

    if max_frames <= 0:
        raise ValueError("max_frames must be greater than 0")

    if not model_path.exists():
        raise FileNotFoundError(f"Pose model does not exist: {model_path}")

    videos = discover_sample_videos(sample_dir)
    if limit is not None:
        videos = videos[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    pose_jsonl_dir.mkdir(parents=True, exist_ok=True)

    runner = pose_runner or _run_pose_extraction
    rows: list[PoseAuditRow] = []

    for video_path in videos:
        split, shot_type = infer_sample_video_labels(video_path, sample_dir)
        video_id = make_video_id(split, shot_type, video_path)
        output_jsonl_path = pose_jsonl_dir / f"{video_id}.pose.jsonl"

        try:
            pose_summary = runner(
                video_path,
                model_path,
                output_jsonl_path,
                every_n_frames,
                max_frames,
            )
            frames_seen = int(pose_summary.get("frames_seen", 0))
            frames_processed = int(pose_summary.get("frames_processed", 0))
            frames_with_pose = int(pose_summary.get("frames_with_pose", 0))
            pose_detection_rate = (
                frames_with_pose / frames_processed if frames_processed else 0.0
            )
            status = "pose_detected" if frames_with_pose > 0 else "no_pose_detected"
            error = ""
        except Exception as exc:  # noqa: BLE001
            frames_seen = 0
            frames_processed = 0
            frames_with_pose = 0
            pose_detection_rate = 0.0
            status = "failed"
            error = str(exc)

        rows.append(
            PoseAuditRow(
                video_id=video_id,
                split=split,
                shot_type=shot_type,
                video_path=str(video_path),
                output_jsonl_path=str(output_jsonl_path),
                frames_seen=frames_seen,
                frames_processed=frames_processed,
                frames_with_pose=frames_with_pose,
                pose_detection_rate=round(pose_detection_rate, 4),
                status=status,
                error=error,
            )
        )

    output_csv_path = output_dir / "pose_audit.csv"
    output_summary_path = output_dir / "pose_audit_summary.json"

    row_dicts = [asdict(row) for row in rows]
    pd.DataFrame(row_dicts).to_csv(output_csv_path, index=False)

    summary = _build_summary(
        sample_dir=sample_dir,
        output_csv_path=output_csv_path,
        output_summary_path=output_summary_path,
        rows=rows,
    )
    output_summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n")

    return summary


def _build_summary(
    sample_dir: Path,
    output_csv_path: Path,
    output_summary_path: Path,
    rows: list[PoseAuditRow],
) -> PoseAuditSummary:
    attempted_rows = [row for row in rows if row.status != "failed"]
    failed_rows = [row for row in rows if row.status == "failed"]
    pose_rows = [row for row in rows if row.frames_with_pose > 0]

    mean_pose_detection_rate = (
        sum(row.pose_detection_rate for row in attempted_rows) / len(attempted_rows)
        if attempted_rows
        else 0.0
    )

    best_row = max(
        attempted_rows,
        key=lambda row: (row.pose_detection_rate, row.frames_with_pose),
        default=None,
    )

    return PoseAuditSummary(
        sample_dir=str(sample_dir),
        output_csv_path=str(output_csv_path),
        output_summary_path=str(output_summary_path),
        total_videos=len(rows),
        attempted_videos=len(attempted_rows),
        failed_videos=len(failed_rows),
        videos_with_pose=len(pose_rows),
        mean_pose_detection_rate=round(mean_pose_detection_rate, 4),
        best_video=asdict(best_row) if best_row else None,
        limitation=(
            "Pose audit uses MediaPipe detections on sampled public cricket videos. "
            "Detection rate measures landmark availability only; it is not a "
            "biomechanics, coaching-quality, or shot-correctness score."
        ),
    )


def _run_pose_extraction(
    video_path: Path,
    model_path: Path,
    output_jsonl_path: Path,
    every_n_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "cricform.pose.mediapipe_pose",
        str(video_path),
        "--model-path",
        str(model_path),
        "--output-jsonl",
        str(output_jsonl_path),
        "--every-n-frames",
        str(every_n_frames),
        "--max-frames",
        str(max_frames),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(message[-1000:]) from exc

    return _parse_json_from_stdout(completed.stdout)


def _parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    start_index = stdout.find("{")
    end_index = stdout.rfind("}")

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise ValueError("Pose extraction did not emit a JSON summary.")

    return json.loads(stdout[start_index : end_index + 1])


def _is_real_video_file(path: Path) -> bool:
    if path.name.startswith("._"):
        return False

    if "__MACOSX" in path.parts:
        return False

    return path.suffix.lower() in VIDEO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-audit pose detection on videos.")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pose-jsonl-dir", type=Path, required=True)
    parser.add_argument("--every-n-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = run_batch_pose_audit(
        sample_dir=args.sample_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        pose_jsonl_dir=args.pose_jsonl_dir,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        limit=args.limit,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
