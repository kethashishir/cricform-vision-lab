from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CommandRunner = Callable[[list[str]], dict[str, Any]]


@dataclass(frozen=True)
class RealDemoSummary:
    pose_audit_csv_path: str
    pose_audit_summary_path: str
    selected_video_id: str
    selected_shot_type: str
    selected_video_path: str
    selected_pose_jsonl_path: str
    selected_pose_detection_rate: float
    output_landmark_parquet_path: str
    output_overlay_video_path: str
    output_pose_quality_summary_path: str
    output_phase_summary_path: str
    output_movement_summary_path: str
    output_baseline_profile_path: str
    output_comparison_json_path: str
    output_report_markdown_path: str
    output_report_chart_path: str
    baseline_manifest_path: str
    baseline_video_count: int
    limitation: str


def select_best_pose_audit_row(
    audit_csv_path: Path,
    audit_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Select the best real sample based on pose detection rate."""

    if audit_summary_path is not None and audit_summary_path.exists():
        summary = json.loads(audit_summary_path.read_text())
        best_video = summary.get("best_video")
        if best_video:
            return best_video

    dataframe = pd.read_csv(audit_csv_path)

    if dataframe.empty:
        raise ValueError("Pose audit CSV is empty.")

    usable = dataframe[dataframe["status"] == "pose_detected"].copy()
    if usable.empty:
        raise ValueError("No pose-detected videos are available for real demo.")

    usable = usable.sort_values(
        by=["pose_detection_rate", "frames_with_pose"],
        ascending=False,
    )
    return usable.iloc[0].to_dict()


def build_real_demo_artifacts(
    pose_audit_csv_path: Path,
    pose_audit_summary_path: Path,
    output_root: Path,
    overlay_output_dir: Path,
    command_runner: CommandRunner | None = None,
) -> RealDemoSummary:
    """Build real demo artifacts from the best audited cricket sample."""

    runner = command_runner or _run_json_command
    output_root.mkdir(parents=True, exist_ok=True)
    overlay_output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_best_pose_audit_row(
        audit_csv_path=pose_audit_csv_path,
        audit_summary_path=pose_audit_summary_path,
    )

    selected_video_id = str(selected["video_id"])
    selected_shot_type = str(selected["shot_type"])
    selected_video_path = Path(str(selected["video_path"]))
    selected_pose_jsonl_path = Path(str(selected["output_jsonl_path"]))
    selected_pose_detection_rate = float(selected["pose_detection_rate"])

    if not selected_video_path.exists():
        raise FileNotFoundError(f"Selected video does not exist: {selected_video_path}")

    if not selected_pose_jsonl_path.exists():
        raise FileNotFoundError(
            f"Selected pose JSONL does not exist: {selected_pose_jsonl_path}"
        )

    pose_dir = output_root / "pose"
    feature_dir = output_root / "features"
    baseline_dir = output_root / "baselines"
    report_dir = output_root / "reports"

    for directory in [pose_dir, feature_dir, baseline_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    landmark_parquet_path = pose_dir / f"{selected_video_id}.landmarks.parquet"
    overlay_video_path = overlay_output_dir / f"{selected_video_id}_pose_overlay.mp4"

    landmark_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.pose.landmark_schema",
            str(selected_pose_jsonl_path),
            "--output-parquet",
            str(landmark_parquet_path),
        ]
    )

    quality_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.features.quality_features",
            str(selected_pose_jsonl_path),
            "--output-dir",
            str(feature_dir),
        ]
    )

    phase_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.phases.detect_phases",
            str(landmark_parquet_path),
            "--output-dir",
            str(feature_dir),
        ]
    )

    movement_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.features.motion_features",
            str(landmark_parquet_path),
            "--phase-timeline-csv",
            str(phase_summary["output_phase_csv_path"]),
            "--output-dir",
            str(feature_dir),
        ]
    )

    overlay_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.video.overlay",
            str(selected_video_path),
            str(selected_pose_jsonl_path),
            "--output-video",
            str(overlay_video_path),
        ]
    )

    baseline_manifest_path = baseline_dir / "real_demo_baseline_manifest.csv"
    baseline_profile_path = baseline_dir / "real_demo_baseline_profile.json"

    baseline_video_count = write_mixed_real_sample_manifest(
        pose_audit_csv_path=pose_audit_csv_path,
        baseline_manifest_path=baseline_manifest_path,
        feature_csv_path=Path(str(movement_summary["output_features_csv_path"])),
    )

    baseline_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.baseline.build_baseline",
            str(baseline_manifest_path),
            "--output-profile",
            str(baseline_profile_path),
        ]
    )

    comparison_json_path = report_dir / f"{selected_video_id}_comparison.json"
    report_markdown_path = report_dir / f"{selected_video_id}_report.md"
    report_chart_path = report_dir / f"{selected_video_id}_metric_comparison.png"

    comparison_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.baseline.compare_shot",
            str(movement_summary["output_summary_json_path"]),
            str(baseline_profile_path),
            "--shot-type",
            "real_sample_mixed",
            "--output-comparison",
            str(comparison_json_path),
        ]
    )

    report_summary = runner(
        [
            sys.executable,
            "-m",
            "cricform.reports.render_report",
            str(comparison_json_path),
            "--output-markdown",
            str(report_markdown_path),
            "--output-chart",
            str(report_chart_path),
        ]
    )

    summary = RealDemoSummary(
        pose_audit_csv_path=str(pose_audit_csv_path),
        pose_audit_summary_path=str(pose_audit_summary_path),
        selected_video_id=selected_video_id,
        selected_shot_type=selected_shot_type,
        selected_video_path=str(selected_video_path),
        selected_pose_jsonl_path=str(selected_pose_jsonl_path),
        selected_pose_detection_rate=round(selected_pose_detection_rate, 4),
        output_landmark_parquet_path=str(landmark_summary["output_parquet_path"]),
        output_overlay_video_path=str(overlay_summary["output_video_path"]),
        output_pose_quality_summary_path=str(quality_summary["output_summary_json_path"]),
        output_phase_summary_path=str(phase_summary["output_summary_json_path"]),
        output_movement_summary_path=str(movement_summary["output_summary_json_path"]),
        output_baseline_profile_path=str(baseline_summary["output_profile_path"]),
        output_comparison_json_path=str(comparison_summary["output_comparison_path"]),
        output_report_markdown_path=str(report_summary["output_markdown_path"]),
        output_report_chart_path=str(report_summary["output_chart_path"]),
        baseline_manifest_path=str(baseline_manifest_path),
        baseline_video_count=baseline_video_count,
        limitation=(
            "Real demo artifacts are generated from sampled public cricket clips. "
            "The mixed baseline is tiny and exists to demonstrate pipeline behavior. "
            "It is not a coaching standard, population norm, biomechanics validation, "
            "medical assessment, or shot-correctness score."
        ),
    )

    summary_path = output_root / "real_demo_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n")

    return summary


def write_mixed_real_sample_manifest(
    pose_audit_csv_path: Path,
    baseline_manifest_path: Path,
    feature_csv_path: Path,
) -> int:
    """Write a small manifest for a mixed real-sample demo baseline."""

    dataframe = pd.read_csv(pose_audit_csv_path)
    usable = dataframe[dataframe["status"] == "pose_detected"].copy()

    baseline_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in usable.iterrows():
        rows.append(
            {
                "shot_id": row["video_id"],
                "shot_type": "real_sample_mixed",
                "movement_features_csv": _relative_or_absolute(
                    feature_csv_path,
                    baseline_manifest_path.parent,
                ),
            }
        )

    pd.DataFrame(rows).to_csv(baseline_manifest_path, index=False)
    return len(rows)


def _relative_or_absolute(path: Path, start: Path) -> str:
    """Return a path that can be resolved relative to the manifest directory."""

    try:
        return os.path.relpath(path.resolve(), start.resolve())
    except OSError:
        return str(path)


def _run_json_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(message[-1500:]) from exc

    return _parse_json_from_stdout(completed.stdout)


def _parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    start_index = stdout.find("{")
    end_index = stdout.rfind("}")

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise ValueError("Command did not emit a JSON object.")

    return json.loads(stdout[start_index : end_index + 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real cricket demo artifacts.")
    parser.add_argument("--pose-audit-csv", type=Path, required=True)
    parser.add_argument("--pose-audit-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overlay-output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = build_real_demo_artifacts(
        pose_audit_csv_path=args.pose_audit_csv,
        pose_audit_summary_path=args.pose_audit_summary,
        output_root=args.output_root,
        overlay_output_dir=args.overlay_output_dir,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
