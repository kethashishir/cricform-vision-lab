from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cricform.pipeline.real_demo import (
    _parse_json_from_stdout,
    build_real_demo_artifacts,
    select_best_pose_audit_row,
    write_real_sample_baseline_manifest,
)


def test_parse_json_from_stdout_with_logs() -> None:
    stdout = 'INFO something happened\n{"ok": true, "value": 3}\n'

    assert _parse_json_from_stdout(stdout) == {"ok": True, "value": 3}


def test_select_best_pose_audit_row_prefers_summary(tmp_path: Path) -> None:
    audit_csv = tmp_path / "pose_audit.csv"
    audit_summary = tmp_path / "pose_audit_summary.json"

    pd.DataFrame(
        [
            {
                "video_id": "low",
                "shot_type": "cover",
                "video_path": "cover.avi",
                "output_jsonl_path": "cover.jsonl",
                "pose_detection_rate": 0.1,
                "frames_with_pose": 1,
                "status": "pose_detected",
            }
        ]
    ).to_csv(audit_csv, index=False)

    audit_summary.write_text(
        json.dumps(
            {
                "best_video": {
                    "video_id": "best",
                    "shot_type": "pull",
                    "video_path": "pull.avi",
                    "output_jsonl_path": "pull.jsonl",
                    "pose_detection_rate": 0.9,
                    "frames_with_pose": 9,
                    "status": "pose_detected",
                }
            }
        )
    )

    selected = select_best_pose_audit_row(audit_csv, audit_summary)

    assert selected["video_id"] == "best"
    assert selected["shot_type"] == "pull"


def test_select_best_pose_audit_row_falls_back_to_csv(tmp_path: Path) -> None:
    audit_csv = tmp_path / "pose_audit.csv"

    pd.DataFrame(
        [
            {
                "video_id": "cover",
                "shot_type": "cover",
                "video_path": "cover.avi",
                "output_jsonl_path": "cover.jsonl",
                "pose_detection_rate": 0.2,
                "frames_with_pose": 2,
                "status": "pose_detected",
            },
            {
                "video_id": "pull",
                "shot_type": "pull",
                "video_path": "pull.avi",
                "output_jsonl_path": "pull.jsonl",
                "pose_detection_rate": 0.6,
                "frames_with_pose": 6,
                "status": "pose_detected",
            },
        ]
    ).to_csv(audit_csv, index=False)

    selected = select_best_pose_audit_row(audit_csv)

    assert selected["video_id"] == "pull"


def test_write_real_sample_baseline_manifest_uses_per_video_features(
    tmp_path: Path,
) -> None:
    audit_csv = tmp_path / "pose_audit.csv"
    manifest_path = tmp_path / "baselines" / "manifest.csv"
    baseline_feature_dir = tmp_path / "baselines" / "real_sample_features"
    known_feature_csv = tmp_path / "features" / "pull_selected.movement_features.csv"
    cover_pose_jsonl = tmp_path / "pose" / "cover.pose.jsonl"
    pull_pose_jsonl = tmp_path / "pose" / "pull.pose.jsonl"

    known_feature_csv.parent.mkdir(parents=True)
    known_feature_csv.write_text("frame_index,value\n0,1\n")
    cover_pose_jsonl.parent.mkdir(parents=True)
    cover_pose_jsonl.write_text("")
    pull_pose_jsonl.write_text("")

    pd.DataFrame(
        [
            {
                "video_id": "cover",
                "status": "pose_detected",
                "output_jsonl_path": str(cover_pose_jsonl),
            },
            {
                "video_id": "pull",
                "status": "pose_detected",
                "output_jsonl_path": str(pull_pose_jsonl),
            },
            {
                "video_id": "failed",
                "status": "failed",
                "output_jsonl_path": "",
            },
        ]
    ).to_csv(audit_csv, index=False)

    called_modules: list[str] = []

    def fake_runner(command: list[str]) -> dict[str, str]:
        module_name = command[command.index("-m") + 1]
        called_modules.append(module_name)

        if module_name == "cricform.pose.landmark_schema":
            output_path = Path(command[command.index("--output-parquet") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("")
            return {"output_parquet_path": str(output_path)}

        if module_name == "cricform.phases.detect_phases":
            output_dir = Path(command[command.index("--output-dir") + 1])
            input_stem = Path(command[3]).stem
            return {
                "output_phase_csv_path": str(output_dir / f"{input_stem}.phase_timeline.csv"),
                "output_summary_json_path": str(output_dir / f"{input_stem}.phase_summary.json"),
            }

        if module_name == "cricform.features.motion_features":
            output_dir = Path(command[command.index("--output-dir") + 1])
            input_stem = Path(command[3]).stem
            return {
                "output_features_csv_path": str(output_dir / f"{input_stem}.movement_features.csv"),
                "output_summary_json_path": str(output_dir / f"{input_stem}.movement_summary.json"),
            }

        raise AssertionError(f"Unexpected module: {module_name}")

    count = write_real_sample_baseline_manifest(
        pose_audit_csv_path=audit_csv,
        baseline_manifest_path=manifest_path,
        baseline_feature_dir=baseline_feature_dir,
        command_runner=fake_runner,
        known_feature_paths={"pull": known_feature_csv},
    )

    manifest = pd.read_csv(manifest_path)
    feature_paths = set(manifest["movement_features_csv"])

    assert count == 2
    assert len(manifest) == 2
    assert set(manifest["shot_id"]) == {"cover", "pull"}
    assert set(manifest["shot_type"]) == {"real_sample_mixed"}
    assert len(feature_paths) == 2
    assert "../features/pull_selected.movement_features.csv" in feature_paths
    assert any(path.startswith("real_sample_features/cover.landmarks") for path in feature_paths)
    assert called_modules == [
        "cricform.pose.landmark_schema",
        "cricform.phases.detect_phases",
        "cricform.features.motion_features",
    ]


def test_build_real_demo_artifacts_accepts_quality_output_summary_path(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample" / "test" / "pull" / "pull_0025.avi"
    pose_jsonl_path = tmp_path / "pose" / "test_pull_pull_0025.pose.jsonl"
    audit_csv = tmp_path / "audit" / "pose_audit.csv"
    audit_summary = tmp_path / "audit" / "pose_audit_summary.json"
    output_root = tmp_path / "real_demo"
    overlay_dir = tmp_path / "overlays"

    video_path.parent.mkdir(parents=True)
    pose_jsonl_path.parent.mkdir(parents=True)
    audit_csv.parent.mkdir(parents=True)

    video_path.write_bytes(b"fake-video")
    pose_jsonl_path.write_text("")

    best_video = {
        "video_id": "test_pull_pull_0025",
        "split": "test",
        "shot_type": "pull",
        "video_path": str(video_path),
        "output_jsonl_path": str(pose_jsonl_path),
        "frames_seen": 34,
        "frames_processed": 7,
        "frames_with_pose": 4,
        "pose_detection_rate": 0.5714,
        "status": "pose_detected",
        "error": "",
    }

    pd.DataFrame([best_video]).to_csv(audit_csv, index=False)
    audit_summary.write_text(json.dumps({"best_video": best_video}))

    def fake_runner(command: list[str]) -> dict[str, str]:
        module_name = command[command.index("-m") + 1]

        if module_name == "cricform.pose.landmark_schema":
            return {"output_parquet_path": str(output_root / "pose" / "demo.parquet")}

        if module_name == "cricform.features.quality_features":
            return {
                "output_summary_path": str(
                    output_root / "features" / "demo.pose_quality_summary.json"
                )
            }

        if module_name == "cricform.phases.detect_phases":
            return {
                "output_phase_csv_path": str(output_root / "features" / "demo.phase_timeline.csv"),
                "output_summary_json_path": str(
                    output_root / "features" / "demo.phase_summary.json"
                ),
            }

        if module_name == "cricform.features.motion_features":
            return {
                "output_features_csv_path": str(
                    output_root / "features" / "demo.movement_features.csv"
                ),
                "output_summary_json_path": str(
                    output_root / "features" / "demo.movement_summary.json"
                ),
            }

        if module_name == "cricform.video.overlay":
            return {"output_video_path": str(overlay_dir / "demo_overlay.mp4")}

        if module_name == "cricform.baseline.build_baseline":
            return {"output_profile_path": str(output_root / "baselines" / "demo_baseline.json")}

        if module_name == "cricform.baseline.compare_shot":
            return {"output_comparison_path": str(output_root / "reports" / "demo_comparison.json")}

        if module_name == "cricform.reports.render_report":
            return {
                "output_markdown_path": str(output_root / "reports" / "demo_report.md"),
                "output_chart_path": str(output_root / "reports" / "demo_chart.png"),
            }

        raise AssertionError(f"Unexpected module: {module_name}")

    summary = build_real_demo_artifacts(
        pose_audit_csv_path=audit_csv,
        pose_audit_summary_path=audit_summary,
        output_root=output_root,
        overlay_output_dir=overlay_dir,
        command_runner=fake_runner,
    )

    assert summary.selected_video_id == "test_pull_pull_0025"
    assert summary.output_pose_quality_summary_path.endswith("demo.pose_quality_summary.json")
    assert (output_root / "real_demo_summary.json").exists()
