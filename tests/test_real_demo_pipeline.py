from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cricform.pipeline.real_demo import (
    _parse_json_from_stdout,
    select_best_pose_audit_row,
    write_mixed_real_sample_manifest,
)


def test_parse_json_from_stdout_with_logs() -> None:
    stdout = "INFO something happened\n{\"ok\": true, \"value\": 3}\n"

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


def test_write_mixed_real_sample_manifest(tmp_path: Path) -> None:
    audit_csv = tmp_path / "pose_audit.csv"
    manifest_path = tmp_path / "baselines" / "manifest.csv"
    feature_csv = tmp_path / "features" / "movement.csv"

    pd.DataFrame(
        [
            {"video_id": "cover", "status": "pose_detected"},
            {"video_id": "pull", "status": "pose_detected"},
            {"video_id": "failed", "status": "failed"},
        ]
    ).to_csv(audit_csv, index=False)

    feature_csv.parent.mkdir(parents=True)
    feature_csv.write_text("frame_index,value\n0,1\n")

    count = write_mixed_real_sample_manifest(
        pose_audit_csv_path=audit_csv,
        baseline_manifest_path=manifest_path,
        feature_csv_path=feature_csv,
    )

    manifest = pd.read_csv(manifest_path)

    assert count == 2
    assert len(manifest) == 2
    assert set(manifest["shot_id"]) == {"cover", "pull"}
    assert set(manifest["shot_type"]) == {"real_sample_mixed"}
    assert set(manifest["movement_features_csv"]) == {"../features/movement.csv"}
