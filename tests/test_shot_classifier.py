from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cricform.models.shot_classifier import (
    build_labelled_feature_table,
    evaluate_leave_one_out_nearest_neighbor,
    load_labelled_real_sample_manifest,
    write_shot_classifier_evaluation,
)


def _write_movement_csv(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "frame_index": 0,
                "usable_for_motion": True,
                "backlift_height_proxy": value,
                "follow_through_height_proxy": value,
                "head_displacement_proxy": value,
                "front_knee_bend_proxy_deg": value,
                "shoulder_hip_separation_proxy_deg": value,
                "wrist_speed_proxy": value,
                "wrist_path_smoothness_proxy": value,
            }
        ]
    ).to_csv(path, index=False)


def test_load_labelled_real_sample_manifest_uses_audit_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "baselines" / "manifest.csv"
    audit = tmp_path / "audit" / "pose_audit.csv"

    manifest.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "shot_id": "test_cover_cover_0001",
                "shot_type": "real_sample_mixed",
                "movement_features_csv": "features/cover.csv",
            },
            {
                "shot_id": "test_pull_pull_0001",
                "shot_type": "real_sample_mixed",
                "movement_features_csv": "features/pull.csv",
            },
        ]
    ).to_csv(manifest, index=False)

    pd.DataFrame(
        [
            {"video_id": "test_cover_cover_0001", "shot_type": "cover"},
            {"video_id": "test_pull_pull_0001", "shot_type": "pull"},
        ]
    ).to_csv(audit, index=False)

    labelled = load_labelled_real_sample_manifest(manifest, audit)

    assert labelled["shot_type"].tolist() == ["cover", "pull"]
    assert labelled["movement_features_csv"].tolist() == [
        "features/cover.csv",
        "features/pull.csv",
    ]


def test_evaluate_leave_one_out_nearest_neighbor_has_confusion_matrix(
    tmp_path: Path,
) -> None:
    feature_table = pd.DataFrame(
        [
            {
                "shot_id": "cover_a",
                "shot_type": "cover",
                "max_backlift_height_proxy": 0.0,
                "follow_through_height_proxy": 0.0,
                "mean_head_displacement_proxy": 0.0,
                "min_front_knee_bend_proxy_deg": 0.0,
                "mean_shoulder_hip_separation_proxy_deg": 0.0,
                "mean_wrist_speed_proxy": 0.0,
                "mean_wrist_path_smoothness_proxy": 0.0,
            },
            {
                "shot_id": "cover_b",
                "shot_type": "cover",
                "max_backlift_height_proxy": 0.1,
                "follow_through_height_proxy": 0.1,
                "mean_head_displacement_proxy": 0.1,
                "min_front_knee_bend_proxy_deg": 0.1,
                "mean_shoulder_hip_separation_proxy_deg": 0.1,
                "mean_wrist_speed_proxy": 0.1,
                "mean_wrist_path_smoothness_proxy": 0.1,
            },
            {
                "shot_id": "pull_a",
                "shot_type": "pull",
                "max_backlift_height_proxy": 10.0,
                "follow_through_height_proxy": 10.0,
                "mean_head_displacement_proxy": 10.0,
                "min_front_knee_bend_proxy_deg": 10.0,
                "mean_shoulder_hip_separation_proxy_deg": 10.0,
                "mean_wrist_speed_proxy": 10.0,
                "mean_wrist_path_smoothness_proxy": 10.0,
            },
            {
                "shot_id": "pull_b",
                "shot_type": "pull",
                "max_backlift_height_proxy": 10.1,
                "follow_through_height_proxy": 10.1,
                "mean_head_displacement_proxy": 10.1,
                "min_front_knee_bend_proxy_deg": 10.1,
                "mean_shoulder_hip_separation_proxy_deg": 10.1,
                "mean_wrist_speed_proxy": 10.1,
                "mean_wrist_path_smoothness_proxy": 10.1,
            },
        ]
    )

    predictions, confusion, metrics = evaluate_leave_one_out_nearest_neighbor(
        feature_table
    )

    assert len(predictions) == 4
    assert metrics["evaluation_status"] == "ok"
    assert metrics["accuracy"] == 1.0
    assert confusion.loc["cover", "cover"] == 2
    assert confusion.loc["pull", "pull"] == 2


def test_write_shot_classifier_evaluation_outputs_files(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baselines"
    manifest = baseline_dir / "manifest.csv"
    audit = tmp_path / "audit" / "pose_audit.csv"
    output_dir = tmp_path / "classification"

    cover_a = baseline_dir / "features" / "cover_a.csv"
    cover_b = baseline_dir / "features" / "cover_b.csv"
    pull_a = baseline_dir / "features" / "pull_a.csv"
    pull_b = baseline_dir / "features" / "pull_b.csv"

    _write_movement_csv(cover_a, 0.0)
    _write_movement_csv(cover_b, 0.1)
    _write_movement_csv(pull_a, 10.0)
    _write_movement_csv(pull_b, 10.1)

    pd.DataFrame(
        [
            {
                "shot_id": "cover_a",
                "shot_type": "real_sample_mixed",
                "movement_features_csv": "features/cover_a.csv",
            },
            {
                "shot_id": "cover_b",
                "shot_type": "real_sample_mixed",
                "movement_features_csv": "features/cover_b.csv",
            },
            {
                "shot_id": "pull_a",
                "shot_type": "real_sample_mixed",
                "movement_features_csv": "features/pull_a.csv",
            },
            {
                "shot_id": "pull_b",
                "shot_type": "real_sample_mixed",
                "movement_features_csv": "features/pull_b.csv",
            },
        ]
    ).to_csv(manifest, index=False)

    audit.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"video_id": "cover_a", "shot_type": "cover"},
            {"video_id": "cover_b", "shot_type": "cover"},
            {"video_id": "pull_a", "shot_type": "pull"},
            {"video_id": "pull_b", "shot_type": "pull"},
        ]
    ).to_csv(audit, index=False)

    summary = write_shot_classifier_evaluation(
        baseline_manifest_path=manifest,
        pose_audit_csv_path=audit,
        output_dir=output_dir,
    )

    assert summary.samples == 4
    assert summary.evaluation_status == "ok"
    assert summary.accuracy == 1.0
    assert Path(summary.output_feature_table_path).exists()
    assert Path(summary.output_predictions_path).exists()
    assert Path(summary.output_metrics_path).exists()
    assert Path(summary.output_confusion_matrix_csv_path).exists()
    assert Path(summary.output_confusion_matrix_png_path).exists()

    metrics = json.loads(Path(summary.output_metrics_path).read_text())
    assert metrics["accuracy"] == 1.0


def test_build_labelled_feature_table_skips_unusable_features(tmp_path: Path) -> None:
    movement_csv = tmp_path / "features" / "empty.csv"
    movement_csv.parent.mkdir(parents=True)
    pd.DataFrame(
        columns=[
            "frame_index",
            "usable_for_motion",
            "backlift_height_proxy",
            "follow_through_height_proxy",
            "head_displacement_proxy",
            "front_knee_bend_proxy_deg",
            "shoulder_hip_separation_proxy_deg",
            "wrist_speed_proxy",
            "wrist_path_smoothness_proxy",
        ]
    ).to_csv(movement_csv, index=False)

    manifest = pd.DataFrame(
        [
            {
                "shot_id": "empty",
                "shot_type": "cover",
                "movement_features_csv": "features/empty.csv",
            }
        ]
    )

    feature_table = build_labelled_feature_table(
        labelled_manifest=manifest,
        manifest_base_dir=tmp_path,
    )

    assert feature_table.empty
