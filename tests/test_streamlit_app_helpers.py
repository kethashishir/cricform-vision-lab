from __future__ import annotations

import json
from pathlib import Path

import pytest

from cricform.app.streamlit_app import (
    artifact_paths_for_mode,
    artifact_status,
    baseline_manifest_evidence,
    clean_report_markdown_for_streamlit,
    comparison_badge_status,
    key_metric_cards,
    load_csv,
    load_json,
)


def test_load_json_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_json(tmp_path / "missing.json") is None


def test_load_json_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"ok": True}))

    assert load_json(path) == {"ok": True}


def test_load_csv_returns_empty_dataframe_for_missing_file(tmp_path: Path) -> None:
    dataframe = load_csv(tmp_path / "missing.csv")

    assert dataframe.empty


def test_artifact_status(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("hello")

    status = artifact_status(
        {
            "existing": existing,
            "missing": tmp_path / "missing.txt",
        }
    )

    assert len(status) == 2
    assert status.loc[status["artifact"] == "existing", "exists"].iloc[0] == True  # noqa: E712
    assert status.loc[status["artifact"] == "missing", "exists"].iloc[0] == False  # noqa: E712


def test_comparison_badge_status() -> None:
    assert comparison_badge_status(None) == "missing"
    assert (
        comparison_badge_status({"comparison_status": "insufficient_shot_pose_data"})
        == "insufficient_shot_pose_data"
    )


def test_baseline_manifest_evidence_counts_unique_feature_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "shot_id,shot_type,movement_features_csv\n"
        "a,real_sample_mixed,features/a.csv\n"
        "b,real_sample_mixed,features/b.csv\n"
        "c,real_sample_mixed,features/b.csv\n"
    )

    evidence = baseline_manifest_evidence(manifest_path)

    assert evidence == {
        "status": "ok",
        "manifest_rows": 3,
        "unique_movement_feature_paths": 2,
    }


def test_baseline_manifest_evidence_handles_missing_file(tmp_path: Path) -> None:
    evidence = baseline_manifest_evidence(tmp_path / "missing.csv")

    assert evidence == {
        "status": "missing_or_empty",
        "manifest_rows": 0,
        "unique_movement_feature_paths": 0,
    }


def test_key_metric_cards() -> None:
    cards = key_metric_cards(
        comparison={"comparison_status": "comparison_available", "usable_motion_frames": 12},
        quality_summary={"pose_detection_rate": 0.75, "mean_frame_quality_score": 0.6},
        movement_summary={"usable_frames": 10},
    )

    assert cards["comparison_status"] == "comparison_available"
    assert cards["usable_motion_frames"] == "12"
    assert cards["pose_detection_rate"] == "0.750"
    assert cards["mean_frame_quality"] == "0.600"


def test_artifact_paths_for_mode_real() -> None:
    paths = artifact_paths_for_mode("real")

    assert (
        paths["overlay_video"]
        .as_posix()
        .endswith("outputs/real_demo/test_pull_pull_0025_pose_overlay_slow.mp4")
    )
    assert (
        paths["original_overlay_video"]
        .as_posix()
        .endswith("outputs/real_demo/test_pull_pull_0025_pose_overlay.mp4")
    )
    assert (
        paths["real_demo_summary"]
        .as_posix()
        .endswith("data/processed/real_demo/real_demo_summary.json")
    )
    assert (
        paths["baseline_manifest"]
        .as_posix()
        .endswith("data/processed/real_demo/baselines/real_demo_baseline_manifest.csv")
    )


def test_artifact_paths_for_mode_synthetic() -> None:
    paths = artifact_paths_for_mode("synthetic")

    assert (
        paths["overlay_video"]
        .as_posix()
        .endswith("outputs/sample_overlays/synthetic_batting_sample_pose_overlay.mp4")
    )


def test_artifact_paths_for_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported artifact mode"):
        artifact_paths_for_mode("unknown")


def test_clean_report_markdown_for_streamlit_removes_local_image() -> None:
    markdown = """# Report

## Metric comparison chart

![Shot vs baseline](data/processed/real_demo/reports/chart.png)

## Notes

- Good.
"""

    cleaned = clean_report_markdown_for_streamlit(markdown)

    assert "![Shot vs baseline]" not in cleaned
    assert "## Metric comparison chart" in cleaned
    assert "## Notes" in cleaned
