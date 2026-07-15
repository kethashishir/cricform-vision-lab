from __future__ import annotations

import json
from pathlib import Path

import pytest

from cricform.baseline.compare_shot import (
    compare_metric_to_baseline,
    compare_shot_to_baseline,
    write_shot_comparison,
)


def _movement_summary() -> dict:
    return {
        "rows": 10,
        "usable_frames": 10,
        "max_backlift_height_proxy": 0.30,
        "follow_through_height_proxy": 0.25,
        "mean_head_displacement_proxy": 0.05,
        "min_front_knee_bend_proxy_deg": 155.0,
        "mean_shoulder_hip_separation_proxy_deg": 12.0,
        "mean_wrist_speed_proxy": 2.2,
        "mean_wrist_path_smoothness_proxy": 0.7,
    }


def _baseline_profile() -> dict:
    return {
        "baseline_version": "movement_proxy_baseline_v1",
        "shot_types": {
            "cover": {
                "manifest_shots": 2,
                "usable_shots": 2,
                "features": {
                    "max_backlift_height_proxy": {
                        "count": 2,
                        "mean": 0.25,
                        "std": 0.05,
                        "min": 0.20,
                        "max": 0.30,
                    },
                    "follow_through_height_proxy": {
                        "count": 2,
                        "mean": 0.20,
                        "std": 0.05,
                        "min": 0.15,
                        "max": 0.25,
                    },
                },
                "shot_ids": ["shot_1", "shot_2"],
            }
        },
    }


def test_compare_metric_to_baseline_with_z_score() -> None:
    result = compare_metric_to_baseline(
        metric_name="max_backlift_height_proxy",
        shot_value=0.30,
        baseline_stats={"count": 2, "mean": 0.25, "std": 0.05},
    )

    assert result["delta_from_mean"] == pytest.approx(0.05)
    assert result["z_score"] == pytest.approx(1.0)
    assert result["status"] == "within_baseline"


def test_compare_metric_handles_missing_shot_value() -> None:
    result = compare_metric_to_baseline(
        metric_name="max_backlift_height_proxy",
        shot_value=None,
        baseline_stats={"count": 2, "mean": 0.25, "std": 0.05},
    )

    assert result["status"] == "shot_metric_unavailable"


def test_compare_metric_handles_missing_baseline() -> None:
    result = compare_metric_to_baseline(
        metric_name="max_backlift_height_proxy",
        shot_value=0.30,
        baseline_stats=None,
    )

    assert result["status"] == "baseline_metric_unavailable"


def test_compare_shot_to_baseline_available() -> None:
    comparison = compare_shot_to_baseline(
        movement_summary=_movement_summary(),
        baseline_profile=_baseline_profile(),
        shot_type="cover",
    )

    assert comparison["comparison_status"] == "comparison_available"
    assert comparison["shot_type"] == "cover"
    assert comparison["usable_motion_frames"] == 10
    assert len(comparison["metric_comparisons"]) > 0


def test_compare_shot_to_baseline_missing_shot_type() -> None:
    comparison = compare_shot_to_baseline(
        movement_summary=_movement_summary(),
        baseline_profile=_baseline_profile(),
        shot_type="pull",
    )

    assert comparison["comparison_status"] == "missing_shot_type_baseline"


def test_compare_shot_to_baseline_insufficient_pose_data() -> None:
    movement_summary = _movement_summary()
    movement_summary["usable_frames"] = 0

    comparison = compare_shot_to_baseline(
        movement_summary=movement_summary,
        baseline_profile=_baseline_profile(),
        shot_type="cover",
    )

    assert comparison["comparison_status"] == "insufficient_shot_pose_data"


def test_write_shot_comparison(tmp_path: Path) -> None:
    movement_summary_path = tmp_path / "movement_summary.json"
    baseline_profile_path = tmp_path / "baseline_profile.json"
    output_path = tmp_path / "comparison.json"

    movement_summary_path.write_text(json.dumps(_movement_summary()))
    baseline_profile_path.write_text(json.dumps(_baseline_profile()))

    summary = write_shot_comparison(
        movement_summary_path=movement_summary_path,
        baseline_profile_path=baseline_profile_path,
        shot_type="cover",
        output_comparison_path=output_path,
    )

    assert output_path.exists()
    assert summary.comparison_status == "comparison_available"
    assert summary.comparable_metrics >= 1

    comparison = json.loads(output_path.read_text())
    assert comparison["source_paths"]["movement_summary_path"] == str(movement_summary_path)
