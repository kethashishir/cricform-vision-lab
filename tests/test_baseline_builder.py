from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cricform.baseline.build_baseline import (
    BASELINE_METRIC_SPECS,
    build_baseline_profile,
    load_baseline_manifest,
    summarize_shot_features,
    write_baseline_profile,
)


def _movement_features_dataframe(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": ["shot"] * 4,
            "frame_index": [0, 1, 2, 3],
            "timestamp_ms": [0, 42, 84, 126],
            "phase": ["stance", "backlift", "downswing", "follow_through"],
            "usable_for_motion": [True, True, True, True],
            "head_x": [0.5, 0.51, 0.52, 0.53],
            "head_y": [0.2, 0.2, 0.21, 0.21],
            "head_displacement_proxy": [0.0, 0.01, 0.02, 0.03 + offset],
            "left_knee_angle_deg": [170.0, 165.0, 160.0, 158.0],
            "right_knee_angle_deg": [175.0, 172.0, 171.0, 170.0],
            "front_knee_bend_proxy_deg": [170.0, 165.0, 160.0, 158.0 - offset],
            "shoulder_hip_separation_proxy_deg": [5.0, 8.0, 12.0, 15.0 + offset],
            "wrist_center_x": [0.4, 0.41, 0.42, 0.43],
            "wrist_center_y": [0.6, 0.5, 0.4, 0.3],
            "wrist_speed_proxy": [0.0, 2.0, 2.2, 2.4 + offset],
            "wrist_acceleration_proxy": [0.0, 1.0, 0.5, 0.3],
            "wrist_path_smoothness_proxy": [1.0, 0.5, 0.67, 0.77],
            "backlift_height_proxy": [-0.1, 0.0, 0.1, 0.2 + offset],
            "follow_through_height_proxy": [None, None, None, 0.2 + offset],
        }
    )


def test_load_baseline_manifest_validates_required_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("shot_id,shot_type\nshot_1,cover\n")

    with pytest.raises(ValueError):
        load_baseline_manifest(manifest_path)


def test_summarize_shot_features(tmp_path: Path) -> None:
    feature_path = tmp_path / "shot_1.csv"
    _movement_features_dataframe().to_csv(feature_path, index=False)

    summary = summarize_shot_features(
        shot_id="shot_1",
        shot_type="cover",
        movement_features_path=feature_path,
    )

    assert summary["shot_id"] == "shot_1"
    assert summary["shot_type"] == "cover"
    assert summary["rows"] == 4
    assert summary["usable_frames"] == 4
    assert summary["is_usable"] is True

    metric_names = {metric_name for metric_name, _, _ in BASELINE_METRIC_SPECS}
    assert set(summary["metrics"]) == metric_names
    assert summary["metrics"]["max_backlift_height_proxy"] == pytest.approx(0.2)
    assert summary["metrics"]["min_front_knee_bend_proxy_deg"] == pytest.approx(158.0)


def test_summarize_shot_features_handles_empty_csv(tmp_path: Path) -> None:
    feature_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["usable_for_motion"]).to_csv(feature_path, index=False)

    summary = summarize_shot_features(
        shot_id="empty",
        shot_type="unknown",
        movement_features_path=feature_path,
    )

    assert summary["rows"] == 0
    assert summary["usable_frames"] == 0
    assert summary["is_usable"] is False
    assert summary["metrics"] == {}


def test_build_baseline_profile_groups_by_shot_type(tmp_path: Path) -> None:
    shot_1_path = tmp_path / "shot_1.csv"
    shot_2_path = tmp_path / "shot_2.csv"
    shot_3_path = tmp_path / "shot_3.csv"

    _movement_features_dataframe(offset=0.0).to_csv(shot_1_path, index=False)
    _movement_features_dataframe(offset=0.1).to_csv(shot_2_path, index=False)
    _movement_features_dataframe(offset=0.2).to_csv(shot_3_path, index=False)

    manifest = pd.DataFrame(
        {
            "shot_id": ["shot_1", "shot_2", "shot_3"],
            "shot_type": ["cover", "cover", "pull"],
            "movement_features_csv": [
                "shot_1.csv",
                "shot_2.csv",
                "shot_3.csv",
            ],
        }
    )

    profile = build_baseline_profile(manifest, manifest_base_dir=tmp_path)

    assert profile["baseline_version"] == "movement_proxy_baseline_v1"
    assert profile["total_manifest_rows"] == 3
    assert profile["usable_shots"] == 3
    assert set(profile["shot_types"]) == {"cover", "pull"}

    cover = profile["shot_types"]["cover"]
    assert cover["manifest_shots"] == 2
    assert cover["usable_shots"] == 2
    assert cover["features"]["max_backlift_height_proxy"]["count"] == 2


def test_write_baseline_profile(tmp_path: Path) -> None:
    feature_path = tmp_path / "shot_1.csv"
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "baseline.json"

    _movement_features_dataframe().to_csv(feature_path, index=False)

    pd.DataFrame(
        {
            "shot_id": ["shot_1"],
            "shot_type": ["cover"],
            "movement_features_csv": ["shot_1.csv"],
        }
    ).to_csv(manifest_path, index=False)

    summary = write_baseline_profile(
        manifest_path=manifest_path,
        output_profile_path=output_path,
    )

    assert output_path.exists()
    assert summary.total_manifest_rows == 1
    assert summary.usable_shots == 1
    assert summary.shot_types == ["cover"]

    profile = json.loads(output_path.read_text())
    assert profile["source_manifest_path"] == str(manifest_path)
    assert "not population norms" in profile["limitation"]
