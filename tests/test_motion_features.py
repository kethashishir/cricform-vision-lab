from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cricform.features.motion_features import (
    MOVEMENT_FEATURE_COLUMNS,
    build_movement_feature_table,
    summarize_movement_features,
    write_movement_feature_outputs,
)


def _synthetic_landmark_dataframe() -> pd.DataFrame:
    rows = []
    hand_y_values = [0.58, 0.53, 0.46, 0.36, 0.30, 0.40, 0.55, 0.44, 0.34, 0.29]

    landmark_index_by_name = {
        "nose": 0,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    for frame_index, hand_y in enumerate(hand_y_values):
        timestamp_ms = frame_index * 42

        landmark_specs = {
            "nose": (0.50 + frame_index * 0.001, 0.20),
            "left_wrist": (0.42, hand_y),
            "right_wrist": (0.46, hand_y + 0.01),
            "left_shoulder": (0.40, 0.45),
            "right_shoulder": (0.52, 0.46),
            "left_hip": (0.42, 0.70),
            "right_hip": (0.50, 0.71),
            "left_knee": (0.43, 0.82),
            "right_knee": (0.49, 0.82),
            "left_ankle": (0.44, 0.95),
            "right_ankle": (0.48, 0.95),
        }

        for landmark_name, (x_value, y_value) in landmark_specs.items():
            rows.append(
                {
                    "video_id": "sample",
                    "source_video_path": "data/raw/videos/sample.mp4",
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "pose_index": 0,
                    "coordinate_space": "image",
                    "landmark_index": landmark_index_by_name[landmark_name],
                    "landmark_name": landmark_name,
                    "x": x_value,
                    "y": y_value,
                    "z": 0.0,
                    "visibility": 0.9,
                    "presence": 0.9,
                }
            )

    return pd.DataFrame(rows)


def _synthetic_phase_dataframe() -> pd.DataFrame:
    phases = [
        "stance",
        "backlift",
        "backlift",
        "downswing",
        "contact_zone",
        "follow_through",
        "follow_through",
        "recovery",
        "recovery",
        "recovery",
    ]

    return pd.DataFrame(
        {
            "frame_index": list(range(len(phases))),
            "phase": phases,
        }
    )


def test_build_movement_feature_table_has_expected_columns() -> None:
    features = build_movement_feature_table(
        landmark_dataframe=_synthetic_landmark_dataframe(),
        phase_timeline_dataframe=_synthetic_phase_dataframe(),
    )

    assert list(features.columns) == list(MOVEMENT_FEATURE_COLUMNS)
    assert len(features) == 10
    assert features["usable_for_motion"].all()


def test_build_movement_feature_table_computes_key_proxies() -> None:
    features = build_movement_feature_table(
        landmark_dataframe=_synthetic_landmark_dataframe(),
        phase_timeline_dataframe=_synthetic_phase_dataframe(),
    )

    assert features["backlift_height_proxy"].max() > 0
    assert features["head_displacement_proxy"].max() > 0
    assert features["front_knee_bend_proxy_deg"].notna().all()
    assert features["shoulder_hip_separation_proxy_deg"].notna().all()
    assert features["wrist_speed_proxy"].max() > 0
    assert features["wrist_path_smoothness_proxy"].between(0, 1).all()


def test_build_movement_feature_table_handles_empty_landmarks() -> None:
    features = build_movement_feature_table(pd.DataFrame())

    assert list(features.columns) == list(MOVEMENT_FEATURE_COLUMNS)
    assert features.empty


def test_summarize_movement_features() -> None:
    features = build_movement_feature_table(
        landmark_dataframe=_synthetic_landmark_dataframe(),
        phase_timeline_dataframe=_synthetic_phase_dataframe(),
    )

    summary = summarize_movement_features(features)

    assert summary["rows"] == 10
    assert summary["usable_frames"] == 10
    assert summary["max_backlift_height_proxy"] is not None
    assert summary["follow_through_height_proxy"] is not None
    assert summary["mean_head_displacement_proxy"] is not None
    assert summary["min_front_knee_bend_proxy_deg"] is not None


def test_write_movement_feature_outputs(tmp_path: Path) -> None:
    landmark_parquet_path = tmp_path / "sample.landmarks.parquet"
    phase_csv_path = tmp_path / "sample.phase_timeline.csv"
    output_dir = tmp_path / "features"

    _synthetic_landmark_dataframe().to_parquet(landmark_parquet_path, index=False)
    _synthetic_phase_dataframe().to_csv(phase_csv_path, index=False)

    summary = write_movement_feature_outputs(
        landmark_parquet_path=landmark_parquet_path,
        phase_timeline_csv_path=phase_csv_path,
        output_dir=output_dir,
    )

    assert Path(summary.output_features_csv_path).exists()
    assert Path(summary.output_summary_json_path).exists()
    assert summary.rows == 10
    assert summary.usable_frames == 10
    assert summary.max_backlift_height_proxy is not None

    summary_json = json.loads(Path(summary.output_summary_json_path).read_text())
    assert "not validated biomechanics" in summary_json["limitation"]
