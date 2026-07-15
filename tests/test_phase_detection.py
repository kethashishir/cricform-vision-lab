from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cricform.phases.detect_phases import (
    PHASE_ORDER,
    PHASE_TIMELINE_COLUMNS,
    build_frame_motion_table,
    detect_phase_timeline,
    write_phase_detection_outputs,
)


def _synthetic_landmark_dataframe() -> pd.DataFrame:
    rows = []

    # Hand y pattern:
    # early stable -> hands rise for backlift -> hands move down toward contact
    # -> hands rise again for follow-through -> stabilize in recovery.
    hand_y_values = [
        0.58,
        0.57,
        0.56,
        0.52,
        0.47,
        0.40,
        0.34,
        0.30,
        0.33,
        0.40,
        0.49,
        0.59,
        0.54,
        0.46,
        0.38,
        0.31,
        0.28,
        0.30,
        0.34,
        0.38,
        0.42,
        0.44,
        0.45,
        0.45,
    ]

    for frame_index, hand_y in enumerate(hand_y_values):
        timestamp_ms = frame_index * 42

        landmark_specs = {
            "left_wrist": (0.42, hand_y),
            "right_wrist": (0.46, hand_y + 0.01),
            "left_shoulder": (0.40, 0.45),
            "right_shoulder": (0.50, 0.45),
        }

        for landmark_index, (landmark_name, (x_value, y_value)) in enumerate(
            landmark_specs.items()
        ):
            rows.append(
                {
                    "video_id": "sample",
                    "source_video_path": "data/raw/videos/sample.mp4",
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "pose_index": 0,
                    "coordinate_space": "image",
                    "landmark_index": landmark_index,
                    "landmark_name": landmark_name,
                    "x": x_value,
                    "y": y_value,
                    "z": 0.0,
                    "visibility": 0.9,
                    "presence": 0.9,
                }
            )

    return pd.DataFrame(rows)


def test_build_frame_motion_table_from_landmarks() -> None:
    landmarks = _synthetic_landmark_dataframe()

    frame_motion = build_frame_motion_table(landmarks)

    assert len(frame_motion) == 24
    assert frame_motion["usable_for_phase"].all()
    assert "hand_y" in frame_motion.columns
    assert "hand_height_proxy" in frame_motion.columns
    assert "wrist_speed_y" in frame_motion.columns


def test_detect_phase_timeline_has_expected_columns() -> None:
    landmarks = _synthetic_landmark_dataframe()
    frame_motion = build_frame_motion_table(landmarks)

    timeline = detect_phase_timeline(frame_motion)

    assert list(timeline.columns) == list(PHASE_TIMELINE_COLUMNS)
    assert len(timeline) == 24


def test_detect_phase_timeline_includes_core_phase_sequence() -> None:
    landmarks = _synthetic_landmark_dataframe()
    frame_motion = build_frame_motion_table(landmarks)

    timeline = detect_phase_timeline(frame_motion)

    phases = timeline["phase"].tolist()

    for phase in PHASE_ORDER:
        assert phase in phases

    first_seen = {phase: phases.index(phase) for phase in PHASE_ORDER}

    assert first_seen["stance"] < first_seen["backlift"]
    assert first_seen["backlift"] < first_seen["downswing"]
    assert first_seen["downswing"] < first_seen["contact_zone"]
    assert first_seen["contact_zone"] < first_seen["follow_through"]
    assert first_seen["follow_through"] < first_seen["recovery"]


def test_detect_phase_timeline_marks_one_contact_proxy() -> None:
    landmarks = _synthetic_landmark_dataframe()
    frame_motion = build_frame_motion_table(landmarks)

    timeline = detect_phase_timeline(frame_motion)

    assert int(timeline["is_contact_proxy"].sum()) == 1
    contact_row = timeline[timeline["is_contact_proxy"]].iloc[0]
    assert contact_row["phase"] == "contact_zone"


def test_detect_phase_timeline_handles_empty_input() -> None:
    timeline = detect_phase_timeline(pd.DataFrame())

    assert list(timeline.columns) == list(PHASE_TIMELINE_COLUMNS)
    assert timeline.empty


def test_detect_phase_timeline_handles_too_few_usable_frames() -> None:
    frame_motion = pd.DataFrame(
        [
            {
                "video_id": "sample",
                "frame_index": 0,
                "timestamp_ms": 0,
                "hand_y": 0.5,
                "hand_height_proxy": 0.1,
                "wrist_speed_y": 0.0,
                "usable_for_phase": True,
            }
        ]
    )

    timeline = detect_phase_timeline(frame_motion)

    assert len(timeline) == 1
    assert timeline.iloc[0]["phase"] == "unavailable"


def test_write_phase_detection_outputs(tmp_path: Path) -> None:
    landmark_parquet_path = tmp_path / "sample.landmarks.parquet"
    output_dir = tmp_path / "features"

    landmarks = _synthetic_landmark_dataframe()
    landmarks.to_parquet(landmark_parquet_path, index=False)

    summary = write_phase_detection_outputs(
        landmark_parquet_path=landmark_parquet_path,
        output_dir=output_dir,
    )

    assert Path(summary.output_phase_csv_path).exists()
    assert Path(summary.output_summary_json_path).exists()
    assert summary.rows == 24
    assert summary.usable_frames == 24
    assert summary.method == "rule_based_wrist_motion_proxy_v1"
    assert "contact_zone" in summary.detected_phases

    summary_json = json.loads(Path(summary.output_summary_json_path).read_text())
    assert summary_json["rows"] == 24
    assert "Not validated" in summary_json["limitation"]
