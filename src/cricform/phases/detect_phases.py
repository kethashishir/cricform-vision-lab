from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data/processed/features")

PHASE_ORDER: tuple[str, ...] = (
    "stance",
    "backlift",
    "downswing",
    "contact_zone",
    "follow_through",
    "recovery",
)

PHASE_TIMELINE_COLUMNS: tuple[str, ...] = (
    "video_id",
    "frame_index",
    "timestamp_ms",
    "phase",
    "phase_confidence",
    "phase_reason",
    "hand_y",
    "hand_height_proxy",
    "wrist_speed_y",
    "is_contact_proxy",
)

REQUIRED_LANDMARKS: tuple[str, ...] = (
    "left_wrist",
    "right_wrist",
    "left_shoulder",
    "right_shoulder",
)


@dataclass(frozen=True)
class PhaseDetectionSummary:
    input_landmark_parquet_path: str
    output_phase_csv_path: str
    output_summary_json_path: str
    rows: int
    usable_frames: int
    detected_phases: list[str]
    phase_counts: dict[str, int]
    method: str
    limitation: str


def build_frame_motion_table(landmark_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build frame-level hand motion proxies from tidy landmark rows."""

    if landmark_dataframe.empty:
        return _empty_frame_motion_table()

    image_df = landmark_dataframe[
        (landmark_dataframe["coordinate_space"] == "image")
        & (landmark_dataframe["pose_index"] == 0)
    ].copy()

    if image_df.empty:
        return _empty_frame_motion_table()

    pivot = image_df.pivot_table(
        index=["video_id", "frame_index", "timestamp_ms"],
        columns="landmark_name",
        values=["x", "y"],
        aggfunc="first",
    )

    pivot.columns = [f"{axis}_{landmark_name}" for axis, landmark_name in pivot.columns]
    pivot = pivot.reset_index().sort_values("frame_index").reset_index(drop=True)

    for required_landmark in REQUIRED_LANDMARKS:
        y_column = f"y_{required_landmark}"
        if y_column not in pivot.columns:
            pivot[y_column] = pd.NA

    pivot["hand_y"] = pivot[["y_left_wrist", "y_right_wrist"]].mean(axis=1, skipna=True)
    pivot["shoulder_y"] = pivot[["y_left_shoulder", "y_right_shoulder"]].mean(
        axis=1,
        skipna=True,
    )
    pivot["hand_height_proxy"] = pivot["shoulder_y"] - pivot["hand_y"]

    timestamp_delta_sec = pivot["timestamp_ms"].diff() / 1000.0
    frame_delta = pivot["frame_index"].diff()

    fallback_delta_sec = frame_delta.where(frame_delta > 0, 1).fillna(1) / 24.0
    timestamp_delta_sec = timestamp_delta_sec.where(timestamp_delta_sec > 0, fallback_delta_sec)

    pivot["wrist_speed_y"] = pivot["hand_y"].diff() / timestamp_delta_sec
    pivot["wrist_speed_y"] = pivot["wrist_speed_y"].fillna(0.0)

    pivot["usable_for_phase"] = pivot[
        [
            "y_left_wrist",
            "y_right_wrist",
            "y_left_shoulder",
            "y_right_shoulder",
            "hand_y",
            "hand_height_proxy",
        ]
    ].notna().all(axis=1)

    return pivot[
        [
            "video_id",
            "frame_index",
            "timestamp_ms",
            "hand_y",
            "hand_height_proxy",
            "wrist_speed_y",
            "usable_for_phase",
        ]
    ]


def detect_phase_timeline(frame_motion_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Assign batting phases using transparent wrist-motion heuristics."""

    if frame_motion_dataframe.empty:
        return pd.DataFrame(columns=list(PHASE_TIMELINE_COLUMNS))

    timeline = frame_motion_dataframe.copy().sort_values("frame_index").reset_index(drop=True)
    timeline["phase"] = "unavailable"
    timeline["phase_confidence"] = 0.0
    timeline["phase_reason"] = "missing required wrist/shoulder landmarks"
    timeline["is_contact_proxy"] = False

    usable_mask = timeline["usable_for_phase"].astype(bool)

    if usable_mask.sum() < 6:
        return _timeline_output_columns(timeline)

    usable = timeline[usable_mask].copy()
    usable_positions = usable.index.to_list()
    usable_count = len(usable_positions)

    backlift_position = _estimate_backlift_position(usable)
    contact_position = _estimate_contact_position(usable, backlift_position)

    contact_radius = max(1, round(usable_count * 0.04))
    recovery_start_rank = max(contact_position + contact_radius + 1, round(usable_count * 0.85))
    stance_end_rank = max(1, min(round(usable_count * 0.18), backlift_position - 1))

    for rank, dataframe_index in enumerate(usable_positions):
        if rank < stance_end_rank:
            phase = "stance"
            confidence = 0.55
            reason = "early low-motion setup window"
        elif rank <= backlift_position:
            phase = "backlift"
            confidence = 0.70
            reason = "hand height proxy rising toward backlift apex"
        elif abs(rank - contact_position) <= contact_radius:
            phase = "contact_zone"
            confidence = 0.65
            reason = "contact proxy near strongest downward wrist motion"
        elif rank < contact_position:
            phase = "downswing"
            confidence = 0.70
            reason = "after backlift apex and before contact proxy"
        elif rank >= recovery_start_rank:
            phase = "recovery"
            confidence = 0.55
            reason = "late post-shot stabilization window"
        else:
            phase = "follow_through"
            confidence = 0.65
            reason = "after contact proxy before recovery window"

        timeline.loc[dataframe_index, "phase"] = phase
        timeline.loc[dataframe_index, "phase_confidence"] = confidence
        timeline.loc[dataframe_index, "phase_reason"] = reason

    contact_dataframe_index = usable_positions[contact_position]
    timeline.loc[contact_dataframe_index, "is_contact_proxy"] = True

    return _timeline_output_columns(timeline)


def write_phase_detection_outputs(
    landmark_parquet_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> PhaseDetectionSummary:
    """Read landmark Parquet, detect phases, and write CSV/JSON outputs."""

    if not landmark_parquet_path.exists():
        raise FileNotFoundError(f"Landmark Parquet file does not exist: {landmark_parquet_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = _video_stem_from_landmark_parquet_path(landmark_parquet_path)
    output_phase_csv_path = output_dir / f"{video_stem}.phase_timeline.csv"
    output_summary_json_path = output_dir / f"{video_stem}.phase_summary.json"

    landmark_dataframe = pd.read_parquet(landmark_parquet_path)
    frame_motion_dataframe = build_frame_motion_table(landmark_dataframe)
    phase_timeline = detect_phase_timeline(frame_motion_dataframe)

    phase_timeline.to_csv(output_phase_csv_path, index=False)

    phase_counts = {
        str(phase): int(count)
        for phase, count in phase_timeline["phase"].value_counts().sort_index().items()
    } if not phase_timeline.empty else {}

    detected_phases = [
        phase for phase in PHASE_ORDER
        if phase in set(phase_timeline["phase"].tolist())
    ] if not phase_timeline.empty else []

    usable_frames = int((phase_timeline["phase"] != "unavailable").sum()) if not phase_timeline.empty else 0

    summary_values = {
        "rows": int(len(phase_timeline)),
        "usable_frames": usable_frames,
        "detected_phases": detected_phases,
        "phase_counts": phase_counts,
        "method": "rule_based_wrist_motion_proxy_v1",
        "limitation": (
            "Heuristic phase labels based on pose landmarks and wrist-motion proxies. "
            "Not validated against expert cricket batting annotations."
        ),
    }

    output_summary_json_path.write_text(json.dumps(summary_values, indent=2) + "\n")

    return PhaseDetectionSummary(
        input_landmark_parquet_path=str(landmark_parquet_path),
        output_phase_csv_path=str(output_phase_csv_path),
        output_summary_json_path=str(output_summary_json_path),
        rows=int(summary_values["rows"]),
        usable_frames=int(summary_values["usable_frames"]),
        detected_phases=list(summary_values["detected_phases"]),
        phase_counts=dict(summary_values["phase_counts"]),
        method=str(summary_values["method"]),
        limitation=str(summary_values["limitation"]),
    )


def _estimate_backlift_position(usable: pd.DataFrame) -> int:
    search_end = max(2, round(len(usable) * 0.60))
    search_window = usable.iloc[:search_end]

    # MediaPipe image y increases downward. Higher hands relative to shoulders
    # means a larger shoulder_y - hand_y value.
    return int(search_window["hand_height_proxy"].astype(float).idxmax())


def _estimate_contact_position(usable: pd.DataFrame, backlift_position: int) -> int:
    minimum_contact_rank = min(backlift_position + 1, len(usable) - 1)
    search_start = minimum_contact_rank
    search_end = max(search_start + 1, round(len(usable) * 0.85))

    search_window = usable.iloc[search_start:search_end]

    if search_window.empty:
        return min(round(len(usable) * 0.55), len(usable) - 1)

    # Positive y velocity means hands are moving downward in image space.
    return int(search_window["wrist_speed_y"].astype(float).idxmax())


def _timeline_output_columns(timeline: pd.DataFrame) -> pd.DataFrame:
    for column in PHASE_TIMELINE_COLUMNS:
        if column not in timeline.columns:
            timeline[column] = pd.NA

    return timeline[list(PHASE_TIMELINE_COLUMNS)]


def _empty_frame_motion_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "video_id",
            "frame_index",
            "timestamp_ms",
            "hand_y",
            "hand_height_proxy",
            "wrist_speed_y",
            "usable_for_phase",
        ]
    )


def _video_stem_from_landmark_parquet_path(landmark_parquet_path: Path) -> str:
    name = landmark_parquet_path.name

    if name.endswith(".landmarks.parquet"):
        return name.removesuffix(".landmarks.parquet")

    return landmark_parquet_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect batting phases from landmark Parquet.")
    parser.add_argument("landmark_parquet_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = write_phase_detection_outputs(
        landmark_parquet_path=args.landmark_parquet_path,
        output_dir=args.output_dir,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
