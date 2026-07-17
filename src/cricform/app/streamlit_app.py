from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SYNTHETIC_ARTIFACT_PATHS = {
    "overlay_video": PROJECT_ROOT
    / "outputs"
    / "sample_overlays"
    / "synthetic_batting_sample_pose_overlay.mp4",
    "report_markdown": PROJECT_ROOT
    / "data"
    / "processed"
    / "reports"
    / "synthetic_batting_sample_report.md",
    "comparison_json": PROJECT_ROOT
    / "data"
    / "processed"
    / "reports"
    / "synthetic_batting_sample_comparison.json",
    "report_chart": PROJECT_ROOT
    / "outputs"
    / "sample_reports"
    / "synthetic_batting_sample_metric_comparison.png",
    "pose_quality_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "synthetic_batting_sample.pose_quality_summary.json",
    "pose_quality_csv": PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "synthetic_batting_sample.pose_quality.csv",
    "phase_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "synthetic_batting_sample.phase_summary.json",
    "phase_timeline_csv": PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "synthetic_batting_sample.phase_timeline.csv",
    "movement_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "synthetic_batting_sample.movement_summary.json",
    "movement_features_csv": PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "synthetic_batting_sample.movement_features.csv",
    "baseline_profile": PROJECT_ROOT
    / "data"
    / "processed"
    / "baselines"
    / "synthetic_baseline_profile.json",
}

REAL_DEMO_ARTIFACT_PATHS = {
    "overlay_video": PROJECT_ROOT
    / "outputs"
    / "real_demo"
    / "test_pull_pull_0025_pose_overlay_slow.mp4",
    "original_overlay_video": PROJECT_ROOT
    / "outputs"
    / "real_demo"
    / "test_pull_pull_0025_pose_overlay.mp4",
    "report_markdown": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "reports"
    / "test_pull_pull_0025_report.md",
    "comparison_json": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "reports"
    / "test_pull_pull_0025_comparison.json",
    "report_chart": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "reports"
    / "test_pull_pull_0025_metric_comparison.png",
    "real_demo_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "real_demo_summary.json",
    "pose_audit_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_sample_pose_audit"
    / "pose_audit_summary.json",
    "pose_audit_csv": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_sample_pose_audit"
    / "pose_audit.csv",
    "pose_quality_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "features"
    / "test_pull_pull_0025.pose_quality_summary.json",
    "pose_quality_csv": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "features"
    / "test_pull_pull_0025.pose_quality.csv",
    "phase_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "features"
    / "test_pull_pull_0025.phase_summary.json",
    "phase_timeline_csv": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "features"
    / "test_pull_pull_0025.phase_timeline.csv",
    "movement_summary": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "features"
    / "test_pull_pull_0025.movement_summary.json",
    "movement_features_csv": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "features"
    / "test_pull_pull_0025.movement_features.csv",
    "baseline_profile": PROJECT_ROOT
    / "data"
    / "processed"
    / "real_demo"
    / "baselines"
    / "real_demo_baseline_profile.json",
}


def artifact_paths_for_mode(mode: str) -> dict[str, Path]:
    """Return artifact paths for the selected demo mode."""

    if mode == "real":
        return dict(REAL_DEMO_ARTIFACT_PATHS)

    if mode == "synthetic":
        return dict(SYNTHETIC_ARTIFACT_PATHS)

    raise ValueError(f"Unsupported artifact mode: {mode}")


def artifact_status(paths: dict[str, Path]) -> pd.DataFrame:
    """Return existence and size information for app artifacts."""

    rows = []

    for name, path in paths.items():
        exists = path.exists()
        rows.append(
            {
                "artifact": name,
                "path": _display_path(path),
                "exists": exists,
                "size_kb": round(path.stat().st_size / 1024, 2) if exists else 0.0,
            }
        )

    return pd.DataFrame(rows)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""

    if not path.exists():
        return None

    return json.loads(path.read_text())


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file if it exists, otherwise return an empty DataFrame."""

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def comparison_badge_status(comparison: dict[str, Any] | None) -> str:
    """Return a compact human-readable comparison status."""

    if comparison is None:
        return "missing"

    return str(comparison.get("comparison_status", "unknown"))


def key_metric_cards(
    comparison: dict[str, Any] | None,
    quality_summary: dict[str, Any] | None,
    movement_summary: dict[str, Any] | None,
) -> dict[str, str]:
    """Build display-ready metric card values."""

    return {
        "comparison_status": comparison_badge_status(comparison),
        "usable_motion_frames": str(
            (comparison or {}).get(
                "usable_motion_frames",
                (movement_summary or {}).get("usable_frames", "N/A"),
            )
        ),
        "pose_detection_rate": _format_optional_number(
            (quality_summary or {}).get("pose_detection_rate")
        ),
        "mean_frame_quality": _format_optional_number(
            (quality_summary or {}).get("mean_frame_quality_score")
        ),
    }


def app() -> None:
    st.set_page_config(
        page_title="CricForm Vision Lab",
        page_icon="🏏",
        layout="wide",
    )

    st.title("🏏 CricForm Vision Lab")
    st.caption(
        "Explainable cricket batting computer-vision pipeline: pose extraction, "
        "quality checks, phase timeline, movement proxies, baseline comparison, "
        "and honest limitations."
    )

    mode = st.sidebar.radio(
        "Demo mode",
        options=["real", "synthetic"],
        format_func=lambda value: "Real cricket demo" if value == "real" else "Synthetic sample",
        index=0,
    )
    paths = _sidebar_paths(mode)

    comparison = load_json(paths["comparison_json"])
    quality_summary = load_json(paths["pose_quality_summary"])
    phase_summary = load_json(paths["phase_summary"])
    movement_summary = load_json(paths["movement_summary"])
    baseline_profile = load_json(paths["baseline_profile"])
    real_demo_summary = load_json(paths.get("real_demo_summary", Path("__missing__")))
    pose_audit_summary = load_json(paths.get("pose_audit_summary", Path("__missing__")))

    cards = key_metric_cards(comparison, quality_summary, movement_summary)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Comparison status", cards["comparison_status"])
    col2.metric("Usable motion frames", cards["usable_motion_frames"])
    col3.metric("Pose detection rate", _display_pose_detection_rate(cards, real_demo_summary))
    col4.metric("Mean frame quality", cards["mean_frame_quality"])

    if mode == "real":
        st.success(
            "Real demo mode uses the best sampled cricket clip from the pose audit. "
            "Run `make slow-real-demo` if artifacts are missing."
        )
    else:
        st.info(
            "Synthetic mode is useful for smoke testing. It usually has zero pose "
            "detections and intentionally shows insufficient data."
        )

    tabs = st.tabs(
        [
            "Overview",
            "Overlay video",
            "Report",
            "Pose audit",
            "Quality",
            "Phases",
            "Movement",
            "Baseline",
            "Artifacts",
        ]
    )

    with tabs[0]:
        _render_overview(
            comparison=comparison,
            quality_summary=quality_summary,
            phase_summary=phase_summary,
            movement_summary=movement_summary,
            real_demo_summary=real_demo_summary,
        )

    with tabs[1]:
        _render_overlay(paths["overlay_video"])

    with tabs[2]:
        _render_report(paths["report_markdown"], paths.get("report_chart"))

    with tabs[3]:
        _render_pose_audit(
            paths.get("pose_audit_summary"),
            paths.get("pose_audit_csv"),
            pose_audit_summary,
        )

    with tabs[4]:
        _render_quality(paths["pose_quality_summary"], paths["pose_quality_csv"])

    with tabs[5]:
        _render_phases(paths["phase_summary"], paths["phase_timeline_csv"])

    with tabs[6]:
        _render_movement(paths["movement_summary"], paths["movement_features_csv"])

    with tabs[7]:
        _render_baseline(baseline_profile)

    with tabs[8]:
        st.subheader("Artifact status")
        st.dataframe(artifact_status(paths), use_container_width=True)
        st.code("make real-demo\nmake app", language="bash")

    st.divider()
    st.caption(
        "Limitations: This is not coaching-grade, medical, biomechanics-validated, "
        "or injury-risk advice. Metrics are 2D pose-derived proxies."
    )


def _sidebar_paths(mode: str) -> dict[str, Path]:
    st.sidebar.header("Artifact paths")
    st.sidebar.caption("Defaults point to the selected workflow.")

    paths = {}

    for key, default_path in artifact_paths_for_mode(mode).items():
        user_value = st.sidebar.text_input(
            label=key,
            value=str(default_path.relative_to(PROJECT_ROOT)),
        )
        paths[key] = PROJECT_ROOT / user_value

    return paths


def _render_overview(
    comparison: dict[str, Any] | None,
    quality_summary: dict[str, Any] | None,
    phase_summary: dict[str, Any] | None,
    movement_summary: dict[str, Any] | None,
    real_demo_summary: dict[str, Any] | None,
) -> None:
    st.subheader("Pipeline overview")

    if real_demo_summary:
        st.markdown("### Selected real demo clip")
        st.json(
            {
                "selected_video_id": real_demo_summary.get("selected_video_id"),
                "selected_shot_type": real_demo_summary.get("selected_shot_type"),
                "selected_pose_detection_rate": real_demo_summary.get(
                    "selected_pose_detection_rate"
                ),
                "baseline_video_count": real_demo_summary.get("baseline_video_count"),
                "limitation": real_demo_summary.get("limitation"),
            }
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Comparison")
        if comparison:
            st.json(
                {
                    "comparison_status": comparison.get("comparison_status"),
                    "shot_type": comparison.get("shot_type"),
                    "movement_rows": comparison.get("movement_rows"),
                    "usable_motion_frames": comparison.get("usable_motion_frames"),
                    "notes": comparison.get("notes"),
                }
            )
        else:
            st.warning("Comparison JSON not found.")

    with col2:
        st.markdown("### Summaries")
        st.json(
            {
                "pose_quality": quality_summary or "missing",
                "phase_summary": phase_summary or "missing",
                "movement_summary": movement_summary or "missing",
            }
        )


def _render_overlay(video_path: Path) -> None:
    st.subheader("Pose overlay video")

    if not video_path.exists():
        st.warning(
            "Overlay video not found. Run `make slow-real-demo` for real mode "
            "or `make overlay-sample` for synthetic mode."
        )
        return

    st.video(str(video_path))
    st.caption(str(video_path.relative_to(PROJECT_ROOT)))


def _render_report(report_path: Path, chart_path: Path | None) -> None:
    st.subheader("Generated report")

    if chart_path is not None and chart_path.exists():
        st.image(str(chart_path), caption="Shot vs baseline metric comparison")

    if not report_path.exists():
        st.warning("Markdown report not found. Run `make real-demo` or `make report-sample`.")
        return

    st.markdown(clean_report_markdown_for_streamlit(report_path.read_text()))


def _render_pose_audit(
    summary_path: Path | None,
    csv_path: Path | None,
    summary: dict[str, Any] | None,
) -> None:
    st.subheader("Real sample pose audit")

    if summary_path is None or csv_path is None:
        st.info("Pose audit is only available in real demo mode.")
        return

    if summary is None:
        st.warning("Pose audit summary not found. Run `make real-demo`.")
    else:
        st.json(summary)

    audit_df = load_csv(csv_path)
    if audit_df.empty:
        st.info("No pose-audit rows available.")
        return

    st.dataframe(audit_df, use_container_width=True)

    if "pose_detection_rate" in audit_df.columns and "shot_type" in audit_df.columns:
        chart_df = audit_df.sort_values("pose_detection_rate", ascending=False)
        st.bar_chart(chart_df.set_index("shot_type")["pose_detection_rate"])


def _render_quality(summary_path: Path, csv_path: Path) -> None:
    st.subheader("Pose quality")

    summary = load_json(summary_path)
    quality_df = load_csv(csv_path)

    if summary is None:
        st.warning("Pose quality summary not found.")
    else:
        st.json(summary)

    if quality_df.empty:
        st.info("No frame-level quality rows available.")
    else:
        st.dataframe(quality_df, use_container_width=True)
        if "frame_quality_score" in quality_df.columns:
            st.line_chart(quality_df.set_index("frame_index")["frame_quality_score"])


def _render_phases(summary_path: Path, csv_path: Path) -> None:
    st.subheader("Phase timeline")

    summary = load_json(summary_path)
    phase_df = load_csv(csv_path)

    if summary is None:
        st.warning("Phase summary not found.")
    else:
        st.json(summary)

    if phase_df.empty:
        st.info("No phase rows available.")
    else:
        st.dataframe(phase_df, use_container_width=True)
        if "phase" in phase_df.columns:
            st.bar_chart(phase_df["phase"].value_counts())


def _render_movement(summary_path: Path, csv_path: Path) -> None:
    st.subheader("Movement features")

    summary = load_json(summary_path)
    movement_df = load_csv(csv_path)

    if summary is None:
        st.warning("Movement summary not found.")
    else:
        st.json(summary)

    if movement_df.empty:
        st.info("No movement-feature rows available.")
        return

    st.dataframe(movement_df, use_container_width=True)

    chart_columns = [
        column
        for column in [
            "head_displacement_proxy",
            "backlift_height_proxy",
            "wrist_speed_proxy",
            "wrist_path_smoothness_proxy",
        ]
        if column in movement_df.columns
    ]

    if chart_columns:
        st.line_chart(movement_df.set_index("frame_index")[chart_columns])


def _render_baseline(baseline_profile: dict[str, Any] | None) -> None:
    st.subheader("Baseline profile")

    if baseline_profile is None:
        st.warning("Baseline profile not found.")
        return

    st.json(
        {
            "baseline_version": baseline_profile.get("baseline_version"),
            "total_manifest_rows": baseline_profile.get("total_manifest_rows"),
            "usable_shots": baseline_profile.get("usable_shots"),
            "shot_types": list((baseline_profile.get("shot_types") or {}).keys()),
            "limitation": baseline_profile.get("limitation"),
        }
    )

    shot_types = baseline_profile.get("shot_types") or {}
    if not shot_types:
        return

    rows = []
    for shot_type, payload in shot_types.items():
        rows.append(
            {
                "shot_type": shot_type,
                "manifest_shots": payload.get("manifest_shots"),
                "usable_shots": payload.get("usable_shots"),
                "feature_count": len(payload.get("features") or {}),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def clean_report_markdown_for_streamlit(markdown_text: str) -> str:
    """Remove local Markdown image references already rendered by Streamlit."""

    lines = markdown_text.splitlines()
    cleaned_lines: list[str] = []
    skip_blank_after_image = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            skip_blank_after_image = True
            continue

        if skip_blank_after_image and stripped == "":
            skip_blank_after_image = False
            continue

        skip_blank_after_image = False
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip() + "\n"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _display_pose_detection_rate(
    cards: dict[str, str],
    real_demo_summary: dict[str, Any] | None,
) -> str:
    if real_demo_summary and real_demo_summary.get("selected_pose_detection_rate") is not None:
        return _format_optional_number(real_demo_summary.get("selected_pose_detection_rate"))

    return cards["pose_detection_rate"]


def _format_optional_number(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


if __name__ == "__main__":
    app()
