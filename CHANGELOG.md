# Changelog

## v0.1.0 - CricForm Vision Lab real-demo milestone

This release marks the first portfolio-ready version of CricForm Vision Lab.

### Added

- Python project scaffold with package structure, Makefile workflow, pytest, Ruff, and GitHub Actions CI.
- Synthetic cricket-style video fixture for local smoke testing.
- Video metadata inspection and deterministic frame extraction.
- MediaPipe Pose Landmarker video inference workflow.
- Pose landmark JSONL output and tidy Parquet landmark schema.
- Frame-level and video-level pose quality scoring.
- Annotated pose overlay video generation.
- Rule-based batting phase timeline detection.
- Interpretable movement-proxy feature extraction.
- Small empirical baseline profile builder.
- Shot-vs-baseline comparison JSON output.
- Markdown report and metric comparison chart rendering.
- Streamlit artifact viewer.
- Public cricket-shot dataset metadata and archive audit workflow.
- AppleDouble sidecar filtering for dataset archive hygiene.
- Real sampled-video pose audit across public cricket clips.
- Best real-demo clip selection using pose detection rate.
- Real cricket demo artifact pipeline.
- Slow-motion overlay export for clearer demo presentation.
- Streamlit real demo mode with synthetic fallback.
- Demo walkthrough and project talking points documentation.

### Real demo snapshot

    selected_video_id: test_pull_pull_0025
    selected_shot_type: pull
    selected_pose_detection_rate: 0.5714
    comparison_status: comparison_available
    usable_motion_frames: 4
    baseline_video_count: 10

### Known limitations

- Movement features are 2D pose-derived proxies.
- Phase labels are heuristic and not expert-annotated.
- The real-demo baseline is tiny and exists to demonstrate pipeline behavior.
- Results are not coaching-grade, biomechanics-validated, medical, injury-risk, or shot-correctness advice.
- Pose quality depends heavily on camera angle, occlusion, motion blur, body scale, and video quality.

### Generated artifacts

Large/generated artifacts are intentionally ignored by Git, including:

- downloaded cricket-shot archives
- sampled real videos
- MediaPipe model files
- pose JSONL files
- Parquet outputs
- reports
- charts
- overlay videos
