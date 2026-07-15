# CricForm Vision Lab

CricForm Vision Lab is an explainable cricket batting computer-vision pipeline.

It takes a cricket batting video, extracts pose landmarks, audits pose quality, detects batting phases, computes interpretable movement features, compares a shot against a small baseline, and generates visual overlays and reports.

## One-sentence pitch

Build a system that takes a cricket batting video, extracts pose landmarks, detects batting phases, computes interpretable movement features, compares the shot against a small baseline, and produces a visual report with overlays, timelines, quality checks, and honest limitations.

## What this project is

This is a portfolio-grade computer-vision and feature-engineering project focused on:

- Pose extraction
- Pose quality auditing
- Rule-based batting phase detection
- Interpretable motion features
- Baseline comparison
- Visual reports
- Reproducible engineering workflow

## What this project is not

This is not a professional biomechanics product.

This is not medical, injury, or coaching-grade advice.

This does not claim validated cricket coaching accuracy.

This does not make expert claims unless future validation is performed with expert-labeled data.

## Planned pipeline

1. Ingest or sample cricket shot videos.
2. Extract frames.
3. Run pose extraction.
4. Store landmarks in a clean schema.
5. Compute pose quality metrics.
6. Smooth landmarks where appropriate.
7. Detect batting phases with rule-based logic first.
8. Compute interpretable movement features.
9. Build small baseline profiles by shot type.
10. Compare a test shot against baseline.
11. Generate annotated video overlay.
12. Generate visual report.
13. Build Streamlit demo.
14. Add tests, CI, screenshots, and portfolio polish.

## Current status

Phase 9: movement feature extraction.

The project currently verifies:

- Python package structure
- Local installation
- Basic import test
- Data/output directory layout
- Makefile workflow
- Synthetic local video fixture generation
- Video metadata inspection
- Deterministic frame extraction
- MediaPipe Pose Landmarker smoke workflow
- Tidy landmark schema
- Parquet landmark storage
- Frame-level pose quality scoring
- Video-level pose quality summary
- Annotated pose overlay video generation
- Rule-based batting phase timeline detection
- Interpretable movement-feature proxies

## Dataset plan

Primary candidate:

- Hugging Face rokmr/cricket-shot

Known risks:

- The dataset is designed for shot classification, not biomechanics.
- Camera angle, occlusion, broadcast cuts, lighting, scale, and motion blur may hurt pose extraction.
- Dataset access and file format must be verified locally before pipeline assumptions are made.
- Public video provenance and license limitations must be documented honestly.

## Environment

Use Python 3.12.

Python 3.14 is currently avoided because MediaPipe Pose Landmarker support is the future dependency constraint.

## Setup

Run:

    pyenv local 3.12.13
    make setup
    make check

## Development commands

Run:

    make setup
    make test
    make lint
    make format
    make check
    make clean


## Synthetic sample video

Generate a tiny local MP4 fixture with:

    make sample-video

This creates:

    data/raw/videos/synthetic_batting_sample.mp4

The video is synthetic and exists only to test video I/O and pipeline plumbing. It is not real cricket footage and must not be used for coaching, biomechanics, or dataset-quality claims.


## Video metadata and frame extraction

Inspect the synthetic sample video:

    make video-info

Extract sampled frames:

    make extract-frames

This writes frames to:

    data/interim/frames/synthetic_batting_sample/

Extracted frames are generated artifacts and should not be committed.


## Pose extraction

Download the MediaPipe Pose Landmarker model:

    make download-pose-model

Run pose extraction on the synthetic sample video:

    make pose-sample

This writes:

    data/interim/pose_landmarks/synthetic_batting_sample.pose.jsonl

Important: the synthetic stick-figure video may produce zero detected poses. That is not a pipeline failure. It only proves the MediaPipe video-processing path runs. Real pose-quality evaluation requires real human batting footage.


## Landmark schema and Parquet

Convert pose JSONL into a tidy landmark table:

    make landmarks-sample

This writes:

    data/interim/pose_landmarks/synthetic_batting_sample.landmarks.parquet

The table uses one row per landmark per frame per coordinate space.

Expected columns include:

    video_id
    source_video_path
    frame_index
    timestamp_ms
    pose_index
    coordinate_space
    landmark_index
    landmark_name
    x
    y
    z
    visibility
    presence

The synthetic stick-figure sample may produce an empty Parquet table because MediaPipe may detect zero human poses. That is acceptable for the smoke workflow.


## Pose quality scoring

Compute frame-level and video-level pose quality metrics:

    make pose-quality-sample

This writes:

    data/processed/features/synthetic_batting_sample.pose_quality.csv
    data/processed/features/synthetic_batting_sample.pose_quality_summary.json

Quality metrics include:

    pose_detection_rate
    landmark_coverage
    mean_visibility
    mean_presence
    coordinate_valid_ratio
    frame_quality_score
    low_quality_frame_rate

The synthetic stick-figure sample is expected to score poorly because MediaPipe detects no real human pose. This is useful because it proves that low-quality pose data is not silently trusted.


## Pose overlay video

Render an annotated overlay video:

    make overlay-sample

This writes:

    outputs/sample_overlays/synthetic_batting_sample_pose_overlay.mp4

The overlay draws detected pose landmarks when available and labels frames where pose was not sampled or no pose was detected.

For the synthetic sample, MediaPipe usually detects zero poses. That is expected. The overlay still proves that the project can generate visual audit artifacts and communicate pose failure clearly.


## Rule-based phase detection

Detect a transparent batting phase timeline from landmark Parquet:

    make phase-sample

This writes:

    data/processed/features/synthetic_batting_sample.phase_timeline.csv
    data/processed/features/synthetic_batting_sample.phase_summary.json

Phase labels include:

    stance
    backlift
    downswing
    contact_zone
    follow_through
    recovery
    unavailable

The current detector is a heuristic based on wrist/hand motion proxies. It is not validated against expert cricket annotations and must not be described as coaching-grade.


## Movement feature extraction

Compute interpretable movement-feature proxies:

    make movement-features-sample

This writes:

    data/processed/features/synthetic_batting_sample.movement_features.csv
    data/processed/features/synthetic_batting_sample.movement_summary.json

Current movement proxies include:

    head_displacement_proxy
    front_knee_bend_proxy_deg
    shoulder_hip_separation_proxy_deg
    wrist_speed_proxy
    wrist_path_smoothness_proxy
    backlift_height_proxy
    follow_through_height_proxy

These are 2D pose-derived proxies. They are not validated biomechanics, coaching-grade metrics, or injury-risk assessments.

## Roadmap

- Phase 0: dataset and tool feasibility
- Phase 1: repo scaffold
- Phase 2: tiny local sample video workflow
- Phase 3: frame extraction and video metadata
- Phase 4: MediaPipe pose extraction
- Phase 5: landmark schema and Parquet storage
- Phase 6: pose quality scoring
- Phase 7: overlay video generation
- Phase 8: rule-based phase detection
- Phase 9: movement feature extraction
- Phase 10: baseline builder
- Phase 11: test-shot comparison report
- Phase 12: Streamlit app
- Phase 13: tests and CI
- Phase 14: README, screenshots, demo GIF, LinkedIn post, resume bullets

## License

To be decided.

Dataset licenses and sample-video provenance will be documented separately.
