# CricForm Vision Lab

CricForm Vision Lab is an explainable cricket batting computer-vision pipeline.

It takes a cricket batting video, extracts pose landmarks, audits pose quality, detects batting phases, computes interpretable movement features, compares a shot against a small baseline, and generates visual overlays and reports.

## One-sentence pitch

A reproducible cricket computer-vision lab that audits pose quality on real batting clips, selects the best demo sample, computes interpretable movement proxies, and visualizes the result with reports, charts, overlays, and honest limitations.

## Demo snapshot

Current real demo candidate:

```text
selected_video_id: test_pull_pull_0025
selected_shot_type: pull
selected_pose_detection_rate: 0.5714
comparison_status: comparison_available
usable_motion_frames: 4
baseline_video_count: 10
```

The Streamlit app defaults to the real cricket demo and uses a slow-motion overlay so the skeleton is easier to inspect.

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

This is not medical, injury-risk, or coaching-grade advice.

This does not claim validated cricket coaching accuracy.

This does not make expert claims unless future validation is performed with expert-labeled data.

## Pipeline overview

1. Ingest or sample cricket shot videos.
2. Extract frames or process video directly.
3. Run MediaPipe pose extraction.
4. Store landmarks in JSONL and tidy Parquet formats.
5. Compute pose quality metrics.
6. Detect rough batting phases with rule-based heuristics.
7. Compute interpretable movement-proxy features.
8. Build a small empirical baseline.
9. Compare a selected shot against the baseline.
10. Generate annotated overlay videos.
11. Generate Markdown reports and metric charts.
12. Display artifacts in Streamlit.
13. Protect the workflow with tests, linting, and GitHub Actions CI.

## Current status

Phase 19: Streamlit slow overlay default.

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
- Small empirical baseline profile builder
- Test-shot comparison JSON and Markdown report
- Streamlit artifact viewer demo UI
- GitHub Actions CI for linting and tests
- Public cricket-shot dataset archive audit and sampling
- Real sampled-video pose detection audit
- Real cricket demo artifact generation from best audited clip
- Streamlit real demo mode for visualizing real cricket artifacts
- Slow-motion presentation overlay for easier demo viewing
- Streamlit defaults to the slow real overlay for clearer demos

## Dataset plan

Primary candidate:

- Hugging Face rokmr/cricket-shot

Known risks:

- The dataset is designed for shot classification, not biomechanics.
- Camera angle, occlusion, broadcast cuts, lighting, scale, and motion blur may hurt pose extraction.
- Dataset access and file format must be verified locally before pipeline assumptions are made.
- Public video provenance and license limitations must be documented honestly.

## Demo walkthrough

For interview/demo talking points, see:

    docs/demo_walkthrough.md

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


## Baseline builder

Build a small empirical baseline profile from movement-feature CSVs:

    make baseline-sample

This writes:

    data/processed/baselines/synthetic_baseline_manifest.csv
    data/processed/baselines/synthetic_baseline_profile.json

The baseline manifest has columns:

    shot_id
    shot_type
    movement_features_csv

The baseline profile groups usable shots by shot_type and computes simple mean, standard deviation, min, max, and count statistics for movement-feature proxies.

The synthetic sample usually has no usable pose landmarks, so the generated sample baseline will show zero usable shots. That is acceptable for the smoke workflow. Real baselines require real pose-detected clips.


## Test-shot comparison report

Generate a shot-vs-baseline comparison report:

    make report-sample

This writes:

    data/processed/reports/synthetic_batting_sample_comparison.json
    data/processed/reports/synthetic_batting_sample_report.md
    outputs/sample_reports/synthetic_batting_sample_metric_comparison.png

The synthetic sample usually has no usable pose landmarks, so the report should clearly state that comparison is not meaningful yet. This is expected and honest.

Reports include:

    comparison_status
    usable_motion_frames
    metric comparison table
    notes
    honest limitations
    optional metric comparison chart when enough data exists


## Streamlit demo UI

Launch the local demo app:

    make app

The app defaults to real cricket demo mode. Before opening it, generate the slow real demo artifacts:

    make slow-real-demo

The app displays:

    slow real overlay video
    Markdown report
    metric comparison chart
    comparison status
    real pose audit summary
    pose quality summary
    phase timeline
    movement feature table
    baseline profile
    artifact status

Synthetic sample mode is still available in the sidebar for smoke testing. The synthetic sample is expected to show insufficient pose data. That is honest behavior, not a UI failure.


## Continuous integration

GitHub Actions runs CI on pushes to main and pull requests.

CI runs:

    ruff check src tests
    pytest -q

CI intentionally does not download the full dataset, the MediaPipe model asset, or generated demo artifacts. The test suite uses tiny synthetic fixtures so the workflow stays fast and reproducible.


## Public cricket-shot dataset audit

Document the public Hugging Face dataset metadata:

    make dataset-info

Download the public cricket-shot archive:

    make download-cricket-shot

Warning: the archive is large, roughly several GB. It is ignored by Git.

Audit the archive without extracting everything:

    make audit-cricket-shot

Extract a tiny balanced real-video sample from the test split:

    make sample-cricket-shot

This writes sampled videos under:

    data/raw/videos/hf_cricket_shot/

Downloaded archives and extracted video samples are generated data artifacts and should not be committed.

Dataset limitation: this public dataset is designed for cricket shot classification, not validated biomechanics or coaching-grade movement analysis.

Archive note: the downloaded tar archive may contain macOS AppleDouble files such as `._cover_0023.avi`. The sampler ignores those metadata sidecar files and extracts only real video files.


## Real sample pose audit

After downloading and sampling the public cricket-shot archive, run pose detection on the sampled real clips:

    make real-pose-audit

This writes:

    data/processed/real_sample_pose_audit/pose_audit.csv
    data/processed/real_sample_pose_audit/pose_audit_summary.json
    data/interim/pose_landmarks/real_samples/

The audit reports:

    frames_processed
    frames_with_pose
    pose_detection_rate
    best_video

The pose detection rate measures whether landmarks were available. It is not a coaching-quality, biomechanics, or shot-correctness score.


## Real cricket demo artifacts

Build demo-ready artifacts from the best real sampled cricket clip:

    make real-demo

This runs the real pose audit, selects the best detected clip, and generates:

    data/processed/real_demo/real_demo_summary.json
    data/processed/real_demo/features/
    data/processed/real_demo/reports/
    outputs/real_demo/

The generated real demo uses a tiny mixed sample baseline only to demonstrate the reporting pipeline. It is not a cricket coaching standard or biomechanics validation.


## Slow-motion demo overlay

Create a slower presentation copy of the real overlay video:

    make slow-real-demo

This writes:

    outputs/real_demo/test_pull_pull_0025_pose_overlay_slow.mp4

The slow overlay is for viewing only. It does not change pose timestamps, movement features, phase labels, baseline comparison, or any analysis result.

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
