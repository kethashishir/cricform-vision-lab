# CricForm Vision Lab Demo Walkthrough

This document explains how to present CricForm Vision Lab in interviews, portfolio reviews, and project demos.

## 1. Short project explanation

CricForm Vision Lab is an explainable cricket batting computer-vision pipeline.

It takes cricket batting clips through pose extraction, pose-quality auditing, rule-based batting phase detection, movement-proxy feature extraction, baseline comparison, report generation, and Streamlit visualization.

The project is intentionally honest: the metrics are 2D pose-derived proxies, not professional biomechanics, medical advice, injury-risk assessment, or coaching-grade shot correction.

## 2. Thirty-second pitch

I built an explainable cricket batting computer-vision pipeline that processes batting videos through pose extraction, pose-quality auditing, movement-proxy feature generation, baseline comparison, report generation, and Streamlit visualization.

The project uses a public cricket-shot video dataset, filters invalid sidecar files, audits MediaPipe pose quality across real samples, selects the best detected clip, and generates slow-motion overlay videos and comparison reports.

I focused on reproducibility, testing, CI, artifact management, and honest limitations instead of overclaiming coaching-grade accuracy.

## 3. Demo setup

Before opening the app, run:

    make slow-real-demo
    make app

The app should open in real cricket demo mode.

Expected selected clip:

    selected_video_id: test_pull_pull_0025
    selected_shot_type: pull
    selected_pose_detection_rate: 0.5714
    comparison_status: comparison_available
    usable_motion_frames: 4
    baseline_video_count: 10

## 4. What to show in the Streamlit app

### Overview tab

Show:

- selected real demo clip
- selected shot type
- pose detection rate
- comparison status
- usable motion frames
- honest limitation text

Explain that the selected clip was chosen from a real pose audit, not hardcoded randomly.

### Overlay video tab

Show the slow-motion skeleton overlay.

Say:

The slow video is only for presentation. It does not change timestamps, pose extraction, movement features, phase labels, or analysis results.

### Pose audit tab

Show the pose-audit table and chart.

Explain:

- the pipeline sampled real cricket videos
- MediaPipe Pose was run on each sampled clip
- the best clip was selected by pose detection rate
- pose detection rate means landmark availability, not shot quality

### Report tab

Show:

- metric comparison chart
- movement rows
- usable motion frames
- comparison status
- metric table
- honest limitations

Do not describe the report as coaching feedback.

### Synthetic mode

Switch to synthetic mode briefly.

Explain:

The synthetic sample is useful for smoke testing the pipeline, but it usually has zero pose detections. The app surfaces that honestly instead of pretending there is useful data.

## 5. What makes the project technically strong

Key engineering points:

- Python package structure with modular components
- MediaPipe Pose Landmarker video workflow
- JSONL pose storage and tidy Parquet landmark schema
- pose-quality scoring before feature interpretation
- real dataset audit and AppleDouble sidecar filtering
- batch pose audit across sampled real cricket videos
- best-clip selection based on measured detection quality
- overlay video rendering
- slow-motion presentation copy
- movement-proxy feature extraction
- baseline profile and comparison report
- Markdown report and metric chart generation
- Streamlit app for artifact visualization
- generated data ignored by Git
- pytest and Ruff checks
- GitHub Actions CI

## 6. What not to claim

Do not say:

- this is a professional coaching product
- this validates batting technique
- this detects injury risk
- this gives medical or biomechanics advice
- this proves whether a shot is correct or incorrect
- this baseline represents population-level cricket norms

Say instead:

- this is an explainable computer-vision pipeline
- the metrics are 2D pose-derived proxies
- the baseline is small and empirical
- the goal is reproducible analysis and inspection
- pose quality is measured before interpreting features

## 7. Resume bullets

Use one or two of these depending on the role.

Software engineering / data engineering version:

- Built an explainable cricket computer-vision pipeline in Python that processes batting videos through MediaPipe pose extraction, pose-quality auditing, movement-proxy feature generation, baseline comparison, report generation, and Streamlit visualization.
- Engineered reproducible data workflows for public cricket-shot video sampling, archive auditing, AppleDouble sidecar filtering, pose landmark JSONL/Parquet storage, generated artifact management, and GitHub Actions CI.
- Developed tested modular components for video ingestion, pose auditing, landmark schema conversion, feature extraction, baseline building, report rendering, overlay generation, and slow-motion demo export.

ML / computer vision version:

- Built a real-video cricket pose-analysis lab using MediaPipe Pose, OpenCV, pandas, and Streamlit to audit landmark availability, select the best detected clip, and generate interpretable movement-proxy reports.
- Implemented pose-quality scoring and batch sample auditing across public cricket-shot clips, selecting a real pull-shot demo with 0.5714 pose detection rate and generating overlay/report artifacts with explicit model limitations.
- Designed rule-based batting phase heuristics and movement-proxy features while clearly separating pose-derived inspection metrics from coaching-grade or biomechanics claims.

Portfolio summary version:

- CricForm Vision Lab: explainable cricket batting CV pipeline with real dataset sampling, pose extraction, quality audit, movement proxies, baseline comparison, report generation, slow-motion overlay, Streamlit demo, and CI-tested modular Python code.

## 8. LinkedIn project post draft

I built CricForm Vision Lab, an explainable cricket batting computer-vision pipeline.

The project takes batting clips through pose extraction, pose-quality auditing, rough batting phase detection, movement-proxy feature extraction, baseline comparison, report generation, and Streamlit visualization.

A big focus was not overclaiming. Instead of calling it an AI coach, I treated it as an inspectable CV pipeline: measure pose quality first, reject weak data, surface limitations clearly, and only interpret movement metrics as 2D pose-derived proxies.

Some engineering pieces I built:

- public cricket-shot dataset audit and sampling
- AppleDouble sidecar filtering
- MediaPipe pose extraction
- JSONL and Parquet landmark storage
- batch pose-quality audit on real cricket clips
- best real-clip selection
- overlay and slow-motion demo video generation
- movement-proxy features and baseline comparison
- Markdown report and metric chart rendering
- Streamlit artifact viewer
- pytest, Ruff, and GitHub Actions CI

The current real demo selects a pull-shot clip from sampled cricket videos and shows pose overlays, audit results, feature summaries, and a comparison report.

## 9. Interview explanation

If asked why this project is useful:

This project shows that I can build an end-to-end ML-adjacent engineering workflow, not just a notebook. I handled data ingestion, messy archive contents, model inference, quality auditing, feature engineering, artifact generation, UI presentation, tests, CI, and limitations.

If asked why not train a model:

The project first needed a reliable data and pose-quality pipeline. Training a model before verifying pose availability and data quality would be premature. This version builds the foundation for future modeling.

If asked what you would improve next:

I would compute real movement features across all sampled clips, build stronger baselines by shot type, improve phase detection using expert-labeled examples, and add richer temporal visualizations in Streamlit.

## 10. Current limitation statement

CricForm Vision Lab is not a coaching-grade biomechanics system. It uses MediaPipe pose landmarks and 2D movement proxies to inspect pipeline behavior and relative feature patterns. Results depend heavily on camera angle, occlusion, body scale, motion blur, and landmark quality.
