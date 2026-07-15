# Phase 0 Feasibility Notes

## Project

CricForm Vision Lab — an explainable cricket batting computer-vision pipeline.

## Feasibility decision

Feasible as an explainable computer-vision and feature-engineering portfolio project.

Not feasible to claim professional coaching accuracy, injury insight, or validated biomechanics without expert labels and validation.

## Primary dataset candidate

Dataset: rokmr/cricket-shot  
Source: Hugging Face  
Task: video classification  
License shown on Hugging Face: Apache-2.0  
Archive size shown: about 2.89 GB  
Classes: cover, defense, flick, hook, late_cut, lofted, pull, square_cut, straight, sweep  
Splits: train 1250, validation 250, test 250  

## Dataset risks

- Dataset is designed for shot classification, not biomechanics.
- Camera angle, lighting, batter scale, occlusion, broadcast cuts, and motion blur may hurt pose extraction.
- Dataset card/code may have naming mismatch: repository is rokmr/cricket-shot, while README usage mentions rokmr/cricketshot.
- Need to verify actual download method locally.
- Need to inspect a few clips before trusting it.
- Public video provenance/copyright should be documented carefully.
- Reports must include limitations.

## Secondary datasets

CricShot10:
- 10 cricket batting shot classes.
- Access appears email-based.
- Developed using YouTube videos.
- Useful as background, not ideal as reproducible primary source.

CricShot10k:
- 10086 clips across 15 classes.
- Larger but more complex.
- Google Drive distribution.
- Non-commercial research use statement.
- Better later, not for initial build.

Kaggle cricket shot datasets:
- Many appear image-only.
- Not suitable for temporal pose mechanics.

SportsMOT fallback:
- Strong sports video dataset.
- Not cricket batting.
- Use only if cricket pose extraction quality is unusable.

## Pose extraction tool

MediaPipe Pose Landmarker is the first pose extractor.

Use offline VIDEO mode with detect_for_video for reproducible frame-by-frame processing.

Avoid live-stream async mode for dataset processing because it may drop frames.

## Phase 1 recommendation

Create a clean Python package scaffold with:

- pyproject.toml
- Makefile
- README.md
- src/cricform/
- tests/
- data/ directories with .gitkeep files
- outputs/ directories with .gitkeep files
- lightweight dependencies only
- pytest smoke test
- no MediaPipe processing yet
- no Streamlit yet
- no Docker yet
