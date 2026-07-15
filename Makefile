.PHONY: setup test lint format check ci clean sample-video video-info extract-frames download-pose-model pose-sample landmarks-sample pose-quality-sample overlay-sample phase-sample movement-features-sample baseline-sample report-sample app dataset-info download-cricket-shot audit-cricket-shot sample-cricket-shot

SAMPLE_VIDEO=data/raw/videos/synthetic_batting_sample.mp4
POSE_MODEL_DIR=models/pose_landmarker
POSE_MODEL=$(POSE_MODEL_DIR)/pose_landmarker_lite.task
POSE_MODEL_URL=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
POSE_OUTPUT=data/interim/pose_landmarks/synthetic_batting_sample.pose.jsonl
LANDMARK_PARQUET=data/interim/pose_landmarks/synthetic_batting_sample.landmarks.parquet
FEATURE_OUTPUT_DIR=data/processed/features
PHASE_TIMELINE=data/processed/features/synthetic_batting_sample.phase_timeline.csv
MOVEMENT_FEATURES=data/processed/features/synthetic_batting_sample.movement_features.csv
MOVEMENT_SUMMARY=data/processed/features/synthetic_batting_sample.movement_summary.json
BASELINE_DIR=data/processed/baselines
BASELINE_MANIFEST=$(BASELINE_DIR)/synthetic_baseline_manifest.csv
BASELINE_PROFILE=$(BASELINE_DIR)/synthetic_baseline_profile.json
REPORT_DIR=data/processed/reports
COMPARISON_JSON=$(REPORT_DIR)/synthetic_batting_sample_comparison.json
REPORT_MD=$(REPORT_DIR)/synthetic_batting_sample_report.md
REPORT_CHART=outputs/sample_reports/synthetic_batting_sample_metric_comparison.png
OVERLAY_OUTPUT=outputs/sample_overlays/synthetic_batting_sample_pose_overlay.mp4
DATASET_DIR=data/raw/datasets
CRICKET_SHOT_ARCHIVE=$(DATASET_DIR)/cricketshot.tar.gz
HF_SAMPLE_DIR=data/raw/videos/hf_cricket_shot

setup:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -e ".[dev]"

sample-video:
	. .venv/bin/activate && python -m cricform.ingest.create_sample_video --overwrite

video-info:
	. .venv/bin/activate && python -m cricform.video.metadata $(SAMPLE_VIDEO) --json

extract-frames:
	. .venv/bin/activate && python -m cricform.video.frame_extract $(SAMPLE_VIDEO) --every-n-frames 6 --overwrite

download-pose-model:
	mkdir -p $(POSE_MODEL_DIR)
	test -f $(POSE_MODEL) || curl -L $(POSE_MODEL_URL) -o $(POSE_MODEL)
	ls -lh $(POSE_MODEL)

pose-sample: download-pose-model sample-video
	. .venv/bin/activate && python -m cricform.pose.mediapipe_pose $(SAMPLE_VIDEO) --model-path $(POSE_MODEL) --output-jsonl $(POSE_OUTPUT) --every-n-frames 3 --max-frames 24

landmarks-sample: pose-sample
	. .venv/bin/activate && python -m cricform.pose.landmark_schema $(POSE_OUTPUT) --output-parquet $(LANDMARK_PARQUET)

pose-quality-sample: pose-sample
	. .venv/bin/activate && python -m cricform.features.quality_features $(POSE_OUTPUT) --output-dir $(FEATURE_OUTPUT_DIR)

overlay-sample: pose-sample
	. .venv/bin/activate && python -m cricform.video.overlay $(SAMPLE_VIDEO) $(POSE_OUTPUT) --output-video $(OVERLAY_OUTPUT)

phase-sample: landmarks-sample
	. .venv/bin/activate && python -m cricform.phases.detect_phases $(LANDMARK_PARQUET) --output-dir $(FEATURE_OUTPUT_DIR)

movement-features-sample: phase-sample
	. .venv/bin/activate && python -m cricform.features.motion_features $(LANDMARK_PARQUET) --phase-timeline-csv $(PHASE_TIMELINE) --output-dir $(FEATURE_OUTPUT_DIR)

baseline-sample: movement-features-sample
	mkdir -p $(BASELINE_DIR)
	printf "shot_id,shot_type,movement_features_csv\nsynthetic_batting_sample,unknown,../features/synthetic_batting_sample.movement_features.csv\n" > $(BASELINE_MANIFEST)
	. .venv/bin/activate && python -m cricform.baseline.build_baseline $(BASELINE_MANIFEST) --output-profile $(BASELINE_PROFILE)

report-sample: baseline-sample
	mkdir -p $(REPORT_DIR)
	mkdir -p outputs/sample_reports
	. .venv/bin/activate && python -m cricform.baseline.compare_shot $(MOVEMENT_SUMMARY) $(BASELINE_PROFILE) --shot-type unknown --output-comparison $(COMPARISON_JSON)
	. .venv/bin/activate && python -m cricform.reports.render_report $(COMPARISON_JSON) --output-markdown $(REPORT_MD) --output-chart $(REPORT_CHART)

dataset-info:
	. .venv/bin/activate && python -m cricform.ingest.download_dataset --metadata-output $(DATASET_DIR)/cricket_shot_dataset_info.json

download-cricket-shot:
	. .venv/bin/activate && python -m cricform.ingest.download_dataset --download --output-dir $(DATASET_DIR)

audit-cricket-shot:
	. .venv/bin/activate && python -m cricform.ingest.sample_dataset $(CRICKET_SHOT_ARCHIVE) --audit-only

sample-cricket-shot:
	. .venv/bin/activate && python -m cricform.ingest.sample_dataset $(CRICKET_SHOT_ARCHIVE) --output-dir $(HF_SAMPLE_DIR) --split test --samples-per-class 1 --overwrite

app:
	. .venv/bin/activate && streamlit run src/cricform/app/streamlit_app.py

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check src tests

format:
	. .venv/bin/activate && ruff format src tests

check: lint test

ci:
	ruff check src tests
	pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
