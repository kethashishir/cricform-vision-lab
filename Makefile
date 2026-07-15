.PHONY: setup test lint format check clean sample-video video-info extract-frames download-pose-model pose-sample landmarks-sample pose-quality-sample

SAMPLE_VIDEO=data/raw/videos/synthetic_batting_sample.mp4
POSE_MODEL_DIR=models/pose_landmarker
POSE_MODEL=$(POSE_MODEL_DIR)/pose_landmarker_lite.task
POSE_MODEL_URL=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
POSE_OUTPUT=data/interim/pose_landmarks/synthetic_batting_sample.pose.jsonl
LANDMARK_PARQUET=data/interim/pose_landmarks/synthetic_batting_sample.landmarks.parquet
FEATURE_OUTPUT_DIR=data/processed/features

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

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check src tests

format:
	. .venv/bin/activate && ruff format src tests

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
