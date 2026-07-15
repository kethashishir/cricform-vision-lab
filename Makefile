.PHONY: setup test lint format check clean sample-video video-info extract-frames

SAMPLE_VIDEO=data/raw/videos/synthetic_batting_sample.mp4

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
