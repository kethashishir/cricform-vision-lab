.PHONY: setup test lint format check clean

setup:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -e ".[dev]"

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
