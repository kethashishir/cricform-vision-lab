from pathlib import Path

CI_WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def test_ci_workflow_exists() -> None:
    assert CI_WORKFLOW_PATH.exists()


def test_ci_workflow_runs_lint_and_tests() -> None:
    workflow = CI_WORKFLOW_PATH.read_text()

    assert "ruff check src tests" in workflow
    assert "pytest -q" in workflow


def test_ci_workflow_uses_python_312_and_does_not_download_large_artifacts() -> None:
    workflow = CI_WORKFLOW_PATH.read_text()

    assert 'python-version: "3.12.13"' in workflow
    assert "download-pose-model" not in workflow
    assert "report-sample" not in workflow
    assert "overlay-sample" not in workflow


def test_ci_workflow_uses_current_official_action_majors() -> None:
    workflow = CI_WORKFLOW_PATH.read_text()

    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v6" in workflow
