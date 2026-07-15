from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw" / "videos"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_REPORTS_DIR = OUTPUTS_DIR / "sample_reports"
SAMPLE_OVERLAYS_DIR = OUTPUTS_DIR / "sample_overlays"


def project_paths() -> dict[str, Path]:
    """Return important project paths for scripts and tests."""
    return {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "raw_video_dir": RAW_VIDEO_DIR,
        "interim_dir": INTERIM_DIR,
        "processed_dir": PROCESSED_DIR,
        "outputs_dir": OUTPUTS_DIR,
        "sample_reports_dir": SAMPLE_REPORTS_DIR,
        "sample_overlays_dir": SAMPLE_OVERLAYS_DIR,
    }
