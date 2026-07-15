from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DATASET_REPO_ID = "rokmr/cricket-shot"
DATASET_REPO_TYPE = "dataset"
DATASET_ARCHIVE_FILENAME = "cricketshot.tar.gz"
DEFAULT_DATASET_DIR = Path("data/raw/datasets")

SHOT_CLASSES: tuple[str, ...] = (
    "cover",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight",
    "sweep",
)


@dataclass(frozen=True)
class CricketShotDatasetInfo:
    repo_id: str
    repo_type: str
    archive_filename: str
    license: str
    task: str
    expected_classes: list[str]
    expected_train_samples: int
    expected_validation_samples: int
    expected_test_samples: int
    known_risks: list[str]


def cricket_shot_dataset_info() -> CricketShotDatasetInfo:
    """Return documented metadata for the public cricket-shot dataset."""

    return CricketShotDatasetInfo(
        repo_id=DATASET_REPO_ID,
        repo_type=DATASET_REPO_TYPE,
        archive_filename=DATASET_ARCHIVE_FILENAME,
        license="apache-2.0",
        task="video-classification",
        expected_classes=list(SHOT_CLASSES),
        expected_train_samples=1250,
        expected_validation_samples=250,
        expected_test_samples=250,
        known_risks=[
            "Dataset is designed for shot classification, not biomechanics.",
            "Camera angle, occlusion, motion blur, and scale may hurt pose extraction.",
            "Dataset viewer may be unreliable; archive inspection is preferred.",
            "Pose-derived outputs must not be described as coaching-grade validation.",
        ],
    )


def write_dataset_info(output_path: Path) -> CricketShotDatasetInfo:
    """Write dataset metadata to JSON for reproducibility notes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = cricket_shot_dataset_info()
    output_path.write_text(json.dumps(asdict(info), indent=2) + "\n")
    return info


def download_cricket_shot_archive(output_dir: Path = DEFAULT_DATASET_DIR) -> Path:
    """Download the cricket-shot archive from Hugging Face Hub."""

    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        repo_type=DATASET_REPO_TYPE,
        filename=DATASET_ARCHIVE_FILENAME,
        local_dir=output_dir,
    )

    return Path(archive_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document or download cricket-shot dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "cricket_shot_dataset_info.json",
    )
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    info = write_dataset_info(args.metadata_output)
    print(json.dumps(asdict(info), indent=2))

    if args.download:
        archive_path = download_cricket_shot_archive(args.output_dir)
        print(f"Downloaded archive: {archive_path}")


if __name__ == "__main__":
    main()
