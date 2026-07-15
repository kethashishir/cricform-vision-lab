from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from cricform.ingest.download_dataset import SHOT_CLASSES

VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
)

KNOWN_SPLITS: tuple[str, ...] = (
    "train",
    "val",
    "validation",
    "test",
)


@dataclass(frozen=True)
class ArchiveAuditSummary:
    archive_path: str
    total_video_files: int
    counts_by_split: dict[str, int]
    counts_by_shot_type: dict[str, int]
    counts_by_split_and_shot_type: dict[str, dict[str, int]]
    sample_members: list[str]


@dataclass(frozen=True)
class SampleExtractionSummary:
    archive_path: str
    output_dir: str
    split: str
    samples_per_class: int
    extracted_files: list[str]


def audit_cricket_shot_archive(
    archive_path: Path,
    max_sample_members: int = 30,
) -> ArchiveAuditSummary:
    """Audit a cricket-shot tar archive without extracting all files."""

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")

    counts_by_split: dict[str, int] = defaultdict(int)
    counts_by_shot_type: dict[str, int] = defaultdict(int)
    counts_by_split_and_shot_type: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sample_members: list[str] = []
    total_video_files = 0

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not _is_video_file(member.name):
                continue

            split, shot_type = infer_split_and_shot_type(member.name)

            total_video_files += 1
            counts_by_split[split] += 1
            counts_by_shot_type[shot_type] += 1
            counts_by_split_and_shot_type[split][shot_type] += 1

            if len(sample_members) < max_sample_members:
                sample_members.append(member.name)

    return ArchiveAuditSummary(
        archive_path=str(archive_path),
        total_video_files=total_video_files,
        counts_by_split=dict(sorted(counts_by_split.items())),
        counts_by_shot_type=dict(sorted(counts_by_shot_type.items())),
        counts_by_split_and_shot_type={
            split: dict(sorted(counts.items()))
            for split, counts in sorted(counts_by_split_and_shot_type.items())
        },
        sample_members=sample_members,
    )


def extract_sample_videos(
    archive_path: Path,
    output_dir: Path,
    split: str = "test",
    samples_per_class: int = 1,
    overwrite: bool = False,
) -> SampleExtractionSummary:
    """Extract a tiny balanced sample of videos from the archive."""

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than 0")

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_counts: dict[str, int] = defaultdict(int)
    extracted_files: list[str] = []

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not _is_video_file(member.name):
                continue

            member_split, shot_type = infer_split_and_shot_type(member.name)

            if member_split != split:
                continue

            if shot_type not in SHOT_CLASSES:
                continue

            if extracted_counts[shot_type] >= samples_per_class:
                continue

            output_path = output_dir / split / shot_type / Path(member.name).name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists() and not overwrite:
                extracted_counts[shot_type] += 1
                extracted_files.append(str(output_path))
                continue

            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue

            output_path.write_bytes(file_obj.read())

            extracted_counts[shot_type] += 1
            extracted_files.append(str(output_path))

            if all(extracted_counts[label] >= samples_per_class for label in SHOT_CLASSES):
                break

    return SampleExtractionSummary(
        archive_path=str(archive_path),
        output_dir=str(output_dir),
        split=split,
        samples_per_class=samples_per_class,
        extracted_files=extracted_files,
    )


def infer_split_and_shot_type(member_name: str) -> tuple[str, str]:
    """Infer split and shot type from an archive member path."""

    parts = Path(member_name).parts

    for index, part in enumerate(parts):
        normalized = part.lower()

        if normalized not in KNOWN_SPLITS:
            continue

        split = "val" if normalized == "validation" else normalized
        shot_type = parts[index + 1].lower() if index + 1 < len(parts) else "unknown"
        return split, shot_type

    for shot_type in SHOT_CLASSES:
        if shot_type in [part.lower() for part in parts]:
            return "unknown", shot_type

    return "unknown", "unknown"


def _is_video_file(member_name: str) -> bool:
    path = Path(member_name)

    if path.name.startswith("._"):
        return False

    if "__MACOSX" in path.parts:
        return False

    return path.suffix.lower() in VIDEO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit or sample cricket-shot archive.")
    parser.add_argument("archive_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/videos/hf_cricket_shot"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audit = audit_cricket_shot_archive(args.archive_path)
    print(json.dumps(asdict(audit), indent=2))

    if args.audit_only:
        return

    extraction = extract_sample_videos(
        archive_path=args.archive_path,
        output_dir=args.output_dir,
        split=args.split,
        samples_per_class=args.samples_per_class,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(extraction), indent=2))


if __name__ == "__main__":
    main()
