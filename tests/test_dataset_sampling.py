from __future__ import annotations

import tarfile
from pathlib import Path

from cricform.ingest.download_dataset import cricket_shot_dataset_info
from cricform.ingest.sample_dataset import (
    audit_cricket_shot_archive,
    extract_sample_videos,
    infer_split_and_shot_type,
)


def _create_fake_cricket_archive(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    archive_path = tmp_path / "cricketshot.tar.gz"

    files = [
        source_dir / "dataset" / "test" / "cover" / "cover_001.mp4",
        source_dir / "dataset" / "test" / "pull" / "pull_001.mp4",
        source_dir / "dataset" / "train" / "cover" / "cover_002.mp4",
        source_dir / "dataset" / "val" / "sweep" / "sweep_001.mp4",
    ]

    for file_path in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"fake-video-bytes")

    with tarfile.open(archive_path, mode="w:gz") as archive:
        for file_path in files:
            archive.add(file_path, arcname=file_path.relative_to(source_dir))

    return archive_path


def test_cricket_shot_dataset_info() -> None:
    info = cricket_shot_dataset_info()

    assert info.repo_id == "rokmr/cricket-shot"
    assert info.archive_filename == "cricketshot.tar.gz"
    assert info.license == "apache-2.0"
    assert "cover" in info.expected_classes
    assert "sweep" in info.expected_classes


def test_infer_split_and_shot_type() -> None:
    split, shot_type = infer_split_and_shot_type("dataset/test/cover/example.mp4")

    assert split == "test"
    assert shot_type == "cover"


def test_audit_cricket_shot_archive(tmp_path: Path) -> None:
    archive_path = _create_fake_cricket_archive(tmp_path)

    audit = audit_cricket_shot_archive(archive_path)

    assert audit.total_video_files == 4
    assert audit.counts_by_split["test"] == 2
    assert audit.counts_by_shot_type["cover"] == 2
    assert audit.counts_by_split_and_shot_type["test"]["cover"] == 1
    assert len(audit.sample_members) > 0


def test_extract_sample_videos(tmp_path: Path) -> None:
    archive_path = _create_fake_cricket_archive(tmp_path)
    output_dir = tmp_path / "sampled"

    summary = extract_sample_videos(
        archive_path=archive_path,
        output_dir=output_dir,
        split="test",
        samples_per_class=1,
    )

    assert len(summary.extracted_files) == 2

    extracted_paths = [Path(path) for path in summary.extracted_files]

    assert any("cover" in path.parts for path in extracted_paths)
    assert any("pull" in path.parts for path in extracted_paths)

    for path in extracted_paths:
        assert path.exists()
        assert path.read_bytes() == b"fake-video-bytes"
