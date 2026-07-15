from __future__ import annotations

from pathlib import Path

import pandas as pd

from cricform.pose.batch_pose_audit import (
    discover_sample_videos,
    infer_sample_video_labels,
    make_video_id,
    run_batch_pose_audit,
)


def _create_sample_video_tree(tmp_path: Path) -> Path:
    sample_dir = tmp_path / "hf_cricket_shot"

    real_files = [
        sample_dir / "test" / "cover" / "cover_001.avi",
        sample_dir / "test" / "pull" / "pull_001.avi",
    ]
    ignored_files = [
        sample_dir / "test" / "cover" / "._cover_001.avi",
        sample_dir / "__MACOSX" / "test" / "pull" / "pull_001.avi",
        sample_dir / "test" / "cover" / "notes.txt",
    ]

    for path in real_files + ignored_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-content")

    return sample_dir


def test_discover_sample_videos_ignores_sidecars(tmp_path: Path) -> None:
    sample_dir = _create_sample_video_tree(tmp_path)

    videos = discover_sample_videos(sample_dir)

    assert len(videos) == 2
    assert all(not path.name.startswith("._") for path in videos)
    assert all("__MACOSX" not in path.parts for path in videos)


def test_infer_sample_video_labels(tmp_path: Path) -> None:
    sample_dir = _create_sample_video_tree(tmp_path)
    video_path = sample_dir / "test" / "cover" / "cover_001.avi"

    split, shot_type = infer_sample_video_labels(video_path, sample_dir)

    assert split == "test"
    assert shot_type == "cover"


def test_make_video_id_is_filesystem_safe() -> None:
    video_id = make_video_id(
        split="test",
        shot_type="late cut",
        video_path=Path("late cut/example clip 01.avi"),
    )

    assert video_id == "test_late_cut_example_clip_01"


def test_run_batch_pose_audit_with_fake_runner(tmp_path: Path) -> None:
    sample_dir = _create_sample_video_tree(tmp_path)
    model_path = tmp_path / "pose.task"
    output_dir = tmp_path / "audit"
    pose_jsonl_dir = tmp_path / "pose_jsonl"

    model_path.write_bytes(b"fake-model")

    def fake_runner(
        video_path: Path,
        model_path: Path,
        output_jsonl_path: Path,
        every_n_frames: int,
        max_frames: int,
    ) -> dict[str, int]:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl_path.write_text("")
        frames_with_pose = 3 if "cover" in video_path.parts else 0
        return {
            "frames_seen": 20,
            "frames_processed": 10,
            "frames_with_pose": frames_with_pose,
        }

    summary = run_batch_pose_audit(
        sample_dir=sample_dir,
        model_path=model_path,
        output_dir=output_dir,
        pose_jsonl_dir=pose_jsonl_dir,
        every_n_frames=5,
        max_frames=10,
        pose_runner=fake_runner,
    )

    assert summary.total_videos == 2
    assert summary.attempted_videos == 2
    assert summary.failed_videos == 0
    assert summary.videos_with_pose == 1
    assert summary.best_video is not None
    assert summary.best_video["shot_type"] == "cover"

    csv_path = output_dir / "pose_audit.csv"
    dataframe = pd.read_csv(csv_path)

    assert len(dataframe) == 2
    assert set(dataframe["status"]) == {"pose_detected", "no_pose_detected"}
