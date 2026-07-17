# CricForm Vision Lab Release Checklist

This checklist records the project health criteria for the v0.1.0 portfolio milestone.

## Code health

- [ ] make test passes.
- [ ] make lint passes.
- [ ] make check passes.
- [ ] GitHub Actions CI passes on main.
- [ ] No generated artifacts are staged.
- [ ] git status is clean after commit and merge.

## Demo health

- [ ] make slow-real-demo runs locally.
- [ ] outputs/real_demo/test_pull_pull_0025_pose_overlay_slow.mp4 is generated locally.
- [ ] make app opens the Streamlit UI.
- [ ] Real cricket demo mode is selected by default.
- [ ] Overlay video tab shows the slow real overlay.
- [ ] Report tab shows the metric chart and Markdown report.
- [ ] Pose audit tab shows 10 sampled real cricket clips.
- [ ] Synthetic fallback mode still works.

## Documentation health

- [ ] README has a clear one-sentence pitch.
- [ ] README includes the real demo snapshot.
- [ ] README states honest limitations.
- [ ] docs/demo_walkthrough.md explains how to demo the project.
- [ ] CHANGELOG.md describes the v0.1.0 milestone.

## Artifact policy

The following should remain ignored and uncommitted:

- data/raw/datasets/cricketshot.tar.gz
- data/raw/videos/hf_cricket_shot/
- data/interim/pose_landmarks/real_samples/
- data/processed/real_sample_pose_audit/
- data/processed/real_demo/
- outputs/real_demo/
- models/pose_landmarker/*.task

## Release tag

After checks pass and main is clean:

    git tag -a v0.1.0 -m "CricForm Vision Lab real-demo milestone"
    git push origin v0.1.0
