from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_REPORT_DIR = Path("data/processed/reports")

COMPARISON_METRICS: tuple[str, ...] = (
    "max_backlift_height_proxy",
    "follow_through_height_proxy",
    "mean_head_displacement_proxy",
    "min_front_knee_bend_proxy_deg",
    "mean_shoulder_hip_separation_proxy_deg",
    "mean_wrist_speed_proxy",
    "mean_wrist_path_smoothness_proxy",
)


@dataclass(frozen=True)
class ShotComparisonSummary:
    movement_summary_path: str
    baseline_profile_path: str
    output_comparison_path: str
    shot_type: str
    comparison_status: str
    comparable_metrics: int
    limitation: str


def compare_shot_to_baseline(
    movement_summary: dict[str, Any],
    baseline_profile: dict[str, Any],
    shot_type: str,
) -> dict[str, Any]:
    """Compare one shot movement summary against a shot-type baseline profile."""

    shot_type_baseline = baseline_profile.get("shot_types", {}).get(shot_type)

    if shot_type_baseline is None:
        return _comparison_payload(
            movement_summary=movement_summary,
            baseline_profile=baseline_profile,
            shot_type=shot_type,
            metric_comparisons=[],
            comparison_status="missing_shot_type_baseline",
            notes=[
                f"No baseline exists for shot_type='{shot_type}'.",
                "Comparison requires a baseline built from usable shots of the same type.",
            ],
        )

    metric_comparisons = []

    for metric_name in COMPARISON_METRICS:
        shot_value = _optional_float(movement_summary.get(metric_name))
        baseline_stats = shot_type_baseline.get("features", {}).get(metric_name)

        metric_comparisons.append(
            compare_metric_to_baseline(
                metric_name=metric_name,
                shot_value=shot_value,
                baseline_stats=baseline_stats,
            )
        )

    comparable_metrics = sum(
        comparison["status"] in {"within_baseline", "above_baseline", "below_baseline"}
        for comparison in metric_comparisons
    )

    if int(movement_summary.get("usable_frames", 0) or 0) == 0:
        comparison_status = "insufficient_shot_pose_data"
        notes = [
            "The test shot has zero usable motion frames.",
            "This usually means pose extraction failed or no valid landmarks were available.",
        ]
    elif int(shot_type_baseline.get("usable_shots", 0) or 0) == 0:
        comparison_status = "insufficient_baseline_data"
        notes = [
            "The selected shot-type baseline has zero usable shots.",
            "Add real clips with successful pose extraction before interpreting "
            "comparison results.",
        ]
    elif comparable_metrics == 0:
        comparison_status = "insufficient_metric_overlap"
        notes = [
            "The shot and baseline do not share enough numeric movement metrics.",
        ]
    else:
        comparison_status = "comparison_available"
        notes = [
            "Comparison is available, but metrics are still 2D pose-derived proxies.",
        ]

    return _comparison_payload(
        movement_summary=movement_summary,
        baseline_profile=baseline_profile,
        shot_type=shot_type,
        metric_comparisons=metric_comparisons,
        comparison_status=comparison_status,
        notes=notes,
    )


def compare_metric_to_baseline(
    metric_name: str,
    shot_value: float | None,
    baseline_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare one shot-level metric against baseline distribution stats."""

    if shot_value is None:
        return {
            "metric": metric_name,
            "shot_value": None,
            "baseline_mean": None,
            "baseline_std": None,
            "baseline_count": 0,
            "delta_from_mean": None,
            "z_score": None,
            "status": "shot_metric_unavailable",
            "interpretation": "The shot metric is unavailable.",
        }

    if not baseline_stats:
        return {
            "metric": metric_name,
            "shot_value": shot_value,
            "baseline_mean": None,
            "baseline_std": None,
            "baseline_count": 0,
            "delta_from_mean": None,
            "z_score": None,
            "status": "baseline_metric_unavailable",
            "interpretation": "The baseline does not contain this metric.",
        }

    baseline_mean = _optional_float(baseline_stats.get("mean"))
    baseline_std = _optional_float(baseline_stats.get("std"))
    baseline_count = int(baseline_stats.get("count", 0) or 0)

    if baseline_mean is None or baseline_count <= 0:
        return {
            "metric": metric_name,
            "shot_value": shot_value,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "baseline_count": baseline_count,
            "delta_from_mean": None,
            "z_score": None,
            "status": "baseline_metric_unavailable",
            "interpretation": "The baseline metric has no usable mean/count.",
        }

    delta_from_mean = shot_value - baseline_mean

    if baseline_std is None or baseline_std == 0.0:
        return {
            "metric": metric_name,
            "shot_value": shot_value,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "baseline_count": baseline_count,
            "delta_from_mean": delta_from_mean,
            "z_score": None,
            "status": "baseline_variance_unavailable",
            "interpretation": (
                "The baseline has no variance estimate, so a z-score cannot be computed."
            ),
        }

    z_score = delta_from_mean / baseline_std

    if z_score > 1.0:
        status = "above_baseline"
        interpretation = "Shot value is more than 1 baseline standard deviation above the mean."
    elif z_score < -1.0:
        status = "below_baseline"
        interpretation = "Shot value is more than 1 baseline standard deviation below the mean."
    else:
        status = "within_baseline"
        interpretation = "Shot value is within 1 baseline standard deviation of the mean."

    return {
        "metric": metric_name,
        "shot_value": shot_value,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_count": baseline_count,
        "delta_from_mean": delta_from_mean,
        "z_score": z_score,
        "status": status,
        "interpretation": interpretation,
    }


def write_shot_comparison(
    movement_summary_path: Path,
    baseline_profile_path: Path,
    shot_type: str,
    output_comparison_path: Path | None = None,
) -> ShotComparisonSummary:
    """Write a comparison JSON for one shot against a shot-type baseline."""

    if not movement_summary_path.exists():
        raise FileNotFoundError(f"Movement summary does not exist: {movement_summary_path}")

    if not baseline_profile_path.exists():
        raise FileNotFoundError(f"Baseline profile does not exist: {baseline_profile_path}")

    if output_comparison_path is None:
        output_comparison_path = DEFAULT_REPORT_DIR / "shot_comparison.json"

    output_comparison_path.parent.mkdir(parents=True, exist_ok=True)

    movement_summary = json.loads(movement_summary_path.read_text())
    baseline_profile = json.loads(baseline_profile_path.read_text())

    comparison = compare_shot_to_baseline(
        movement_summary=movement_summary,
        baseline_profile=baseline_profile,
        shot_type=shot_type,
    )

    comparison["source_paths"] = {
        "movement_summary_path": str(movement_summary_path),
        "baseline_profile_path": str(baseline_profile_path),
    }

    output_comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

    comparable_metrics = sum(
        metric["status"] in {"within_baseline", "above_baseline", "below_baseline"}
        for metric in comparison["metric_comparisons"]
    )

    return ShotComparisonSummary(
        movement_summary_path=str(movement_summary_path),
        baseline_profile_path=str(baseline_profile_path),
        output_comparison_path=str(output_comparison_path),
        shot_type=shot_type,
        comparison_status=str(comparison["comparison_status"]),
        comparable_metrics=comparable_metrics,
        limitation=str(comparison["limitation"]),
    )


def _comparison_payload(
    movement_summary: dict[str, Any],
    baseline_profile: dict[str, Any],
    shot_type: str,
    metric_comparisons: list[dict[str, Any]],
    comparison_status: str,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "comparison_version": "shot_vs_baseline_v1",
        "shot_type": shot_type,
        "comparison_status": comparison_status,
        "movement_rows": int(movement_summary.get("rows", 0) or 0),
        "usable_motion_frames": int(movement_summary.get("usable_frames", 0) or 0),
        "baseline_version": baseline_profile.get("baseline_version"),
        "metric_comparisons": metric_comparisons,
        "notes": notes,
        "limitation": (
            "Comparison uses 2D pose-derived movement proxies and a small empirical baseline. "
            "It is not coaching-grade, biomechanics-validated, medical, or injury-risk advice."
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a shot against a baseline profile.")
    parser.add_argument("movement_summary_path", type=Path)
    parser.add_argument("baseline_profile_path", type=Path)
    parser.add_argument("--shot-type", required=True)
    parser.add_argument("--output-comparison", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = write_shot_comparison(
        movement_summary_path=args.movement_summary_path,
        baseline_profile_path=args.baseline_profile_path,
        shot_type=args.shot_type,
        output_comparison_path=args.output_comparison,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
