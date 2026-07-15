from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BASELINE_DIR = Path("data/processed/baselines")
REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "shot_type",
    "movement_features_csv",
)

BASELINE_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("max_backlift_height_proxy", "backlift_height_proxy", "max"),
    ("follow_through_height_proxy", "follow_through_height_proxy", "max"),
    ("mean_head_displacement_proxy", "head_displacement_proxy", "mean"),
    ("min_front_knee_bend_proxy_deg", "front_knee_bend_proxy_deg", "min"),
    (
        "mean_shoulder_hip_separation_proxy_deg",
        "shoulder_hip_separation_proxy_deg",
        "mean",
    ),
    ("mean_wrist_speed_proxy", "wrist_speed_proxy", "mean"),
    ("mean_wrist_path_smoothness_proxy", "wrist_path_smoothness_proxy", "mean"),
)


@dataclass(frozen=True)
class BaselineBuildSummary:
    manifest_path: str
    output_profile_path: str
    total_manifest_rows: int
    usable_shots: int
    shot_types: list[str]
    baseline_version: str
    limitation: str


def load_baseline_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load and validate a baseline manifest CSV."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Baseline manifest does not exist: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    missing_columns = [
        column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")

    manifest = manifest[list(REQUIRED_MANIFEST_COLUMNS)].copy()
    manifest["shot_id"] = manifest["shot_id"].astype(str)
    manifest["shot_type"] = manifest["shot_type"].astype(str)
    manifest["movement_features_csv"] = manifest["movement_features_csv"].astype(str)

    return manifest


def build_baseline_profile(
    manifest: pd.DataFrame,
    manifest_base_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a small shot-type baseline profile from movement feature CSVs."""

    if manifest_base_dir is None:
        manifest_base_dir = Path(".")

    _validate_manifest_dataframe(manifest)

    shot_summaries: list[dict[str, Any]] = []

    for row in manifest.to_dict(orient="records"):
        movement_features_path = _resolve_features_path(
            str(row["movement_features_csv"]),
            manifest_base_dir,
        )

        shot_summary = summarize_shot_features(
            shot_id=str(row["shot_id"]),
            shot_type=str(row["shot_type"]),
            movement_features_path=movement_features_path,
        )
        shot_summaries.append(shot_summary)

    return _aggregate_shot_type_baselines(shot_summaries)


def summarize_shot_features(
    shot_id: str,
    shot_type: str,
    movement_features_path: Path,
) -> dict[str, Any]:
    """Summarize one shot's movement-feature CSV into shot-level metrics."""

    if not movement_features_path.exists():
        raise FileNotFoundError(f"Movement feature CSV does not exist: {movement_features_path}")

    features = pd.read_csv(movement_features_path)

    summary: dict[str, Any] = {
        "shot_id": shot_id,
        "shot_type": shot_type,
        "movement_features_csv": str(movement_features_path),
        "rows": int(len(features)),
        "usable_frames": 0,
        "is_usable": False,
        "metrics": {},
    }

    if features.empty:
        return summary

    if "usable_for_motion" in features.columns:
        usable = features[features["usable_for_motion"].astype(bool)].copy()
    else:
        usable = features.copy()

    summary["usable_frames"] = int(len(usable))

    if usable.empty:
        return summary

    metric_values: dict[str, float | None] = {}

    for metric_name, source_column, aggregation in BASELINE_METRIC_SPECS:
        metric_values[metric_name] = _aggregate_series(
            usable[source_column] if source_column in usable.columns else pd.Series(dtype=float),
            aggregation,
        )

    non_null_metric_count = sum(value is not None for value in metric_values.values())

    summary["is_usable"] = non_null_metric_count > 0
    summary["metrics"] = metric_values

    return summary


def write_baseline_profile(
    manifest_path: Path,
    output_profile_path: Path | None = None,
) -> BaselineBuildSummary:
    """Build and write a baseline profile JSON from a manifest CSV."""

    manifest = load_baseline_manifest(manifest_path)

    if output_profile_path is None:
        output_profile_path = DEFAULT_BASELINE_DIR / "baseline_profile.json"

    output_profile_path.parent.mkdir(parents=True, exist_ok=True)

    profile = build_baseline_profile(
        manifest=manifest,
        manifest_base_dir=manifest_path.parent,
    )

    profile["source_manifest_path"] = str(manifest_path)
    output_profile_path.write_text(json.dumps(profile, indent=2) + "\n")

    return BaselineBuildSummary(
        manifest_path=str(manifest_path),
        output_profile_path=str(output_profile_path),
        total_manifest_rows=int(profile["total_manifest_rows"]),
        usable_shots=int(profile["usable_shots"]),
        shot_types=list(profile["shot_types"].keys()),
        baseline_version=str(profile["baseline_version"]),
        limitation=str(profile["limitation"]),
    )


def _aggregate_shot_type_baselines(
    shot_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    shot_types: dict[str, Any] = {}

    for shot_summary in shot_summaries:
        shot_type = str(shot_summary["shot_type"])
        shot_types.setdefault(
            shot_type,
            {
                "manifest_shots": 0,
                "usable_shots": 0,
                "features": {},
                "shot_ids": [],
            },
        )

        shot_type_record = shot_types[shot_type]
        shot_type_record["manifest_shots"] += 1
        shot_type_record["shot_ids"].append(str(shot_summary["shot_id"]))

        if not shot_summary["is_usable"]:
            continue

        shot_type_record["usable_shots"] += 1

        for metric_name, value in shot_summary["metrics"].items():
            if value is None:
                continue

            shot_type_record["features"].setdefault(metric_name, []).append(float(value))

    for shot_type_record in shot_types.values():
        feature_lists = shot_type_record["features"]
        shot_type_record["features"] = {
            metric_name: _distribution_stats(values)
            for metric_name, values in feature_lists.items()
        }

    usable_shots = sum(1 for shot_summary in shot_summaries if shot_summary["is_usable"])

    return {
        "baseline_version": "movement_proxy_baseline_v1",
        "total_manifest_rows": len(shot_summaries),
        "usable_shots": usable_shots,
        "shot_types": shot_types,
        "shot_summaries": shot_summaries,
        "limitation": (
            "Baseline values are computed from the provided movement-feature files only. "
            "They are not population norms, coaching standards, or validated biomechanics."
        ),
    }


def _distribution_stats(values: list[float]) -> dict[str, float | int | None]:
    series = pd.Series(values, dtype=float)

    return {
        "count": int(series.count()),
        "mean": _float_or_none(series.mean(skipna=True)),
        "std": _float_or_none(series.std(skipna=True)),
        "min": _float_or_none(series.min(skipna=True)),
        "max": _float_or_none(series.max(skipna=True)),
    }


def _aggregate_series(series: pd.Series, aggregation: str) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        return None

    if aggregation == "mean":
        return float(numeric.mean())

    if aggregation == "max":
        return float(numeric.max())

    if aggregation == "min":
        return float(numeric.min())

    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None

    return float(value)


def _resolve_features_path(path_value: str, manifest_base_dir: Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return manifest_base_dir / path


def _validate_manifest_dataframe(manifest: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build movement-feature baseline profiles.")
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--output-profile", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = write_baseline_profile(
        manifest_path=args.manifest_path,
        output_profile_path=args.output_profile,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
