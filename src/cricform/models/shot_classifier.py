from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cricform.baseline.build_baseline import (
    BASELINE_METRIC_SPECS,
    load_baseline_manifest,
    summarize_shot_features,
)

DEFAULT_OUTPUT_DIR = Path("data/processed/real_demo/classification")

CLASSIFIER_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    metric_name for metric_name, _, _ in BASELINE_METRIC_SPECS
)


@dataclass(frozen=True)
class ShotClassifierEvaluationSummary:
    baseline_manifest_path: str
    pose_audit_csv_path: str
    output_dir: str
    output_feature_table_path: str
    output_predictions_path: str
    output_metrics_path: str
    output_confusion_matrix_csv_path: str
    output_confusion_matrix_png_path: str
    evaluation_method: str
    evaluation_status: str
    samples: int
    classes: list[str]
    accuracy: float | None
    limitation: str


def load_labelled_real_sample_manifest(
    baseline_manifest_path: Path,
    pose_audit_csv_path: Path,
) -> pd.DataFrame:
    """Join real-demo feature paths with actual shot labels from the pose audit."""

    baseline_manifest = load_baseline_manifest(baseline_manifest_path)

    if not pose_audit_csv_path.exists():
        raise FileNotFoundError(f"Pose audit CSV does not exist: {pose_audit_csv_path}")

    pose_audit = pd.read_csv(pose_audit_csv_path)

    required_audit_columns = {"video_id", "shot_type"}
    missing = sorted(required_audit_columns - set(pose_audit.columns))
    if missing:
        raise ValueError(f"Pose audit CSV missing required columns: {missing}")

    labels = (
        pose_audit[["video_id", "shot_type"]]
        .dropna(subset=["video_id", "shot_type"])
        .drop_duplicates(subset=["video_id"])
        .copy()
    )
    labels["video_id"] = labels["video_id"].astype(str)
    labels["shot_type"] = labels["shot_type"].astype(str)

    labelled = baseline_manifest.merge(
        labels,
        left_on="shot_id",
        right_on="video_id",
        how="left",
        suffixes=("_baseline", ""),
    )

    if labelled["shot_type"].isna().any():
        missing_ids = labelled.loc[labelled["shot_type"].isna(), "shot_id"].tolist()
        raise ValueError(f"Missing shot labels for manifest rows: {missing_ids}")

    return labelled[["shot_id", "shot_type", "movement_features_csv"]].copy()


def build_labelled_feature_table(
    labelled_manifest: pd.DataFrame,
    manifest_base_dir: Path,
) -> pd.DataFrame:
    """Convert per-video movement CSVs into one shot-level feature table."""

    rows: list[dict[str, Any]] = []

    for row in labelled_manifest.to_dict(orient="records"):
        movement_features_path = _resolve_features_path(
            str(row["movement_features_csv"]),
            manifest_base_dir,
        )

        shot_summary = summarize_shot_features(
            shot_id=str(row["shot_id"]),
            shot_type=str(row["shot_type"]),
            movement_features_path=movement_features_path,
        )

        if not shot_summary["is_usable"]:
            continue

        output_row: dict[str, Any] = {
            "shot_id": shot_summary["shot_id"],
            "shot_type": shot_summary["shot_type"],
            "movement_features_csv": str(movement_features_path),
            "rows": shot_summary["rows"],
            "usable_frames": shot_summary["usable_frames"],
        }

        for feature_name in CLASSIFIER_FEATURE_COLUMNS:
            output_row[feature_name] = shot_summary["metrics"].get(feature_name)

        rows.append(output_row)

    return pd.DataFrame(
        rows,
        columns=[
            "shot_id",
            "shot_type",
            "movement_features_csv",
            "rows",
            "usable_frames",
            *CLASSIFIER_FEATURE_COLUMNS,
        ],
    )


def evaluate_leave_one_out_nearest_neighbor(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate a transparent nearest-neighbor shot classifier."""

    if feature_table.empty:
        metrics = {
            "evaluation_status": "no_usable_features",
            "evaluation_method": "leave_one_out_nearest_neighbor_on_movement_proxies",
            "samples": 0,
            "classes": [],
            "class_counts": {},
            "accuracy": None,
            "limitation": _classification_limitation(),
        }
        return pd.DataFrame(), pd.DataFrame(), metrics

    labels = sorted(feature_table["shot_type"].astype(str).unique().tolist())
    class_counts = {
        str(label): int(count)
        for label, count in feature_table["shot_type"].value_counts().sort_index().items()
    }

    numeric_features = feature_table[list(CLASSIFIER_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    usable_feature_columns = [
        column for column in numeric_features.columns if numeric_features[column].notna().any()
    ]

    if len(feature_table) < 2 or not usable_feature_columns:
        metrics = {
            "evaluation_status": "insufficient_samples_or_features",
            "evaluation_method": "leave_one_out_nearest_neighbor_on_movement_proxies",
            "samples": int(len(feature_table)),
            "classes": labels,
            "class_counts": class_counts,
            "accuracy": None,
            "limitation": _classification_limitation(),
        }
        confusion = pd.DataFrame(0, index=labels, columns=labels)
        return pd.DataFrame(), confusion, metrics

    matrix_values = numeric_features[usable_feature_columns].copy()
    matrix_values = matrix_values.fillna(matrix_values.mean(numeric_only=True))

    means = matrix_values.mean()
    stds = matrix_values.std(ddof=0).replace(0, 1)
    scaled = ((matrix_values - means) / stds).to_numpy(dtype=float)

    predictions: list[dict[str, Any]] = []

    for index in range(len(feature_table)):
        distances = np.linalg.norm(scaled - scaled[index], axis=1)
        distances[index] = np.inf

        nearest_index = int(np.argmin(distances))
        current = feature_table.iloc[index]
        nearest = feature_table.iloc[nearest_index]

        predictions.append(
            {
                "shot_id": current["shot_id"],
                "true_shot_type": current["shot_type"],
                "predicted_shot_type": nearest["shot_type"],
                "nearest_neighbor_shot_id": nearest["shot_id"],
                "distance": float(distances[nearest_index]),
            }
        )

    predictions_df = pd.DataFrame(predictions)
    confusion = pd.DataFrame(0, index=labels, columns=labels)

    for row in predictions:
        confusion.loc[str(row["true_shot_type"]), str(row["predicted_shot_type"])] += 1

    accuracy = float(
        (
            predictions_df["true_shot_type"]
            == predictions_df["predicted_shot_type"]
        ).mean()
    )

    min_class_count = min(class_counts.values()) if class_counts else 0
    status = (
        "ok"
        if min_class_count >= 2
        else "insufficient_class_replicates_for_reliable_evaluation"
    )

    metrics = {
        "evaluation_status": status,
        "evaluation_method": "leave_one_out_nearest_neighbor_on_movement_proxies",
        "samples": int(len(feature_table)),
        "classes": labels,
        "class_counts": class_counts,
        "feature_columns": usable_feature_columns,
        "accuracy": accuracy,
        "limitation": _classification_limitation(),
    }

    return predictions_df, confusion, metrics


def write_shot_classifier_evaluation(
    baseline_manifest_path: Path,
    pose_audit_csv_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ShotClassifierEvaluationSummary:
    """Write shot-classification feature table, predictions, metrics, and matrix."""

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table_path = output_dir / "shot_classifier_feature_table.csv"
    predictions_path = output_dir / "shot_classifier_predictions.csv"
    metrics_path = output_dir / "shot_classifier_metrics.json"
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    confusion_png_path = output_dir / "confusion_matrix.png"

    labelled_manifest = load_labelled_real_sample_manifest(
        baseline_manifest_path=baseline_manifest_path,
        pose_audit_csv_path=pose_audit_csv_path,
    )

    feature_table = build_labelled_feature_table(
        labelled_manifest=labelled_manifest,
        manifest_base_dir=baseline_manifest_path.parent,
    )
    feature_table.to_csv(feature_table_path, index=False)

    predictions, confusion, metrics = evaluate_leave_one_out_nearest_neighbor(
        feature_table
    )

    predictions.to_csv(predictions_path, index=False)
    confusion.to_csv(confusion_csv_path)
    _render_confusion_matrix(confusion, confusion_png_path)

    metrics["output_feature_table_path"] = str(feature_table_path)
    metrics["output_predictions_path"] = str(predictions_path)
    metrics["output_confusion_matrix_csv_path"] = str(confusion_csv_path)
    metrics["output_confusion_matrix_png_path"] = str(confusion_png_path)

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    return ShotClassifierEvaluationSummary(
        baseline_manifest_path=str(baseline_manifest_path),
        pose_audit_csv_path=str(pose_audit_csv_path),
        output_dir=str(output_dir),
        output_feature_table_path=str(feature_table_path),
        output_predictions_path=str(predictions_path),
        output_metrics_path=str(metrics_path),
        output_confusion_matrix_csv_path=str(confusion_csv_path),
        output_confusion_matrix_png_path=str(confusion_png_path),
        evaluation_method=str(metrics["evaluation_method"]),
        evaluation_status=str(metrics["evaluation_status"]),
        samples=int(metrics["samples"]),
        classes=list(metrics["classes"]),
        accuracy=metrics["accuracy"],
        limitation=str(metrics["limitation"]),
    )


def _render_confusion_matrix(confusion: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(confusion.columns) * 0.75), 5))

    if confusion.empty:
        ax.text(0.5, 0.5, "No confusion matrix available", ha="center", va="center")
        ax.axis("off")
    else:
        ax.imshow(confusion.to_numpy(dtype=float))
        ax.set_xticks(range(len(confusion.columns)))
        ax.set_xticklabels(confusion.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(confusion.index)))
        ax.set_yticklabels(confusion.index)
        ax.set_xlabel("Predicted shot type")
        ax.set_ylabel("True shot type")
        ax.set_title("Shot classification confusion matrix")

        for row_index, true_label in enumerate(confusion.index):
            for column_index, predicted_label in enumerate(confusion.columns):
                value = int(confusion.loc[true_label, predicted_label])
                ax.text(column_index, row_index, str(value), ha="center", va="center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _resolve_features_path(path_value: str, manifest_base_dir: Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return manifest_base_dir / path


def _classification_limitation() -> str:
    return (
        "This classifier is a small movement-proxy baseline using MediaPipe-derived "
        "2D features. It is not a coaching-grade, biomechanics-validated, or "
        "production shot-recognition model. With one sample per class, results are "
        "only a feature-space diagnostic, not reliable generalization evidence."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a simple shot classifier on real-sample movement features."
    )
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--pose-audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = write_shot_classifier_evaluation(
        baseline_manifest_path=args.baseline_manifest,
        pose_audit_csv_path=args.pose_audit_csv,
        output_dir=args.output_dir,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
