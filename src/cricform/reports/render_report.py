from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_REPORT_DIR = Path("data/processed/reports")
DEFAULT_SAMPLE_REPORT_DIR = Path("outputs/sample_reports")


@dataclass(frozen=True)
class RenderedReportSummary:
    comparison_json_path: str
    output_markdown_path: str
    output_chart_path: str | None
    comparison_status: str
    chart_created: bool


def render_shot_report(
    comparison_json_path: Path,
    output_markdown_path: Path | None = None,
    output_chart_path: Path | None = None,
) -> RenderedReportSummary:
    """Render a Markdown report and optional metric comparison chart."""

    if not comparison_json_path.exists():
        raise FileNotFoundError(f"Comparison JSON does not exist: {comparison_json_path}")

    comparison = json.loads(comparison_json_path.read_text())

    if output_markdown_path is None:
        output_markdown_path = DEFAULT_REPORT_DIR / "shot_report.md"

    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)

    chart_created = False

    if output_chart_path is not None:
        output_chart_path.parent.mkdir(parents=True, exist_ok=True)
        chart_created = render_metric_comparison_chart(
            comparison=comparison,
            output_chart_path=output_chart_path,
        )

    markdown = _build_markdown_report(
        comparison=comparison,
        chart_path=output_chart_path if chart_created else None,
    )

    output_markdown_path.write_text(markdown)

    return RenderedReportSummary(
        comparison_json_path=str(comparison_json_path),
        output_markdown_path=str(output_markdown_path),
        output_chart_path=str(output_chart_path) if output_chart_path is not None else None,
        comparison_status=str(comparison["comparison_status"]),
        chart_created=chart_created,
    )


def render_metric_comparison_chart(
    comparison: dict[str, Any],
    output_chart_path: Path,
) -> bool:
    """Render a simple shot-vs-baseline mean chart when enough data exists."""

    rows = [
        metric
        for metric in comparison.get("metric_comparisons", [])
        if metric.get("shot_value") is not None and metric.get("baseline_mean") is not None
    ]

    if not rows:
        return False

    metric_names = [_short_metric_name(row["metric"]) for row in rows]
    shot_values = [float(row["shot_value"]) for row in rows]
    baseline_means = [float(row["baseline_mean"]) for row in rows]

    x_positions = range(len(metric_names))
    bar_width = 0.38

    plt.figure(figsize=(max(8, len(metric_names) * 1.25), 5))
    plt.bar([x - bar_width / 2 for x in x_positions], shot_values, width=bar_width, label="Shot")
    plt.bar(
        [x + bar_width / 2 for x in x_positions],
        baseline_means,
        width=bar_width,
        label="Baseline mean",
    )
    plt.xticks(list(x_positions), metric_names, rotation=35, ha="right")
    plt.ylabel("Proxy value")
    plt.title("Shot vs baseline movement proxies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=160)
    plt.close()

    return True


def _build_markdown_report(
    comparison: dict[str, Any],
    chart_path: Path | None,
) -> str:
    lines = [
        "# CricForm Vision Lab Shot Report",
        "",
        "## Summary",
        "",
        f"- Shot type: `{comparison['shot_type']}`",
        f"- Comparison status: `{comparison['comparison_status']}`",
        f"- Movement rows: `{comparison['movement_rows']}`",
        f"- Usable motion frames: `{comparison['usable_motion_frames']}`",
        f"- Baseline version: `{comparison.get('baseline_version')}`",
        "",
    ]

    if chart_path is not None:
        lines.extend(
            [
                "## Metric comparison chart",
                "",
                f"![Shot vs baseline metric comparison]({chart_path})",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
        ]
    )

    for note in comparison.get("notes", []):
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Metric comparisons",
            "",
            "| Metric | Shot value | Baseline mean | Baseline std | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )

    metric_comparisons = comparison.get("metric_comparisons", [])

    if not metric_comparisons:
        lines.append("| No comparable metrics |  |  |  | insufficient data |")
    else:
        for metric in metric_comparisons:
            lines.append(
                "| "
                f"{metric['metric']} | "
                f"{_format_number(metric.get('shot_value'))} | "
                f"{_format_number(metric.get('baseline_mean'))} | "
                f"{_format_number(metric.get('baseline_std'))} | "
                f"{metric['status']} |"
            )

    lines.extend(
        [
            "",
            "## Honest limitations",
            "",
            comparison["limitation"],
            "",
            "This report is generated from pose-derived proxy features. It should be used "
            "to inspect pipeline behavior, signal quality, and relative feature patterns. "
            "It should not be used as professional cricket coaching, medical, "
            "biomechanics, or injury-risk advice.",
            "",
        ]
    )

    return "\n".join(lines)


def _short_metric_name(metric_name: str) -> str:
    replacements = {
        "_proxy": "",
        "_deg": "",
        "mean_": "",
        "max_": "",
        "min_": "",
    }

    short_name = metric_name

    for old, new in replacements.items():
        short_name = short_name.replace(old, new)

    return short_name.replace("_", " ")


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a CricForm shot report.")
    parser.add_argument("comparison_json_path", type=Path)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--output-chart", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = render_shot_report(
        comparison_json_path=args.comparison_json_path,
        output_markdown_path=args.output_markdown,
        output_chart_path=args.output_chart,
    )

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
