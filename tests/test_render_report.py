from __future__ import annotations

import json
from pathlib import Path

from cricform.reports.render_report import render_metric_comparison_chart, render_shot_report


def _comparison_payload() -> dict:
    return {
        "comparison_version": "shot_vs_baseline_v1",
        "shot_type": "cover",
        "comparison_status": "comparison_available",
        "movement_rows": 10,
        "usable_motion_frames": 10,
        "baseline_version": "movement_proxy_baseline_v1",
        "metric_comparisons": [
            {
                "metric": "max_backlift_height_proxy",
                "shot_value": 0.30,
                "baseline_mean": 0.25,
                "baseline_std": 0.05,
                "baseline_count": 2,
                "delta_from_mean": 0.05,
                "z_score": 1.0,
                "status": "within_baseline",
                "interpretation": "Within range.",
            }
        ],
        "notes": ["Comparison is available."],
        "limitation": "Not coaching-grade.",
    }


def test_render_metric_comparison_chart(tmp_path: Path) -> None:
    chart_path = tmp_path / "chart.png"

    created = render_metric_comparison_chart(
        comparison=_comparison_payload(),
        output_chart_path=chart_path,
    )

    assert created is True
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0


def test_render_metric_comparison_chart_skips_when_no_data(tmp_path: Path) -> None:
    chart_path = tmp_path / "chart.png"
    comparison = _comparison_payload()
    comparison["metric_comparisons"] = []

    created = render_metric_comparison_chart(
        comparison=comparison,
        output_chart_path=chart_path,
    )

    assert created is False
    assert not chart_path.exists()


def test_render_shot_report(tmp_path: Path) -> None:
    comparison_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "report.md"
    chart_path = tmp_path / "chart.png"

    comparison_path.write_text(json.dumps(_comparison_payload()))

    summary = render_shot_report(
        comparison_json_path=comparison_path,
        output_markdown_path=markdown_path,
        output_chart_path=chart_path,
    )

    assert markdown_path.exists()
    assert chart_path.exists()
    assert summary.chart_created is True
    assert summary.comparison_status == "comparison_available"

    markdown = markdown_path.read_text()
    assert "# CricForm Vision Lab Shot Report" in markdown
    assert "Honest limitations" in markdown
    assert "max_backlift_height_proxy" in markdown
