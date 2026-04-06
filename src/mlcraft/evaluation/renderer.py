"""HTML renderer for evaluation results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import EvaluationResult
from mlcraft.reporting.html import render_d3_card, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_evaluation_context


class EvaluationReportRenderer:
    """Render an evaluation result as a D3-based HTML report."""

    def __init__(self, *, palette: dict[str, str] | None = None) -> None:
        self.palette = get_report_palette(palette)

    def build_context(self, result: EvaluationResult, *, title: str | None = "mlcraft Evaluation Report") -> dict[str, Any]:
        """Build the evaluation view context used by the HTML renderer."""

        return build_evaluation_context(result, title=title)

    def render(self, result: EvaluationResult, *, title: str | None = "mlcraft Evaluation Report", output_path=None) -> str:
        """Render a complete evaluation report."""

        context = self.build_context(result, title=title)
        return self.render_context(context, output_path=output_path)

    def render_context(self, context: dict[str, Any], *, output_path=None) -> str:
        """Render an evaluation report from a pre-built dictionary context."""

        sections: list[str] = []
        if context.get("title"):
            sections.append(f"<h1>{escape(str(context['title']))}</h1>")
        sections.append(self._render_summary_panel(context))
        sections.append(self._render_metric_explorer(context))
        if context.get("curve_groups"):
            sections.append(self._render_curve_explorer(context))
        html = wrap_html(str(context.get("title") or "mlcraft Evaluation Report"), "".join(sections), palette=self.palette)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_summary_panel(self, context: dict[str, Any]) -> str:
        summary = context["summary"]
        leader = summary["leader"]
        if leader is None:
            leader_name = "n/a"
            leader_value = "n/a"
            direction = "No metrics available"
        else:
            leader_name = leader["prediction_name"]
            leader_value = f"{context['primary_metric_name']} = {leader['value']:.6f}"
            direction = "Higher is better" if leader["higher_is_better"] else "Lower is better"
        lead_margin = f"{summary['lead_margin']:.6f}"
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Evaluation Overview</span>"
            f"<h2>{escape(str(context['task_type']).title())} comparison</h2>"
            "<p class='muted'>The report is now driven by in-browser D3 charts so you can change the metric focus without regenerating the document.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Compared models', str(summary['model_count']), 'Prediction bundles in this run.')}"
            f"{self._metric_card('Focus metric', escape(str(context['primary_metric_name'])), direction)}"
            f"{self._metric_card('Leader', escape(leader_name), leader_value)}"
            f"{self._metric_card('Lead margin', lead_margin, 'Gap vs the next best score.')}"
            "</div>"
            "</section>"
        )

    def _render_metric_explorer(self, context: dict[str, Any]) -> str:
        metric_names = []
        for row in context["metric_rows"]:
            if row["metric_name"] not in metric_names:
                metric_names.append(row["metric_name"])
        buttons = []
        panels = []
        for index, metric_name in enumerate(metric_names):
            panel_id = self._metric_panel_id(metric_name)
            default_attr = " data-toggle-default='true'" if index == 0 else ""
            buttons.append(
                f"<button class='segmented-button' type='button' data-toggle-button data-toggle-group='evaluation-metrics' data-toggle-target='{escape(panel_id)}'{default_attr}>{escape(metric_name)}</button>"
            )
            panels.append(self._render_metric_panel(context, metric_name, panel_id))

        heatmap_payload = self._build_heatmap_payload(context)
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Graphical Leaderboard</span>"
            "<h2>Metric explorer</h2>"
            "<p class='muted'>Select any metric to re-rank predictions instantly. The heatmap stays as the dense comparison view across all metrics.</p>"
            "</div>"
            f"<div class='segmented' role='group' aria-label='Evaluation metric selector'>{''.join(buttons)}</div>"
            + "".join(panels)
            + "<div class='viz-grid'>"
            + render_d3_card("Metric Comparison Map", "mountHeatmap", heatmap_payload, wide=True)
            + "</div></section>"
        )

    def _render_metric_panel(self, context: dict[str, Any], metric_name: str, panel_id: str) -> str:
        rows = [row for row in context["metric_rows"] if row["metric_name"] == metric_name]
        rows = sorted(rows, key=lambda row: row["score"], reverse=True)
        chart_payload = {
            "rows": [
                {
                    "label": row["prediction_name"],
                    "value": float(row["value"]),
                    "color": self.palette["accent"] if index == 0 else self.palette["series_muted"],
                }
                for index, row in enumerate(rows)
            ],
            "metricLabel": metric_name,
        }
        return (
            f"<div class='toggle-panel' data-toggle-panel data-toggle-group='evaluation-metrics' data-toggle-panel='{escape(panel_id)}' hidden>"
            "<div class='viz-grid'>"
            + render_d3_card(f"{metric_name} ranking", "mountBarChart", chart_payload, wide=True, chart_id=f"{panel_id}-chart")
            + "</div></div>"
        )

    def _render_curve_explorer(self, context: dict[str, Any]) -> str:
        buttons = []
        panels = []
        for index, group in enumerate(context["curve_groups"]):
            panel_id = self._curve_panel_id(group["curve_name"])
            default_attr = " data-toggle-default='true'" if index == 0 else ""
            buttons.append(
                f"<button class='segmented-button' type='button' data-toggle-button data-toggle-group='evaluation-curves' data-toggle-target='{escape(panel_id)}'{default_attr}>{escape(group['title'])}</button>"
            )
            chart_payload = {
                "xLabel": group["x_label"],
                "yLabel": group["y_label"],
                "series": [
                    {
                        "name": series["prediction_name"],
                        "color": chart_colors(self.palette)[series_index % len(chart_colors(self.palette))],
                        "points": [
                            {"x": float(x_value), "y": float(y_value)}
                            for x_value, y_value in zip(series["x"], series["y"])
                        ],
                    }
                    for series_index, series in enumerate(group["series"])
                ],
            }
            if group["curve_name"] in {"roc", "calibration", "poisson_calibration"}:
                chart_payload["diagonal"] = [[0.0, 0.0], [1.0, 1.0]]
            panels.append(
                f"<div class='toggle-panel' data-toggle-panel data-toggle-group='evaluation-curves' data-toggle-panel='{escape(panel_id)}' hidden>"
                + render_d3_card(group["title"], "mountSeriesChart", chart_payload, wide=True, chart_id=f"{panel_id}-chart")
                + "</div>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Curves on shared axes</span>"
            "<h2>Interactive curve explorer</h2>"
            "</div>"
            f"<div class='segmented' role='group' aria-label='Evaluation curve selector'>{''.join(buttons)}</div>"
            + "".join(panels)
            + "</section>"
        )

    def _build_heatmap_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        metrics_by_prediction = context["metrics_by_prediction"]
        predictions = sorted(
            metrics_by_prediction,
            key=lambda prediction_name: self._primary_score(metrics_by_prediction[prediction_name], context["primary_metric_name"]),
            reverse=True,
        )
        metrics: list[str] = []
        for prediction_name in predictions:
            for row in metrics_by_prediction[prediction_name]:
                if row["metric_name"] not in metrics:
                    metrics.append(row["metric_name"])
        matrix = []
        for prediction_name in predictions:
            row_map = {row["metric_name"]: row["value"] for row in metrics_by_prediction[prediction_name]}
            matrix.append([row_map.get(metric_name, np.nan) for metric_name in metrics])
        return {
            "xLabels": metrics,
            "yLabels": predictions,
            "matrix": matrix,
            "lowColor": self.palette["accent_soft"],
            "highColor": self.palette["accent"],
        }

    def _primary_score(self, rows: list[dict[str, Any]], primary_metric: str) -> float:
        row = next((candidate for candidate in rows if candidate["metric_name"] == primary_metric), None)
        return float(row["score"]) if row is not None else float("-inf")

    def _metric_panel_id(self, metric_name: str) -> str:
        safe = "".join(character if character.isalnum() else "-" for character in str(metric_name).lower()).strip("-")
        return f"evaluation-metric-{safe or 'metric'}"

    def _curve_panel_id(self, curve_name: str) -> str:
        safe = "".join(character if character.isalnum() else "-" for character in str(curve_name).lower()).strip("-")
        return f"evaluation-curve-{safe or 'curve'}"

    def _metric_card(self, title: str, value: str, subtitle: str) -> str:
        return (
            "<div class='card metric-card'>"
            f"<span class='eyebrow'>{title}</span>"
            f"<strong class='metric-big'>{value}</strong>"
            f"<span class='metric-subtle'>{subtitle}</span>"
            "</div>"
        )
