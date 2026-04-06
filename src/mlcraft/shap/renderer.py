"""HTML renderer for SHAP results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import ShapResult
from mlcraft.reporting.html import render_d3_card, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_shap_context


class ShapReportRenderer:
    """Render SHAP artifacts as an interactive D3 dashboard."""

    def __init__(self, *, palette: dict[str, str] | None = None) -> None:
        self.palette = get_report_palette(palette)

    def build_context(self, result: ShapResult, *, title: str | None = "mlcraft SHAP Report", top_n: int = 10) -> dict[str, Any]:
        """Build the SHAP view context used by the HTML renderer."""

        return build_shap_context(result, title=title, top_n=top_n)

    def render(self, result: ShapResult, *, title: str | None = "mlcraft SHAP Report", output_path=None, top_n: int = 10) -> str:
        """Render a complete SHAP report."""

        context = self.build_context(result, title=title, top_n=top_n)
        return self.render_context(context, output_path=output_path)

    def render_context(self, context: dict[str, Any], *, output_path=None) -> str:
        """Render a SHAP report from a pre-built dictionary context."""

        sections: list[str] = []
        if context.get("title"):
            sections.append(f"<h1>{escape(str(context['title']))}</h1>")
        sections.append(self._render_summary_panel(context))
        sections.append(self._render_importance_section(context))
        sections.append(self._render_scatter_section(context))
        if context.get("interaction_matrix") is not None:
            sections.append(self._render_interaction_section(context))
        html = wrap_html(str(context.get("title") or "mlcraft SHAP Report"), "".join(sections), palette=self.palette)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_summary_panel(self, context: dict[str, Any]) -> str:
        ordered = context.get("ordered_feature_names") or []
        top_feature = ordered[0] if ordered else "n/a"
        sample_count = int(np.asarray(context["shap_values"]).shape[0])
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Explainability Overview</span>"
            "<h2>SHAP feature perspective</h2>"
            "<p class='muted'>The report stays model-agnostic and interactive: importance, distribution, scatter, and interactions all come from the serialized SHAP payload.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Samples', str(sample_count), 'Rows available in the SHAP payload.')}"
            f"{self._metric_card('Feature count', str(context['feature_count']), 'Features available in the SHAP payload.')}"
            f"{self._metric_card('Top feature', escape(top_feature), 'Highest mean absolute SHAP contribution.')}"
            f"{self._metric_card('Top features shown', str(context['top_n']), 'Maximum features used in summary views.')}"
            "</div>"
            "</section>"
        )

    def _render_importance_section(self, context: dict[str, Any]) -> str:
        colors = chart_colors(self.palette)
        importance_payload = {
            "rows": [
                {"label": name, "value": float(value), "color": colors[index % len(colors)]}
                for index, (name, value) in enumerate(zip(context["top_feature_names"], context["importance_values"]))
            ],
            "metricLabel": "mean(|SHAP|)",
        }
        beeswarm_payload = {
            "groups": [
                {
                    "label": feature_payload["feature_name"],
                    "values": [float(value) for value in feature_payload["values"]],
                    "color": colors[index % len(colors)],
                }
                for index, feature_payload in enumerate(context["beeswarm"])
            ]
        }
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Feature Structure</span>"
            "<h2>Importance and distribution</h2>"
            "<p class='muted'>Use the ranking to understand global magnitude, then read the beeswarm for the signed SHAP spread feature by feature.</p>"
            "</div>"
            "<div class='two-column-grid'>"
            + render_d3_card("Mean absolute SHAP importance", "mountBarChart", importance_payload, chart_id="shap-importance")
            + render_d3_card("SHAP beeswarm", "mountBeeswarm", beeswarm_payload, chart_id="shap-beeswarm")
            + "</div>"
            "</section>"
        )

    def _render_scatter_section(self, context: dict[str, Any]) -> str:
        scatter_plots = context.get("scatter_plots") or []
        if not scatter_plots:
            return ""
        buttons = []
        panels = []
        for index, scatter in enumerate(scatter_plots):
            panel_id = self._scatter_panel_id(scatter["feature_name"])
            default_attr = " data-toggle-default='true'" if index == 0 else ""
            buttons.append(
                f"<button class='segmented-button' type='button' data-toggle-button data-toggle-group='shap-scatter' data-toggle-target='{escape(panel_id)}'{default_attr}>{escape(scatter['feature_name'])}</button>"
            )
            payload = {
                "points": [{"x": float(x_value), "y": float(y_value)} for x_value, y_value in zip(scatter["x"], scatter["y"])],
                "xLabel": scatter["feature_name"],
                "yLabel": "SHAP value",
                "label": scatter["feature_name"],
                "color": self.palette["series_2"],
            }
            panels.append(
                f"<div class='toggle-panel' data-toggle-panel data-toggle-group='shap-scatter' data-toggle-panel='{escape(panel_id)}' hidden>"
                + render_d3_card(f"SHAP scatter - {scatter['feature_name']}", "mountScatter", payload, wide=True, chart_id=f"{panel_id}-chart")
                + "</div>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Dependence Explorer</span>"
            "<h2>Inspect feature-level SHAP scatter views</h2>"
            "</div>"
            f"<div class='segmented' role='group' aria-label='SHAP scatter selector'>{''.join(buttons)}</div>"
            + "".join(panels)
            + "</section>"
        )

    def _render_interaction_section(self, context: dict[str, Any]) -> str:
        payload = {
            "xLabels": context["top_feature_names"],
            "yLabels": context["top_feature_names"],
            "matrix": context["interaction_matrix"],
            "lowColor": self.palette["interaction_low"],
            "highColor": self.palette["interaction_high"],
        }
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Interaction Structure</span>"
            "<h2>Mean absolute SHAP interaction strength</h2>"
            "</div>"
            "<div class='viz-grid'>"
            + render_d3_card("SHAP interaction heatmap", "mountHeatmap", payload, wide=True, chart_id="shap-interactions")
            + "</div>"
            "</section>"
        )

    def _scatter_panel_id(self, feature_name: str) -> str:
        safe = "".join(character if character.isalnum() else "-" for character in str(feature_name).lower()).strip("-")
        return f"shap-scatter-{safe or 'feature'}"

    def _metric_card(self, title: str, value: str, subtitle: str) -> str:
        return (
            "<div class='card metric-card'>"
            f"<span class='eyebrow'>{title}</span>"
            f"<strong class='metric-big'>{value}</strong>"
            f"<span class='metric-subtle'>{subtitle}</span>"
            "</div>"
        )
