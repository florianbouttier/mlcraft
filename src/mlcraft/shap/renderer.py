"""HTML renderer for SHAP results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import ShapResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_shap_context


class ShapReportRenderer:
    """Render SHAP artifacts with a lightweight visual dashboard."""

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

        import matplotlib.pyplot as plt

        sections: list[str] = []
        if context.get("title"):
            sections.append(f"<h1>{escape(str(context['title']))}</h1>")
        sections.append(self._render_summary_panel(context))
        sections.append("<section class='panel section-stack'>")
        sections.append("<div><span class='eyebrow'>Feature Structure</span><h2>Importance and distribution</h2><p class='muted'>The SHAP report keeps the same reporting palette and consumes a serializable context before plotting.</p></div>")
        sections.append("<div class='viz-grid viz-grid--compact'>")
        for title, figure in self._figures(context):
            sections.append(self._figure_card(title, figure))
            plt.close(figure)
        sections.append("</div></section>")
        html = wrap_html(str(context.get("title") or "mlcraft SHAP Report"), "".join(sections), palette=self.palette)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_summary_panel(self, context: dict[str, Any]) -> str:
        top_feature = context["top_feature_names"][0] if context["top_feature_names"] else "n/a"
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Explainability Overview</span>"
            "<h2>SHAP feature perspective</h2>"
            "<p class='muted'>Summary plots and top feature scatters are generated from a context dictionary, then styled consistently with the rest of the reporting stack.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Feature count', str(context['feature_count']), 'Features available in the SHAP payload.')}"
            f"{self._metric_card('Top feature', escape(top_feature), 'Highest mean absolute SHAP contribution.')}"
            f"{self._metric_card('Top features shown', str(context['top_n']), 'Maximum features used in summary views.')}"
            "</div>"
            "</section>"
        )

    def _figures(self, context: dict[str, Any]) -> list[tuple[str, object]]:
        figures = [
            ("Mean Absolute SHAP Importance", self._importance_plot(context)),
            ("SHAP Beeswarm", self._beeswarm_plot(context)),
        ]
        if context["interaction_matrix"] is not None:
            figures.append(("Mean Absolute SHAP Interactions", self._interaction_plot(context)))
        figures.extend((f"SHAP Scatter - {scatter['feature_name']}", self._scatter_plot(scatter)) for scatter in context["scatter_plots"])
        return figures

    def _importance_plot(self, context: dict[str, Any]):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, max(4.6, 0.42 * len(context["top_feature_names"]) + 2.8)))
        colors = chart_colors(self.palette)
        ax.barh(context["top_feature_names"][::-1], context["importance_values"][::-1], color=colors[0])
        ax.set_title("Mean absolute SHAP importance", loc="left", fontsize=16, fontweight="bold")
        ax.set_xlabel("|SHAP value|")
        ax.grid(axis="x", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _beeswarm_plot(self, context: dict[str, Any]):
        import matplotlib.pyplot as plt

        colors = chart_colors(self.palette)
        fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.5 * len(context["beeswarm"]) + 2.8)))
        for display_index, feature_payload in enumerate(context["beeswarm"]):
            values = np.asarray(feature_payload["values"], dtype=float)
            jitter = np.linspace(-0.25, 0.25, num=values.shape[0]) if values.size else np.array([])
            ax.scatter(values, np.full(values.shape[0], display_index) + jitter, alpha=0.5, s=18, color=colors[display_index % len(colors)])
        ax.set_yticks(range(len(context["beeswarm"])))
        ax.set_yticklabels([payload["feature_name"] for payload in context["beeswarm"]])
        ax.set_title("SHAP beeswarm", loc="left", fontsize=16, fontweight="bold")
        ax.set_xlabel("SHAP value")
        ax.grid(axis="x", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _interaction_plot(self, context: dict[str, Any]):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        matrix = np.asarray(context["interaction_matrix"], dtype=float)
        image = ax.imshow(matrix, cmap="YlGnBu")
        labels = context["top_feature_names"]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title("Mean absolute SHAP interactions", loc="left", fontsize=16, fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
        return fig

    def _scatter_plot(self, scatter: dict[str, Any]):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.scatter(scatter["x"], scatter["y"], alpha=0.55, s=18, color=self.palette["series_2"])
        ax.set_title(f"SHAP scatter - {scatter['feature_name']}", loc="left", fontsize=16, fontweight="bold")
        ax.set_xlabel(scatter["feature_name"])
        ax.set_ylabel("SHAP value")
        ax.grid(alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _metric_card(self, title: str, value: str, subtitle: str) -> str:
        return (
            "<div class='card metric-card'>"
            f"<span class='eyebrow'>{title}</span>"
            f"<strong class='metric-big'>{value}</strong>"
            f"<span class='metric-subtle'>{subtitle}</span>"
            "</div>"
        )

    def _figure_card(self, title: str, figure) -> str:
        return (
            "<div class='card'>"
            f"<span class='eyebrow'>{escape(title)}</span>"
            f"<div class='plot-frame'><img alt='{escape(title)}' src='{figure_to_data_uri(figure)}' /></div>"
            "</div>"
        )
