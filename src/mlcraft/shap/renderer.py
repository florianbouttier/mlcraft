"""HTML renderer for SHAP results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import ShapResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_shap_context
from mlcraft.utils.optional import OptionalDependencyError, optional_import


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
        sections.append("<div><span class='eyebrow'>Feature Structure</span><h2>Importance and distribution</h2></div>")
        sections.append("<div class='viz-grid viz-grid--compact'>")
        for title, figure in self._iter_figures(context):
            sections.append(self._figure_card(title, figure))
            plt.close(figure)
        sections.append("</div></section>")
        html = wrap_html(str(context.get("title") or "mlcraft SHAP Report"), "".join(sections), palette=self.palette)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_summary_panel(self, context: dict[str, Any]) -> str:
        top_feature = context["ordered_feature_names"][0] if context["ordered_feature_names"] else "n/a"
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Explainability Overview</span>"
            "<h2>SHAP feature perspective</h2>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Feature count', str(context['feature_count']), 'Features available in the SHAP payload.')}"
            f"{self._metric_card('Top feature', escape(top_feature), 'Highest mean absolute SHAP contribution.')}"
            f"{self._metric_card('Top features shown', str(context['top_n']), 'Maximum features used in summary views.')}"
            "</div>"
            "</section>"
        )

    def _iter_figures(self, context: dict[str, Any]):
        try:
            shap = optional_import("shap")
        except OptionalDependencyError:
            yield from self._iter_fallback_figures(context)
            return
        explanation = self._build_explanation(shap, context)
        yield ("SHAP Beeswarm", self._official_beeswarm_plot(shap, explanation, context))
        if context["interaction_values"] is not None:
            yield ("SHAP Interaction Plot", self._interaction_plot(context))
        for feature_name in context["ordered_feature_names"]:
            yield (f"SHAP Scatter - {feature_name}", self._official_scatter_plot(shap, explanation, feature_name))

    def _iter_fallback_figures(self, context: dict[str, Any]):
        yield ("Mean Absolute SHAP Importance", self._importance_plot(context))
        yield ("SHAP Beeswarm", self._beeswarm_plot(context))
        if context["interaction_matrix"] is not None:
            yield ("Mean Absolute SHAP Interactions", self._interaction_plot(context))
        for scatter in context["scatter_plots"]:
            yield (f"SHAP Scatter - {scatter['feature_name']}", self._scatter_plot(scatter))

    def _build_explanation(self, shap, context: dict[str, Any]):
        base_values = context["base_values"]
        if isinstance(base_values, np.ndarray) and base_values.ndim == 0:
            base_values = base_values.item()
        elif isinstance(base_values, np.ndarray) and base_values.size == 1:
            base_values = base_values.reshape(-1)[0].item()
        return shap.Explanation(
            values=np.asarray(context["shap_values"]),
            base_values=base_values,
            data=context["feature_values"],
            feature_names=list(context["feature_names"]),
        )

    def _official_beeswarm_plot(self, shap, explanation, context: dict[str, Any]):
        import matplotlib.pyplot as plt

        shap.plots.beeswarm(explanation, max_display=int(context["top_n"]), show=False)
        return plt.gcf()

    def _official_scatter_plot(self, shap, explanation, feature_name: str):
        import matplotlib.pyplot as plt

        shap.plots.scatter(explanation[:, feature_name], color=explanation, show=False)
        return plt.gcf()

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
        from matplotlib.colors import LinearSegmentedColormap

        interaction_values = np.asarray(context["interaction_values"], dtype=float)
        interaction_matrix = np.abs(interaction_values).mean(0)
        ranking_matrix = np.array(interaction_matrix, copy=True)
        for index in range(ranking_matrix.shape[0]):
            ranking_matrix[index, index] = 0
        indices = np.argsort(-ranking_matrix.sum(0))[:12]
        sorted_matrix = interaction_matrix[indices, :][:, indices]
        labels = np.asarray(context["feature_names"])[indices].tolist()
        cell_count = max(int(sorted_matrix.shape[0]), 1)
        figure_size = float(np.clip(1.05 * cell_count + 2.6, 9.0, 18.0))
        font_size = float(np.clip(17.0 - 0.55 * cell_count, 8.0, 15.0))
        max_abs_interaction = float(np.max(sorted_matrix)) if sorted_matrix.size else 0.0
        interaction_cmap = LinearSegmentedColormap.from_list(
            "mlcraft_interaction_white_red",
            [
                self.palette["interaction_low"],
                self.palette["interaction_mid"],
                self.palette["interaction_high"],
            ],
        )

        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        image = ax.imshow(sorted_matrix, cmap=interaction_cmap, vmin=0.0, vmax=max_abs_interaction if max_abs_interaction > 0.0 else 1.0)
        ax.set_yticks(
            range(sorted_matrix.shape[0]),
            labels,
            rotation=50.4,
            horizontalalignment="right",
        )
        ax.set_xticks(
            range(sorted_matrix.shape[0]),
            labels,
            rotation=50.4,
            horizontalalignment="left",
        )
        ax.xaxis.tick_top()
        for row_index in range(sorted_matrix.shape[0]):
            for col_index in range(sorted_matrix.shape[1]):
                abs_value = float(sorted_matrix[row_index, col_index])
                text_color = self.palette["text_main"] if sorted_matrix[row_index, col_index] < 0.58 * max_abs_interaction else "white"
                ax.text(
                    col_index,
                    row_index,
                    self._format_interaction_value(abs_value),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    fontweight="bold",
                    color=text_color,
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
        colorbar.set_label("mean(|SHAP interaction value|)")
        ax.set_title("Mean absolute SHAP interaction strength", loc="left", fontsize=16, fontweight="bold")
        return fig

    def _format_interaction_value(self, value: float) -> str:
        abs_value = abs(float(value))
        if not np.isfinite(abs_value):
            return "n/a"
        if abs_value == 0.0:
            return "0"
        if abs_value >= 100.0:
            return f"{abs_value:.0f}"
        if abs_value >= 10.0:
            return f"{abs_value:.1f}"
        if abs_value >= 1.0:
            return f"{abs_value:.2f}"
        if abs_value >= 0.1:
            return f"{abs_value:.3f}"
        if abs_value >= 0.01:
            return f"{abs_value:.4f}"
        return f"{abs_value:.2g}"

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
