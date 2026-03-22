"""HTML renderer for SHAP results."""

from __future__ import annotations

import numpy as np

from mlcraft.core.results import ShapResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html


class ShapReportRenderer:
    """Render SHAP artifacts with lightweight matplotlib plots."""

    def render(self, result: ShapResult, *, title: str = "mlcraft SHAP Report", output_path=None, top_n: int = 10) -> str:
        """Render a complete SHAP report.

        Args:
            result: SHAP output to render.
            title: Title displayed in the HTML document.
            output_path: Optional file path used to persist the rendered HTML.
            top_n: Number of top features to keep in summary plots.

        Returns:
            str: Standalone HTML document.
        """

        import matplotlib.pyplot as plt

        sections = [f"<h1>{title}</h1>"]
        sections.append("<h2>Feature Importance</h2>")
        sections.append("<div class='grid'>")
        for fig in self._figures(result, top_n=top_n):
            sections.append(f"<div class='card'><img alt='shap-plot' src='{figure_to_data_uri(fig)}' /></div>")
            plt.close(fig)
        sections.append("</div>")
        html = wrap_html(title, "".join(sections))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _figures(self, result: ShapResult, *, top_n: int) -> list:
        figures = [self._importance_plot(result, top_n=top_n), self._beeswarm_plot(result, top_n=top_n)]
        if result.interaction_values is not None:
            figures.append(self._interaction_plot(result, top_n=top_n))
        figures.extend(self._feature_scatter_plots(result, top_n=min(top_n, 5)))
        return figures

    def _importance_plot(self, result: ShapResult, *, top_n: int):
        import matplotlib.pyplot as plt

        importance = result.importance if result.importance is not None else np.mean(np.abs(result.shap_values), axis=0)
        order = np.argsort(importance)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(np.array(result.feature_names)[order][::-1], importance[order][::-1])
        ax.set_title("Mean absolute SHAP importance")
        ax.set_xlabel("|SHAP value|")
        return fig

    def _beeswarm_plot(self, result: ShapResult, *, top_n: int):
        import matplotlib.pyplot as plt

        importance = result.importance if result.importance is not None else np.mean(np.abs(result.shap_values), axis=0)
        order = np.argsort(importance)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
        for display_index, feature_index in enumerate(order):
            values = result.shap_values[:, feature_index]
            jitter = np.linspace(-0.25, 0.25, num=values.shape[0])
            ax.scatter(values, np.full(values.shape[0], display_index) + jitter, alpha=0.4, s=12)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(np.array(result.feature_names)[order])
        ax.set_title("SHAP beeswarm")
        ax.set_xlabel("SHAP value")
        return fig

    def _interaction_plot(self, result: ShapResult, *, top_n: int):
        import matplotlib.pyplot as plt

        importance = result.importance if result.importance is not None else np.mean(np.abs(result.shap_values), axis=0)
        order = np.argsort(importance)[::-1][:top_n]
        interactions = np.mean(np.abs(result.interaction_values), axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        matrix = interactions[np.ix_(order, order)]
        image = ax.imshow(matrix, cmap="viridis")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        labels = np.array(result.feature_names)[order]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title("Mean absolute SHAP interactions")
        fig.colorbar(image, ax=ax)
        return fig

    def _feature_scatter_plots(self, result: ShapResult, *, top_n: int) -> list:
        import matplotlib.pyplot as plt

        if result.feature_values is None:
            return []
        importance = result.importance if result.importance is not None else np.mean(np.abs(result.shap_values), axis=0)
        order = np.argsort(importance)[::-1][:top_n]
        figures = []
        for feature_index in order:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(result.feature_values[:, feature_index], result.shap_values[:, feature_index], alpha=0.5, s=12)
            ax.set_title(f"SHAP scatter - {result.feature_names[feature_index]}")
            ax.set_xlabel(result.feature_names[feature_index])
            ax.set_ylabel("SHAP value")
            figures.append(fig)
        return figures
