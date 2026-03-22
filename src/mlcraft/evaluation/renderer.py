"""HTML renderer for evaluation results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import EvaluationResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_evaluation_context


class EvaluationReportRenderer:
    """Render an evaluation result as a comparison-first HTML report."""

    def __init__(self, *, palette: dict[str, str] | None = None) -> None:
        self.palette = get_report_palette(palette)

    def build_context(self, result: EvaluationResult, *, title: str | None = "mlcraft Evaluation Report") -> dict[str, Any]:
        """Build the evaluation view context used by the HTML renderer.

        Args:
            result: Evaluation output to transform into a render context.
            title: Optional report title.

        Returns:
            dict[str, Any]: Dictionary of report data that can be serialized
            or rendered later.
        """

        return build_evaluation_context(result, title=title)

    def render(self, result: EvaluationResult, *, title: str | None = "mlcraft Evaluation Report", output_path=None) -> str:
        """Render a complete evaluation report."""

        context = self.build_context(result, title=title)
        return self.render_context(context, output_path=output_path)

    def render_context(self, context: dict[str, Any], *, output_path=None) -> str:
        """Render an evaluation report from a pre-built dictionary context.

        Args:
            context: Dictionary returned by `build_context()`.
            output_path: Optional file path used to persist the rendered HTML.

        Returns:
            str: Standalone HTML document.
        """

        sections: list[str] = []
        if context.get("title"):
            sections.append(f"<h1>{escape(str(context['title']))}</h1>")
        sections.append(self._render_summary_panel(context))
        sections.append(self._render_model_comparison(context))
        if context.get("curve_groups"):
            sections.append(self._render_curve_comparison(context))
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
            "<div>"
            "<span class='eyebrow'>Evaluation Overview</span>"
            f"<h2>{escape(str(context['task_type']).title())} comparison</h2>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Compared models', str(summary['model_count']), 'Prediction bundles in this run.')}"
            f"{self._metric_card('Focus metric', escape(str(context['primary_metric_name'])), direction)}"
            f"{self._metric_card('Leader', escape(leader_name), leader_value)}"
            f"{self._metric_card('Lead margin', lead_margin, 'Gap vs the next best score.')}"
            "</div>"
            "</section>"
        )

    def _render_model_comparison(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        leaderboard_fig = self._plot_primary_metric_leaderboard(context)
        heatmap_fig = self._plot_metric_heatmap(context)
        try:
            return (
                "<section class='panel section-stack'>"
                "<div>"
                "<span class='eyebrow'>Model Comparison</span>"
                "<h2>Graphical leaderboard</h2>"
                "</div>"
                "<div class='viz-grid'>"
                f"{self._figure_card('Primary Metric Ranking', leaderboard_fig, wide=True)}"
                f"{self._figure_card('Metric Comparison Map', heatmap_fig)}"
                "</div>"
                "</section>"
            )
        finally:
            plt.close(leaderboard_fig)
            plt.close(heatmap_fig)

    def _render_curve_comparison(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        sections = [
            "<section class='panel section-stack'>",
            "<div>",
            "<span class='eyebrow'>Curve Comparison</span>",
            "<h2>Curves on shared axes</h2>",
            "</div>",
            "<div class='viz-grid'>",
        ]
        figures = self._plot_curve_groups(context["curve_groups"])
        try:
            for title, figure in figures:
                sections.append(self._figure_card(title, figure))
        finally:
            for _, figure in figures:
                plt.close(figure)
        sections.append("</div></section>")
        return "".join(sections)

    def _plot_primary_metric_leaderboard(self, context: dict[str, Any]):
        import matplotlib.pyplot as plt

        primary_metric = context["primary_metric_name"]
        rows = [row for row in context["metric_rows"] if row["metric_name"] == primary_metric]
        rows = sorted(rows, key=lambda row: row["score"], reverse=True)
        values = np.asarray([row["value"] for row in rows], dtype=float)
        labels = [row["prediction_name"] for row in rows]
        if not rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No metrics available.", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig
        baseline = values.max() if not rows[0]["higher_is_better"] else values.min()
        positions = np.arange(len(rows))

        fig, ax = plt.subplots(figsize=(10.5, max(4.6, 1.0 + 0.9 * len(rows))))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fbfcfd")
        colors = [self.palette["accent"] if idx == 0 else self.palette["series_muted"] for idx in range(len(rows))]
        start = np.full_like(values, baseline, dtype=float)
        ax.hlines(positions, start, values, color=self.palette["line_soft"], linewidth=5, zorder=1)
        ax.scatter(values, positions, s=170, color=colors, edgecolors="white", linewidth=1.6, zorder=3)
        for idx, (value, label) in enumerate(zip(values, labels)):
            ax.text(value, idx + 0.16, f"{value:.6f}", color=self.palette["text_main"], fontsize=10, fontweight="bold")
            ax.text(baseline, idx - 0.18, label, color=self.palette["text_soft"], fontsize=10, ha="left", va="center")
        ax.set_yticks([])
        ax.set_xlabel(f"{primary_metric} ({'higher' if rows[0]['higher_is_better'] else 'lower'} is better)")
        ax.set_title("Primary metric ranking", loc="left", fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _plot_metric_heatmap(self, context: dict[str, Any]):
        import matplotlib.pyplot as plt

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

        raw = np.full((len(predictions), len(metrics)), np.nan, dtype=float)
        normalized = np.full_like(raw, 0.5)
        for row_idx, prediction_name in enumerate(predictions):
            row_map = {row["metric_name"]: row for row in metrics_by_prediction[prediction_name]}
            for col_idx, metric_name in enumerate(metrics):
                if metric_name in row_map:
                    raw[row_idx, col_idx] = row_map[metric_name]["value"]

        for col_idx, metric_name in enumerate(metrics):
            column_rows = [
                next(row for row in metrics_by_prediction[prediction_name] if row["metric_name"] == metric_name)
                for prediction_name in predictions
                if any(row["metric_name"] == metric_name for row in metrics_by_prediction[prediction_name])
            ]
            scores = np.asarray([row["score"] for row in column_rows], dtype=float)
            scaled = (scores - scores.min()) / (scores.max() - scores.min()) if scores.size and float(scores.max()) != float(scores.min()) else np.full(scores.shape, 0.5)
            row_pointer = 0
            for row_idx, prediction_name in enumerate(predictions):
                if any(row["metric_name"] == metric_name for row in metrics_by_prediction[prediction_name]):
                    normalized[row_idx, col_idx] = scaled[row_pointer]
                    row_pointer += 1

        fig, ax = plt.subplots(figsize=(max(7.8, 1.35 * len(metrics)), max(4.8, 0.9 * len(predictions) + 2.0)))
        fig.patch.set_facecolor("#ffffff")
        heatmap = ax.imshow(normalized, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(metrics)), labels=metrics, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(predictions)), labels=predictions)
        ax.set_title("Metric comparison map", loc="left", fontsize=16, fontweight="bold")
        for row_idx in range(len(predictions)):
            for col_idx in range(len(metrics)):
                if np.isnan(raw[row_idx, col_idx]):
                    continue
                text_color = self.palette["text_main"] if normalized[row_idx, col_idx] < 0.62 else "white"
                ax.text(col_idx, row_idx, f"{raw[row_idx, col_idx]:.3f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.02)
        colorbar.ax.set_ylabel("Normalized performance", rotation=270, labelpad=18)
        return fig

    def _plot_curve_groups(self, curve_groups: list[dict[str, Any]]) -> list[tuple[str, object]]:
        import matplotlib.pyplot as plt

        figures: list[tuple[str, object]] = []
        palette = chart_colors(self.palette)
        for curve_group in curve_groups:
            fig, ax = plt.subplots(figsize=(8.8, 5.6))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#fbfcfd")
            for idx, series in enumerate(curve_group["series"]):
                color = palette[idx % len(palette)]
                x_values = np.asarray(series["x"], dtype=float)
                y_values = np.asarray(series["y"], dtype=float)
                if curve_group["curve_name"] == "residuals":
                    ax.plot(x_values, y_values, color=color, linewidth=2.5, label=series["prediction_name"])
                    ax.fill_between(x_values, y_values, alpha=0.12, color=color)
                else:
                    ax.plot(x_values, y_values, color=color, linewidth=2.7, label=series["prediction_name"])
            if curve_group["curve_name"] in {"calibration", "poisson_calibration"}:
                diagonal_min = min(min(series["x"] or [0.0]) for series in curve_group["series"])
                diagonal_min = min(diagonal_min, min(min(series["y"] or [0.0]) for series in curve_group["series"]))
                diagonal_max = max(max(series["x"] or [1.0]) for series in curve_group["series"])
                diagonal_max = max(diagonal_max, max(max(series["y"] or [1.0]) for series in curve_group["series"]))
                ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color=self.palette["grid_soft"], linewidth=1.6)
            ax.set_title(curve_group["title"], loc="left", fontsize=16, fontweight="bold")
            ax.set_xlabel(curve_group["x_label"])
            ax.set_ylabel(curve_group["y_label"])
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, loc="best")
            for spine in ax.spines.values():
                spine.set_visible(False)
            figures.append((curve_group["title"], fig))
        return figures

    def _metric_card(self, title: str, value: str, subtitle: str) -> str:
        return (
            "<div class='card metric-card'>"
            f"<span class='eyebrow'>{title}</span>"
            f"<strong class='metric-big'>{value}</strong>"
            f"<span class='metric-subtle'>{subtitle}</span>"
            "</div>"
        )

    def _figure_card(self, title: str, figure, *, wide: bool = False) -> str:
        wide_class = " card--wide" if wide else ""
        return (
            f"<div class='card{wide_class}'>"
            f"<span class='eyebrow'>{escape(title)}</span>"
            f"<div class='plot-frame'><img alt='{escape(title)}' src='{figure_to_data_uri(figure)}' /></div>"
            "</div>"
        )

    def _primary_score(self, rows: list[dict[str, Any]], primary_metric: str) -> float:
        for row in rows:
            if row["metric_name"] == primary_metric:
                return float(row["score"])
        return max(float(row["score"]) for row in rows)
