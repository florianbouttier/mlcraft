"""HTML renderer for evaluation results."""

from __future__ import annotations

from collections import defaultdict
from html import escape

import numpy as np

from mlcraft.core.results import CurveData, EvaluationResult, MetricRow
from mlcraft.reporting.html import figure_to_data_uri, wrap_html


class EvaluationReportRenderer:
    """Render an evaluation result as a comparison-first HTML report.

    The renderer favors large, visual comparisons over dense tables so model
    differences are immediately visible when several prediction bundles are
    evaluated together.
    """

    def render(self, result: EvaluationResult, *, title: str | None = "mlcraft Evaluation Report", output_path=None) -> str:
        """Render a complete evaluation report.

        Args:
            result: Evaluation output to render.
            title: Title displayed in the standalone HTML document. When
                `None`, no top-level heading is added to the body.
            output_path: Optional file path used to persist the rendered HTML.

        Returns:
            str: Standalone HTML document.

        Example:
            >>> renderer = EvaluationReportRenderer()
            >>> html = renderer.render(result)
            >>> html.startswith("<!doctype html>")
            True
        """

        metrics_by_prediction = self._metrics_by_prediction(result.metric_rows)
        primary_metric = self._primary_metric_name(result)
        sections: list[str] = []
        if title:
            sections.append(f"<h1>{escape(title)}</h1>")
        sections.append(self._render_summary_panel(result, metrics_by_prediction, primary_metric))
        sections.append(self._render_model_comparison(result, metrics_by_prediction, primary_metric))
        if result.curves:
            sections.append(self._render_curve_comparison(result))
        html = wrap_html(title or "mlcraft Evaluation Report", "".join(sections))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _metrics_by_prediction(self, metric_rows: list[MetricRow]) -> dict[str, list[MetricRow]]:
        grouped: dict[str, list[MetricRow]] = defaultdict(list)
        for row in metric_rows:
            grouped[row.prediction_name].append(row)
        return dict(grouped)

    def _primary_metric_name(self, result: EvaluationResult) -> str:
        preferred = result.task_spec.eval_metric
        metric_names = [row.metric_name for row in result.metric_rows]
        if preferred in metric_names:
            return preferred
        return metric_names[0]

    def _render_summary_panel(
        self,
        result: EvaluationResult,
        metrics_by_prediction: dict[str, list[MetricRow]],
        primary_metric: str,
    ) -> str:
        primary_rows = [row for row in result.metric_rows if row.metric_name == primary_metric]
        ordered = sorted(primary_rows, key=lambda row: row.score, reverse=True)
        leader = ordered[0]
        lead_margin = ordered[0].score - ordered[1].score if len(ordered) > 1 else 0.0
        direction = "Higher is better" if leader.higher_is_better else "Lower is better"
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div>"
            f"<span class='eyebrow'>Evaluation Overview</span>"
            f"<h2>{escape(result.task_spec.task_type.value.title())} comparison</h2>"
            "<p class='muted'>The report ranks prediction bundles first, then shows how their curves differ on the same axes.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Compared models', str(len(metrics_by_prediction)), 'Prediction bundles in this run.')}"
            f"{self._metric_card('Focus metric', escape(primary_metric), direction)}"
            f"{self._metric_card('Leader', escape(leader.prediction_name), f'{primary_metric} = {leader.value:.6f}')}"
            f"{self._metric_card('Lead margin', f'{lead_margin:.6f}', 'Gap vs the next best score.')}"
            "</div>"
            "</section>"
        )

    def _render_model_comparison(
        self,
        result: EvaluationResult,
        metrics_by_prediction: dict[str, list[MetricRow]],
        primary_metric: str,
    ) -> str:
        import matplotlib.pyplot as plt

        leaderboard_fig = self._plot_primary_metric_leaderboard(result.metric_rows, primary_metric)
        heatmap_fig = self._plot_metric_heatmap(metrics_by_prediction, primary_metric)
        try:
            return (
                "<section class='panel section-stack'>"
                "<div>"
                "<span class='eyebrow'>Model Comparison</span>"
                "<h2>Graphical leaderboard</h2>"
                "<p class='muted'>The main metric comes first, then a normalized map highlights where each framework wins or falls behind.</p>"
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

    def _render_curve_comparison(self, result: EvaluationResult) -> str:
        import matplotlib.pyplot as plt

        sections = [
            "<section class='panel section-stack'>",
            "<div>",
            "<span class='eyebrow'>Curve Comparison</span>",
            "<h2>Curves on shared axes</h2>",
            "<p class='muted'>Each plot overlays all prediction bundles so framework differences stay visible without thumbnail noise.</p>",
            "</div>",
            "<div class='viz-grid'>",
        ]
        figures = self._plot_curve_groups(result.curves)
        try:
            for title, figure in figures:
                sections.append(self._figure_card(title, figure))
        finally:
            for _, figure in figures:
                plt.close(figure)
        sections.append("</div></section>")
        return "".join(sections)

    def _plot_primary_metric_leaderboard(self, metric_rows: list[MetricRow], primary_metric: str):
        import matplotlib.pyplot as plt

        rows = [row for row in metric_rows if row.metric_name == primary_metric]
        rows = sorted(rows, key=lambda row: row.score, reverse=True)
        values = np.asarray([row.value for row in rows], dtype=float)
        labels = [row.prediction_name for row in rows]
        baseline = values.max() if rows and not rows[0].higher_is_better else values.min()
        positions = np.arange(len(rows))

        fig, ax = plt.subplots(figsize=(10.5, max(4.6, 1.0 + 0.9 * len(rows))))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fbfcfd")
        colors = ["#0f766e" if idx == 0 else "#7aa6c2" for idx in range(len(rows))]
        start = np.full_like(values, baseline, dtype=float)
        ax.hlines(positions, start, values, color="#d7e3ec", linewidth=5, zorder=1)
        ax.scatter(values, positions, s=170, color=colors, edgecolors="white", linewidth=1.6, zorder=3)
        for idx, (value, label) in enumerate(zip(values, labels)):
            ax.text(value, idx + 0.16, f"{value:.6f}", color="#16324f", fontsize=10, fontweight="bold")
            ax.text(baseline, idx - 0.18, label, color="#486581", fontsize=10, ha="left", va="center")
        ax.set_yticks([])
        ax.set_xlabel(
            f"{primary_metric} ({'higher' if rows[0].higher_is_better else 'lower'} is better)"
        )
        ax.set_title("Primary metric ranking", loc="left", fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _plot_metric_heatmap(
        self,
        metrics_by_prediction: dict[str, list[MetricRow]],
        primary_metric: str,
    ):
        import matplotlib.pyplot as plt

        def _primary_score(prediction_name: str) -> float:
            for row in metrics_by_prediction[prediction_name]:
                if row.metric_name == primary_metric:
                    return row.score
            return max(row.score for row in metrics_by_prediction[prediction_name])

        predictions = sorted(
            metrics_by_prediction,
            key=_primary_score,
            reverse=True,
        )
        metrics: list[str] = []
        for prediction_name in predictions:
            for row in metrics_by_prediction[prediction_name]:
                if row.metric_name not in metrics:
                    metrics.append(row.metric_name)

        raw = np.full((len(predictions), len(metrics)), np.nan, dtype=float)
        normalized = np.full_like(raw, 0.5)
        for row_idx, prediction_name in enumerate(predictions):
            row_map = {row.metric_name: row for row in metrics_by_prediction[prediction_name]}
            for col_idx, metric_name in enumerate(metrics):
                if metric_name in row_map:
                    raw[row_idx, col_idx] = row_map[metric_name].value

        for col_idx, metric_name in enumerate(metrics):
            column_rows = [
                next(row for row in metrics_by_prediction[prediction_name] if row.metric_name == metric_name)
                for prediction_name in predictions
                if any(row.metric_name == metric_name for row in metrics_by_prediction[prediction_name])
            ]
            scores = np.asarray([row.score for row in column_rows], dtype=float)
            if scores.size and float(scores.max()) != float(scores.min()):
                scaled = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scaled = np.full(scores.shape, 0.5)
            row_pointer = 0
            for row_idx, prediction_name in enumerate(predictions):
                if any(row.metric_name == metric_name for row in metrics_by_prediction[prediction_name]):
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
                text_color = "#102a43" if normalized[row_idx, col_idx] < 0.62 else "white"
                ax.text(col_idx, row_idx, f"{raw[row_idx, col_idx]:.3f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.02)
        colorbar.ax.set_ylabel("Normalized performance", rotation=270, labelpad=18)
        return fig

    def _plot_curve_groups(self, curves_by_prediction: dict[str, list[CurveData]]) -> list[tuple[str, object]]:
        import matplotlib.pyplot as plt

        grouped: dict[str, list[tuple[str, CurveData]]] = defaultdict(list)
        for prediction_name, curves in curves_by_prediction.items():
            for curve in curves:
                grouped[curve.name].append((prediction_name, curve))

        figures: list[tuple[str, object]] = []
        palette = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#c2410c", "#0891b2"]
        for curve_name, entries in grouped.items():
            fig, ax = plt.subplots(figsize=(8.8, 5.6))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#fbfcfd")
            for idx, (prediction_name, curve) in enumerate(entries):
                color = palette[idx % len(palette)]
                if curve.name == "residuals":
                    ax.plot(curve.x, curve.y, color=color, linewidth=2.5, label=prediction_name)
                    ax.fill_between(curve.x, curve.y, alpha=0.12, color=color)
                else:
                    ax.plot(curve.x, curve.y, color=color, linewidth=2.7, label=prediction_name)
            if curve_name in {"calibration", "poisson_calibration"}:
                diagonal_min = min(
                    min(float(curve.x.min()) if curve.x.size else 0.0 for _, curve in entries),
                    min(float(curve.y.min()) if curve.y.size else 0.0 for _, curve in entries),
                )
                diagonal_max = max(
                    max(float(curve.x.max()) if curve.x.size else 1.0 for _, curve in entries),
                    max(float(curve.y.max()) if curve.y.size else 1.0 for _, curve in entries),
                )
                ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="#94a3b8", linewidth=1.6)
            reference_curve = entries[0][1]
            ax.set_title(curve_name.replace("_", " ").title(), loc="left", fontsize=16, fontweight="bold")
            ax.set_xlabel(reference_curve.x_label)
            ax.set_ylabel(reference_curve.y_label)
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, loc="best")
            for spine in ax.spines.values():
                spine.set_visible(False)
            figures.append((curve_name.replace("_", " ").title(), fig))
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
