"""HTML renderer for Optuna tuning results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import TuningResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_tuning_context


class TuningReportRenderer:
    """Render a tuning result as a comparison-first HTML dashboard."""

    def __init__(self, *, palette: dict[str, str] | None = None) -> None:
        self.palette = get_report_palette(palette)

    def build_context(self, result: TuningResult, *, title: str | None = "mlcraft Tuning Report") -> dict[str, Any]:
        """Build the tuning view context used by the HTML renderer."""

        return build_tuning_context(result, title=title)

    def render(self, result: TuningResult, *, title: str | None = "mlcraft Tuning Report", output_path=None) -> str:
        """Render a complete tuning report."""

        context = self.build_context(result, title=title)
        return self.render_context(context, output_path=output_path)

    def render_context(self, context: dict[str, Any], *, output_path=None) -> str:
        """Render a tuning report from a pre-built dictionary context."""

        sections: list[str] = []
        if context.get("title"):
            sections.append(f"<h1>{escape(str(context['title']))}</h1>")
        sections.append(self._render_metric_matrix(context))
        sections.append(self._render_summary_panel(context))
        if context.get("backend_summary_rows"):
            sections.append(self._render_backend_overview(context))
        sections.append(self._render_metric_explorer(context))
        sections.append(self._render_search_dynamics(context))
        sections.append(self._render_configuration_panel(context))
        if context.get("holdout_curve_groups"):
            sections.append(self._render_holdout_curves(context))
        if context.get("fold_curve_groups"):
            sections.append(self._render_fold_curves(context))
        html = wrap_html(str(context.get("title") or "mlcraft Tuning Report"), "".join(sections), palette=self.palette)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_metric_matrix(self, context: dict[str, Any]) -> str:
        rows = []
        has_test = any(metric["test_value"] is not None for metric in context["metric_catalog"])
        no_value = "<span class='muted'>n/a</span>"
        for metric in context["metric_catalog"]:
            gap = self._format_delta(metric["score_gap"])
            holdout = self._format_delta(metric["holdout_delta"])
            test_cell = self._format_metric(metric["test_value"]) if has_test else no_value
            holdout_cell = holdout if has_test else no_value
            rows.append(
                "<tr>"
                "<td>"
                f"<button class='table-action' type='button' data-toggle-button data-toggle-group='tuning-metrics' data-toggle-target='{escape(self._metric_panel_id(metric['metric_name']))}' data-toggle-scroll='metric-explorer'>{escape(metric['metric_name'])}</button>"
                "</td>"
                f"<td>{self._format_metric(metric['train_value'])}</td>"
                f"<td>{self._format_metric(metric['val_value'])}</td>"
                f"<td>{test_cell}</td>"
                f"<td>{gap}</td>"
                f"<td>{holdout_cell}</td>"
                f"<td>{'higher' if metric['higher_is_better'] else 'lower'}</td>"
                "</tr>"
            )
        test_header = "<th>Test</th><th>Test vs val</th>" if has_test else "<th>Test</th><th>Test vs val</th>"
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>KPI Matrix</span>"
            "<h2>Train, validation, and holdout summary for every tracked metric</h2>"
            "<p class='muted'>Use any metric name below to jump directly to its detailed fold-level view.</p>"
            "</div>"
            "<div class='summary-table'>"
            "<table><thead><tr><th>Metric</th><th>Train</th><th>Validation</th>"
            + test_header
            + "<th>Validation vs train</th><th>Direction</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
            "</section>"
        )

    def _render_summary_panel(self, context: dict[str, Any]) -> str:
        gap_badge = (
            "<span class='badge badge--alert'>Validation drop is material</span>"
            if context["generalization_gap"] >= 0.05
            else "<span class='badge'>Validation behavior is contained</span>"
        )
        holdout_value = "n/a" if context["test_metric_value"] is None else f"{context['test_metric_value']:.6f}"
        train_metric = context["split_points"][0]["value"] if context.get("split_points") else np.nan
        validation_metric = context["split_points"][1]["value"] if len(context.get("split_points", [])) > 1 else np.nan
        generalization_gap = f"{context['generalization_gap']:.6f}"
        metric_label = escape(str(context["metric_name"]))
        backend_label = escape(str(context.get("selected_model_type") or context["backend_summary_rows"][0]["backend_name"]))
        alpha_label = f"alpha = {context['alpha']:.4f}"
        best_score_label = f"{context['best_score']:.6f}"
        train_metric_label = f"{train_metric:.6f}"
        validation_metric_label = f"{validation_metric:.6f}"
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Tuning Overview</span>"
            f"<h2>{metric_label} stays front and center</h2>"
            "<p class='muted'>The summary stays compact, while the detailed KPI explorer below lets you switch metrics without redrawing the whole report.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Selected backend', backend_label, 'Winning framework after the Optuna search.')}"
            f"{self._metric_card('Optimized metric', metric_label, alpha_label)}"
            f"{self._metric_card('Best penalized score', best_score_label, 'The internal objective always maximized during tuning.')}"
            f"{self._metric_card('Average train', train_metric_label, f'Average train {metric_label}.')}"
            f"{self._metric_card('Average validation', validation_metric_label, f'Average validation {metric_label}.')}"
            f"{self._metric_card('Average test', holdout_value, 'Holdout metric after refit on the full train partition.')}"
            f"{self._metric_card('Generalization gap', generalization_gap, 'Absolute gap between average train and validation score.')}"
            "</div>"
            f"{gap_badge}"
            "</section>"
        )

    def _render_backend_overview(self, context: dict[str, Any]) -> str:
        rows = context.get("backend_summary_rows") or []
        if not rows:
            return ""
        scores = [row["best_score"] for row in rows if not np.isnan(row["best_score"])]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        comparison_rows = []
        for row in rows:
            width = self._normalized_width(row["best_score"], min_score, max_score)
            metric_value = row["val_metric_value"]
            test_value = row["test_metric_value"]
            title = escape(row["backend_name"])
            badge = "<span class='badge'>selected</span>" if row["is_selected"] else ""
            comparison_rows.append(
                "<div class='comparison-row'>"
                "<div class='comparison-head'>"
                f"<div class='comparison-title'>{title}{badge}</div>"
                f"<div class='muted'>best score {row['best_score']:.6f} | val {metric_value:.6f} | test {self._format_metric(test_value)}</div>"
                "</div>"
                "<div class='comparison-track'>"
                f"<div class='comparison-fill' style='width:{width:.2f}%; background:{self.palette['accent'] if row['is_selected'] else self.palette['series_muted']};'></div>"
                "</div>"
                "</div>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Backend Comparison</span>"
            "<h2>Frameworks ranked by penalized validation score</h2>"
            "<p class='muted'>This stays visual and compact: the winning backend is highlighted, while the remaining frameworks stay directly comparable.</p>"
            "</div>"
            f"<div class='comparison-list'>{''.join(comparison_rows)}</div>"
            "</section>"
        )

    def _render_metric_explorer(self, context: dict[str, Any]) -> str:
        buttons = []
        panels = []
        for index, metric in enumerate(context["metric_catalog"]):
            panel_id = self._metric_panel_id(metric["metric_name"])
            default_attr = " data-toggle-default='true'" if index == 0 else ""
            buttons.append(
                f"<button class='segmented-button' type='button' data-toggle-button data-toggle-group='tuning-metrics' data-toggle-target='{escape(panel_id)}'{default_attr}>{escape(metric['metric_name'])}</button>"
            )
            panels.append(self._render_metric_panel(metric))
        return (
            "<section id='metric-explorer' class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Metric Explorer</span>"
            "<h2>Pick the KPI you want to inspect</h2>"
            "<p class='muted'>Switching the selected KPI only toggles HTML panels. The report stays fast and readable even when many metrics are present.</p>"
            "</div>"
            f"<div class='segmented' role='group' aria-label='Metric selector'>{''.join(buttons)}</div>"
            + "".join(panels)
            + "</section>"
        )

    def _render_metric_panel(self, metric: dict[str, Any]) -> str:
        panel_id = self._metric_panel_id(metric["metric_name"])
        fold_viz = self._render_fold_dumbbell(metric)
        holdout_delta_text = self._format_signed(metric["holdout_delta"])
        holdout_badge = (
            "Holdout is not available for this metric."
            if metric["test_value"] is None
            else f"Holdout vs validation = {holdout_delta_text}"
        )
        return (
            f"<div class='metric-panel toggle-panel' data-toggle-panel data-toggle-group='tuning-metrics' data-toggle-panel='{escape(panel_id)}' hidden>"
            "<div class='metric-meta'>"
            f"<span class='badge'>{escape(metric['metric_name'])}</span>"
            f"<span class='badge'>{'higher is better' if metric['higher_is_better'] else 'lower is better'}</span>"
            f"<span class='badge'>{escape(holdout_badge)}</span>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Average train', self._format_metric(metric['train_value']), 'Average train value across folds.')}"
            f"{self._metric_card('Average validation', self._format_metric(metric['val_value']), 'Average validation value across folds.')}"
            f"{self._metric_card('Average test', self._format_metric(metric['test_value']), 'Final holdout value after refit.')}"
            f"{self._metric_card('Validation vs train', self._format_delta(metric['score_gap']), 'Normalized so positive means the validation side improved.')}"
            "</div>"
            "<div class='two-column-grid'>"
            f"<div class='card'>{fold_viz}</div>"
            "<div class='card'>"
            "<span class='eyebrow'>Interpretation</span>"
            "<div class='notes-list'>"
            f"<div class='note-item'><strong>Validation view</strong><p class='muted'>Use the fold visual to see where train and validation stay close, and where they drift apart.</p></div>"
            f"<div class='note-item'><strong>Direction</strong><p class='muted'>This metric is interpreted with <code>{'higher' if metric['higher_is_better'] else 'lower'}</code> values considered better.</p></div>"
            f"<div class='note-item'><strong>Holdout check</strong><p class='muted'>{escape(holdout_badge)}</p></div>"
            "</div>"
            "</div>"
            "</div>"
            "</div>"
        )

    def _render_search_dynamics(self, context: dict[str, Any]) -> str:
        history_rows = context.get("history_rows") or []
        if not history_rows:
            return ""
        ranked_trials = sorted(history_rows, key=lambda row: row["penalized_score"], reverse=True)
        top_trials = []
        for row in ranked_trials[: min(5, len(ranked_trials))]:
            params = ", ".join(f"{key}={self._format_value(value)}" for key, value in list(row["params"].items())[:4])
            top_trials.append(
                "<div class='note-item'>"
                f"<strong>Trial {row['trial_number']}</strong>"
                f"<p class='muted'>penalized {row['penalized_score']:.6f} | train {row['train_score']:.6f} | validation {row['val_score']:.6f}</p>"
                f"<p class='muted'>{escape(params) if params else 'No parameters recorded.'}</p>"
                "</div>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Search Dynamics</span>"
            "<h2>How the search moved across trials</h2>"
            "<p class='muted'>This keeps the study readable without embedding a heavy plotting runtime.</p>"
            "</div>"
            "<div class='two-column-grid'>"
            f"<div class='card'>{self._render_trial_history(history_rows)}</div>"
            "<div class='card'>"
            "<span class='eyebrow'>Top Trials</span>"
            f"<div class='notes-list'>{''.join(top_trials)}</div>"
            "</div>"
            "</div>"
            "</section>"
        )

    def _render_configuration_panel(self, context: dict[str, Any]) -> str:
        narrative_cards = []
        if context.get("best_fold") is not None:
            narrative_cards.append(
                self._metric_card(
                    "Best validation fold",
                    f"Fold {context['best_fold']['fold_index']}",
                    f"validation score = {context['best_fold']['val_score']:.6f}",
                )
            )
        if context.get("worst_fold") is not None:
            narrative_cards.append(
                self._metric_card(
                    "Most fragile fold",
                    f"Fold {context['worst_fold']['fold_index']}",
                    f"validation score = {context['worst_fold']['val_score']:.6f}",
                )
            )
        chips = "".join(
            f"<span class='chip'><strong>{escape(str(key))}</strong><span>{escape(self._format_value(value))}</span></span>"
            for key, value in context["best_params"].items()
        )
        empty_chip = "<span class='muted'>No tuned parameters were recorded.</span>"
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Winning Configuration</span>"
            "<h2>Best trial configuration</h2>"
            "</div>"
            f"<div class='chip-cloud'>{chips or empty_chip}</div>"
            f"<div class='kpi-grid'>{''.join(narrative_cards)}</div>"
            "</section>"
        )

    def _render_holdout_curves(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        figures = self._plot_curve_groups(context["holdout_curve_groups"])
        if not figures:
            return ""
        try:
            return (
                "<section class='panel section-stack'>"
                "<div class='section-head'>"
                "<span class='eyebrow'>Final Test Curves</span>"
                "<h2>Holdout behavior</h2>"
                "</div>"
                "<div class='viz-grid'>"
                + "".join(self._figure_card(title, figure, wide=True) for title, figure in figures)
                + "</div></section>"
            )
        finally:
            for _, figure in figures:
                plt.close(figure)

    def _render_fold_curves(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        curve_groups = context.get("fold_curve_groups") or []
        figures = self._plot_curve_groups(curve_groups)
        if not figures:
            return ""
        try:
            return (
                "<section class='panel section-stack'>"
                "<div class='section-head'>"
                "<span class='eyebrow'>Fold Curves</span>"
                "<h2>Train and validation curves by fold</h2>"
                "</div>"
                "<div class='viz-grid'>"
                + "".join(self._figure_card(title, figure, wide=True) for title, figure in figures)
                + "</div></section>"
            )
        finally:
            for _, figure in figures:
                plt.close(figure)

    def _render_trial_history(self, history_rows: list[dict[str, Any]]) -> str:
        penalized = [row["penalized_score"] for row in history_rows]
        validation = [row["val_score"] for row in history_rows]
        train = [row["train_score"] for row in history_rows]
        svg = self._svg_multi_line_chart(
            [
                {"name": "Penalized", "values": penalized, "color": self.palette["accent"]},
                {"name": "Validation", "values": validation, "color": self.palette["series_2"]},
                {"name": "Train", "values": train, "color": self.palette["series_3"]},
            ],
            x_labels=[str(row["trial_number"]) for row in history_rows],
            y_label="score",
        )
        legend = (
            "<div class='legend-row'>"
            f"{self._legend_item('Penalized', self.palette['accent'])}"
            f"{self._legend_item('Validation', self.palette['series_2'])}"
            f"{self._legend_item('Train', self.palette['series_3'])}"
            "</div>"
        )
        return "<span class='eyebrow'>Trial History</span>" + svg + legend

    def _render_fold_dumbbell(self, metric: dict[str, Any]) -> str:
        rows = metric["fold_rows"]
        if not rows:
            return "<span class='eyebrow'>Fold Comparison</span><p class='muted'>No fold-level values were recorded for this metric.</p>"
        svg = self._svg_dumbbell_chart(rows)
        legend = (
            "<div class='legend-row'>"
            f"{self._legend_item('Train', self.palette['accent'])}"
            f"{self._legend_item('Validation', self.palette['series_2'])}"
            "</div>"
        )
        return "<span class='eyebrow'>Fold Comparison</span>" + svg + legend

    def _svg_multi_line_chart(self, series_list: list[dict[str, Any]], *, x_labels: list[str], y_label: str) -> str:
        width = 820
        height = 320
        left = 58
        right = 24
        top = 24
        bottom = 42
        all_values = [float(value) for series in series_list for value in series["values"] if not np.isnan(value)]
        if not all_values:
            return "<p class='muted'>No history values available.</p>"
        min_value = min(all_values)
        max_value = max(all_values)
        if abs(max_value - min_value) < 1e-12:
            max_value += 1.0
            min_value -= 1.0

        def scale_x(index: int) -> float:
            if len(x_labels) == 1:
                return left + (width - left - right) / 2
            return left + (width - left - right) * index / (len(x_labels) - 1)

        def scale_y(value: float) -> float:
            return top + (height - top - bottom) * (1.0 - ((value - min_value) / (max_value - min_value)))

        grid_lines = []
        for fraction in np.linspace(0.0, 1.0, 4):
            y = top + (height - top - bottom) * fraction
            value = max_value - (max_value - min_value) * fraction
            grid_lines.append(
                f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' stroke='{self.palette['line_soft']}' stroke-width='1' />"
                f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' fill='{self.palette['text_soft']}' font-size='12'>{value:.4f}</text>"
            )

        series_markup = []
        for series in series_list:
            points = []
            circles = []
            for index, value in enumerate(series["values"]):
                if np.isnan(value):
                    continue
                x = scale_x(index)
                y = scale_y(float(value))
                points.append(f"{x:.2f},{y:.2f}")
                circles.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.5' fill='{series['color']}' />")
            if points:
                series_markup.append(
                    f"<polyline fill='none' stroke='{series['color']}' stroke-width='3' points='{' '.join(points)}' stroke-linecap='round' stroke-linejoin='round' />"
                    + "".join(circles)
                )

        x_axis_labels = []
        for index, label in enumerate(x_labels):
            x = scale_x(index)
            x_axis_labels.append(
                f"<text x='{x:.2f}' y='{height - 10:.2f}' text-anchor='middle' fill='{self.palette['text_soft']}' font-size='12'>{escape(label)}</text>"
            )

        return (
            "<div class='viz-shell'>"
            f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Trial history chart'>"
            + "".join(grid_lines)
            + f"<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='{self.palette['border_strong']}' stroke-width='1.5' />"
            + "".join(series_markup)
            + "".join(x_axis_labels)
            + f"<text x='18' y='{top + 10}' fill='{self.palette['text_soft']}' font-size='12' transform='rotate(-90 18,{top + 10})'>{escape(y_label)}</text>"
            + "</svg></div>"
        )

    def _svg_dumbbell_chart(self, rows: list[dict[str, Any]]) -> str:
        width = 820
        row_height = 42
        height = 58 + len(rows) * row_height
        left = 130
        right = 70
        top = 24
        values = [
            float(value)
            for row in rows
            for value in (row["train_value"], row["val_value"])
            if value is not None and not np.isnan(value)
        ]
        min_value = min(values) if values else 0.0
        max_value = max(values) if values else 1.0
        if abs(max_value - min_value) < 1e-12:
            min_value -= 1.0
            max_value += 1.0

        def scale_x(value: float) -> float:
            return left + (width - left - right) * ((value - min_value) / (max_value - min_value))

        grid = []
        for fraction in np.linspace(0.0, 1.0, 4):
            x = left + (width - left - right) * fraction
            value = min_value + (max_value - min_value) * fraction
            grid.append(
                f"<line x1='{x:.2f}' y1='{top - 6}' x2='{x:.2f}' y2='{height - 18}' stroke='{self.palette['line_soft']}' stroke-width='1' />"
                f"<text x='{x:.2f}' y='{height - 2:.2f}' text-anchor='middle' fill='{self.palette['text_soft']}' font-size='12'>{value:.4f}</text>"
            )

        rows_markup = []
        for index, row in enumerate(rows):
            y = top + index * row_height + 14
            train_value = row["train_value"]
            val_value = row["val_value"]
            if train_value is None or val_value is None:
                continue
            train_x = scale_x(float(train_value))
            val_x = scale_x(float(val_value))
            color = self.palette["accent_2"] if (row["score_gap"] or 0.0) >= 0 else self.palette["danger"]
            rows_markup.append(
                f"<text x='18' y='{y + 4:.2f}' fill='{self.palette['text_main']}' font-size='13' font-weight='700'>Fold {row['fold_index']}</text>"
                f"<line x1='{train_x:.2f}' y1='{y:.2f}' x2='{val_x:.2f}' y2='{y:.2f}' stroke='{color}' stroke-width='4' stroke-linecap='round' />"
                f"<circle cx='{train_x:.2f}' cy='{y:.2f}' r='6' fill='{self.palette['accent']}' />"
                f"<circle cx='{val_x:.2f}' cy='{y:.2f}' r='6' fill='{self.palette['series_2']}' />"
                f"<text x='{width - right + 8:.2f}' y='{y + 4:.2f}' fill='{color}' font-size='12' font-weight='700'>{self._format_signed(row['score_gap'])}</text>"
            )

        return (
            "<div class='viz-shell'>"
            f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Fold comparison chart'>"
            + "".join(grid)
            + "".join(rows_markup)
            + "</svg></div>"
        )

    def _legend_item(self, label: str, color: str) -> str:
        return (
            "<span class='legend-item'>"
            f"<span class='legend-swatch' style='background:{color};'></span>"
            f"{escape(label)}"
            "</span>"
        )

    def _plot_curve_groups(self, curve_groups: list[dict[str, Any]]) -> list[tuple[str, object]]:
        import matplotlib.pyplot as plt

        figures: list[tuple[str, object]] = []
        palette = chart_colors(self.palette)
        for curve_group in curve_groups:
            fig, ax = plt.subplots(figsize=(8.8, 5.6))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#fbfcfd")
            for idx, series in enumerate(curve_group["series"]):
                color = palette[(series.get("fold_index", idx) or idx) % len(palette)]
                dash = "--" if series.get("split") == "validation" else "-"
                x_values = np.asarray(series["x"], dtype=float)
                y_values = np.asarray(series["y"], dtype=float)
                ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linewidth=2.6,
                    linestyle=dash,
                    label=series.get("series_name", series.get("prediction_name", f"series {idx + 1}")),
                )
            if curve_group["curve_name"] in {"calibration", "poisson_calibration", "roc"}:
                diagonal_min = min(min(series["x"] or [0.0]) for series in curve_group["series"])
                diagonal_min = min(diagonal_min, min(min(series["y"] or [0.0]) for series in curve_group["series"]))
                diagonal_max = max(max(series["x"] or [1.0]) for series in curve_group["series"])
                diagonal_max = max(diagonal_max, max(max(series["y"] or [1.0]) for series in curve_group["series"]))
                ax.plot(
                    [diagonal_min, diagonal_max],
                    [diagonal_min, diagonal_max],
                    linestyle=":",
                    color=self.palette["grid_soft"],
                    linewidth=1.5,
                )
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

    def _metric_panel_id(self, metric_name: str) -> str:
        safe = "".join(character if character.isalnum() else "-" for character in str(metric_name).lower()).strip("-")
        return f"metric-{safe or 'metric'}"

    def _format_metric(self, value) -> str:
        if value is None or np.isnan(value):
            return "<span class='muted'>n/a</span>"
        return f"{float(value):.6f}"

    def _format_signed(self, value) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{float(value):+0.4f}"

    def _format_delta(self, value) -> str:
        if value is None or np.isnan(value):
            return "<span class='muted'>n/a</span>"
        css_class = "delta delta--positive" if float(value) >= 0 else "delta delta--negative"
        return f"<span class='{css_class}'>{self._format_signed(value)}</span>"

    def _normalized_width(self, value: float, lower: float, upper: float) -> float:
        if np.isnan(value):
            return 0.0
        if abs(upper - lower) < 1e-12:
            return 100.0
        return 16.0 + 84.0 * ((value - lower) / (upper - lower))

    def _format_value(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
