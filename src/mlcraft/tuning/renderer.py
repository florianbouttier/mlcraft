"""HTML renderer for Optuna tuning results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import TuningResult
from mlcraft.reporting.html import render_d3_card, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_tuning_context


class TuningReportRenderer:
    """Render a tuning result as a D3-based comparison dashboard."""

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
        if context.get("history_rows"):
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
            panel_id = self._metric_panel_id(metric["metric_name"])
            rows.append(
                "".join(
                    [
                        "<tr>",
                        "<td>",
                        (
                            f"<button class='table-action' type='button' data-toggle-button "
                            f"data-toggle-group='tuning-metrics' data-toggle-target='{escape(panel_id)}' "
                            "data-toggle-scroll='metric-explorer'>"
                            f"{escape(metric['metric_name'])}</button>"
                        ),
                        "</td>",
                        f"<td>{self._format_metric(metric['train_value'])}</td>",
                        f"<td>{self._format_metric(metric['val_value'])}</td>",
                        f"<td>{self._format_metric(metric['test_value']) if has_test else no_value}</td>",
                        f"<td>{self._format_delta(metric['score_gap'])}</td>",
                        f"<td>{self._format_delta(metric['holdout_delta']) if has_test else no_value}</td>",
                        f"<td>{'higher' if metric['higher_is_better'] else 'lower'}</td>",
                        "</tr>",
                    ]
                )
            )
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>KPI Matrix</span>"
            "<h2>Train, validation, and holdout summary for every tracked metric</h2>"
            "<p class='muted'>Start here, then jump into the metric explorer for fold-level behavior.</p>"
            "</div>"
            "<div class='summary-table'>"
            "<table><thead><tr>"
            "<th>Metric</th><th>Train</th><th>Validation</th><th>Test</th><th>Validation vs train</th><th>Test vs val</th><th>Direction</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
            "</section>"
        )

    def _render_summary_panel(self, context: dict[str, Any]) -> str:
        split_points = context.get("split_points") or []
        train_metric = split_points[0]["value"] if split_points else np.nan
        validation_metric = split_points[1]["value"] if len(split_points) > 1 else np.nan
        holdout_metric = context.get("test_metric_value")
        backend_name = context.get("selected_model_type") or (
            context["backend_summary_rows"][0]["backend_name"] if context.get("backend_summary_rows") else "model"
        )
        alpha_label = f"alpha = {context['alpha']:.4f}"
        best_score_label = f"{context['best_score']:.6f}"
        generalization_gap_label = f"{context['generalization_gap']:.6f}"
        gap_badge = (
            "<span class='badge badge--alert'>Validation gap needs attention</span>"
            if context["generalization_gap"] >= 0.05
            else "<span class='badge'>Validation gap is contained</span>"
        )
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Tuning Overview</span>"
            f"<h2>{escape(str(context['metric_name']))} stays in focus</h2>"
            "<p class='muted'>The report is intentionally interactive: pick a KPI, inspect folds, then compare final holdout behavior without leaving the page.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Selected backend', escape(str(backend_name)), 'Winning framework after the search.')}"
            f"{self._metric_card('Optimized metric', escape(str(context['metric_name'])), alpha_label)}"
            f"{self._metric_card('Best penalized score', best_score_label, 'Internal Optuna objective, always maximized.')}"
            f"{self._metric_card('Average train', self._plain_metric(train_metric), 'Primary KPI averaged across train folds.')}"
            f"{self._metric_card('Average validation', self._plain_metric(validation_metric), 'Primary KPI averaged across validation folds.')}"
            f"{self._metric_card('Average test', self._plain_metric(holdout_metric), 'Final holdout KPI after refit.')}"
            f"{self._metric_card('Generalization gap', generalization_gap_label, 'Absolute gap between train and validation scores.')}"
            "</div>"
            f"{gap_badge}"
            "</section>"
        )

    def _render_backend_overview(self, context: dict[str, Any]) -> str:
        rows = context.get("backend_summary_rows") or []
        if not rows:
            return ""
        payload = {
            "rows": [
                {
                    "label": row["backend_name"],
                    "value": float(row["best_score"]),
                    "color": self.palette["accent"] if row["is_selected"] else self.palette["series_muted"],
                }
                for row in rows
            ],
            "metricLabel": "penalized score",
        }
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Backend Comparison</span>"
            "<h2>Frameworks ranked by penalized validation score</h2>"
            "</div>"
            "<div class='viz-grid'>"
            + render_d3_card("Backend ranking", "mountBarChart", payload, wide=True, chart_id="tuning-backend-ranking")
            + "</div>"
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
            "<h2>Select the KPI you want to inspect</h2>"
            "<p class='muted'>Each KPI view compares fold-level train and validation values and keeps the holdout delta visible when available.</p>"
            "</div>"
            f"<div class='segmented' role='group' aria-label='Tuning metric selector'>{''.join(buttons)}</div>"
            + "".join(panels)
            + "</section>"
        )

    def _render_metric_panel(self, metric: dict[str, Any]) -> str:
        panel_id = self._metric_panel_id(metric["metric_name"])
        fold_payload = {
            "rows": [
                {
                    "label": f"Fold {row['fold_index']}",
                    "train_value": float(row["train_value"]),
                    "val_value": float(row["val_value"]),
                    "color": self.palette["accent_2"] if (row["score_gap"] or 0.0) >= 0 else self.palette["danger"],
                }
                for row in metric["fold_rows"]
                if row["train_value"] is not None and row["val_value"] is not None
            ],
            "trainColor": self.palette["accent"],
            "validationColor": self.palette["series_2"],
        }
        holdout_badge = (
            "No holdout metric available."
            if metric["test_value"] is None
            else f"Test vs validation = {self._plain_signed(metric['holdout_delta'])}"
        )
        return (
            f"<div class='metric-panel toggle-panel' data-toggle-panel data-toggle-group='tuning-metrics' data-toggle-panel='{escape(panel_id)}' hidden>"
            "<div class='metric-meta'>"
            f"<span class='badge'>{escape(metric['metric_name'])}</span>"
            f"<span class='badge'>{'higher is better' if metric['higher_is_better'] else 'lower is better'}</span>"
            f"<span class='badge'>{escape(holdout_badge)}</span>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Average train', self._plain_metric(metric['train_value']), 'Average train value across folds.')}"
            f"{self._metric_card('Average validation', self._plain_metric(metric['val_value']), 'Average validation value across folds.')}"
            f"{self._metric_card('Average test', self._plain_metric(metric['test_value']), 'Final holdout value after refit.')}"
            f"{self._metric_card('Validation vs train', self._plain_signed(metric['score_gap']), 'Positive means validation is better after direction normalization.')}"
            "</div>"
            "<div class='two-column-grid'>"
            + render_d3_card(f"{metric['metric_name']} by fold", "mountDumbbell", fold_payload, chart_id=f"{panel_id}-folds")
            + (
                "<div class='card'>"
                "<span class='eyebrow'>Reading Guide</span>"
                "<div class='notes-list'>"
                "<div class='note-item'><strong>Fold alignment</strong><p class='muted'>Shorter connectors mean train and validation stay close on that fold.</p></div>"
                f"<div class='note-item'><strong>Direction</strong><p class='muted'>{'Higher' if metric['higher_is_better'] else 'Lower'} values are better for this KPI.</p></div>"
                f"<div class='note-item'><strong>Holdout</strong><p class='muted'>{escape(holdout_badge)}</p></div>"
                "</div>"
                "</div>"
            )
            + "</div></div>"
        )

    def _render_search_dynamics(self, context: dict[str, Any]) -> str:
        history_rows = context.get("history_rows") or []
        if not history_rows:
            return ""
        history_payload = {
            "xLabel": "trial",
            "yLabel": "score",
            "series": [
                {
                    "name": "Penalized",
                    "color": self.palette["accent"],
                    "points": [{"x": float(row["trial_number"]), "y": float(row["penalized_score"])} for row in history_rows],
                },
                {
                    "name": "Validation",
                    "color": self.palette["series_2"],
                    "points": [{"x": float(row["trial_number"]), "y": float(row["val_score"])} for row in history_rows],
                },
                {
                    "name": "Train",
                    "color": self.palette["series_3"],
                    "points": [{"x": float(row["trial_number"]), "y": float(row["train_score"])} for row in history_rows],
                },
            ],
        }
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
            "</div>"
            "<div class='two-column-grid'>"
            + render_d3_card("Trial History", "mountSeriesChart", history_payload, chart_id="tuning-trial-history")
            + "<div class='card'><span class='eyebrow'>Top Trials</span><div class='notes-list'>"
            + "".join(top_trials)
            + "</div></div></div></section>"
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
        if not chips:
            chips = "<span class='muted'>No tuned parameters were recorded.</span>"
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Winning Configuration</span>"
            "<h2>Best trial configuration</h2>"
            "</div>"
            f"<div class='chip-cloud'>{chips}</div>"
            f"<div class='kpi-grid'>{''.join(narrative_cards)}</div>"
            "</section>"
        )

    def _render_holdout_curves(self, context: dict[str, Any]) -> str:
        groups = context["holdout_curve_groups"]
        buttons = []
        panels = []
        for index, group in enumerate(groups):
            panel_id = f"holdout-{self._metric_panel_id(group['curve_name'])}"
            default_attr = " data-toggle-default='true'" if index == 0 else ""
            buttons.append(
                f"<button class='segmented-button' type='button' data-toggle-button data-toggle-group='tuning-holdout-curves' data-toggle-target='{escape(panel_id)}'{default_attr}>{escape(group['title'])}</button>"
            )
            panels.append(
                f"<div class='toggle-panel' data-toggle-panel data-toggle-group='tuning-holdout-curves' data-toggle-panel='{escape(panel_id)}' hidden>"
                + render_d3_card(group["title"], "mountSeriesChart", self._curve_payload(group), wide=True, chart_id=f"{panel_id}-chart")
                + "</div>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Final Test Curves</span>"
            "<h2>Holdout behavior</h2>"
            "</div>"
            f"<div class='segmented' role='group' aria-label='Holdout curve selector'>{''.join(buttons)}</div>"
            + "".join(panels)
            + "</section>"
        )

    def _render_fold_curves(self, context: dict[str, Any]) -> str:
        groups = context.get("fold_curve_groups") or []
        buttons = []
        panels = []
        for index, group in enumerate(groups):
            panel_id = f"fold-{self._metric_panel_id(group['curve_name'])}"
            default_attr = " data-toggle-default='true'" if index == 0 else ""
            buttons.append(
                f"<button class='segmented-button' type='button' data-toggle-button data-toggle-group='tuning-fold-curves' data-toggle-target='{escape(panel_id)}'{default_attr}>{escape(group['title'])}</button>"
            )
            panels.append(
                f"<div class='toggle-panel' data-toggle-panel data-toggle-group='tuning-fold-curves' data-toggle-panel='{escape(panel_id)}' hidden>"
                + render_d3_card(group["title"], "mountSeriesChart", self._curve_payload(group), wide=True, chart_id=f"{panel_id}-chart")
                + "</div>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div class='section-head'>"
            "<span class='eyebrow'>Fold Curves</span>"
            "<h2>Train and validation curves by fold</h2>"
            "</div>"
            f"<div class='segmented' role='group' aria-label='Fold curve selector'>{''.join(buttons)}</div>"
            + "".join(panels)
            + "</section>"
        )

    def _curve_payload(self, curve_group: dict[str, Any]) -> dict[str, Any]:
        palette = chart_colors(self.palette)
        payload = {
            "xLabel": curve_group["x_label"],
            "yLabel": curve_group["y_label"],
            "series": [
                {
                    "name": series.get("series_name", series.get("prediction_name", "series")),
                    "color": palette[(series.get("fold_index", index) or index) % len(palette)],
                    "dash": "6 6" if series.get("split") == "validation" else None,
                    "points": [{"x": float(x_value), "y": float(y_value)} for x_value, y_value in zip(series["x"], series["y"])],
                }
                for index, series in enumerate(curve_group["series"])
            ],
        }
        if curve_group["curve_name"] in {"roc", "calibration", "poisson_calibration"}:
            payload["diagonal"] = [[0.0, 0.0], [1.0, 1.0]]
        return payload

    def _metric_panel_id(self, metric_name: str) -> str:
        safe = "".join(character if character.isalnum() else "-" for character in str(metric_name).lower()).strip("-")
        return f"metric-{safe or 'metric'}"

    def _metric_card(self, title: str, value: str, subtitle: str) -> str:
        return (
            "<div class='card metric-card'>"
            f"<span class='eyebrow'>{title}</span>"
            f"<strong class='metric-big'>{value}</strong>"
            f"<span class='metric-subtle'>{subtitle}</span>"
            "</div>"
        )

    def _format_metric(self, value: float | None) -> str:
        if value is None or np.isnan(value):
            return "<span class='muted'>n/a</span>"
        return f"{float(value):.6f}"

    def _plain_metric(self, value: float | None) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{float(value):.6f}"

    def _plain_signed(self, value: float | None) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{float(value):+0.4f}"

    def _format_delta(self, value: float | None) -> str:
        if value is None or np.isnan(value):
            return "<span class='muted'>n/a</span>"
        css_class = "delta delta--positive" if float(value) >= 0 else "delta delta--negative"
        return f"<span class='{css_class}'>{self._plain_signed(value)}</span>"

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
