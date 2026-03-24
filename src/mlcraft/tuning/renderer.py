"""HTML renderer for Optuna tuning results."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

from mlcraft.core.results import TuningResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_tuning_context
from mlcraft.utils.optional import optional_import


class TuningReportRenderer:
    """Render a tuning result as a visual HTML dashboard."""

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
        """Render a tuning report from a pre-built dictionary context.

        Args:
            context: Dictionary returned by `build_context()`.
            output_path: Optional file path used to persist the rendered HTML.

        Returns:
            str: Standalone HTML document.
        """

        sections: list[str] = []
        if context.get("title"):
            sections.append(f"<h1>{escape(str(context['title']))}</h1>")
        if context.get("backend_summary_rows"):
            sections.append(self._render_backend_overview(context))
        sections.append(self._render_summary_panel(context))
        sections.append(self._render_generalization_overview(context))
        if context.get("fold_curve_groups"):
            sections.append(self._render_fold_curves(context))
        if context.get("study") is not None:
            sections.append(self._render_optuna_visualizations(context))
        sections.append(self._render_configuration_panel(context))
        if context.get("holdout_curve_groups"):
            sections.append(self._render_holdout_curves(context))
        html = wrap_html(str(context.get("title") or "mlcraft Tuning Report"), "".join(sections), palette=self.palette)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_summary_panel(self, context: dict[str, Any]) -> str:
        gap_badge = (
            "<span class='badge badge--alert'>Overfitting watch</span>"
            if context["generalization_gap"] >= 0.05
            else "<span class='badge'>Gap under control</span>"
        )
        holdout_value = "n/a" if context["test_metric_value"] is None else f"{context['test_metric_value']:.6f}"
        alpha_value = f"alpha = {context['alpha']:.4f}"
        best_score = f"{context['best_score']:.6f}"
        train_metric = context["split_points"][0]["value"] if context.get("split_points") else np.nan
        validation_metric = context["split_points"][1]["value"] if len(context.get("split_points", [])) > 1 else np.nan
        generalization_gap = f"{context['generalization_gap']:.6f}"
        metric_label = escape(str(context["metric_name"]))
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Tuning Overview</span>"
            f"<h2>{metric_label} averages, folds, and backend comparison</h2>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Optimized metric', metric_label, alpha_value)}"
            f"{self._metric_card('Best penalized score', best_score, 'Optuna maximizes this internal score.')}"
            f"{self._metric_card(f'Average train {metric_label}', f'{train_metric:.6f}', 'Mean train metric across CV folds.')}"
            f"{self._metric_card(f'Average validation {metric_label}', f'{validation_metric:.6f}', 'Mean validation metric across CV folds.')}"
            f"{self._metric_card(f'Average test {metric_label}', holdout_value, 'Holdout metric after refit on the full train partition.')}"
            f"{self._metric_card('Generalization gap', generalization_gap, 'Absolute train-vs-validation score gap.')}"
            "</div>"
            f"{gap_badge}"
            "</section>"
        )

    def _render_generalization_overview(self, context: dict[str, Any]) -> str:
        try:
            return self._render_generalization_overview_plotly(context)
        except Exception:
            return self._render_generalization_overview_matplotlib(context)

    def _render_generalization_overview_plotly(self, context: dict[str, Any]) -> str:
        figures = [
            ("Optimized Metric by Fold", self._plot_fold_metric_lines(context), True),
            ("Generalization Gap by Fold", self._plot_generalization_gap(context), True),
        ]
        figures.extend(
            (f"{metric_name} by Fold", self._plot_metric_by_fold(context, metric_name), True)
            for metric_name in self._ordered_fold_metrics(context)
        )
        return (
            "<section class='panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Generalization Overview</span>"
            "<h2>Each KPI shows train and validation side by side for every fold</h2>"
            "</div>"
            "<div class='viz-grid'>"
            f"{''.join(self._plotly_cards(figures))}"
            "</div>"
            "</section>"
        )

    def _render_generalization_overview_matplotlib(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        split_fig = self._plot_split_journey(context)
        fold_fig = self._plot_fold_gaps(context)
        try:
            return (
                "<section class='panel section-stack'>"
                "<div>"
                "<span class='eyebrow'>Generalization Overview</span>"
                "<h2>What the model keeps on validation and test</h2>"
                "</div>"
                "<div class='viz-grid'>"
                f"{self._figure_card('Train / Validation / Test Journey', split_fig, wide=True)}"
                f"{self._figure_card('Fold Generalization Gap', fold_fig)}"
                "</div>"
                "</section>"
            )
        finally:
            plt.close(split_fig)
            plt.close(fold_fig)

    def _render_backend_overview(self, context: dict[str, Any]) -> str:
        rows = context.get("backend_summary_rows") or []
        if not rows:
            return ""
        selected_row = next((row for row in rows if row["is_selected"]), rows[0])
        metric_label = escape(str(context["metric_name"]))
        selected_best_score = f"{selected_row['best_score']:.6f}"
        selected_validation = f"{selected_row['val_metric_value']:.6f}"
        selected_test = "n/a" if selected_row["test_metric_value"] is None else f"{selected_row['test_metric_value']:.6f}"
        try:
            figures = [
                ("Backend Metric Averages", self._plot_backend_metric_averages(context), True),
                ("Backend Penalized Score Ranking", self._plot_backend_penalized_scores(context), False),
            ]
            charts = "".join(self._plotly_cards(figures))
        except Exception:
            charts = "<div class='card'><p class='muted'>Backend visuals could not be rendered in this environment.</p></div>"

        table_rows = []
        for row in rows:
            selected_badge = " <span class='badge'>selected</span>" if row["is_selected"] else ""
            test_value = "n/a" if row["test_metric_value"] is None else f"{row['test_metric_value']:.6f}"
            table_rows.append(
                "<tr>"
                f"<td>{escape(row['backend_name'])}{selected_badge}</td>"
                f"<td>{row['best_score']:.6f}</td>"
                f"<td>{row['train_metric_value']:.6f}</td>"
                f"<td>{row['val_metric_value']:.6f}</td>"
                f"<td>{test_value}</td>"
                "</tr>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Backend Comparison</span>"
            "<h2>Best backend first, then the full framework comparison</h2>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Best backend', escape(selected_row['backend_name']), 'Winning framework before fold-level inspection.')}"
            f"{self._metric_card('Best penalized score', selected_best_score, 'The backend winner on the Optuna objective.')}"
            f"{self._metric_card(f'Validation {metric_label}', selected_validation, 'Mean validation metric for the winning backend.')}"
            f"{self._metric_card(f'Test {metric_label}', selected_test, 'Mean holdout metric after final refit.')}"
            "</div>"
            "<div class='viz-grid'>"
            f"{charts}"
            "<div class='card card--wide'>"
            "<span class='eyebrow'>Backend Summary Table</span>"
            "<table><thead><tr><th>Backend</th><th>Best penalized score</th><th>Average train metric</th><th>Average validation metric</th><th>Average test metric</th></tr></thead><tbody>"
            + "".join(table_rows)
            + "</tbody></table></div>"
            "</div>"
            "</section>"
        )

    def _render_optuna_visualizations(self, context: dict[str, Any]) -> str:
        sections = [
            "<section class='panel section-stack'>",
            "<div>",
            "<span class='eyebrow'>Optuna Visualizations</span>",
            "<h2>Optuna Plotly</h2>",
            "</div>",
            "<div class='viz-grid'>",
        ]
        sections.extend(self._optuna_plots(context))
        sections.append("</div></section>")
        return "".join(sections)

    def _render_configuration_panel(self, context: dict[str, Any]) -> str:
        narrative_cards = []
        if context.get("selected_model_type"):
            narrative_cards.append(
                self._metric_card(
                    "Selected backend",
                    str(context["selected_model_type"]),
                    "Best validation-penalized Optuna search across candidate frameworks.",
                )
            )
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
        comparison_rows = []
        for model_type, payload in context.get("backend_comparison", {}).items():
            comparison_rows.append(
                "<tr>"
                f"<td>{escape(str(model_type))}</td>"
                f"<td>{payload['best_score']:.6f}</td>"
                f"<td>{payload['val_metrics'].get(context['metric_name'], next(iter(payload['val_metrics'].values()), float('nan'))):.6f}</td>"
                f"<td>{payload['test_metrics'].get(context['metric_name'], next(iter(payload['test_metrics'].values()), float('nan'))) if payload['test_metrics'] else float('nan'):.6f}</td>"
                "</tr>"
            )
        comparison_table = ""
        if comparison_rows:
            comparison_table = (
                "<div class='card card--wide'>"
                "<span class='eyebrow'>Backend Comparison</span>"
                "<table><thead><tr><th>Backend</th><th>Best score</th><th>Validation metric</th><th>Test metric</th></tr></thead><tbody>"
                + "".join(comparison_rows)
                + "</tbody></table></div>"
            )
        return (
            "<section class='panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Winning Configuration</span>"
            "<h2>Best trial configuration</h2>"
            "</div>"
            f"<div class='chip-cloud'>{chips or empty_chip}</div>"
            f"<div class='kpi-grid'>{''.join(narrative_cards)}</div>"
            f"{comparison_table}"
            "</section>"
        )

    def _render_holdout_curves(self, context: dict[str, Any]) -> str:
        try:
            return (
                "<section class='panel section-stack'>"
                "<div>"
                "<span class='eyebrow'>Final Test Curves</span>"
                "<h2>Holdout behavior</h2>"
                "</div>"
                "<div class='viz-grid'>"
                f"{''.join(self._plotly_cards([(group['title'], self._plot_curve_group_figure(group), True) for group in context['holdout_curve_groups']]))}"
                "</div>"
                "</section>"
            )
        except Exception:
            return self._render_holdout_curves_matplotlib(context)

    def _render_holdout_curves_matplotlib(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        figures = self._plot_curve_groups(context["holdout_curve_groups"])
        sections = [
            "<section class='panel section-stack'>",
            "<div>",
            "<span class='eyebrow'>Final Test Curves</span>",
            "<h2>Holdout behavior</h2>",
            "</div>",
            "<div class='viz-grid'>",
        ]
        try:
            for title, figure in figures:
                sections.append(self._figure_card(title, figure))
        finally:
            for _, figure in figures:
                plt.close(figure)
        sections.append("</div></section>")
        return "".join(sections)

    def _render_fold_curves(self, context: dict[str, Any]) -> str:
        curve_groups = context.get("fold_curve_groups") or []
        if not curve_groups:
            return ""
        try:
            cards = self._plotly_cards([(group["title"], self._plot_curve_group_figure(group), True) for group in curve_groups])
        except Exception:
            cards = ["<div class='card'><p class='muted'>Fold-level curves could not be rendered in this environment.</p></div>"]
        return (
            "<section class='panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Fold Curves</span>"
            "<h2>ROC, PR, calibration, and task-specific curves on train and validation for each fold</h2>"
            "</div>"
            "<div class='viz-grid'>"
            f"{''.join(cards)}"
            "</div>"
            "</section>"
        )

    def _plot_split_journey(self, context: dict[str, Any]):
        import matplotlib.pyplot as plt

        points = context["split_points"]
        y_values = np.asarray([point["value"] for point in points], dtype=float)
        x_values = np.arange(len(points))
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fbfcfd")
        ax.plot(x_values, y_values, color=self.palette["accent"], linewidth=3.2, marker="o", markersize=9)
        ax.fill_between(x_values, y_values, alpha=0.08, color=self.palette["accent"])
        for idx, value in enumerate(y_values):
            ax.text(x_values[idx], value, f"  {value:.6f}", va="center", fontsize=10, fontweight="bold", color=self.palette["text_main"])
        ax.set_xticks(x_values, labels=[point["label"].replace("_", " ").title() for point in points])
        ax.set_ylabel(f"{context['metric_name']} (raw metric value)")
        ax.set_title("Train to validation to test", loc="left", fontsize=16, fontweight="bold")
        ax.grid(alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _plot_fold_gaps(self, context: dict[str, Any]):
        import matplotlib.pyplot as plt

        folds = context["fold_points"]
        fig, ax = plt.subplots(figsize=(9.0, max(4.8, 1.0 + 0.9 * max(len(folds), 1))))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fbfcfd")
        if not folds:
            ax.text(0.5, 0.5, "No fold summaries available.", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        positions = np.arange(len(folds))
        for idx, fold in enumerate(folds):
            color = self.palette["danger"] if fold["gap"] > 0.05 else self.palette["accent"]
            ax.plot([fold["val_score"], fold["train_score"]], [idx, idx], color=color, linewidth=4, solid_capstyle="round")
            ax.scatter(fold["val_score"], idx, color=self.palette["series_2"], s=90, zorder=3, label="Validation" if idx == 0 else None)
            ax.scatter(fold["train_score"], idx, color=self.palette["accent"], s=90, zorder=3, label="Train" if idx == 0 else None)
            ax.scatter(fold["penalized_score"], idx, color=self.palette["danger"], s=100, marker="D", zorder=3, label="Penalized" if idx == 0 else None)
        ax.set_yticks(positions, labels=[f"Fold {fold['fold_index']}" for fold in folds])
        ax.set_xlabel("Internal score (higher is always better)")
        ax.set_title("Fold generalization gap", loc="left", fontsize=16, fontweight="bold")
        ax.grid(alpha=0.18, axis="x")
        ax.legend(frameon=False, loc="lower right")
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _optuna_plots(self, context: dict[str, Any]) -> list[str]:
        try:
            return self._optuna_plotly_cards(context)
        except Exception:
            return ["<div class='card'><p class='muted'>Official Optuna visualizations could not be rendered in this environment.</p></div>"]

    def _optuna_plotly_cards(self, context: dict[str, Any]) -> list[str]:
        optional_import("plotly", extra_name="reporting")
        import plotly.io as pio
        from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice

        plotters = {
            "optimization_history": ("Optimization History", plot_optimization_history),
            "parallel_coordinate": ("Parallel Coordinates", plot_parallel_coordinate),
            "slice": ("Slice Plot", plot_slice),
        }
        cards: list[str] = []
        include_js = True
        for plot_name in context["optuna_plots"]:
            title, plotter = plotters[plot_name]
            try:
                figure = plotter(context["study"])
                fragment = pio.to_html(
                    figure,
                    full_html=False,
                    include_plotlyjs="inline" if include_js else False,
                    config={"displayModeBar": False, "responsive": True},
                )
                include_js = False
                cards.append(
                    "<div class='card plotly-card'>"
                    f"<span class='eyebrow'>{escape(title)}</span>"
                    f"{fragment}"
                    "</div>"
                )
            except Exception:
                continue
        if not cards:
            raise RuntimeError("Official Optuna Plotly visualizations could not be rendered.")
        return cards

    def _plotly_cards(self, figures: list[tuple[str, object, bool]]) -> list[str]:
        optional_import("plotly", extra_name="reporting")
        import plotly.io as pio

        cards: list[str] = []
        include_js = True
        for title, figure, wide in figures:
            fragment = pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs="inline" if include_js else False,
                config={"displayModeBar": True, "responsive": True, "scrollZoom": True},
            )
            include_js = False
            wide_class = " card--wide" if wide else ""
            cards.append(
                f"<div class='card plotly-card{wide_class}'>"
                f"<span class='eyebrow'>{escape(title)}</span>"
                f"{fragment}"
                "</div>"
            )
        return cards

    def _plot_backend_metric_averages(self, context: dict[str, Any]):
        optional_import("plotly", extra_name="reporting")
        import plotly.graph_objects as go

        rows = context["backend_summary_rows"]
        backends = [row["backend_name"] for row in rows]
        figure = go.Figure()
        figure.add_bar(name="Train", x=backends, y=[row["train_metric_value"] for row in rows], marker_color=self.palette["accent"])
        figure.add_bar(name="Validation", x=backends, y=[row["val_metric_value"] for row in rows], marker_color=self.palette["series_2"])
        test_values = [row["test_metric_value"] for row in rows]
        if any(value is not None for value in test_values):
            figure.add_bar(
                name="Test",
                x=backends,
                y=[np.nan if value is None else value for value in test_values],
                marker_color=self.palette["series_3"],
            )
        figure.update_layout(
            barmode="group",
            title=f"Average {context['metric_name']} by backend",
            xaxis_title="Backend",
            yaxis_title=context["metric_name"],
            legend_title="Split",
            template="plotly_white",
        )
        return figure

    def _plot_backend_penalized_scores(self, context: dict[str, Any]):
        optional_import("plotly", extra_name="reporting")
        import plotly.graph_objects as go

        rows = context["backend_summary_rows"]
        colors = [self.palette["accent"] if row["is_selected"] else self.palette["series_muted"] for row in rows]
        figure = go.Figure(
            go.Bar(
                x=[row["best_score"] for row in rows],
                y=[row["backend_name"] for row in rows],
                orientation="h",
                marker_color=colors,
                text=[f"{row['best_score']:.6f}" for row in rows],
                textposition="outside",
            )
        )
        figure.update_layout(
            title="Best penalized score by backend",
            xaxis_title="Best penalized score",
            yaxis_title="Backend",
            template="plotly_white",
        )
        return figure

    def _ordered_fold_metrics(self, context: dict[str, Any]) -> list[str]:
        ordered = [str(context["metric_name"])]
        for row in context.get("fold_metric_rows", []):
            metric_name = str(row["metric_name"])
            if metric_name not in ordered:
                ordered.append(metric_name)
        return ordered

    def _plot_fold_metric_lines(self, context: dict[str, Any]):
        optional_import("plotly", extra_name="reporting")
        import plotly.graph_objects as go

        folds = context["fold_points"]
        fold_index = [f"Fold {point['fold_index']}" for point in folds]
        train_values = [point["train_metric_value"] for point in folds]
        val_values = [point["val_metric_value"] for point in folds]
        penalized_values = [point["penalized_score"] for point in folds]
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=fold_index,
                y=train_values,
                mode="lines+markers+text",
                name="Train",
                line={"color": self.palette["accent"], "width": 3},
                text=[f"{value:.6f}" for value in train_values],
                textposition="top center",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=fold_index,
                y=val_values,
                mode="lines+markers+text",
                name="Validation",
                line={"color": self.palette["series_2"], "width": 3},
                text=[f"{value:.6f}" for value in val_values],
                textposition="top center",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=fold_index,
                y=penalized_values,
                mode="lines+markers",
                name="Penalized score",
                line={"color": self.palette["danger"], "width": 2, "dash": "dot"},
            )
        )
        figure.update_layout(
            title=f"Optimized metric ({context['metric_name']}) by fold",
            xaxis_title="Fold",
            yaxis_title=context["metric_name"],
            template="plotly_white",
        )
        return figure

    def _plot_generalization_gap(self, context: dict[str, Any]):
        optional_import("plotly", extra_name="reporting")
        import plotly.graph_objects as go

        folds = context["fold_points"]
        figure = go.Figure(
            go.Bar(
                x=[f"Fold {point['fold_index']}" for point in folds],
                y=[abs(float(point["gap"])) for point in folds],
                marker_color=self.palette["danger"],
                text=[f"{abs(float(point['gap'])):.6f}" for point in folds],
                textposition="outside",
            )
        )
        figure.update_layout(
            title="Generalization gap by fold",
            xaxis_title="Fold",
            yaxis_title="Absolute train-vs-validation score gap",
            template="plotly_white",
        )
        return figure

    def _plot_metric_by_fold(self, context: dict[str, Any], metric_name: str):
        optional_import("plotly", extra_name="reporting")
        import plotly.graph_objects as go

        rows = [row for row in context["fold_metric_rows"] if str(row["metric_name"]) == str(metric_name)]
        fold_labels = sorted({int(row["fold_index"]) for row in rows})
        train_map = {int(row["fold_index"]): float(row["value"]) for row in rows if row["split"] == "train"}
        validation_map = {int(row["fold_index"]): float(row["value"]) for row in rows if row["split"] == "validation"}
        x_values = [f"Fold {fold_index}" for fold_index in fold_labels]
        train_values = [train_map.get(fold_index, np.nan) for fold_index in fold_labels]
        validation_values = [validation_map.get(fold_index, np.nan) for fold_index in fold_labels]

        figure = go.Figure()
        figure.add_bar(
            name="Train",
            x=x_values,
            y=train_values,
            marker_color=self.palette["accent"],
            text=[("" if np.isnan(value) else f"{value:.6f}") for value in train_values],
            textposition="outside",
        )
        figure.add_bar(
            name="Validation",
            x=x_values,
            y=validation_values,
            marker_color=self.palette["series_2"],
            text=[("" if np.isnan(value) else f"{value:.6f}") for value in validation_values],
            textposition="outside",
        )
        figure.update_layout(
            barmode="group",
            title=f"{metric_name} by fold",
            xaxis_title="Fold",
            yaxis_title=metric_name,
            template="plotly_white",
        )
        return figure

    def _plot_curve_group_figure(self, curve_group: dict[str, Any]):
        optional_import("plotly", extra_name="reporting")
        import plotly.graph_objects as go

        figure = go.Figure()
        colors = chart_colors(self.palette)
        for index, series in enumerate(curve_group["series"]):
            fold_index = series.get("fold_index")
            split_name = series.get("split")
            color = colors[int(fold_index) % len(colors)] if fold_index is not None else colors[index % len(colors)]
            dash = "dash" if split_name == "validation" else "solid"
            legend_group = f"fold_{fold_index}" if fold_index is not None else f"series_{index}"
            figure.add_trace(
                go.Scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines",
                    name=series.get("series_name", series.get("prediction_name", "series")),
                    line={"color": color, "dash": dash, "width": 2.8},
                    legendgroup=legend_group,
                    hovertemplate=(
                        f"{escape(series.get('series_name', 'series'))}<br>"
                        + f"{escape(curve_group['x_label'])}=%{{x:.6f}}<br>"
                        + f"{escape(curve_group['y_label'])}=%{{y:.6f}}<extra></extra>"
                    ),
                )
            )
        if curve_group["curve_name"] in {"roc", "calibration", "poisson_calibration"}:
            diagonal_start = min(min(series["x"] or [0.0]) for series in curve_group["series"])
            diagonal_end = max(max(series["x"] or [1.0]) for series in curve_group["series"])
            figure.add_trace(
                go.Scatter(
                    x=[diagonal_start, diagonal_end],
                    y=[diagonal_start, diagonal_end],
                    mode="lines",
                    name="Diagonal",
                    line={"color": self.palette["grid_soft"], "dash": "dot"},
                )
            )
        figure.update_layout(
            title=curve_group["title"],
            xaxis_title=curve_group["x_label"],
            yaxis_title=curve_group["y_label"],
            template="plotly_white",
        )
        return figure

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

    def _format_value(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
