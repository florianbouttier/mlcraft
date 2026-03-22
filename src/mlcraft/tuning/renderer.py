"""HTML renderer for Optuna tuning results."""

from __future__ import annotations

from html import escape
from typing import Any
import warnings

import numpy as np

from mlcraft.core.results import TuningResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html
from mlcraft.reporting.palette import chart_colors, get_report_palette
from mlcraft.reporting.view_models import build_tuning_context


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
        sections.append(self._render_summary_panel(context))
        sections.append(self._render_generalization_overview(context))
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
        validation_score = f"{context['val_score']:.6f}"
        generalization_gap = f"{context['generalization_gap']:.6f}"
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Tuning Overview</span>"
            f"<h2>{escape(str(context['metric_name']))} across train, validation, and test</h2>"
            "<p class='muted'>The dashboard highlights generalization first, then shows the official Optuna visualizations without recreating them.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Optimized metric', escape(str(context['metric_name'])), alpha_value)}"
            f"{self._metric_card('Best penalized score', best_score, 'Optuna maximizes this internal score.')}"
            f"{self._metric_card('Validation score', validation_score, 'Fold-average internal validation score.')}"
            f"{self._metric_card('Generalization gap', generalization_gap, 'Absolute train-vs-validation score gap.')}"
            f"{self._metric_card('Final test metric', holdout_value, 'Available only when a holdout set is provided.')}"
            "</div>"
            f"{gap_badge}"
            "</section>"
        )

    def _render_generalization_overview(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        split_fig = self._plot_split_journey(context)
        fold_fig = self._plot_fold_gaps(context)
        try:
            return (
                "<section class='panel section-stack'>"
                "<div>"
                "<span class='eyebrow'>Generalization Overview</span>"
                "<h2>What the model keeps on validation and test</h2>"
                "<p class='muted'>The first chart tracks metric drift from train to validation to final holdout. The second one surfaces overfitting fold by fold.</p>"
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

    def _render_optuna_visualizations(self, context: dict[str, Any]) -> str:
        sections = [
            "<section class='panel section-stack'>",
            "<div>",
            "<span class='eyebrow'>Optuna Visualizations</span>",
            "<h2>Official study plots</h2>",
            "<p class='muted'>These plots come from Optuna itself. The renderer does not replace them with homemade versions.</p>",
            "</div>",
            "<div class='viz-grid'>",
        ]
        sections.extend(self._optuna_plots(context))
        sections.append("</div></section>")
        return "".join(sections)

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
            "<div>"
            "<span class='eyebrow'>Winning Configuration</span>"
            "<h2>Best trial configuration</h2>"
            "<p class='muted'>Parameter values stay visible, but in a compact chip cloud instead of a wide table.</p>"
            "</div>"
            f"<div class='chip-cloud'>{chips or empty_chip}</div>"
            f"<div class='kpi-grid'>{''.join(narrative_cards)}</div>"
            "</section>"
        )

    def _render_holdout_curves(self, context: dict[str, Any]) -> str:
        import matplotlib.pyplot as plt

        figures = self._plot_curve_groups(context["holdout_curve_groups"])
        sections = [
            "<section class='panel section-stack'>",
            "<div>",
            "<span class='eyebrow'>Final Test Curves</span>",
            "<h2>Holdout behavior</h2>",
            "<p class='muted'>Final holdout curves stay graphical and use shared axes when several predictions are present.</p>",
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
        if context.get("study") is None:
            return ["<div class='card'><p class='muted'>The Optuna study object is not attached, so official study plots are unavailable.</p></div>"]

        plotly_cards = self._optuna_plotly_cards(context)
        if plotly_cards:
            return plotly_cards

        matplotlib_cards = self._optuna_matplotlib_cards(context)
        if matplotlib_cards:
            return matplotlib_cards

        return ["<div class='card'><p class='muted'>Official Optuna visualizations could not be rendered in this environment.</p></div>"]

    def _optuna_plotly_cards(self, context: dict[str, Any]) -> list[str]:
        try:
            import plotly.io as pio
            from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
        except Exception:
            return []

        plotters = {
            "optimization_history": ("Optimization History", plot_optimization_history),
            "param_importances": ("Parameter Importance", plot_param_importances),
            "parallel_coordinate": ("Parallel Coordinates", plot_parallel_coordinate),
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
        return cards

    def _optuna_matplotlib_cards(self, context: dict[str, Any]) -> list[str]:
        import matplotlib.pyplot as plt

        try:
            from optuna.visualization.matplotlib import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
        except Exception:
            return []

        plotters = {
            "optimization_history": ("Optimization History", plot_optimization_history),
            "param_importances": ("Parameter Importance", plot_param_importances),
            "parallel_coordinate": ("Parallel Coordinates", plot_parallel_coordinate),
        }
        cards: list[str] = []
        for plot_name in context["optuna_plots"]:
            title, plotter = plotters[plot_name]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    axis = plotter(context["study"])
                figure = axis.figure
                cards.append(self._figure_card(title, figure))
                plt.close(figure)
            except Exception:
                continue
        return cards

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
