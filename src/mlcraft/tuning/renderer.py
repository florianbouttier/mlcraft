"""HTML renderer for Optuna tuning results."""

from __future__ import annotations

from collections import defaultdict
from html import escape
import warnings

import numpy as np

from mlcraft.core.results import CurveData, EvaluationResult, TuningResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html


class TuningReportRenderer:
    """Render a tuning result as a visual HTML dashboard."""

    def render(self, result: TuningResult, *, title: str | None = "mlcraft Tuning Report", output_path=None) -> str:
        """Render a complete tuning report.

        Args:
            result: Tuning output to render.
            title: Title displayed in the standalone HTML document. When
                `None`, no top-level heading is added to the body.
            output_path: Optional file path used to persist the rendered HTML.

        Returns:
            str: Standalone HTML document.
        """

        sections: list[str] = []
        if title:
            sections.append(f"<h1>{escape(title)}</h1>")
        sections.append(self._render_summary_panel(result))
        sections.append(self._render_generalization_overview(result))
        sections.append(self._render_optuna_visualizations(result))
        sections.append(self._render_configuration_panel(result))
        if result.test_evaluation is not None and result.test_evaluation.curves:
            sections.append(self._render_holdout_curves(result.test_evaluation))
        html = wrap_html(title or "mlcraft Tuning Report", "".join(sections))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_summary_panel(self, result: TuningResult) -> str:
        metric_name = result.metric_name or "score"
        generalization_gap = abs(result.best_trial.train_score - result.best_trial.val_score)
        gap_badge = (
            "<span class='badge badge--alert'>Overfitting watch</span>"
            if generalization_gap >= 0.05
            else "<span class='badge'>Gap under control</span>"
        )
        holdout_value = None
        if result.test_metrics and metric_name in result.test_metrics:
            holdout_value = f"{result.test_metrics[metric_name]:.6f}"
        return (
            "<section class='panel hero-panel section-stack'>"
            "<div>"
            "<span class='eyebrow'>Tuning Overview</span>"
            f"<h2>{escape(metric_name)} across train, validation, and test</h2>"
            "<p class='muted'>The dashboard highlights generalization first, then shows the official Optuna visualizations without recreating them.</p>"
            "</div>"
            "<div class='kpi-grid'>"
            f"{self._metric_card('Optimized metric', escape(metric_name), f'alpha = {result.alpha:.4f}')}"
            f"{self._metric_card('Best penalized score', f'{result.best_score:.6f}', 'Optuna maximizes this internal score.')}"
            f"{self._metric_card('Validation score', f'{result.best_trial.val_score:.6f}', 'Fold-average internal validation score.')}"
            f"{self._metric_card('Generalization gap', f'{generalization_gap:.6f}', 'Absolute train-vs-validation score gap.')}"
            f"{self._metric_card('Final test metric', holdout_value or 'n/a', 'Available only when a holdout set is provided.')}"
            "</div>"
            f"{gap_badge}"
            "</section>"
        )

    def _render_generalization_overview(self, result: TuningResult) -> str:
        import matplotlib.pyplot as plt

        split_fig = self._plot_split_journey(result)
        fold_fig = self._plot_fold_gaps(result)
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

    def _render_optuna_visualizations(self, result: TuningResult) -> str:
        sections = [
            "<section class='panel section-stack'>",
            "<div>",
            "<span class='eyebrow'>Optuna Visualizations</span>",
            "<h2>Official study plots</h2>",
            "<p class='muted'>These plots come from Optuna itself. The renderer does not replace them with homemade versions.</p>",
            "</div>",
            "<div class='viz-grid'>",
        ]
        sections.extend(self._optuna_plots(result))
        sections.append("</div></section>")
        return "".join(sections)

    def _render_configuration_panel(self, result: TuningResult) -> str:
        best_fold = max(result.fold_summaries, key=lambda fold: fold.val_score) if result.fold_summaries else None
        worst_fold = min(result.fold_summaries, key=lambda fold: fold.val_score) if result.fold_summaries else None
        narrative_cards = []
        if best_fold is not None:
            narrative_cards.append(
                self._metric_card(
                    "Best validation fold",
                    f"Fold {best_fold.fold_index}",
                    f"validation score = {best_fold.val_score:.6f}",
                )
            )
        if worst_fold is not None:
            narrative_cards.append(
                self._metric_card(
                    "Most fragile fold",
                    f"Fold {worst_fold.fold_index}",
                    f"validation score = {worst_fold.val_score:.6f}",
                )
            )

        chips = "".join(
            f"<span class='chip'><strong>{escape(str(key))}</strong><span>{escape(self._format_value(value))}</span></span>"
            for key, value in result.best_params.items()
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

    def _render_holdout_curves(self, evaluation_result: EvaluationResult) -> str:
        import matplotlib.pyplot as plt

        figures = self._plot_curve_groups(evaluation_result.curves)
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

    def _plot_split_journey(self, result: TuningResult):
        import matplotlib.pyplot as plt

        metric_name = result.metric_name or next(iter(result.train_metrics.keys()), "metric")
        split_labels = ["train", "validation"]
        split_values = [
            self._metric_value(result.train_metrics, metric_name),
            self._metric_value(result.val_metrics, metric_name),
        ]
        if result.test_metrics and metric_name in result.test_metrics:
            split_labels.append("final_test")
            split_values.append(self._metric_value(result.test_metrics, metric_name))
        y_values = np.asarray(split_values, dtype=float)
        x_values = np.arange(len(split_labels))

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fbfcfd")
        ax.plot(x_values, y_values, color="#0f766e", linewidth=3.2, marker="o", markersize=9)
        ax.fill_between(x_values, y_values, alpha=0.08, color="#0f766e")
        for idx, value in enumerate(y_values):
            ax.text(x_values[idx], value, f"  {value:.6f}", va="center", fontsize=10, fontweight="bold", color="#16324f")
        ax.set_xticks(x_values, labels=[label.replace("_", " ").title() for label in split_labels])
        ax.set_ylabel(f"{metric_name} (raw metric value)")
        ax.set_title("Train to validation to test", loc="left", fontsize=16, fontweight="bold")
        ax.grid(alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _plot_fold_gaps(self, result: TuningResult):
        import matplotlib.pyplot as plt

        folds = result.fold_summaries
        fig, ax = plt.subplots(figsize=(9.0, max(4.8, 1.0 + 0.9 * max(len(folds), 1))))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fbfcfd")
        if not folds:
            ax.text(0.5, 0.5, "No fold summaries available.", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        positions = np.arange(len(folds))
        for idx, fold in enumerate(folds):
            gap = fold.train_score - fold.val_score
            color = "#c2410c" if gap > 0.05 else "#0f766e"
            ax.plot([fold.val_score, fold.train_score], [idx, idx], color=color, linewidth=4, solid_capstyle="round")
            ax.scatter(fold.val_score, idx, color="#2563eb", s=90, zorder=3, label="Validation" if idx == 0 else None)
            ax.scatter(fold.train_score, idx, color="#0f766e", s=90, zorder=3, label="Train" if idx == 0 else None)
            ax.scatter(fold.penalized_score, idx, color="#c2410c", s=100, marker="D", zorder=3, label="Penalized" if idx == 0 else None)
        ax.set_yticks(positions, labels=[f"Fold {fold.fold_index}" for fold in folds])
        ax.set_xlabel("Internal score (higher is always better)")
        ax.set_title("Fold generalization gap", loc="left", fontsize=16, fontweight="bold")
        ax.grid(alpha=0.18, axis="x")
        ax.legend(frameon=False, loc="lower right")
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig

    def _optuna_plots(self, result: TuningResult) -> list[str]:
        if result.study is None:
            return ["<div class='card'><p class='muted'>The Optuna study object is not attached, so official study plots are unavailable.</p></div>"]

        plotly_cards = self._optuna_plotly_cards(result)
        if plotly_cards:
            return plotly_cards

        matplotlib_cards = self._optuna_matplotlib_cards(result)
        if matplotlib_cards:
            return matplotlib_cards

        return ["<div class='card'><p class='muted'>Official Optuna visualizations could not be rendered in this environment.</p></div>"]

    def _optuna_plotly_cards(self, result: TuningResult) -> list[str]:
        try:
            import plotly.io as pio
            from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
        except Exception:
            return []

        cards: list[str] = []
        include_js = True
        for title, plotter in (
            ("Optimization History", plot_optimization_history),
            ("Parameter Importance", plot_param_importances),
            ("Parallel Coordinates", plot_parallel_coordinate),
        ):
            try:
                figure = plotter(result.study)
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

    def _optuna_matplotlib_cards(self, result: TuningResult) -> list[str]:
        import matplotlib.pyplot as plt

        try:
            from optuna.visualization.matplotlib import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
        except Exception:
            return []

        cards: list[str] = []
        for title, plotter in (
            ("Optimization History", plot_optimization_history),
            ("Parameter Importance", plot_param_importances),
            ("Parallel Coordinates", plot_parallel_coordinate),
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    axis = plotter(result.study)
                figure = axis.figure
                cards.append(self._figure_card(title, figure))
                plt.close(figure)
            except Exception:
                continue
        return cards

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

    def _metric_value(self, metric_map: dict[str, float], metric_name: str) -> float:
        if metric_name in metric_map:
            return float(metric_map[metric_name])
        return float(next(iter(metric_map.values())))

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
