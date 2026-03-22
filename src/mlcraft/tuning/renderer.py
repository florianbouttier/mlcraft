"""HTML renderer for Optuna tuning results."""

from __future__ import annotations

from mlcraft.core.results import TuningResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html


class TuningReportRenderer:
    """Render a TuningResult as HTML."""

    def render(self, result: TuningResult, *, title: str = "mlcraft Tuning Report", output_path=None) -> str:
        import matplotlib.pyplot as plt

        sections = [f"<h1>{title}</h1>"]
        sections.append("<h2>Best Trial</h2>")
        sections.append(
            f"<p><strong>Metric:</strong> {result.metric_name} | <strong>Alpha:</strong> {result.alpha:.4f} | "
            f"<strong>Best penalized score:</strong> {result.best_score:.6f}</p>"
        )
        sections.append("<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>")
        for key, value in result.best_params.items():
            sections.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
        sections.append("</tbody></table>")
        sections.append("<h2>Aggregated Metrics</h2>")
        sections.append("<table><thead><tr><th>Split</th><th>Metric</th><th>Value</th></tr></thead><tbody>")
        for split_name, metrics in (("train", result.train_metrics), ("validation", result.val_metrics)):
            for metric_name, metric_value in metrics.items():
                sections.append(f"<tr><td>{split_name}</td><td>{metric_name}</td><td>{metric_value:.6f}</td></tr>")
        sections.append("</tbody></table>")
        sections.append("<h2>Fold Details</h2>")
        sections.append("<table><thead><tr><th>Fold</th><th>Train score</th><th>Validation score</th><th>Penalized score</th></tr></thead><tbody>")
        for fold in result.fold_summaries:
            sections.append(
                f"<tr><td>{fold.fold_index}</td><td>{fold.train_score:.6f}</td><td>{fold.val_score:.6f}</td><td>{fold.penalized_score:.6f}</td></tr>"
            )
        sections.append("</tbody></table>")
        sections.append("<h2>Optimization History</h2><div class='grid'>")
        sections.extend(self._render_plots(result))
        sections.append("</div>")
        html = wrap_html(title, "".join(sections))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html

    def _render_plots(self, result: TuningResult) -> list[str]:
        import matplotlib.pyplot as plt

        cards: list[str] = []
        history_fig = self._plot_history(result)
        cards.append(f"<div class='card'><img alt='optimization-history' src='{figure_to_data_uri(history_fig)}' /></div>")
        plt.close(history_fig)
        optuna_plots = self._optuna_plots(result)
        cards.extend(optuna_plots)
        return cards

    def _plot_history(self, result: TuningResult):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        trials = [trial.trial_number for trial in result.history]
        scores = [trial.penalized_score for trial in result.history]
        ax.plot(trials, scores, marker="o")
        ax.set_title("Optimization History")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Penalized score")
        return fig

    def _optuna_plots(self, result: TuningResult) -> list[str]:
        import matplotlib.pyplot as plt

        if result.study is None:
            return []
        try:
            from optuna.visualization.matplotlib import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
        except Exception:
            return []
        cards: list[str] = []
        for plotter in (plot_optimization_history, plot_param_importances, plot_parallel_coordinate):
            try:
                axis = plotter(result.study)
                fig = axis.figure
                cards.append(f"<div class='card'><img alt='{plotter.__name__}' src='{figure_to_data_uri(fig)}' /></div>")
                plt.close(fig)
            except Exception:
                continue
        return cards
