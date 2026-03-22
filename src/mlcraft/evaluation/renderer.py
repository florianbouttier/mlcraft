"""HTML renderer for evaluation results."""

from __future__ import annotations

from collections import defaultdict

from mlcraft.core.results import EvaluationResult
from mlcraft.reporting.html import figure_to_data_uri, wrap_html


class EvaluationReportRenderer:
    """Render an evaluation result as standalone HTML.

    The renderer keeps plotting and templating separate from metric
    computation so `EvaluationResult` stays lightweight and serializable.
    """

    def render(self, result: EvaluationResult, *, title: str = "mlcraft Evaluation Report", output_path=None) -> str:
        """Render a complete evaluation report.

        Args:
            result: Evaluation output to render.
            title: Title displayed in the HTML document.
            output_path: Optional file path used to persist the rendered HTML.

        Returns:
            str: Standalone HTML document.

        Example:
            >>> renderer = EvaluationReportRenderer()
            >>> html = renderer.render(result)
            >>> html.startswith("<!doctype html>")
            True
        """

        import matplotlib.pyplot as plt

        metrics_by_prediction = defaultdict(list)
        for row in result.metric_rows:
            metrics_by_prediction[row.prediction_name].append(row)

        sections: list[str] = [f"<h1>{title}</h1>", f"<p class='muted'>Task: <code>{result.task_spec.task_type.value}</code></p>"]
        sections.append("<h2>Metrics</h2>")
        sections.append("<table><thead><tr><th>Prediction</th><th>Metric</th><th>Value</th><th>Higher is better</th></tr></thead><tbody>")
        for prediction_name, rows in metrics_by_prediction.items():
            for row in rows:
                sections.append(
                    f"<tr><td>{prediction_name}</td><td>{row.metric_name}</td><td>{row.value:.6f}</td><td>{row.higher_is_better}</td></tr>"
                )
        sections.append("</tbody></table>")
        sections.append("<h2>Curves</h2><div class='grid'>")
        for prediction_name, curves in result.curves.items():
            for curve in curves:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(curve.x, curve.y)
                if curve.name in {"calibration", "poisson_calibration"}:
                    diagonal_min = min(float(curve.x.min()) if curve.x.size else 0.0, float(curve.y.min()) if curve.y.size else 0.0)
                    diagonal_max = max(float(curve.x.max()) if curve.x.size else 1.0, float(curve.y.max()) if curve.y.size else 1.0)
                    ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="gray")
                ax.set_title(f"{prediction_name} - {curve.name}")
                ax.set_xlabel(curve.x_label)
                ax.set_ylabel(curve.y_label)
                uri = figure_to_data_uri(fig)
                plt.close(fig)
                sections.append(f"<div class='card'><img alt='{curve.name}' src='{uri}' /></div>")
        sections.append("</div>")
        html = wrap_html(title, "".join(sections))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html
