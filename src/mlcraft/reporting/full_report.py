"""Builder for a combined HTML report."""

from __future__ import annotations

from mlcraft.evaluation.renderer import EvaluationReportRenderer
from mlcraft.reporting.html import wrap_html
from mlcraft.shap.renderer import ShapReportRenderer
from mlcraft.tuning.renderer import TuningReportRenderer


def _extract_body(html: str) -> str:
    start = html.find("<body>")
    end = html.rfind("</body>")
    if start == -1 or end == -1:
        return html
    return html[start + len("<body>") : end]


class FullReportBuilder:
    """Combine multiple result types into one standalone HTML report."""

    def __init__(self) -> None:
        self.evaluation_renderer = EvaluationReportRenderer()
        self.tuning_renderer = TuningReportRenderer()
        self.shap_renderer = ShapReportRenderer()

    def build(self, *, evaluation=None, tuning=None, shap=None, output_path=None) -> str:
        sections = ["<h1>mlcraft Full Report</h1>"]
        if evaluation is not None:
            sections.append("<section><h2>Evaluation</h2>")
            sections.append(_extract_body(self.evaluation_renderer.render(evaluation, title="Evaluation Section")))
            sections.append("</section>")
        if tuning is not None:
            sections.append("<section><h2>Tuning</h2>")
            sections.append(_extract_body(self.tuning_renderer.render(tuning, title="Tuning Section")))
            sections.append("</section>")
        if shap is not None:
            sections.append("<section><h2>SHAP</h2>")
            sections.append(_extract_body(self.shap_renderer.render(shap, title="SHAP Section")))
            sections.append("</section>")
        html = wrap_html("mlcraft Full Report", "".join(sections))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html
