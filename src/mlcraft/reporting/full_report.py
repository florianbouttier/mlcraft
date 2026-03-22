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
    """Combine evaluation, tuning, and SHAP sections into one report.

    The builder reuses the specialized renderers for each result type and
    merges only the HTML bodies, which keeps section-specific logic isolated.

    Example:
        >>> builder = FullReportBuilder()
        >>> html = builder.build(evaluation=evaluation_result)
        >>> "Evaluation" in html
        True
    """

    def __init__(self) -> None:
        self.evaluation_renderer = EvaluationReportRenderer()
        self.tuning_renderer = TuningReportRenderer()
        self.shap_renderer = ShapReportRenderer()

    def build_context(self, *, evaluation=None, tuning=None, shap=None) -> dict:
        """Build the combined report context from sub-report contexts.

        Args:
            evaluation: Optional evaluation result to include.
            tuning: Optional tuning result to include.
            shap: Optional SHAP result to include.

        Returns:
            dict: Combined report context composed of section dictionaries.
        """

        sections = []
        if tuning is not None:
            sections.append({"name": "Tuning", "renderer": "tuning", "context": self.tuning_renderer.build_context(tuning, title=None)})
        if evaluation is not None:
            sections.append({"name": "Evaluation", "renderer": "evaluation", "context": self.evaluation_renderer.build_context(evaluation, title=None)})
        if shap is not None:
            sections.append({"name": "SHAP", "renderer": "shap", "context": self.shap_renderer.build_context(shap, title="SHAP Section")})
        return {"title": "mlcraft Full Report", "sections": sections}

    def build(self, *, evaluation=None, tuning=None, shap=None, output_path=None) -> str:
        """Render a combined HTML report.

        Args:
            evaluation: Optional evaluation result to include.
            tuning: Optional tuning result to include.
            shap: Optional SHAP result to include.
            output_path: Optional file path used to persist the report.

        Returns:
            str: Standalone HTML document containing all requested sections.

        Example:
            >>> html = FullReportBuilder().build(evaluation=evaluation_result)
            >>> "mlcraft Full Report" in html
            True
        """

        return self.render_context(self.build_context(evaluation=evaluation, tuning=tuning, shap=shap), output_path=output_path)

    def render_context(self, context: dict, *, output_path=None) -> str:
        """Render a combined HTML report from a pre-built dictionary context.

        Args:
            context: Dictionary returned by `build_context()`.
            output_path: Optional file path used to persist the report.

        Returns:
            str: Standalone HTML document.
        """

        sections = [f"<h1>{context['title']}</h1>"]
        renderer_map = {
            "evaluation": self.evaluation_renderer.render_context,
            "tuning": self.tuning_renderer.render_context,
            "shap": self.shap_renderer.render_context,
        }
        for section in context["sections"]:
            sections.append(f"<section><h2>{section['name']}</h2>")
            sections.append(_extract_body(renderer_map[section["renderer"]](section["context"])))
            sections.append("</section>")
        html = wrap_html("mlcraft Full Report", "".join(sections))
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
        return html
