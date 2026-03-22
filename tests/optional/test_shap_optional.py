import numpy as np
import pytest

from mlcraft.core.results import ShapResult
from mlcraft.errors import OptionalDependencyError


def test_shap_analyzer_fails_cleanly_when_dependency_missing(monkeypatch):
    from mlcraft.shap.analyzer import ShapAnalyzer

    monkeypatch.setattr("mlcraft.shap.analyzer.optional_import", lambda *args, **kwargs: (_ for _ in ()).throw(OptionalDependencyError("missing")))
    analyzer = ShapAnalyzer()
    with pytest.raises(OptionalDependencyError):
        analyzer.compute(model=object(), X=np.zeros((2, 2)))


def test_shap_renderer_generates_html():
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
    from mlcraft.shap.renderer import ShapReportRenderer

    result = ShapResult(
        feature_names=["a", "b"],
        shap_values=np.array([[0.1, -0.1], [0.2, -0.2]]),
        feature_values=np.array([[1.0, 0.0], [2.0, 1.0]]),
        importance=np.array([0.15, 0.15]),
    )
    renderer = ShapReportRenderer()
    context = renderer.build_context(result)
    assert isinstance(context, dict)
    assert context["feature_count"] == 2
    html = renderer.render_context(context)
    assert "SHAP Report" in html
