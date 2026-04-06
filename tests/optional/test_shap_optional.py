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


def test_shap_renderer_generates_d3_html():
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
    assert "SHAP beeswarm" in html
    assert "cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js" in html
    assert "data-toggle-group='shap-scatter'" in html


def test_write_shap_artifacts_writes_html_and_json(tmp_path):
    pytest.importorskip("jinja2")
    from mlcraft.shap.artifacts import write_shap_artifacts

    result = ShapResult(
        feature_names=["a", "b"],
        shap_values=np.array([[0.1, -0.1], [0.2, -0.2]]),
        feature_values=np.array([[1.0, 0.0], [2.0, 1.0]]),
        importance=np.array([0.15, 0.15]),
    )
    artifacts = write_shap_artifacts(result, output_dir=tmp_path)

    assert artifacts.report_path.exists()
    assert artifacts.result_path.exists()
    assert "SHAP Report" in artifacts.report_path.read_text(encoding="utf-8")
    assert '"feature_names"' in artifacts.result_path.read_text(encoding="utf-8")


def test_run_shap_analysis_uses_standalone_pipeline(monkeypatch, tmp_path):
    from mlcraft.shap.artifacts import run_shap_analysis

    monkeypatch.setattr(
        "mlcraft.shap.analyzer.ShapAnalyzer.compute",
        lambda self, model, X, sample_weight=None, max_samples=None, interaction_values=False: ShapResult(
            feature_names=["a", "b"],
            shap_values=np.array([[0.1, -0.1], [0.2, -0.2]]),
            feature_values=np.array([[1.0, 0.0], [2.0, 1.0]]),
            importance=np.array([0.15, 0.15]),
        ),
    )

    result, artifacts = run_shap_analysis(object(), {"a": np.array([1.0, 2.0]), "b": np.array([0.0, 1.0])}, output_dir=tmp_path)

    assert result.feature_names == ["a", "b"]
    assert artifacts.report_path.exists()
    assert artifacts.result_path.exists()


def test_shap_renderer_exposes_interaction_heatmap():
    pytest.importorskip("jinja2")
    from mlcraft.shap.renderer import ShapReportRenderer

    result = ShapResult(
        feature_names=["age", "bmi", "bp"],
        shap_values=np.array(
            [
                [0.1, -0.2, 0.3],
                [0.2, -0.1, 0.4],
            ]
        ),
        feature_values=np.array(
            [
                [50.0, 20.0, 110.0],
                [60.0, 25.0, 120.0],
            ]
        ),
        importance=np.array([0.15, 0.15, 0.35]),
        interaction_values=np.array(
            [
                [
                    [0.3, 0.2, -0.1],
                    [0.2, 0.1, 0.4],
                    [-0.1, 0.4, 0.2],
                ],
                [
                    [0.2, 0.1, -0.2],
                    [0.1, 0.2, 0.3],
                    [-0.2, 0.3, 0.1],
                ],
            ]
        ),
    )
    renderer = ShapReportRenderer()
    context = renderer.build_context(result)
    html = renderer.render_context(context)

    assert context["interaction_matrix"] is not None
    assert "SHAP interaction heatmap" in html
    assert "Interaction Structure" in html
