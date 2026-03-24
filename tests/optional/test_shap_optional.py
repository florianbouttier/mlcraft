import numpy as np
import pytest
import re

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


def test_write_shap_artifacts_writes_html_and_json(tmp_path):
    pytest.importorskip("matplotlib")
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


def test_shap_renderer_uses_official_beeswarm_and_sorted_scatter(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
    shap = pytest.importorskip("shap")
    from mlcraft.shap.renderer import ShapReportRenderer

    calls = []

    def fake_beeswarm(explanation, max_display=10, show=True, **kwargs):
        import matplotlib.pyplot as plt

        calls.append(("beeswarm", int(max_display)))
        plt.plot([0, 1], [0, 1])

    def fake_scatter(explanation, color=None, show=True, **kwargs):
        import matplotlib.pyplot as plt

        calls.append(("scatter", str(explanation.feature_names), color is not None))
        plt.plot([0, 1], [0, 1])

    monkeypatch.setattr(shap.plots, "beeswarm", fake_beeswarm)
    monkeypatch.setattr(shap.plots, "scatter", fake_scatter)

    result = ShapResult(
        feature_names=["a", "b"],
        shap_values=np.array([[0.1, -0.5], [0.2, -0.1]]),
        feature_values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        importance=np.array([0.15, 0.30]),
        interaction_values=np.zeros((2, 2, 2)),
    )
    html = ShapReportRenderer().render(result)

    assert "SHAP Report" in html
    assert calls[0] == ("beeswarm", 10)
    assert [call[1] for call in calls if call[0] == "scatter"] == ["b", "a"]
    assert all(call[2] for call in calls if call[0] == "scatter")
    assert "SHAP Interaction Plot" in html


def test_shap_interaction_plot_renders_absolute_value_labels():
    pytest.importorskip("matplotlib")
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
    figure = renderer._interaction_plot(context)

    axes = figure.axes[0]
    texts = [text.get_text() for text in axes.texts]
    assert texts
    assert all("%" not in text for text in texts)
    assert all(not text.startswith("-") for text in texts)
    assert all(re.fullmatch(r"(?:0|[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)", text) for text in texts)
    diagonal_texts = [axes.texts[index].get_text() for index in (0, 4, 8)]
    assert any(text != "0" for text in diagonal_texts)


def test_shap_interaction_value_formatter_scales_precision():
    from mlcraft.shap.renderer import ShapReportRenderer

    renderer = ShapReportRenderer()

    assert renderer._format_interaction_value(0.0) == "0"
    assert renderer._format_interaction_value(0.004321) == "0.0043"
    assert renderer._format_interaction_value(0.04567) == "0.0457"
    assert renderer._format_interaction_value(0.4567) == "0.457"
    assert renderer._format_interaction_value(4.567) == "4.57"
    assert renderer._format_interaction_value(45.67) == "45.7"
