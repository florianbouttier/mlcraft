"""Artifact writing helpers for standalone SHAP analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mlcraft.shap.analyzer import ShapAnalyzer
from mlcraft.shap.renderer import ShapReportRenderer


@dataclass
class ShapArtifacts:
    """Describe the files produced for one SHAP run."""

    output_dir: Path
    report_path: Path
    result_path: Path


class ShapArtifactWriter:
    """Write a SHAP result and its standalone HTML report to disk."""

    def __init__(self, *, renderer: ShapReportRenderer | None = None) -> None:
        self.renderer = renderer or ShapReportRenderer()

    def write(
        self,
        result,
        *,
        output_dir: str | Path | None = None,
        report_name: str = "shap_report.html",
        result_name: str = "shap.json",
        title: str | None = "mlcraft SHAP Report",
    ) -> ShapArtifacts:
        resolved_output_dir = Path(output_dir) if output_dir is not None else Path.cwd() / "artifacts" / "mlcraft_shap"
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        report_path = resolved_output_dir / report_name
        result_path = resolved_output_dir / result_name

        self.renderer.render(result, title=title, output_path=report_path)
        result_path.write_text(
            json.dumps(result.to_dict(include_arrays=False), indent=2),
            encoding="utf-8",
        )
        return ShapArtifacts(
            output_dir=resolved_output_dir,
            report_path=report_path,
            result_path=result_path,
        )


def write_shap_artifacts(
    result,
    *,
    output_dir: str | Path | None = None,
    report_name: str = "shap_report.html",
    result_name: str = "shap.json",
    title: str | None = "mlcraft SHAP Report",
) -> ShapArtifacts:
    """Write a standalone SHAP report and JSON payload."""

    return ShapArtifactWriter().write(
        result,
        output_dir=output_dir,
        report_name=report_name,
        result_name=result_name,
        title=title,
    )


def run_shap_analysis(
    model,
    X,
    *,
    sample_weight=None,
    max_samples: int | None = None,
    interaction_values: bool = False,
    output_dir: str | Path | None = None,
    report_name: str = "shap_report.html",
    result_name: str = "shap.json",
    title: str | None = "mlcraft SHAP Report",
    logger=None,
):
    """Compute SHAP values from a fitted model and persist standalone artifacts."""

    analyzer = ShapAnalyzer(logger=logger)
    result = analyzer.compute(
        model,
        X,
        sample_weight=sample_weight,
        max_samples=max_samples,
        interaction_values=interaction_values,
    )
    artifacts = write_shap_artifacts(
        result,
        output_dir=output_dir,
        report_name=report_name,
        result_name=result_name,
        title=title,
    )
    return result, artifacts
