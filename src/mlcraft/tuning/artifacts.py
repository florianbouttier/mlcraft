"""Artifact writing helpers for tuning results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mlcraft.core.results import TuningResult
from mlcraft.reporting.full_report import FullReportBuilder
from mlcraft.shap.renderer import ShapReportRenderer
from mlcraft.tuning.renderer import TuningReportRenderer


@dataclass(slots=True)
class TuningArtifacts:
    """Describe the files produced for one tuning run."""

    output_dir: Path
    report_path: Path
    result_path: Path
    full_report_path: Path | None = None
    shap_report_path: Path | None = None
    shap_result_path: Path | None = None


class TuningArtifactWriter:
    """Write a tuning result and its HTML report to disk."""

    def __init__(self, *, renderer: TuningReportRenderer | None = None) -> None:
        self.renderer = renderer or TuningReportRenderer()
        self.shap_renderer = ShapReportRenderer()
        self.full_report_builder = FullReportBuilder()

    def write(
        self,
        result: TuningResult,
        *,
        output_dir: str | Path | None = None,
        report_name: str = "report.html",
        result_name: str = "tuning.json",
        title: str | None = "mlcraft Tuning Report",
        evaluation=None,
        shap=None,
        full_report_name: str = "full_report.html",
        shap_report_name: str = "shap_report.html",
        shap_result_name: str = "shap.json",
    ) -> TuningArtifacts:
        resolved_output_dir = Path(output_dir) if output_dir is not None else Path.cwd() / "artifacts" / "mlcraft_tuning"
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        report_path = resolved_output_dir / report_name
        result_path = resolved_output_dir / result_name
        full_report_path = None
        shap_report_path = None
        shap_result_path = None

        self.renderer.render(result, title=title, output_path=report_path)
        result_path.write_text(
            json.dumps(result.to_dict(include_arrays=True), indent=2),
            encoding="utf-8",
        )
        if shap is not None:
            shap_report_path = resolved_output_dir / shap_report_name
            shap_result_path = resolved_output_dir / shap_result_name
            self.shap_renderer.render(shap, output_path=shap_report_path)
            shap_result_path.write_text(
                json.dumps(shap.to_dict(include_arrays=False), indent=2),
                encoding="utf-8",
            )
        if evaluation is not None or shap is not None:
            full_report_path = resolved_output_dir / full_report_name
            self.full_report_builder.build(
                tuning=result,
                evaluation=evaluation,
                shap=shap,
                output_path=full_report_path,
            )

        return TuningArtifacts(
            output_dir=resolved_output_dir,
            report_path=report_path,
            result_path=result_path,
            full_report_path=full_report_path,
            shap_report_path=shap_report_path,
            shap_result_path=shap_result_path,
        )


def write_tuning_artifacts(
    result: TuningResult,
    *,
    output_dir: str | Path | None = None,
    report_name: str = "report.html",
    result_name: str = "tuning.json",
    title: str | None = "mlcraft Tuning Report",
    evaluation=None,
    shap=None,
    full_report_name: str = "full_report.html",
    shap_report_name: str = "shap_report.html",
    shap_result_name: str = "shap.json",
) -> TuningArtifacts:
    """Write a tuning result and report with the default artifact writer."""

    return TuningArtifactWriter().write(
        result,
        output_dir=output_dir,
        report_name=report_name,
        result_name=result_name,
        title=title,
        evaluation=evaluation,
        shap=shap,
        full_report_name=full_report_name,
        shap_report_name=shap_report_name,
        shap_result_name=shap_result_name,
    )
