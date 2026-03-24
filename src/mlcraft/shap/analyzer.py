"""Optional SHAP explainability analyzer."""

from __future__ import annotations

import numpy as np

from mlcraft.core.results import ShapResult
from mlcraft.utils.logging import inject_logger
from mlcraft.utils.optional import optional_import


class ShapAnalyzer:
    """Compute SHAP artifacts for fitted tree-based models.

    Args:
        logger: Optional custom logger.

    Example:
        >>> analyzer = ShapAnalyzer()
        >>> analyzer.__class__.__name__
        'ShapAnalyzer'
    """

    def __init__(self, logger=None) -> None:
        self.logger = inject_logger(logger, "shap")

    def compute(self, model, X, *, sample_weight=None, max_samples=None, interaction_values=False) -> ShapResult:
        """Compute SHAP values for a fitted model and dataset.

        Args:
            model: Fitted `BaseGBMModel`-compatible wrapper.
            X: Feature data with shape `(n_samples, n_features)` or a column
                mapping.
            sample_weight: Reserved for future weighted explainability
                strategies.
            max_samples: Optional cap on the number of explained rows.
            interaction_values: Whether to compute SHAP interaction values.

        Returns:
            ShapResult: SHAP values, optional interactions, and derived
            importance values.

        Raises:
            OptionalDependencyError: If `shap` is not installed.

        Example:
            >>> analyzer = ShapAnalyzer()
            >>> analyzer.compute(model, X_test, max_samples=200)
        """

        shap = optional_import("shap")
        matrix, _ = model.transform_features(X)
        feature_names = list(model.adapter_.feature_order)
        if max_samples is not None and matrix.shape[0] > max_samples:
            indices = np.linspace(0, matrix.shape[0] - 1, num=max_samples, dtype=int)
            matrix = matrix[indices]
        explainer = shap.TreeExplainer(model.model_)
        shap_values = explainer.shap_values(matrix)
        base_values = explainer.expected_value
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        if isinstance(base_values, list):
            base_values = base_values[-1]
        interactions = None
        if interaction_values:
            interactions = explainer.shap_interaction_values(matrix)
            if isinstance(interactions, list):
                interactions = interactions[-1]
        importance = np.mean(np.abs(np.asarray(shap_values)), axis=0)
        return ShapResult(
            feature_names=feature_names,
            shap_values=np.asarray(shap_values),
            feature_values=np.array(matrix, copy=True),
            base_values=np.asarray(base_values),
            interaction_values=None if interactions is None else np.asarray(interactions),
            importance=np.asarray(importance),
            metadata={"n_samples": int(np.asarray(shap_values).shape[0])},
        )

    def compute_with_artifacts(
        self,
        model,
        X,
        *,
        sample_weight=None,
        max_samples=None,
        interaction_values=False,
        output_dir=None,
        report_name: str = "shap_report.html",
        result_name: str = "shap.json",
        title: str | None = "mlcraft SHAP Report",
    ):
        """Compute SHAP values and persist standalone HTML/JSON artifacts."""

        from mlcraft.shap.artifacts import write_shap_artifacts

        result = self.compute(
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
