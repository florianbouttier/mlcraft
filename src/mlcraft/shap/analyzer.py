"""Optional SHAP explainability analyzer."""

from __future__ import annotations

import numpy as np

from mlcraft.core.results import ShapResult
from mlcraft.utils.logging import inject_logger
from mlcraft.utils.optional import optional_import


class ShapAnalyzer:
    """Compute SHAP values for fitted tree models."""

    def __init__(self, logger=None) -> None:
        self.logger = inject_logger(logger, "shap")

    def compute(self, model, X, *, sample_weight=None, max_samples=None, interaction_values=False) -> ShapResult:
        shap = optional_import("shap", extra_name="shap")
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
            feature_values=np.asarray(matrix, dtype=float) if matrix.dtype != object else None,
            base_values=np.asarray(base_values),
            interaction_values=None if interactions is None else np.asarray(interactions),
            importance=np.asarray(importance),
            metadata={"n_samples": int(np.asarray(shap_values).shape[0])},
        )

