"""Common abstraction for gradient boosting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from mlcraft.core.prediction import PredictionBundle, resolve_task_spec
from mlcraft.core.task import TaskSpec, TaskType
from mlcraft.data.adapters import FeatureAdapterConfig, fit_feature_adapter
from mlcraft.data.inference import infer_schema
from mlcraft.metrics.registry import MetricRegistry, default_metric_registry
from mlcraft.utils.logging import inject_logger


class BaseGBMModel(ABC):
    """Define the shared contract for gradient boosting wrappers.

    Subclasses only implement backend-specific fitting and prediction. The
    base class owns schema inference, feature adaptation, task propagation,
    and the public prediction interface.
    """

    backend_name: str = "base"

    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        model_params: dict[str, Any] | None = None,
        fit_params: dict[str, Any] | None = None,
        random_state: int | np.random.Generator | None = None,
        metric_registry: MetricRegistry | None = None,
        logger=None,
    ) -> None:
        if isinstance(task_spec, TaskSpec):
            self.task_spec = task_spec
        elif isinstance(task_spec, dict):
            self.task_spec = TaskSpec(**task_spec)
        else:
            self.task_spec = TaskSpec(task_type=task_spec)
        self.model_params = dict(model_params or {})
        self.fit_params = dict(fit_params or {})
        self.random_state = random_state
        self.metric_registry = metric_registry or default_metric_registry
        self.logger = inject_logger(logger, f"models.{self.backend_name}")
        self.model_: Any | None = None
        self.schema_ = None
        self.adapter_ = None

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        exposure=None,
        eval_set=None,
    ) -> "BaseGBMModel":
        """Fit the model on training data.

        Args:
            X: Feature data with shape `(n_samples, n_features)` or a column
                mapping of 1D arrays.
            y: Target array of shape `(n_samples,)`.
            sample_weight: Optional per-row weights.
            exposure: Optional exposure vector of shape `(n_samples,)` used in
                Poisson workflows.
            eval_set: Optional validation set used for early stopping. Each
                item is `(X_val, y_val)` or `(X_val, y_val, exposure_val)`.

        Returns:
            BaseGBMModel: Fitted model instance.

        Example:
            >>> model = ModelFactory.create("xgboost", task_spec=TaskSpec(task_type="regression"))
            >>> fitted = model.fit(X_train, y_train)
            >>> fitted is model
            True
        """

        y_array = np.asarray(y)
        self.schema_ = infer_schema(X)
        self.adapter_ = fit_feature_adapter(X, self.schema_, config=FeatureAdapterConfig())
        X_matrix, metadata = self._transform_features(X, exposure=exposure)
        prepared_eval_set = self._prepare_eval_set(eval_set, exposure=exposure)
        self.model_ = self._fit_backend(
            X_matrix,
            y_array,
            sample_weight=sample_weight,
            exposure=exposure,
            eval_set=prepared_eval_set,
            metadata=metadata,
        )
        return self

    def predict(self, X, *, exposure=None) -> np.ndarray:
        """Predict labels, values, or rates for new data.

        Args:
            X: Feature data with shape `(n_samples, n_features)` or a column
                mapping.
            exposure: Optional exposure vector used for Poisson workflows.

        Returns:
            np.ndarray: Prediction array of shape `(n_samples,)`.
        """

        scores = self._predict_scores(X, exposure=exposure)
        if self.task_spec.task_type == TaskType.CLASSIFICATION:
            return (scores >= 0.5).astype(int)
        return scores

    def predict_proba(self, X, *, exposure=None) -> np.ndarray:
        """Predict class probabilities for binary classification tasks.

        Args:
            X: Feature data with shape `(n_samples, n_features)` or a column
                mapping.
            exposure: Optional exposure vector. Ignored for classification.

        Returns:
            np.ndarray: Probability matrix of shape `(n_samples, 2)`.

        Raises:
            ValueError: If the task is not binary classification.
        """

        if self.task_spec.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        scores = self._predict_scores(X, exposure=exposure)
        return np.column_stack([1.0 - scores, scores])

    def predict_bundle(self, X, *, name: str = "prediction", task_spec: TaskSpec | None = None, exposure=None) -> PredictionBundle:
        """Build a `PredictionBundle` from model outputs.

        Args:
            X: Feature data with shape `(n_samples, n_features)` or a column
                mapping.
            name: Label attached to the prediction bundle.
            task_spec: Optional task specification override.
            exposure: Optional exposure vector used for Poisson workflows.

        Returns:
            PredictionBundle: Bundle containing predictions and optional
            probabilities.
        """

        resolved_task = resolve_task_spec(task_spec, self)
        if resolved_task is None:
            raise ValueError("A TaskSpec is required to build a PredictionBundle.")
        scores = self._predict_scores(X, exposure=exposure)
        y_pred = (scores >= 0.5).astype(int) if resolved_task.task_type == TaskType.CLASSIFICATION else scores
        return PredictionBundle(name=name, y_pred=y_pred, y_score=scores if resolved_task.task_type == TaskType.CLASSIFICATION else None, task_spec=resolved_task)

    def get_params(self) -> dict[str, Any]:
        """Return the public model configuration.

        Returns:
            dict[str, Any]: Serializable configuration payload.
        """

        return {
            "task_spec": self.task_spec.to_dict(),
            "model_params": dict(self.model_params),
            "fit_params": dict(self.fit_params),
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "BaseGBMModel":
        """Update public model parameters in place.

        Args:
            **params: Parameter overrides. Unknown keys are forwarded to
                `model_params`.

        Returns:
            BaseGBMModel: Updated model instance.
        """

        if "task_spec" in params:
            value = params.pop("task_spec")
            self.task_spec = value if isinstance(value, TaskSpec) else TaskSpec(**value)
        if "model_params" in params:
            self.model_params = dict(params.pop("model_params"))
        if "fit_params" in params:
            self.fit_params = dict(params.pop("fit_params"))
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.model_params.update(params)
        return self

    def transform_features(self, X, *, exposure=None) -> tuple[np.ndarray, dict[str, Any]]:
        """Transform raw features using the fitted adapter.

        Args:
            X: Feature data with shape `(n_samples, n_features)` or a column
                mapping.
            exposure: Optional exposure vector used for Poisson workflows.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Backend-ready feature matrix and
            backend-specific metadata.
        """

        return self._transform_features(X, exposure=exposure)

    def _transform_features(self, X, *, exposure=None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.adapter_ is None:
            raise ValueError("The model must be fitted before transforming features.")
        extra_exposure = exposure if self.task_spec.task_type == TaskType.POISSON else None
        return self.adapter_.transform(X, backend=self.backend_name, exposure=extra_exposure)

    def _prepare_eval_set(self, eval_set, *, exposure=None):
        if not eval_set:
            return None
        prepared = []
        for item in eval_set:
            if len(item) == 2:
                X_val, y_val = item
                exp_val = None
            else:
                X_val, y_val, exp_val = item
            matrix, metadata = self._transform_features(X_val, exposure=exp_val if exp_val is not None else exposure)
            prepared.append((matrix, np.asarray(y_val), metadata))
        return prepared

    def _predict_scores(self, X, *, exposure=None) -> np.ndarray:
        matrix, metadata = self._transform_features(X, exposure=exposure)
        return np.asarray(self._predict_backend(matrix, metadata=metadata), dtype=float)

    @abstractmethod
    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        """Fit the native backend model."""

    @abstractmethod
    def _predict_backend(self, X, *, metadata=None):
        """Predict raw scores or values with the native backend model."""
