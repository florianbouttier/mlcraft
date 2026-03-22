"""Central metric registry shared by evaluation and model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mlcraft.metrics import classification, poisson, regression


@dataclass(frozen=True)
class MetricDefinition:
    """Describe one canonical metric and its backend mappings."""

    name: str
    function: Callable[..., float]
    higher_is_better: bool
    backend_names: dict[str, str | None]
    prediction_kind: str = "pred"

    def to_score(self, value: float) -> float:
        """Convert a raw metric value into a score that must be maximized.

        Args:
            value: Raw metric value in its natural direction.

        Returns:
            float: Optimization score where higher is always better.
        """

        return float(value if self.higher_is_better else -value)


class MetricRegistry:
    """Map canonical metric names to implementations and backend aliases.

    Args:
        definitions: Optional initial metric definitions to register.

    Example:
        >>> registry = MetricRegistry()
        >>> registry.register(MetricDefinition("rmse", lambda y_true, y_pred, **_: 0.5, False, {}))
        >>> registry.score("rmse", 0.5)
        -0.5
    """

    def __init__(self, definitions: list[MetricDefinition] | None = None) -> None:
        self._definitions: dict[str, MetricDefinition] = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: MetricDefinition) -> None:
        """Register one metric definition.

        Args:
            definition: Metric definition to store under its canonical name.
        """

        self._definitions[definition.name] = definition

    def get(self, name: str) -> MetricDefinition:
        """Return the definition registered under one canonical name.

        Args:
            name: Canonical metric name.

        Returns:
            MetricDefinition: Registered metric definition.
        """

        return self._definitions[name]

    def names(self) -> list[str]:
        """Return all registered metric names.

        Returns:
            list[str]: Sorted canonical metric names.
        """

        return sorted(self._definitions.keys())

    def backend_name(self, name: str, backend: str) -> str | None:
        """Return the backend-native alias for one metric.

        Args:
            name: Canonical metric name.
            backend: Backend name such as `xgboost`.

        Returns:
            str | None: Backend-native metric alias when defined.
        """

        return self.get(name).backend_names.get(backend)

    def score(self, name: str, value: float) -> float:
        """Convert a raw metric value into an optimization score.

        Args:
            name: Canonical metric name.
            value: Raw metric value.

        Returns:
            float: Maximization-oriented score.
        """

        return self.get(name).to_score(value)

    def evaluate(
        self,
        name: str,
        y_true,
        *,
        y_pred=None,
        y_score=None,
        sample_weight=None,
        exposure=None,
        **options: Any,
    ) -> tuple[float, float]:
        """Evaluate one metric and return both value and optimization score.

        Args:
            name: Canonical metric name.
            y_true: Ground-truth array of shape `(n_samples,)`.
            y_pred: Optional prediction array of shape `(n_samples,)`.
            y_score: Optional score or probability array of shape
                `(n_samples,)`.
            sample_weight: Optional per-row weights.
            exposure: Optional exposure vector for Poisson metrics.
            **options: Additional keyword arguments forwarded to the metric.

        Returns:
            tuple[float, float]: Raw metric value and its maximization score.
        """

        definition = self.get(name)
        value = definition.function(
            y_true,
            y_pred,
            y_score=y_score,
            sample_weight=sample_weight,
            exposure=exposure,
            **options,
        )
        return float(value), definition.to_score(float(value))


default_metric_registry = MetricRegistry(
    definitions=[
        MetricDefinition("mae", regression.mae, False, {"xgboost": "mae", "lightgbm": "l1", "catboost": "MAE"}),
        MetricDefinition("mse", regression.mse, False, {"xgboost": None, "lightgbm": "l2", "catboost": "RMSE"}),
        MetricDefinition("rmse", regression.rmse, False, {"xgboost": "rmse", "lightgbm": "rmse", "catboost": "RMSE"}),
        MetricDefinition("r2", regression.r2, True, {"xgboost": None, "lightgbm": None, "catboost": "R2"}),
        MetricDefinition("medae", regression.medae, False, {"xgboost": None, "lightgbm": None, "catboost": None}),
        MetricDefinition("mape", regression.mape, False, {"xgboost": "mape", "lightgbm": "mape", "catboost": "MAPE"}),
        MetricDefinition("roc_auc", classification.roc_auc, True, {"xgboost": "auc", "lightgbm": "auc", "catboost": "AUC"}, prediction_kind="score"),
        MetricDefinition("pr_auc", classification.pr_auc, True, {"xgboost": "aucpr", "lightgbm": "average_precision", "catboost": "PRAUC"}, prediction_kind="score"),
        MetricDefinition("logloss", classification.logloss, False, {"xgboost": "logloss", "lightgbm": "binary_logloss", "catboost": "Logloss"}, prediction_kind="score"),
        MetricDefinition("accuracy", classification.accuracy, True, {"xgboost": None, "lightgbm": None, "catboost": "Accuracy"}),
        MetricDefinition("precision", classification.precision, True, {"xgboost": None, "lightgbm": None, "catboost": "Precision"}),
        MetricDefinition("recall", classification.recall, True, {"xgboost": None, "lightgbm": None, "catboost": "Recall"}),
        MetricDefinition("f1", classification.f1, True, {"xgboost": None, "lightgbm": None, "catboost": "F1"}),
        MetricDefinition("brier_score", classification.brier_score, False, {"xgboost": None, "lightgbm": None, "catboost": "BrierScore"}, prediction_kind="score"),
        MetricDefinition("gini", classification.gini, True, {"xgboost": None, "lightgbm": None, "catboost": "NormalizedGini"}, prediction_kind="score"),
        MetricDefinition("poisson_deviance", poisson.poisson_deviance, False, {"xgboost": "poisson-nloglik", "lightgbm": "poisson", "catboost": "Poisson"}),
        MetricDefinition("observed_mean", poisson.observed_mean, True, {"xgboost": None, "lightgbm": None, "catboost": None}),
        MetricDefinition("predicted_mean", poisson.predicted_mean, True, {"xgboost": None, "lightgbm": None, "catboost": None}),
    ]
)
