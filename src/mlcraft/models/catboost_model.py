"""CatBoost wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcraft.models.base import BaseGBMModel
from mlcraft.models.objectives import backend_seed_key, resolve_backend_metric, resolve_backend_objective
from mlcraft.utils.optional import optional_import


class CatBoostModel(BaseGBMModel):
    backend_name = "catboost"

    def _default_model_params(self) -> dict[str, Any]:
        return {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.05,
            "verbose": False,
        }

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        catboost = optional_import("catboost", extra_name="catboost")
        params = self._default_model_params()
        params.update(self.model_params)
        params["loss_function"] = resolve_backend_objective(self.backend_name, self.task_spec)
        backend_metric = resolve_backend_metric(self.backend_name, self.task_spec, metric_registry=self.metric_registry)
        if backend_metric is not None:
            params["eval_metric"] = backend_metric
        if self.random_state is not None and not isinstance(self.random_state, np.random.Generator):
            params[backend_seed_key(self.backend_name)] = int(self.random_state)
        estimator_cls = catboost.CatBoostClassifier if self.task_spec.is_classification() else catboost.CatBoostRegressor
        estimator = estimator_cls(**params)
        eval_payload = None
        if eval_set:
            X_val, y_val, val_meta = eval_set[0]
            eval_payload = catboost.Pool(X_val, label=y_val, cat_features=val_meta.get("categorical_indices", []))
        estimator.fit(
            X,
            y,
            sample_weight=sample_weight,
            cat_features=(metadata or {}).get("categorical_indices", []),
            eval_set=eval_payload,
            use_best_model=bool(eval_payload),
            verbose=False,
        )
        return estimator

    def _predict_backend(self, X, *, metadata=None):
        if self.task_spec.is_classification():
            proba = self.model_.predict_proba(X)
            return np.asarray(proba)[:, 1]
        return np.asarray(self.model_.predict(X), dtype=float).reshape(-1)

