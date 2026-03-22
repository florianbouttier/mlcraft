"""LightGBM wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcraft.models.base import BaseGBMModel
from mlcraft.models.objectives import backend_seed_key, resolve_backend_metric, resolve_backend_objective
from mlcraft.utils.optional import optional_import


class LightGBMModel(BaseGBMModel):
    """Wrap LightGBM behind the shared `BaseGBMModel` interface."""

    backend_name = "lightgbm"

    def _default_model_params(self) -> dict[str, Any]:
        return {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
        }

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        lgb = optional_import("lightgbm", extra_name="lightgbm")
        params = self._default_model_params()
        params.update(self.model_params)
        params["objective"] = resolve_backend_objective(self.backend_name, self.task_spec)
        backend_metric = resolve_backend_metric(self.backend_name, self.task_spec, metric_registry=self.metric_registry)
        if backend_metric is not None:
            params["metric"] = backend_metric
        if self.random_state is not None and not isinstance(self.random_state, np.random.Generator):
            params[backend_seed_key(self.backend_name)] = int(self.random_state)
        train_data = lgb.Dataset(X, label=y, weight=sample_weight, free_raw_data=False)
        valid_sets = [train_data]
        callbacks = []
        if self.fit_params.get("early_stopping_rounds") and eval_set:
            callbacks.append(lgb.early_stopping(int(self.fit_params["early_stopping_rounds"]), verbose=False))
        if eval_set:
            for X_val, y_val, _ in eval_set:
                valid_sets.append(lgb.Dataset(X_val, label=y_val, free_raw_data=False))
        num_boost_round = int(self.fit_params.get("num_boost_round", 200))
        return lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

    def _predict_backend(self, X, *, metadata=None):
        return self.model_.predict(X)
