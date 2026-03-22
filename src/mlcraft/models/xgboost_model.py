"""XGBoost wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcraft.models.base import BaseGBMModel
from mlcraft.models.objectives import backend_seed_key, resolve_backend_metric, resolve_backend_objective
from mlcraft.utils.optional import optional_import


class XGBoostModel(BaseGBMModel):
    """Wrap XGBoost behind the shared `BaseGBMModel` interface."""

    backend_name = "xgboost"

    def _default_model_params(self) -> dict[str, Any]:
        return {
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "verbosity": 0,
        }

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        xgb = optional_import("xgboost", extra_name="xgboost")
        params = self._default_model_params()
        params.update(self.model_params)
        params["objective"] = resolve_backend_objective(self.backend_name, self.task_spec)
        backend_metric = resolve_backend_metric(self.backend_name, self.task_spec, metric_registry=self.metric_registry)
        if backend_metric is not None:
            params["eval_metric"] = backend_metric
        if self.random_state is not None and not isinstance(self.random_state, np.random.Generator):
            params[backend_seed_key(self.backend_name)] = int(self.random_state)
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        evals = [(dtrain, "train")]
        if eval_set:
            for idx, (X_val, y_val, _) in enumerate(eval_set):
                evals.append((xgb.DMatrix(X_val, label=y_val), f"valid_{idx}"))
        num_boost_round = int(self.fit_params.get("num_boost_round", 200))
        early_stopping_rounds = self.fit_params.get("early_stopping_rounds")
        return xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

    def _predict_backend(self, X, *, metadata=None):
        xgb = optional_import("xgboost", extra_name="xgboost")
        dmatrix = xgb.DMatrix(X)
        return self.model_.predict(dmatrix)
