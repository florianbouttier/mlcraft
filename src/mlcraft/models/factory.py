"""Factory for gradient boosting wrappers."""

from __future__ import annotations

from mlcraft.models.catboost_model import CatBoostModel
from mlcraft.models.lightgbm_model import LightGBMModel
from mlcraft.models.xgboost_model import XGBoostModel


class ModelFactory:
    """Instantiate backend wrappers from a simple model type name."""

    _MODELS = {
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
    }

    @classmethod
    def create(
        cls,
        model_type,
        *,
        task_spec,
        model_params=None,
        fit_params=None,
        random_state=None,
        logger=None,
    ):
        normalized = str(model_type).lower()
        if normalized not in cls._MODELS:
            raise ValueError(f"Unknown model_type: {model_type}")
        return cls._MODELS[normalized](
            task_spec=task_spec,
            model_params=model_params,
            fit_params=fit_params,
            random_state=random_state,
            logger=logger,
        )
