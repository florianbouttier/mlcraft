"""Factory for gradient boosting wrappers."""

from __future__ import annotations

from mlcraft.models.catboost_model import CatBoostModel
from mlcraft.models.lightgbm_model import LightGBMModel
from mlcraft.models.xgboost_model import XGBoostModel


class ModelFactory:
    """Instantiate backend wrappers from a canonical model name.

    Example:
        >>> model = ModelFactory.create("xgboost", task_spec=TaskSpec(task_type="classification"))
        >>> model.backend_name
        'xgboost'
    """

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
        """Create a backend wrapper with the shared public API.

        Args:
            model_type: Canonical backend name such as `xgboost`.
            task_spec: Shared task specification for the model.
            model_params: Optional backend-native model parameters.
            fit_params: Optional backend-native fit parameters.
            random_state: Optional random seed or generator.
            logger: Optional custom logger.

        Returns:
            BaseGBMModel: Backend wrapper implementing the common interface.

        Raises:
            ValueError: If `model_type` is unknown.

        Example:
            >>> model = ModelFactory.create("lightgbm", task_spec=TaskSpec(task_type="regression"))
            >>> model.backend_name
            'lightgbm'
        """

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
