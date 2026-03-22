"""Public package exports for mlcraft."""

from mlcraft.core.prediction import PredictionBundle
from mlcraft.core.results import EvaluationResult, ShapResult, TuningResult
from mlcraft.core.schema import ColumnSchema, DataSchema
from mlcraft.core.task import TaskSpec
from mlcraft.data.inference import InferenceOptions, SchemaInferer, infer_schema
from mlcraft.evaluation.evaluator import Evaluator
from mlcraft.models.factory import ModelFactory
from mlcraft.tuning.optuna_search import OptunaSearch

__all__ = [
    "ColumnSchema",
    "DataSchema",
    "TaskSpec",
    "PredictionBundle",
    "EvaluationResult",
    "TuningResult",
    "ShapResult",
    "InferenceOptions",
    "SchemaInferer",
    "infer_schema",
    "Evaluator",
    "ModelFactory",
    "OptunaSearch",
]

__version__ = "0.1.0"

