import numpy as np

from mlcraft.core.prediction import PredictionBundle
from mlcraft.core.task import TaskSpec
from mlcraft.data.inference import infer_schema
from mlcraft.evaluation.evaluator import Evaluator

X = {
    "age": np.array([31, 42, 55, 23]),
    "segment": np.array(["a", "b", "a", "c"], dtype=object),
}
y = np.array([100.0, 120.0, 145.0, 90.0])

schema = infer_schema(X)
bundle = PredictionBundle(name="baseline", y_pred=np.array([98.0, 125.0, 140.0, 95.0]), task_spec=TaskSpec(task_type="regression"))
result = Evaluator().evaluate(y, bundle)

print(schema.to_dict())
print(result.to_dict(include_arrays=False))

