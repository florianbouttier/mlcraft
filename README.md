# mlcraft

`mlcraft` is a modular, numpy-first machine learning library designed to stay reusable across projects while keeping a small dependency surface.

Highlights:

- column-oriented data API based on `dict[str, np.ndarray]`
- schema inference and reusable task specifications
- pure-numpy metrics and evaluation pipelines
- unified gradient boosting wrappers for XGBoost, LightGBM, and CatBoost
- Optuna-based tuning with an overfitting penalty
- HTML reporting for evaluation, tuning, and SHAP explainability

## Quick start

```python
import numpy as np

from mlcraft.core.task import TaskSpec
from mlcraft.data.inference import infer_schema
from mlcraft.evaluation.evaluator import Evaluator
from mlcraft.core.prediction import PredictionBundle

X = {
    "f1": np.array([1.0, 2.0, 3.0, 4.0]),
    "f2": np.array([0, 1, 0, 1]),
}
y = np.array([1.2, 2.1, 3.2, 3.8])

schema = infer_schema(X)
task = TaskSpec(task_type="regression")
bundle = PredictionBundle(name="baseline", y_pred=np.array([1.0, 2.0, 3.0, 4.0]), task_spec=task)

result = Evaluator().evaluate(y, bundle)
print(schema.to_dict())
print(result.to_dict())
```

