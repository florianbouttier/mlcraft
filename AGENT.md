# AGENT.md

## 1. Project Overview

`mlcraft` is a modular, object-oriented machine learning library.

Current project goals:

- numpy-first data core
- minimal dependencies
- coherent API across modules
- strict separation between computation and HTML rendering
- reuse across multiple projects
- consistent wrappers for optional gradient boosting backends

The project lives under `src/mlcraft/`, with examples in `examples/` and tests in `tests/`.

## 2. Existing Architecture

This is the current implemented architecture:

- `core/`
  - central project contracts
  - `schema.py` contains `ColumnSchema`, `DataSchema`, `ColumnDType`, `ColumnRole`
  - `task.py` contains `TaskSpec`, `TaskType`, `PredictionType`
  - `prediction.py` contains `PredictionBundle` and `resolve_task_spec`
  - `results.py` contains `EvaluationResult`, `TuningResult`, `ShapResult`, and related structured result objects
- `data/`
  - data container handling, schema inference, and feature adaptation
  - `containers.py` normalizes `dict[str, np.ndarray]` and `np.ndarray`
  - `detection.py` handles dtype and missing-value detection
  - `inference.py` exposes `InferenceOptions`, `SchemaInferer`, `infer_schema`
  - `adapters.py` contains `FeatureAdapterConfig`, `FittedFeatureAdapter`, `fit_feature_adapter`, `transform_feature_data`
- `split/`
  - splitters without a scikit-learn dependency
  - `train_test.py` exposes `train_test_split_random` and `train_test_split_time`
  - `cv.py` exposes `KFoldSplitter`, `StratifiedKFoldSplitter`, `resolve_cv_splitter`
  - `base.py` defines the `BaseCVSplitter` protocol
- `metrics/`
  - pure numpy metrics and the central registry
  - `regression.py`, `classification.py`, `poisson.py`
  - `registry.py` contains `MetricDefinition`, `MetricRegistry`, `default_metric_registry`
- `evaluation/`
  - evaluation computation plus dedicated rendering
  - `evaluator.py` contains `Evaluator`
  - `curves.py` prepares curve data
  - `renderer.py` contains `EvaluationReportRenderer`
- `models/`
  - shared abstraction for model wrappers
  - `base.py` contains `BaseGBMModel`
  - `xgboost_model.py`, `lightgbm_model.py`, `catboost_model.py` contain backend wrappers
  - `factory.py` contains `ModelFactory`
  - `objectives.py` handles backend objective and metric mapping
- `tuning/`
  - Optuna-based optimization and related reporting
  - `optuna_search.py` contains `OptunaSearch`
  - `search_space.py` handles search spaces
  - `renderer.py` contains `TuningReportRenderer`
- `shap/`
  - optional explainability
  - `analyzer.py` contains `ShapAnalyzer`
  - `renderer.py` contains `ShapReportRenderer`
- `reporting/`
  - shared HTML helpers
  - `html.py` contains the shared HTML shell
  - `palette.py` contains the centralized report palette, typography tokens, and surface design tokens
  - `view_models.py` contains dictionary-based report contexts consumed by renderers
  - `full_report.py` contains `FullReportBuilder`
- `utils/`
  - shared utilities
  - `logging.py` centralizes package-level logging
  - `optional.py` handles optional dependencies
  - `random.py` normalizes `random_state`
  - `serialization.py` serializes numpy arrays, dataclasses, and enums

Key existing objects:

- `TaskSpec`
- `ColumnSchema` / `DataSchema`
- `PredictionBundle`
- `EvaluationResult`
- `TuningResult`
- `ShapResult`

Existing interactions that should be preserved:

- `TaskSpec` is the shared contract across models, evaluation, tuning, and predictions.
- `infer_schema()` produces a `DataSchema`, which is then reused by `fit_feature_adapter()`.
- `BaseGBMModel.fit()` infers schema, builds a feature adapter, and trains the native backend.
- `BaseGBMModel.predict_bundle()` produces a `PredictionBundle`.
- `Evaluator.evaluate()` consumes one or more `PredictionBundle` objects and returns an `EvaluationResult`.
- `OptunaSearch.run()` creates models through `ModelFactory`, scores them through `MetricRegistry`, and returns a `TuningResult`.
- When `X_test` and `y_test` are provided to `OptunaSearch.run()`, the best model is refit on the full training data, evaluated on the holdout set, and the final holdout metrics are stored in `TuningResult`.
- `ShapAnalyzer.compute()` returns a `ShapResult`.
- `FullReportBuilder.build()` combines `EvaluationResult`, `TuningResult`, and `ShapResult`.
- Reporting renderers build dictionary contexts first, then render HTML from those contexts.
- All report HTML is now driven by shared D3.js helpers in `reporting/html.py`, with dictionary contexts built in `reporting/view_models.py`.

## 3. Core Design Principles

- numpy-first across the library core
- limited and justified dependencies
- strong defaults with explicit overrides
- reuse existing abstractions before creating new ones
- avoid duplicating logic across modules
- keep a coherent API from one module to another
- separate data, models, metrics, evaluation, and reporting clearly
- when a requested capability is generic to `mlcraft` rather than specific to a downstream project, implement it in `mlcraft` first and keep downstream scripts thin

## 4. Public API Rules

Current top-level exports from `src/mlcraft/__init__.py`:

- `ColumnSchema`
- `DataSchema`
- `TaskSpec`
- `PredictionBundle`
- `EvaluationResult`
- `TuningResult`
- `ShapResult`
- `InferenceOptions`
- `SchemaInferer`
- `infer_schema`
- `Evaluator`
- `ModelFactory`
- `OptunaSearch`

Rules:

- do not automatically expose a new object through `mlcraft/__init__.py`
- only add top-level exports for stable, reusable, cross-module entry points
- avoid public signatures with too many flat parameters
- use existing objects and configuration dictionaries for advanced usage
- existing examples include `model_params`, `fit_params`, `search_space`, `report_options`, and `inference_options`
- treat the top-level API as stable by default

## 5. Dependency Policy

- no `pandas`
- no `polars`
- no `scikit-learn` dependency in the current architecture
- `shap` remains optional
- `xgboost`, `lightgbm`, and `catboost` remain optional
- every new dependency must be justified by a real need not covered by the current stack
- if a dependency is optional, use the existing pattern from `utils.optional`

Current dependencies in the project:

- base: `numpy`, `jinja2`, `matplotlib`, `optuna`
- optional: `plotly`, `shap`, `xgboost`, `lightgbm`, `catboost`

## 6. Logging Policy

- use `logging`, never `print` in library code
- use helpers from `utils.logging`
- current helpers are:
  - `configure_logging`
  - `get_logger`
  - `set_verbosity`
  - `inject_logger`
- keep the `mlcraft` logger namespace
- let users retrieve loggers and change verbosity later
- preserve the current pattern: injectable logger with a package-level fallback

## 7. Metrics & Optimization Rules

- `MetricRegistry` is the source of truth for metrics
- it maps:
  - canonical user-facing metric name
  - internal numpy function
  - backend alias
  - `higher_is_better` direction
- do not spread optimization-direction logic across multiple modules

Current score normalization:

- if `higher_is_better = True`, internal score = `metric`
- if `higher_is_better = False`, internal score = `-metric`

Current Optuna logic in `OptunaSearch`:

- `penalized_score = val_score - alpha * abs(train_score - val_score)`
- Optuna always maximizes this `penalized_score`
- do not change this convention without strong justification and a clear migration path

## 8. Data & Schema Rules

Currently supported dtypes:

- `integer`
- `float`
- `boolean`
- `datetime`
- `categorical`

Rules to preserve:

- missing-value handling is part of schema inference
- missing values must not artificially change the semantic dtype
- do not mix inference and transformation
- inference belongs in `data/detection.py` and `data/inference.py`
- transformation belongs in `data/adapters.py`
- keep `DataSchema` and `ColumnSchema` as the source of truth for column metadata

## 9. Model Wrappers Rules

Current shared interface on `BaseGBMModel`:

- `fit`
- `predict`
- `predict_proba`
- `get_params`
- `set_params`

Other public methods already present:

- `predict_bundle`
- `transform_features`

Rules:

- every new wrapper must follow the `BaseGBMModel` contract
- backend mapping must go through `models/objectives.py` and `MetricRegistry`
- preserve coherent semantics across backends
- support `sample_weight` when the backend supports it
- support `exposure` in the Poisson flow using the current mechanism
- use `ModelFactory` as the uniform entry point

## 10. Evaluation & Reporting Rules

- computation and HTML rendering must remain separate
- `Evaluator` computes, renderers render
- reporting view data should be prepared as dictionaries before HTML rendering whenever possible
- `EvaluationResult` must not contain HTML logic
- `TuningResult` must not contain HTML logic
- `ShapResult` must not contain HTML logic
- dedicated rendering stays in:
  - `evaluation/renderer.py`
  - `tuning/renderer.py`
  - `shap/renderer.py`
  - `reporting/full_report.py`
- shared reporting palette stays in `reporting/palette.py`
- dictionary view-model builders stay in `reporting/view_models.py`
- when changing report design, prefer `reporting/palette.py` and `reporting/html.py` first so the look stays centrally configurable
- keep report interactivity centralized in the shared D3/HTML shell rather than scattering custom JavaScript across modules
- reports should stay fast, lightweight, and browsable with HTML-native controls for metric and view selection

## 11. SHAP Rules

- `shap` is optional
- use `optional_import()` to handle missing dependencies cleanly
- `ShapAnalyzer` only computes results
- `ShapReportRenderer` only renders results
- keep the fallback clear when `shap` is not installed

## 12. Testing Policy

- every new feature must add unit tests
- optional dependencies must use conditional tests
- current pattern: `pytest.importorskip(...)`
- important pipelines should have integration tests
- do not break the structure of existing result objects
- preserve the current test organization:
  - `tests/unit/`
  - `tests/integration/`
  - `tests/optional/`
  - `tests/regression/`

Before considering a change complete:

- verify package compilation when relevant
- run targeted tests or the full `pytest` suite depending on the scope

## 13. Documentation Policy

- use Google-style docstrings exclusively
- keep docstrings compact, useful, and usage-oriented
- add concrete examples on important public APIs
- do not duplicate type hints unnecessarily in prose
- do not leave empty, vague, or decorative docstrings

## 14. Change Workflow

When modifying the code:

1. identify the real module where the feature belongs
2. check whether `TaskSpec`, `DataSchema`, `PredictionBundle`, `MetricRegistry`, or another existing object already covers the need
3. implement the change with the smallest reasonable impact
4. add or update tests
5. add or update docstrings
6. verify cross-module consistency
7. do not break the API without an explicit justification
8. make atomic commits with clear messages
9. update `AGENT.md` when architecture, policies, or public API expectations change

## 15. Anti-patterns to Avoid

- duplicated logic across modules
- new abstractions when an existing object already fits
- unjustified heavy dependencies
- unreadable public signatures with too many flat parameters
- mixing computation and HTML rendering
- duplicating backend logic outside `models/objectives.py` and `MetricRegistry`
- `print` in library code
- adding a top-level export without a strong reason
