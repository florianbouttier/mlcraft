"""Optuna-based Bayesian search with overfitting penalty."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcraft.core.results import FoldSummary, TrialSummary, TuningResult
from mlcraft.core.task import TaskSpec
from mlcraft.data.containers import slice_rows
from mlcraft.evaluation.evaluator import Evaluator
from mlcraft.metrics.registry import MetricRegistry, default_metric_registry
from mlcraft.models.factory import ModelFactory
from mlcraft.split.cv import resolve_cv_splitter
from mlcraft.tuning.search_space import default_search_space, merge_search_spaces, suggest_params
from mlcraft.utils.logging import inject_logger
from mlcraft.utils.optional import optional_import


class OptunaSearch:
    """Run Optuna search with a validation-overfitting penalty.

    The search always maximizes an internal score. Metrics that are naturally
    minimized are negated first, then penalized with
    `val_score - alpha * abs(train_score - val_score)`.

    Args:
        task_spec: Shared task specification for the tuning run.
        model_type: Canonical backend name such as `xgboost`.
        n_trials: Number of Optuna trials to execute.
        cv: Number of folds when no custom splitter is provided.
        cv_splitter: Optional custom splitter implementing `split`.
        alpha: Overfitting penalty applied to the validation score.
        random_state: Optional random seed used by Optuna and the splitters.
        model_params: Optional fixed backend-native model parameters.
        fit_params: Optional fixed backend-native fit parameters.
        search_space: Optional search space overrides.
        report_options: Optional report rendering preferences kept for later
            use.
        metric_registry: Optional custom metric registry.
        logger: Optional custom logger.

    Example:
        >>> search = OptunaSearch(task_spec=TaskSpec(task_type="classification"), model_type="xgboost", n_trials=10)
        >>> search.alpha
        0.0
    """

    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        model_type: str,
        n_trials: int = 20,
        cv: int = 5,
        cv_splitter=None,
        alpha: float = 0.0,
        random_state: int | None = None,
        model_params: dict[str, Any] | None = None,
        fit_params: dict[str, Any] | None = None,
        search_space: dict[str, dict[str, Any]] | None = None,
        report_options: dict[str, Any] | None = None,
        metric_registry: MetricRegistry | None = None,
        logger=None,
    ) -> None:
        self.task_spec = task_spec if isinstance(task_spec, TaskSpec) else TaskSpec(**task_spec)
        self.model_type = str(model_type).lower()
        self.n_trials = int(n_trials)
        self.cv = int(cv)
        self.cv_splitter = cv_splitter
        self.alpha = float(alpha)
        self.random_state = random_state
        self.model_params = dict(model_params or {})
        self.fit_params = dict(fit_params or {})
        self.search_space = dict(search_space or {})
        self.report_options = dict(report_options or {})
        self.metric_registry = metric_registry or default_metric_registry
        self.logger = inject_logger(logger, "tuning")

    def run(
        self,
        X,
        y,
        *,
        sample_weight=None,
        exposure=None,
        X_test=None,
        y_test=None,
        sample_weight_test=None,
        exposure_test=None,
    ) -> TuningResult:
        """Execute the Optuna study and return structured results.

        Args:
            X: Feature data with shape `(n_samples, n_features)` or a column
                mapping.
            y: Target array of shape `(n_samples,)`.
            sample_weight: Optional per-row weights.
            exposure: Optional exposure vector of shape `(n_samples,)` for
                Poisson workflows.
            X_test: Optional holdout feature data used only for final-model
                evaluation after tuning.
            y_test: Optional holdout target array of shape `(n_test_samples,)`.
            sample_weight_test: Optional holdout weights.
            exposure_test: Optional holdout exposure vector for Poisson
                workflows.

        Returns:
            TuningResult: Structured search output containing the best trial,
            trial history, fold aggregates, and optional final holdout
            evaluation.

        Example:
            >>> search = OptunaSearch(task_spec=TaskSpec(task_type="regression"), model_type="xgboost", n_trials=5, cv=3)
            >>> result = search.run(X_train, y_train)
            >>> isinstance(result.best_params, dict)
            True
        """

        optuna = optional_import("optuna", extra_name="tuning")
        if (X_test is None) != (y_test is None):
            raise ValueError("X_test and y_test must be provided together.")
        metric_name = self.task_spec.eval_metric
        splitter = resolve_cv_splitter(
            self.cv,
            cv_splitter=self.cv_splitter,
            task_spec=self.task_spec,
            random_state=self.random_state,
        )
        search_space = merge_search_spaces(default_search_space(self.model_type, self.task_spec), self.search_space)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        trial_summaries: dict[int, TrialSummary] = {}
        y_array = np.asarray(y)
        weight_array = None if sample_weight is None else np.asarray(sample_weight)
        exposure_array = None if exposure is None else np.asarray(exposure)
        y_test_array = None if y_test is None else np.asarray(y_test)
        weight_test_array = None if sample_weight_test is None else np.asarray(sample_weight_test)
        exposure_test_array = None if exposure_test is None else np.asarray(exposure_test)

        def objective(trial) -> float:
            params = suggest_params(trial, search_space)
            fold_summaries: list[FoldSummary] = []
            train_values: list[float] = []
            val_values: list[float] = []
            train_scores: list[float] = []
            val_scores: list[float] = []
            penalized_scores: list[float] = []

            for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y_array)):
                model = ModelFactory.create(
                    self.model_type,
                    task_spec=self.task_spec,
                    model_params={**self.model_params, **params},
                    fit_params=self.fit_params,
                    random_state=self.random_state,
                    logger=self.logger,
                )
                X_train = slice_rows(X, train_idx)
                X_val = slice_rows(X, val_idx)
                y_train = y_array[train_idx]
                y_val = y_array[val_idx]
                sw_train = None if weight_array is None else weight_array[train_idx]
                sw_val = None if weight_array is None else weight_array[val_idx]
                exp_train = None if exposure_array is None else exposure_array[train_idx]
                exp_val = None if exposure_array is None else exposure_array[val_idx]
                eval_set = [(X_val, y_val, exp_val)] if self.fit_params.get("early_stopping_rounds") else None
                model.fit(X_train, y_train, sample_weight=sw_train, exposure=exp_train, eval_set=eval_set)
                train_bundle = model.predict_bundle(X_train, name="train", exposure=exp_train)
                val_bundle = model.predict_bundle(X_val, name="validation", exposure=exp_val)
                train_value, train_score = self.metric_registry.evaluate(
                    metric_name,
                    y_train,
                    y_pred=train_bundle.y_pred,
                    y_score=train_bundle.y_score,
                    sample_weight=sw_train,
                    exposure=exp_train,
                )
                val_value, val_score = self.metric_registry.evaluate(
                    metric_name,
                    y_val,
                    y_pred=val_bundle.y_pred,
                    y_score=val_bundle.y_score,
                    sample_weight=sw_val,
                    exposure=exp_val,
                )
                penalized = val_score - self.alpha * abs(train_score - val_score)
                train_values.append(train_value)
                val_values.append(val_value)
                train_scores.append(train_score)
                val_scores.append(val_score)
                penalized_scores.append(penalized)
                fold_summaries.append(
                    FoldSummary(
                        fold_index=fold_index,
                        train_metrics={metric_name: float(train_value)},
                        val_metrics={metric_name: float(val_value)},
                        train_score=float(train_score),
                        val_score=float(val_score),
                        penalized_score=float(penalized),
                    )
                )

            summary = TrialSummary(
                trial_number=trial.number,
                params=params,
                train_metrics={metric_name: float(np.mean(train_values))},
                val_metrics={metric_name: float(np.mean(val_values))},
                train_score=float(np.mean(train_scores)),
                val_score=float(np.mean(val_scores)),
                penalized_score=float(np.mean(penalized_scores)),
                folds=fold_summaries,
            )
            trial_summaries[trial.number] = summary
            trial.set_user_attr("summary", summary.__dict__)
            return summary.penalized_score

        study.optimize(objective, n_trials=self.n_trials)
        best_summary = trial_summaries[study.best_trial.number]
        history = [trial_summaries[index] for index in sorted(trial_summaries)]
        test_metrics = None
        test_score = None
        test_evaluation = None

        if X_test is not None and y_test is not None:
            final_fit_params = dict(self.fit_params)
            final_fit_params.pop("early_stopping_rounds", None)
            final_model = ModelFactory.create(
                self.model_type,
                task_spec=self.task_spec,
                model_params={**self.model_params, **dict(study.best_params)},
                fit_params=final_fit_params,
                random_state=self.random_state,
                logger=self.logger,
            )
            final_model.fit(X, y_array, sample_weight=weight_array, exposure=exposure_array)
            test_bundle = final_model.predict_bundle(X_test, name="final_test", exposure=exposure_test_array)
            evaluator = Evaluator(metric_registry=self.metric_registry, logger=self.logger)
            test_evaluation = evaluator.evaluate(
                y_test_array,
                test_bundle,
                task_spec=self.task_spec,
                sample_weight=weight_test_array,
                exposure=exposure_test_array,
            )
            test_metrics = test_evaluation.metrics_by_prediction().get("final_test", {})
            _, test_score = self.metric_registry.evaluate(
                metric_name,
                y_test_array,
                y_pred=test_bundle.y_pred,
                y_score=test_bundle.y_score,
                sample_weight=weight_test_array,
                exposure=exposure_test_array,
            )

        return TuningResult(
            task_spec=self.task_spec,
            best_params=dict(study.best_params),
            best_score=float(study.best_value),
            best_trial=best_summary,
            history=history,
            train_metrics=best_summary.train_metrics,
            val_metrics=best_summary.val_metrics,
            penalized_score=best_summary.penalized_score,
            fold_summaries=best_summary.folds,
            alpha=self.alpha,
            metric_name=metric_name,
            test_metrics=test_metrics,
            test_score=test_score,
            test_evaluation=test_evaluation,
            metadata={"model_type": self.model_type, "n_trials": self.n_trials},
            study=study,
        )
