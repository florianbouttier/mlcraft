"""Optuna-based Bayesian search with overfitting penalty."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcraft.core.results import FoldSummary, TrialSummary, TuningResult
from mlcraft.core.task import TaskSpec
from mlcraft.data.containers import slice_rows
from mlcraft.metrics.registry import MetricRegistry, default_metric_registry
from mlcraft.models.factory import ModelFactory
from mlcraft.split.cv import resolve_cv_splitter
from mlcraft.tuning.search_space import default_search_space, merge_search_spaces, suggest_params
from mlcraft.utils.logging import inject_logger
from mlcraft.utils.optional import optional_import


class OptunaSearch:
    """Bayesian hyperparameter search with a validation-overfit penalty."""

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

    def run(self, X, y, *, sample_weight=None, exposure=None) -> TuningResult:
        optuna = optional_import("optuna", extra_name="tuning")
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
            metadata={"model_type": self.model_type, "n_trials": self.n_trials},
            study=study,
        )

