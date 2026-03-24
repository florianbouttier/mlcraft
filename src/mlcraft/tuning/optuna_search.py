"""Optuna-based Bayesian search with overfitting penalty."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mlcraft.core.results import FoldSummary, TrialSummary, TuningResult
from mlcraft.core.task import TaskSpec
from mlcraft.data.containers import slice_rows
from mlcraft.evaluation.evaluator import Evaluator
from mlcraft.metrics.registry import MetricRegistry, default_metric_registry
from mlcraft.models.factory import ModelFactory
from mlcraft.shap.analyzer import ShapAnalyzer
from mlcraft.split.cv import resolve_cv_splitter
from mlcraft.tuning.artifacts import write_tuning_artifacts
from mlcraft.tuning.search_space import default_search_space, merge_search_spaces, split_suggested_params, suggest_params
from mlcraft.utils.logging import inject_logger
from mlcraft.utils.optional import optional_import
from mlcraft.utils.serialization import to_serializable


class OptunaSearch:
    """Run Optuna search with a validation-overfitting penalty.

    The search always maximizes an internal score. Metrics that are naturally
    minimized are negated first, then penalized with
    `val_score - alpha * abs(train_score - val_score)`.

    Args:
        task_spec: Shared task specification for the tuning run.
        model_type: Canonical backend name such as `xgboost`, or a list of
            backend names/aliases to compare by running one Optuna search per
            backend and selecting the best result afterward.
        n_trials: Number of Optuna trials to execute per backend search.
        cv: Number of folds when no custom splitter is provided.
        cv_splitter: Optional custom splitter implementing `split`.
        alpha: Overfitting penalty applied to the validation score.
        random_state: Optional random seed used by Optuna and the splitters.
        model_params: Optional fixed backend-native model parameters.
        fit_params: Optional fixed backend-native fit parameters.
        search_space: Optional search space overrides. Can be a flat search
            space shared by every backend, or a mapping keyed by backend name
            when `model_type` contains several backends.
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
        model_type,
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
        self.model_types = self._normalize_model_types(model_type)
        self.model_type = self.model_types[0]
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

    def _normalize_model_types(self, model_type) -> tuple[str, ...]:
        if isinstance(model_type, (list, tuple, set)):
            requested = list(model_type)
        else:
            requested = [model_type]
        if not requested:
            raise ValueError("model_type must contain at least one backend.")
        normalized: list[str] = []
        for item in requested:
            candidate = ModelFactory.normalize_model_type(item)
            if candidate not in normalized:
                normalized.append(candidate)
        return tuple(normalized)

    def _resolve_search_space(self, model_type: str) -> dict[str, dict[str, Any]]:
        override = dict(self.search_space or {})
        if override:
            try:
                normalized_mapping = {
                    ModelFactory.normalize_model_type(key): value for key, value in override.items()
                }
            except ValueError:
                normalized_mapping = None
            if normalized_mapping is not None:
                override = dict(normalized_mapping.get(model_type, {}))
        return merge_search_spaces(default_search_space(model_type, self.task_spec), override)

    def _resolve_model_params(self, model_type: str) -> dict[str, Any]:
        params = dict(self.model_params or {})
        if params:
            try:
                normalized_mapping = {
                    ModelFactory.normalize_model_type(key): value for key, value in params.items()
                }
            except ValueError:
                normalized_mapping = None
            if normalized_mapping is not None:
                params = dict(normalized_mapping.get(model_type, {}))
        return params

    def _run_single_backend(
        self,
        *,
        model_type: str,
        optuna,
        X,
        y_array: np.ndarray,
        metric_name: str,
        weight_array,
        exposure_array,
        X_test,
        y_test_array,
        weight_test_array,
        exposure_test_array,
    ) -> TuningResult:
        splitter = resolve_cv_splitter(
            self.cv,
            cv_splitter=self.cv_splitter,
            task_spec=self.task_spec,
            random_state=self.random_state,
        )
        search_space = self._resolve_search_space(model_type)
        model_params = self._resolve_model_params(model_type)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        trial_summaries: dict[int, TrialSummary] = {}
        evaluator = Evaluator(metric_registry=self.metric_registry, logger=self.logger)

        def objective(trial) -> float:
            params = suggest_params(trial, search_space)
            trial_model_params, trial_fit_params = split_suggested_params(params, search_space)
            fold_summaries: list[FoldSummary] = []
            train_metric_values: dict[str, list[float]] = defaultdict(list)
            val_metric_values: dict[str, list[float]] = defaultdict(list)
            train_scores: list[float] = []
            val_scores: list[float] = []
            penalized_scores: list[float] = []

            for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y_array)):
                model = ModelFactory.create(
                    model_type,
                    task_spec=self.task_spec,
                    model_params={**model_params, **trial_model_params},
                    fit_params={**self.fit_params, **trial_fit_params},
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
                train_evaluation = evaluator.evaluate(
                    y_train,
                    train_bundle,
                    task_spec=self.task_spec,
                    sample_weight=sw_train,
                    exposure=exp_train,
                )
                val_evaluation = evaluator.evaluate(
                    y_val,
                    val_bundle,
                    task_spec=self.task_spec,
                    sample_weight=sw_val,
                    exposure=exp_val,
                )
                train_metrics = train_evaluation.metrics_by_prediction().get("train", {})
                val_metrics = val_evaluation.metrics_by_prediction().get("validation", {})
                train_value = float(train_metrics[metric_name])
                val_value = float(val_metrics[metric_name])
                train_score = self.metric_registry.score(metric_name, train_value)
                val_score = self.metric_registry.score(metric_name, val_value)
                penalized = val_score - self.alpha * abs(train_score - val_score)
                for metric_key, metric_value in train_metrics.items():
                    train_metric_values[str(metric_key)].append(float(metric_value))
                for metric_key, metric_value in val_metrics.items():
                    val_metric_values[str(metric_key)].append(float(metric_value))
                train_scores.append(train_score)
                val_scores.append(val_score)
                penalized_scores.append(penalized)
                fold_summaries.append(
                    FoldSummary(
                        fold_index=fold_index,
                        train_metrics={str(metric_key): float(metric_value) for metric_key, metric_value in train_metrics.items()},
                        val_metrics={str(metric_key): float(metric_value) for metric_key, metric_value in val_metrics.items()},
                        train_score=float(train_score),
                        val_score=float(val_score),
                        penalized_score=float(penalized),
                        train_evaluation=train_evaluation,
                        val_evaluation=val_evaluation,
                    )
                )

            summary = TrialSummary(
                trial_number=trial.number,
                params=params,
                train_metrics={metric_key: float(np.mean(values)) for metric_key, values in train_metric_values.items()},
                val_metrics={metric_key: float(np.mean(values)) for metric_key, values in val_metric_values.items()},
                train_score=float(np.mean(train_scores)),
                val_score=float(np.mean(val_scores)),
                penalized_score=float(np.mean(penalized_scores)),
                folds=fold_summaries,
            )
            trial_summaries[trial.number] = summary
            trial.set_user_attr("summary", to_serializable(summary, include_arrays=False))
            return summary.penalized_score

        study.optimize(objective, n_trials=self.n_trials)
        best_summary = trial_summaries[study.best_trial.number]
        history = [trial_summaries[index] for index in sorted(trial_summaries)]
        test_metrics = None
        test_score = None
        test_evaluation = None
        best_model_params, best_fit_params = split_suggested_params(dict(study.best_params), search_space)
        final_fit_params = {**self.fit_params, **best_fit_params}
        final_fit_params.pop("early_stopping_rounds", None)
        final_model = ModelFactory.create(
            model_type,
            task_spec=self.task_spec,
            model_params={**model_params, **best_model_params},
            fit_params=final_fit_params,
            random_state=self.random_state,
            logger=self.logger,
        )
        final_model.fit(X, y_array, sample_weight=weight_array, exposure=exposure_array)

        if X_test is not None and y_test_array is not None:
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
            final_model=final_model,
            metadata={"model_type": model_type, "model_types": [model_type], "n_trials": self.n_trials},
            study=study,
        )

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
            evaluation. When several backends are provided, this returns the
            best backend-specific result after running one full Optuna search
            per backend.

        Example:
            >>> search = OptunaSearch(task_spec=TaskSpec(task_type="regression"), model_type="xgboost", n_trials=5, cv=3)
            >>> result = search.run(X_train, y_train)
            >>> isinstance(result.best_params, dict)
            True
        """

        optuna = optional_import("optuna")
        if (X_test is None) != (y_test is None):
            raise ValueError("X_test and y_test must be provided together.")
        metric_name = self.task_spec.eval_metric
        y_array = np.asarray(y)
        weight_array = None if sample_weight is None else np.asarray(sample_weight)
        exposure_array = None if exposure is None else np.asarray(exposure)
        y_test_array = None if y_test is None else np.asarray(y_test)
        weight_test_array = None if sample_weight_test is None else np.asarray(sample_weight_test)
        exposure_test_array = None if exposure_test is None else np.asarray(exposure_test)
        if len(self.model_types) == 1:
            return self._run_single_backend(
                model_type=self.model_types[0],
                optuna=optuna,
                X=X,
                y_array=y_array,
                metric_name=metric_name,
                weight_array=weight_array,
                exposure_array=exposure_array,
                X_test=X_test,
                y_test_array=y_test_array,
                weight_test_array=weight_test_array,
                exposure_test_array=exposure_test_array,
            )

        results_by_model = {
            model_type: self._run_single_backend(
                model_type=model_type,
                optuna=optuna,
                X=X,
                y_array=y_array,
                metric_name=metric_name,
                weight_array=weight_array,
                exposure_array=exposure_array,
                X_test=X_test,
                y_test_array=y_test_array,
                weight_test_array=weight_test_array,
                exposure_test_array=exposure_test_array,
            )
            for model_type in self.model_types
        }
        best_model_type = max(self.model_types, key=lambda name: float(results_by_model[name].best_score))
        best_result = results_by_model[best_model_type]
        comparison = {
            model_type: {
                "result": model_result.to_dict(include_arrays=True),
                "best_params": dict(model_result.best_params),
                "best_score": float(model_result.best_score),
                "train_metrics": dict(model_result.train_metrics),
                "val_metrics": dict(model_result.val_metrics),
                "test_metrics": None if model_result.test_metrics is None else dict(model_result.test_metrics),
                "test_score": None if model_result.test_score is None else float(model_result.test_score),
            }
            for model_type, model_result in results_by_model.items()
        }
        metadata = dict(best_result.metadata)
        metadata.update(
            {
                "model_type": best_model_type,
                "model_types": list(self.model_types),
                "selected_model_type": best_model_type,
                "backend_comparison": comparison,
                "backend_results": {model_type: payload["result"] for model_type, payload in comparison.items()},
            }
        )
        return TuningResult(
            task_spec=best_result.task_spec,
            best_params=dict(best_result.best_params),
            best_score=float(best_result.best_score),
            best_trial=best_result.best_trial,
            history=best_result.history,
            train_metrics=best_result.train_metrics,
            val_metrics=best_result.val_metrics,
            penalized_score=best_result.penalized_score,
            fold_summaries=best_result.fold_summaries,
            alpha=best_result.alpha,
            metric_name=best_result.metric_name,
            test_metrics=best_result.test_metrics,
            test_score=best_result.test_score,
            test_evaluation=best_result.test_evaluation,
            final_model=best_result.final_model,
            metadata=metadata,
            study=best_result.study,
        )

    def run_with_artifacts(
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
        output_dir=None,
        report_name: str = "report.html",
        result_name: str = "tuning.json",
        title: str | None = "mlcraft Tuning Report",
        compute_shap: bool = False,
        X_explain=None,
        sample_weight_explain=None,
        shap_max_samples: int | None = 400,
        shap_interaction_values: bool = False,
        full_report_name: str = "full_report.html",
        shap_report_name: str = "shap_report.html",
        shap_result_name: str = "shap.json",
    ):
        """Run the search and persist the default tuning artifacts.

        When `compute_shap=True`, the method explains `X_explain` when
        provided, otherwise it falls back to `X_test`, then to `X`.
        """

        result = self.run(
            X,
            y,
            sample_weight=sample_weight,
            exposure=exposure,
            X_test=X_test,
            y_test=y_test,
            sample_weight_test=sample_weight_test,
            exposure_test=exposure_test,
        )
        shap_result = None
        if compute_shap:
            explain_data = X_explain if X_explain is not None else (X_test if X_test is not None else X)
            explain_weight = sample_weight_explain if sample_weight_explain is not None else sample_weight_test
            shap_result = ShapAnalyzer(logger=self.logger).compute(
                result.final_model,
                explain_data,
                sample_weight=explain_weight,
                max_samples=shap_max_samples,
                interaction_values=shap_interaction_values,
            )
        artifacts = write_tuning_artifacts(
            result,
            output_dir=output_dir or self.report_options.get("output_dir"),
            report_name=report_name,
            result_name=result_name,
            title=title,
            evaluation=result.test_evaluation,
            shap=shap_result,
            full_report_name=full_report_name,
            shap_report_name=shap_report_name,
            shap_result_name=shap_result_name,
        )
        return result, artifacts
