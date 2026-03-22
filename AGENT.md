# AGENT.md

## 1. Project Overview

`mlcraft` est une librairie de machine learning modulaire orientée objet.

Objectifs actuels du projet :

- coeur de données numpy-first
- dépendances minimales
- API cohérente entre modules
- séparation stricte entre calcul et rendu HTML
- réutilisabilité sur plusieurs projets
- wrappers homogènes pour les backends gradient boosting optionnels

Le projet est structuré sous `src/mlcraft/`, avec des exemples dans `examples/` et des tests dans `tests/`.

## 2. Existing Architecture

Architecture réelle actuellement implémentée :

- `core/`
  - contrats centraux du projet
  - `schema.py` contient `ColumnSchema`, `DataSchema`, `ColumnDType`, `ColumnRole`
  - `task.py` contient `TaskSpec`, `TaskType`, `PredictionType`
  - `prediction.py` contient `PredictionBundle` et `resolve_task_spec`
  - `results.py` contient `EvaluationResult`, `TuningResult`, `ShapResult` et les objets structurés associés
- `data/`
  - gestion des conteneurs, inférence de schéma et adaptation des features
  - `containers.py` normalise `dict[str, np.ndarray]` et `np.ndarray`
  - `detection.py` gère détection de types et NA
  - `inference.py` expose `InferenceOptions`, `SchemaInferer`, `infer_schema`
  - `adapters.py` contient `FeatureAdapterConfig`, `FittedFeatureAdapter`, `fit_feature_adapter`, `transform_feature_data`
- `split/`
  - splitters sans dépendance à scikit-learn
  - `train_test.py` expose `train_test_split_random` et `train_test_split_time`
  - `cv.py` expose `KFoldSplitter`, `StratifiedKFoldSplitter`, `resolve_cv_splitter`
  - `base.py` définit le protocole `BaseCVSplitter`
- `metrics/`
  - métriques numpy pures et registry central
  - `regression.py`, `classification.py`, `poisson.py`
  - `registry.py` contient `MetricDefinition`, `MetricRegistry`, `default_metric_registry`
- `evaluation/`
  - calcul d’évaluation et rendu dédié
  - `evaluator.py` contient `Evaluator`
  - `curves.py` prépare les courbes
  - `renderer.py` contient `EvaluationReportRenderer`
- `models/`
  - abstraction commune des wrappers modèles
  - `base.py` contient `BaseGBMModel`
  - `xgboost_model.py`, `lightgbm_model.py`, `catboost_model.py` contiennent les wrappers backend
  - `factory.py` contient `ModelFactory`
  - `objectives.py` gère le mapping objectifs / métriques backend
- `tuning/`
  - optimisation via Optuna et reporting associé
  - `optuna_search.py` contient `OptunaSearch`
  - `search_space.py` gère les search spaces
  - `renderer.py` contient `TuningReportRenderer`
- `shap/`
  - explainability optionnelle
  - `analyzer.py` contient `ShapAnalyzer`
  - `renderer.py` contient `ShapReportRenderer`
- `reporting/`
  - helpers HTML transverses
  - `html.py` contient le shell HTML partagé
  - `full_report.py` contient `FullReportBuilder`
- `utils/`
  - utilitaires transverses
  - `logging.py` centralise le logging package-level
  - `optional.py` gère les dépendances optionnelles
  - `random.py` normalise `random_state`
  - `serialization.py` sérialise numpy / dataclasses / enums

Objets clés existants :

- `TaskSpec`
- `ColumnSchema` / `DataSchema`
- `PredictionBundle`
- `EvaluationResult`
- `TuningResult`
- `ShapResult`

Interactions existantes à respecter :

- `TaskSpec` est le contrat commun entre modèles, évaluation, tuning et prédictions
- `infer_schema()` produit `DataSchema`, qui est ensuite réutilisé par `fit_feature_adapter()`
- `BaseGBMModel.fit()` infère le schéma, crée un adapter, entraîne le backend natif
- `BaseGBMModel.predict_bundle()` produit un `PredictionBundle`
- `Evaluator.evaluate()` consomme un ou plusieurs `PredictionBundle` et retourne un `EvaluationResult`
- `OptunaSearch.run()` crée des modèles via `ModelFactory`, évalue via `MetricRegistry`, et retourne un `TuningResult`
- `ShapAnalyzer.compute()` retourne un `ShapResult`
- `FullReportBuilder.build()` combine `EvaluationResult`, `TuningResult` et `ShapResult`

## 3. Core Design Principles

- numpy-first partout dans le coeur de la lib
- dépendances limitées et justifiées
- strong defaults avec override explicite
- réutiliser les abstractions existantes avant d’en créer une nouvelle
- éviter la duplication de logique entre modules
- garder une API cohérente d’un module à l’autre
- séparer clairement data, modèles, métriques, évaluation et reporting

## 4. Public API Rules

Exports top-level actuels via `src/mlcraft/__init__.py` :

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

Règles :

- ne pas exposer automatiquement un nouvel objet via `mlcraft/__init__.py`
- n’ajouter au top-level que des entrées stables, réutilisables et cross-module
- éviter les signatures publiques avec trop de paramètres plats
- utiliser les objets existants et des dictionnaires de config pour l’advanced usage
  - exemples existants : `model_params`, `fit_params`, `search_space`, `report_options`, `inference_options`
- considérer l’API top-level comme stable par défaut

## 5. Dependency Policy

- pas de `pandas`
- pas de `polars`
- pas de dépendance à `scikit-learn` dans l’architecture actuelle
- `shap` reste optionnel
- `xgboost`, `lightgbm`, `catboost` restent optionnels
- toute nouvelle dépendance doit être justifiée par un vrai besoin non couvert par l’existant
- si une dépendance est optionnelle, utiliser le pattern déjà en place dans `utils.optional`

Dépendances actuellement présentes dans le projet :

- base : `numpy`, `jinja2`, `matplotlib`, `optuna`
- optionnelles : `shap`, `xgboost`, `lightgbm`, `catboost`

## 6. Logging Policy

- utiliser `logging`, jamais `print` dans le code de librairie
- utiliser les helpers de `utils.logging`
  - `configure_logging`
  - `get_logger`
  - `set_verbosity`
  - `inject_logger`
- respecter le namespace `mlcraft`
- permettre à l’utilisateur de récupérer puis modifier la verbosité plus tard
- conserver le pattern actuel : logger injectable, fallback package-level

## 7. Metrics & Optimization Rules

- `MetricRegistry` est la source de vérité des métriques
- il mappe :
  - nom canonique utilisateur
  - fonction numpy interne
  - alias backend
  - direction `higher_is_better`
- ne pas disperser la logique de direction d’optimisation dans plusieurs modules

Normalisation actuelle des scores :

- si `higher_is_better = True`, score interne = `metric`
- si `higher_is_better = False`, score interne = `-metric`

Logique Optuna actuelle dans `OptunaSearch` :

- `penalized_score = val_score - alpha * abs(train_score - val_score)`
- Optuna maximise toujours ce `penalized_score`
- ne pas casser cette convention sans justification forte et migration claire

## 8. Data & Schema Rules

Types supportés aujourd’hui :

- `integer`
- `float`
- `boolean`
- `datetime`
- `categorical`

Règles à respecter :

- la gestion des NA fait partie de l’inférence de schéma
- les NA ne doivent pas faire dériver artificiellement un type sémantique
- ne pas mélanger inférence et transformation
  - inférence dans `data/detection.py` et `data/inference.py`
  - transformation dans `data/adapters.py`
- conserver `DataSchema` et `ColumnSchema` comme source de vérité des métadonnées colonne

## 9. Model Wrappers Rules

Interface commune existante sur `BaseGBMModel` :

- `fit`
- `predict`
- `predict_proba`
- `get_params`
- `set_params`

Autres méthodes publiques déjà présentes :

- `predict_bundle`
- `transform_features`

Règles :

- tout nouveau wrapper doit hériter du contrat `BaseGBMModel`
- le mapping backend doit passer par `models/objectives.py` et `MetricRegistry`
- conserver une sémantique cohérente entre backends
- supporter `sample_weight` quand le backend le permet
- supporter `exposure` dans le flux Poisson selon le mécanisme actuel
- utiliser `ModelFactory` comme point d’entrée uniforme

## 10. Evaluation & Reporting Rules

- calcul et rendu HTML doivent rester séparés
- `Evaluator` calcule, les renderers rendent
- `EvaluationResult` ne doit pas contenir de logique HTML
- `TuningResult` ne doit pas contenir de logique HTML
- `ShapResult` ne doit pas contenir de logique HTML
- le rendu dédié reste dans :
  - `evaluation/renderer.py`
  - `tuning/renderer.py`
  - `shap/renderer.py`
  - `reporting/full_report.py`

## 11. SHAP Rules

- `shap` est optionnel
- utiliser `optional_import()` pour gérer proprement l’absence de dépendance
- `ShapAnalyzer` calcule uniquement les résultats
- `ShapReportRenderer` s’occupe uniquement du rendu
- garder le fallback propre si `shap` n’est pas installé

## 12. Testing Policy

- toute nouvelle feature doit ajouter des tests unitaires
- les dépendances optionnelles doivent avoir des tests conditionnels
  - pattern actuel : `pytest.importorskip(...)`
- les pipelines importants doivent avoir des tests d’intégration
- ne pas casser la structure des objets de résultats existants
- conserver l’organisation actuelle :
  - `tests/unit/`
  - `tests/integration/`
  - `tests/optional/`
  - `tests/regression/`

Avant de considérer un changement comme terminé :

- vérifier la compilation du package si pertinent
- lancer les tests ciblés ou `pytest` selon l’impact

## 13. Documentation Policy

- utiliser exclusivement des docstrings style Google
- garder les docstrings compactes, utiles, et orientées usage
- ajouter des exemples concrets sur les API publiques importantes
- ne pas dupliquer inutilement les type hints dans le texte
- pas de docstring vide, vague ou décorative

## 14. Change Workflow

Quand tu modifies le code :

1. identifier le module réel où la feature doit s’intégrer
2. vérifier si `TaskSpec`, `DataSchema`, `PredictionBundle`, `MetricRegistry` ou un autre objet existant couvre déjà le besoin
3. implémenter avec l’impact minimal possible
4. ajouter ou adapter les tests
5. ajouter ou corriger les docstrings
6. vérifier la cohérence globale avec les autres modules
7. ne pas casser l’API sans justification explicite
8. faire des commits atomiques avec des messages clairs

## 15. Anti-patterns à éviter

- duplication de logique entre modules
- nouvelle abstraction alors qu’un objet existant convient déjà
- dépendance lourde non justifiée
- signature publique illisible avec trop de paramètres plats
- mélange calcul / rendu HTML
- logique backend dupliquée hors de `models/objectives.py` et `MetricRegistry`
- `print` dans le code librairie
- ajout d’un export top-level sans raison forte
