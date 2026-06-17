import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from app.ml.registry import ModelSpec, get_registry

# Silence Optuna's per-trial logging; pipeline-level logging is sufficient
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ModelResult:
    model_id: str
    display_name: str
    estimator: Any
    cv_score: float
    fit_time_s: float
    best_params: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class TrainingOutput:
    problem_type: str
    scoring: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    results: list[ModelResult]


def _safe_cv(problem_type: str, y: pd.Series, cv_folds: int, random_state: int):
    if problem_type == "classification":
        min_class_count = int(y.value_counts().min())
        folds = max(2, min(cv_folds, min_class_count))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    folds = max(2, min(cv_folds, len(y) // 2 or 2))
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _build_pipeline(spec: ModelSpec, preprocessor: ColumnTransformer, params: dict, random_state: int) -> Pipeline:
    kwargs = dict(spec.fixed_params)
    kwargs.update(params)
    if not spec.no_random_state:
        kwargs["random_state"] = random_state
    return Pipeline([("preprocess", clone(preprocessor)), ("model", spec.factory(**kwargs))])


def _fit_one(
    spec: ModelSpec,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
    scoring: str,
    n_trials: int,
    timeout_s: int,
    random_state: int,
) -> ModelResult:
    start = time.perf_counter()
    try:
        if spec.optuna_space is not None:
            def objective(trial: optuna.Trial) -> float:
                params = spec.optuna_space(trial)
                pipeline = _build_pipeline(spec, preprocessor, params, random_state)
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                return float(np.mean(scores))

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
            best_params = study.best_params
            cv_score = study.best_value
        else:
            # No search space defined — fit with fixed params only
            best_params = {}
            pipeline = _build_pipeline(spec, preprocessor, {}, random_state)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            cv_score = float(np.mean(scores))

        # Refit best configuration on full training set
        best_pipeline = _build_pipeline(spec, preprocessor, best_params, random_state)
        best_pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        return ModelResult(spec.model_id, spec.display_name, best_pipeline, cv_score, fit_time, best_params)
    except Exception as exc:  # noqa: BLE001
        fit_time = time.perf_counter() - start
        return ModelResult(spec.model_id, spec.display_name, None, float("-inf"), fit_time, {}, error=str(exc))


def train_and_tune_all(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    preprocessor: ColumnTransformer,
    *,
    cv_folds: int = 5,
    n_trials: int = 30,
    timeout_s: int = 180,
    random_state: int = 42,
    scoring: str | None = None,
    test_size: float = 0.2,
) -> TrainingOutput:
    scoring = scoring or ("f1_weighted" if problem_type == "classification" else "r2")

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    cv = _safe_cv(problem_type, y_train, cv_folds, random_state)
    specs = get_registry(problem_type, random_state=random_state)

    results: list[ModelResult] = []
    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = {
            executor.submit(
                _fit_one, spec, preprocessor, X_train, y_train, cv, scoring, n_trials, timeout_s, random_state
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            results.append(future.result())

    order = {spec.model_id: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order[r.model_id])

    return TrainingOutput(
        problem_type=problem_type,
        scoring=scoring,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        results=results,
    )
