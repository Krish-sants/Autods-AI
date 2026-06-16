import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from app.ml.registry import ModelSpec, get_registry


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


def _param_space_size(param_distributions: dict[str, list[Any]]) -> int:
    if not param_distributions:
        return 1
    return prod(len(v) for v in param_distributions.values())


def _safe_cv(problem_type: str, y: pd.Series, cv_folds: int, random_state: int):
    if problem_type == "classification":
        min_class_count = int(y.value_counts().min())
        folds = max(2, min(cv_folds, min_class_count))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    folds = max(2, min(cv_folds, len(y) // 2 or 2))
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _fit_one(
    spec: ModelSpec,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
    scoring: str,
    n_iter: int,
    random_state: int,
) -> ModelResult:
    start = time.perf_counter()
    pipeline = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", spec.build(random_state))])
    try:
        if spec.param_distributions:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=spec.param_distributions,
                n_iter=min(n_iter, _param_space_size(spec.param_distributions)),
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            cv_score = float(search.best_score_)
            best_params = {k: v for k, v in search.best_params_.items()}
        else:
            pipeline.fit(X_train, y_train)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            best_estimator = pipeline
            cv_score = float(np.mean(scores))
            best_params = {}
        fit_time = time.perf_counter() - start
        return ModelResult(spec.model_id, spec.display_name, best_estimator, cv_score, fit_time, best_params)
    except Exception as exc:  # noqa: BLE001 - surfaced via ModelResult.error, not raised
        fit_time = time.perf_counter() - start
        return ModelResult(spec.model_id, spec.display_name, None, float("-inf"), fit_time, {}, error=str(exc))


def train_and_tune_all(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    preprocessor: ColumnTransformer,
    *,
    cv_folds: int = 5,
    n_iter: int = 8,
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
            executor.submit(_fit_one, spec, preprocessor, X_train, y_train, cv, scoring, n_iter, random_state): spec
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
