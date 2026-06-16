from dataclasses import dataclass, field
from typing import Any, Callable

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    factory: Callable[..., Any]
    param_distributions: dict[str, list[Any]] = field(default_factory=dict)
    fixed_params: dict[str, Any] = field(default_factory=dict)

    def build(self, random_state: int) -> Any:
        return self.factory(**self.fixed_params, random_state=random_state)


def _classification_registry(random_state: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            "logistic_regression",
            "Logistic Regression",
            LogisticRegression,
            param_distributions={"model__C": [0.01, 0.1, 1.0, 10.0], "model__max_iter": [500, 1000]},
            fixed_params={"class_weight": "balanced", "max_iter": 1000},
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            RandomForestClassifier,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 6, 12, 20],
                "model__min_samples_leaf": [1, 2, 4],
            },
            fixed_params={"class_weight": "balanced"},
        ),
        ModelSpec(
            "extra_trees",
            "Extra Trees",
            ExtraTreesClassifier,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 6, 12, 20],
            },
            fixed_params={"class_weight": "balanced"},
        ),
        ModelSpec(
            "xgboost",
            "XGBoost",
            XGBClassifier,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [3, 6, 9],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            fixed_params={"eval_metric": "logloss", "verbosity": 0},
        ),
        ModelSpec(
            "lightgbm",
            "LightGBM",
            LGBMClassifier,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [-1, 6, 12],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            fixed_params={"class_weight": "balanced", "verbosity": -1},
        ),
    ]


def _regression_registry(random_state: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            "ridge",
            "Ridge Regression",
            Ridge,
            param_distributions={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            RandomForestRegressor,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 6, 12, 20],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        ModelSpec(
            "extra_trees",
            "Extra Trees",
            ExtraTreesRegressor,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 6, 12, 20],
            },
        ),
        ModelSpec(
            "xgboost",
            "XGBoost",
            XGBRegressor,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [3, 6, 9],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            fixed_params={"verbosity": 0},
        ),
        ModelSpec(
            "lightgbm",
            "LightGBM",
            LGBMRegressor,
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [-1, 6, 12],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            fixed_params={"verbosity": -1},
        ),
    ]


def get_registry(problem_type: str, random_state: int = 42) -> list[ModelSpec]:
    if problem_type == "classification":
        return _classification_registry(random_state)
    if problem_type == "regression":
        return _regression_registry(random_state)
    raise ValueError(f"No supervised model registry for problem_type={problem_type!r}")
