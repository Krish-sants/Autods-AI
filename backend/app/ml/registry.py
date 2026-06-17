from dataclasses import dataclass, field
from typing import Any, Callable

import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    factory: Callable[..., Any]
    optuna_space: Callable[[optuna.Trial], dict[str, Any]] | None = None
    fixed_params: dict[str, Any] = field(default_factory=dict)
    no_random_state: bool = False
    # Legacy param_distributions kept so old tests that inspect this field don't break
    param_distributions: dict[str, list[Any]] = field(default_factory=dict)

    def build(self, random_state: int) -> Any:
        kwargs = dict(self.fixed_params)
        if not self.no_random_state:
            kwargs["random_state"] = random_state
        return self.factory(**kwargs)


def _classification_registry(random_state: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            "logistic_regression",
            "Logistic Regression",
            LogisticRegression,
            optuna_space=lambda t: {
                "C": t.suggest_float("C", 1e-3, 1e2, log=True),
                "max_iter": t.suggest_categorical("max_iter", [500, 1000, 2000]),
            },
            fixed_params={"class_weight": "balanced"},
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            RandomForestClassifier,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_categorical("max_depth", [None, 4, 8, 16, 24]),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 8),
            },
            fixed_params={"class_weight": "balanced"},
        ),
        ModelSpec(
            "extra_trees",
            "Extra Trees",
            ExtraTreesClassifier,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_categorical("max_depth", [None, 4, 8, 16, 24]),
            },
            fixed_params={"class_weight": "balanced"},
        ),
        ModelSpec(
            "gradient_boosting",
            "Gradient Boosting",
            GradientBoostingClassifier,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
                "max_depth": t.suggest_int("max_depth", 2, 8),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
            },
        ),
        ModelSpec(
            "xgboost",
            "XGBoost",
            XGBClassifier,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_int("max_depth", 2, 10),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
            },
            fixed_params={"eval_metric": "logloss", "verbosity": 0},
        ),
        ModelSpec(
            "lightgbm",
            "LightGBM",
            LGBMClassifier,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_categorical("max_depth", [-1, 4, 8, 16]),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "num_leaves": t.suggest_int("num_leaves", 16, 128),
            },
            fixed_params={"class_weight": "balanced", "verbosity": -1},
        ),
        ModelSpec(
            "svm",
            "SVM",
            # CalibratedClassifierCV wraps SVC to provide predict_proba without the
            # deprecated probability=True flag (removed in sklearn 1.11)
            lambda **kw: CalibratedClassifierCV(
                SVC(
                    C=kw.pop("C", 1.0),
                    kernel=kw.pop("kernel", "rbf"),
                    gamma=kw.pop("gamma", "scale"),
                    class_weight="balanced",
                ),
                ensemble=False,
            ),
            optuna_space=lambda t: {
                "C": t.suggest_float("C", 1e-2, 1e2, log=True),
                "kernel": t.suggest_categorical("kernel", ["rbf", "linear"]),
                "gamma": t.suggest_categorical("gamma", ["scale", "auto"]),
            },
            no_random_state=True,
        ),
        ModelSpec(
            "knn",
            "K-Nearest Neighbors",
            KNeighborsClassifier,
            optuna_space=lambda t: {
                "n_neighbors": t.suggest_int("n_neighbors", 3, 20),
                "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": t.suggest_categorical("metric", ["euclidean", "manhattan"]),
            },
            no_random_state=True,
        ),
    ]


def _regression_registry(random_state: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            "ridge",
            "Ridge Regression",
            Ridge,
            optuna_space=lambda t: {
                "alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True),
            },
            no_random_state=True,
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            RandomForestRegressor,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_categorical("max_depth", [None, 4, 8, 16, 24]),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 8),
            },
        ),
        ModelSpec(
            "extra_trees",
            "Extra Trees",
            ExtraTreesRegressor,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_categorical("max_depth", [None, 4, 8, 16, 24]),
            },
        ),
        ModelSpec(
            "gradient_boosting",
            "Gradient Boosting",
            GradientBoostingRegressor,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
                "max_depth": t.suggest_int("max_depth", 2, 8),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
            },
        ),
        ModelSpec(
            "xgboost",
            "XGBoost",
            XGBRegressor,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_int("max_depth", 2, 10),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
            },
            fixed_params={"verbosity": 0},
        ),
        ModelSpec(
            "lightgbm",
            "LightGBM",
            LGBMRegressor,
            optuna_space=lambda t: {
                "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": t.suggest_categorical("max_depth", [-1, 4, 8, 16]),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "num_leaves": t.suggest_int("num_leaves", 16, 128),
            },
            fixed_params={"verbosity": -1},
        ),
        ModelSpec(
            "svr",
            "SVR",
            SVR,
            optuna_space=lambda t: {
                "C": t.suggest_float("C", 1e-2, 1e2, log=True),
                "kernel": t.suggest_categorical("kernel", ["rbf", "linear"]),
                "epsilon": t.suggest_float("epsilon", 1e-3, 1.0, log=True),
            },
            no_random_state=True,
        ),
        ModelSpec(
            "knn",
            "K-Nearest Neighbors",
            KNeighborsRegressor,
            optuna_space=lambda t: {
                "n_neighbors": t.suggest_int("n_neighbors", 3, 20),
                "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": t.suggest_categorical("metric", ["euclidean", "manhattan"]),
            },
            no_random_state=True,
        ),
    ]


def get_registry(problem_type: str, random_state: int = 42) -> list[ModelSpec]:
    if problem_type == "classification":
        return _classification_registry(random_state)
    if problem_type == "regression":
        return _regression_registry(random_state)
    raise ValueError(f"No supervised model registry for problem_type={problem_type!r}")
