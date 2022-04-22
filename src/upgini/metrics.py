from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import get_scorer

# from sklearn.metrics._scorer import check_scoring
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target

CATBOOST_PARAMS = {
    "iterations": 700,
    "early_stopping_rounds": 100,
    "one_hot_max_size": 100,
    "verbose": False,
    "random_state": 42,
}


def calculate_cv_metric(X: pd.DataFrame, y, cv=5, scoring: Union[str, Callable, None] = None) -> Tuple[float, str]:
    cat_features_idx, cat_features = get_cat_features(X)
    X[cat_features] = X[cat_features].fillna("")

    estimator = get_estimator(y)

    scorer, metric_name = _get_scorer(y, scoring)

    metrics_by_fold = cross_val_score(
        estimator, X, y, cv=cv, scoring=scorer, fit_params={"cat_features": cat_features_idx}
    )
    metric = np.mean(metrics_by_fold)

    return (
        metric,
        metric_name,
    )


def fit_model(X: pd.DataFrame, y) -> Union[CatBoostClassifier, CatBoostRegressor]:
    cat_features_idx, cat_features = get_cat_features(X)
    X[cat_features] = X[cat_features].fillna("")

    estimator = get_estimator(y)

    return estimator.fit(X, y, cat_features=cat_features_idx)


def calculate_metric(
    model: Union[CatBoostClassifier, CatBoostRegressor], X: pd.DataFrame, y, scoring: Union[str, Callable, None]
) -> float:
    scorer, _ = _get_scorer(y, scoring)
    return scorer(model, X, y)


def _get_scorer(y, scoring: Union[str, Callable, None]) -> Tuple[Callable, str]:
    if scoring is None:
        target_type = type_of_target(y)
        if target_type == "binary":
            metric_name = scoring = "roc_auc"
        elif target_type == "multiclass":
            metric_name = scoring = "accuracy"
        elif target_type == "continuous":
            metric_name = scoring = "neg_root_mean_squared_error"
        else:
            raise Exception(f"Unsupported type of target: {target_type}")
    elif isinstance(scoring, str):
        metric_name = scoring
    else:
        metric_name = "metric"

    return get_scorer(scoring), metric_name


def get_estimator(y):
    target_type = type_of_target(y)
    if target_type in ["multiclass", "binary"]:
        estimator = CatBoostClassifier(**CATBOOST_PARAMS)
    elif target_type == "continuous":
        estimator = CatBoostRegressor(**CATBOOST_PARAMS)
    else:
        raise Exception(f"Unsupported type of target: {target_type}")
    return estimator


def get_cat_features(X: pd.DataFrame) -> Tuple[List[int], List[str]]:
    zipped = [(i, c) for i, c in enumerate(X.columns) if not is_numeric_dtype(X[c])]
    if len(zipped) == 0:
        return ([], [])
    unzipped = list(zip(*zipped))
    return list(unzipped[0]), list(unzipped[1])
