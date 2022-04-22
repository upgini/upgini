from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import get_scorer

# from sklearn.metrics._scorer import check_scoring
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import type_of_target

CATBOOST_PARAMS = {
    "iterations": 700,
    "early_stopping_rounds": 100,
    "one_hot_max_size": 100,
    "verbose": False,
    "random_state": 42,
}


def calculate_cv_score(X: pd.DataFrame, y, cv=5) -> np.ndarray:
    cat_features_idx, cat_features = get_cat_features(X)
    X[cat_features] = X[cat_features].fillna("")

    estimator, method = get_estimator_and_method(y)

    predict = cross_val_predict(estimator, X, y, cv=cv, method=method, fit_params={"cat_features": cat_features_idx})

    if method == "predict_proba":
        return predict[:, 1]
    else:
        return predict


def fit_model(X: pd.DataFrame, y) -> Tuple[Union[CatBoostClassifier, CatBoostRegressor], str]:
    cat_features_idx, cat_features = get_cat_features(X)
    X[cat_features] = X[cat_features].fillna("")

    estimator, method = get_estimator_and_method(y)

    return estimator.fit(X, y, cat_features=cat_features_idx), method


def calculate_score(model: Union[CatBoostClassifier, CatBoostRegressor], X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)  # , prediction_type="") TODO


def calculate_metric(y, scores: np.ndarray, scoring: Union[str, Callable, None]) -> Tuple[float, str]:
    # estimator = get_estimator(y)
    # scoring = check_scoring(estimator, scoring)  # TODO think about None
    if scoring is None:
        target_type = type_of_target(y)
        if target_type == "binary":
            scoring = "roc_auc"
        elif target_type == "multiclass":
            scoring = "accuracy"
        elif target_type == "continuous":
            scoring = "neg_root_mean_squared_error"
        else:
            raise Exception(f"Unsupported type of target: {target_type}")

    metric_function = get_scorer(scoring)._score_func  # type: ignore
    return metric_function(y, scores), scoring


def score_calculate_metric(
    model: Union[CatBoostClassifier, CatBoostRegressor], X: pd.DataFrame, y, scoring: Union[str, Callable, None]
) -> float:
    scores = calculate_score(model, X)
    return calculate_metric(y, scores, scoring)[0]


def get_estimator_and_method(y):
    target_type = type_of_target(y)
    if target_type in ["multiclass", "binary"]:
        estimator = CatBoostClassifier(**CATBOOST_PARAMS)
        method = "predict_proba"
    elif target_type == "continuous":
        estimator = CatBoostRegressor(**CATBOOST_PARAMS)
        method = "predict"
    else:
        raise Exception(f"Unsupported type of target: {target_type}")
    return estimator, method


def get_cat_features(X: pd.DataFrame) -> Tuple[List[int], List[str]]:
    zipped = [(i, c) for i, c in enumerate(X.columns) if not is_numeric_dtype(X[c])]
    if len(zipped) == 0:
        return ([], [])
    unzipped = list(zip(*zipped))
    return list(unzipped[0]), list(unzipped[1])
