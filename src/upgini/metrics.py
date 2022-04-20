from typing import Callable, List, Tuple, Union

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
    "verbose": 50,
    "random_state": 42
}


def calculate_cv_score(X: pd.DataFrame, y, cv=5) -> np.ndarray:
    cat_features_idx, cat_features = get_cat_features(X)
    X[cat_features] = X[cat_features].fillna("")

    estimator = get_estimator(y)

    return cross_val_predict(estimator, X, y, cv=cv, fit_params={"cat_features": cat_features_idx})


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
    unzipped = list(zip(*zipped))
    return list(unzipped[0]), list(unzipped[1])
