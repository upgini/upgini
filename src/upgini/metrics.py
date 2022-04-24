import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target

CATBOOST_PARAMS = {
    "iterations": 200,
    "early_stopping_rounds": 30,
    "one_hot_max_size": 50,
    "verbose": False,
    "random_state": 42,
}


class EstimatorWrapper:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, **kwargs):
        X, fit_params = self._prepare_to_fit(X)
        kwargs.update(fit_params)
        self.estimator.fit(X, **kwargs)
        return self

    def predict(self, **kwargs):
        return self.estimator.predict(**kwargs)

    def _prepare_to_fit(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        raise NotImplementedError()

    def cross_val_predict(self, X, y, cv, scorer):
        X, fit_params = self._prepare_to_fit(X)

        metrics_by_fold = cross_val_score(self.estimator, X, y, cv=cv, scoring=scorer, fit_params=fit_params)

        return np.mean(metrics_by_fold)

    def calculate_metric(self, X: pd.DataFrame, y, scoring: Union[str, Callable, None]) -> float:
        X, _ = self._prepare_to_fit(X)
        scorer, _ = _get_scorer(y, scoring)
        return scorer(self.estimator, X, y)


class CatBoostWrapper(EstimatorWrapper):
    def __init__(self, estimator: Union[CatBoostClassifier, CatBoostRegressor]):
        super(CatBoostWrapper, self).__init__(estimator)

    def _prepare_to_fit(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        cat_features_idx, cat_features = get_cat_features(X)
        X[cat_features] = X[cat_features].fillna("")
        return X, {"cat_features": cat_features_idx}


class LightGBMWrapper(EstimatorWrapper):
    def __init__(self, estimator: Union[LGBMClassifier, LGBMRegressor]):
        super(LightGBMWrapper, self).__init__(estimator)

    def _prepare_to_fit(self, X) -> Tuple[pd.DataFrame, dict]:
        _, cat_features = get_cat_features(X)
        X[cat_features] = X[cat_features].fillna("")
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes

        return X, {}


class OtherEstimatorWrapper(EstimatorWrapper):
    def __init__(self, estimator):
        super(OtherEstimatorWrapper, self).__init__(estimator)

    def _prepare_to_fit(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        cat_features_idx, cat_features = get_cat_features(X)
        # TODO use one-hot encoding if cardinality is less 50
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes
        return X, {}


def calculate_cv_metric(
    X: pd.DataFrame, y, estimator: Optional[Any], cv=5, scoring: Union[str, Callable, None] = None
) -> Tuple[float, str]:
    scorer, metric_name = _get_scorer(y, scoring)
    estimator = get_estimator(estimator, y)
    metric = estimator.cross_val_predict(X, y, cv, scorer)
    return (metric, metric_name)


def fit_model(X: pd.DataFrame, y, estimator: Optional[Any]) -> EstimatorWrapper:
    estimator = get_estimator(estimator, y)

    return estimator.fit(X=X, y=y)


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


def get_estimator(estimator, y) -> EstimatorWrapper:
    target_type = type_of_target(y)
    if estimator is None:
        if target_type in ["multiclass", "binary"]:
            estimator = CatBoostWrapper(CatBoostClassifier(**CATBOOST_PARAMS))
        elif target_type == "continuous":
            estimator = CatBoostWrapper(CatBoostRegressor(**CATBOOST_PARAMS))
        else:
            raise Exception(f"Unsupported type of target: {target_type}")
    else:
        if isinstance(estimator, CatBoostClassifier) or isinstance(estimator, CatBoostRegressor):
            estimator = CatBoostWrapper(estimator)
        elif isinstance(estimator, LGBMClassifier) or isinstance(estimator, LGBMRegressor):
            estimator = LightGBMWrapper(estimator)
        else:
            logging.warning(
                f"Unexpected estimator is used for metrics: {estimator}. "
                "Default strategy for category features will be used"
            )
            estimator = OtherEstimatorWrapper(estimator)
    return estimator


def get_cat_features(X: pd.DataFrame) -> Tuple[List[int], List[str]]:
    zipped = [(i, c) for i, c in enumerate(X.columns) if not is_numeric_dtype(X[c])]
    if len(zipped) == 0:
        return ([], [])
    unzipped = list(zip(*zipped))
    return list(unzipped[0]), list(unzipped[1])
