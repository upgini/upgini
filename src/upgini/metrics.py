import logging
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import SCORERS, get_scorer
from sklearn.model_selection import cross_val_score

from upgini.metadata import ModelTaskType

CATBOOST_PARAMS = {
    "iterations": 300,
    "early_stopping_rounds": 100,
    "one_hot_max_size": 100,
    "verbose": False,
    "random_state": 42,
}


class EstimatorWrapper:
    def __init__(self, estimator, scorer: Callable, metric_name: str, multiplier: int):
        self.estimator = estimator
        self.scorer = scorer
        self.metric_name = metric_name
        self.multiplier = multiplier

    def fit(self, X, y, **kwargs):
        X, fit_params = self._prepare_to_fit(X)
        kwargs.update(fit_params)
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, **kwargs):
        return self.estimator.predict(**kwargs)

    def _prepare_to_fit(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        raise NotImplementedError()

    def cross_val_predict(self, X, y, cv=5):
        X, fit_params = self._prepare_to_fit(X)

        metrics_by_fold = cross_val_score(self.estimator, X, y, cv=cv, scoring=self.scorer, fit_params=fit_params)

        return np.mean(metrics_by_fold) * self.multiplier

    def calculate_metric(self, X: pd.DataFrame, y) -> float:
        X, _ = self._prepare_to_fit(X)
        return self.scorer(self.estimator, X, y) * self.multiplier

    @staticmethod
    def create(estimator, target_type: ModelTaskType, scoring: Union[Callable, str, None] = None) -> "EstimatorWrapper":
        scorer, metric_name, multiplier = _get_scorer(target_type, scoring)
        if estimator is None:
            kwargs = {"scorer": scorer, "metric_name": metric_name, "multiplier": multiplier}
            if target_type in [ModelTaskType.MULTICLASS, ModelTaskType.BINARY]:
                estimator = CatBoostWrapper(CatBoostClassifier(**CATBOOST_PARAMS), **kwargs)
            elif target_type == ModelTaskType.REGRESSION:
                estimator = CatBoostWrapper(CatBoostRegressor(**CATBOOST_PARAMS), **kwargs)
            else:
                raise Exception(f"Unsupported type of target: {target_type}")
        else:
            kwargs = {"estimator": estimator, "scorer": scorer, "metric_name": metric_name, "multiplier": multiplier}
            if isinstance(estimator, CatBoostClassifier) or isinstance(estimator, CatBoostRegressor):
                estimator = CatBoostWrapper(**kwargs)
            elif isinstance(estimator, LGBMClassifier) or isinstance(estimator, LGBMRegressor):
                estimator = LightGBMWrapper(**kwargs)
            else:
                logging.warning(
                    f"Unexpected estimator is used for metrics: {estimator}. "
                    "Default strategy for category features will be used"
                )
                estimator = OtherEstimatorWrapper(**kwargs)
        return estimator


class CatBoostWrapper(EstimatorWrapper):
    def __init__(
        self,
        estimator: Union[CatBoostClassifier, CatBoostRegressor],
        scorer: Callable,
        metric_name: str,
        multiplier: int,
    ):
        super(CatBoostWrapper, self).__init__(estimator, scorer, metric_name, multiplier)

    def _prepare_to_fit(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        cat_features_idx, cat_features = _get_cat_features(X)
        X[cat_features] = X[cat_features].fillna("")
        return X, {"cat_features": cat_features_idx}


class LightGBMWrapper(EstimatorWrapper):
    def __init__(
        self, estimator: Union[LGBMClassifier, LGBMRegressor], scorer: Callable, metric_name: str, multiplier: int
    ):
        super(LightGBMWrapper, self).__init__(estimator, scorer, metric_name, multiplier)

    def _prepare_to_fit(self, X) -> Tuple[pd.DataFrame, dict]:
        _, cat_features = _get_cat_features(X)
        X[cat_features] = X[cat_features].fillna("")
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes

        return X, {}


class OtherEstimatorWrapper(EstimatorWrapper):
    def __init__(self, estimator, scorer: Callable, metric_name: str, multiplier: int):
        super(OtherEstimatorWrapper, self).__init__(estimator, scorer, metric_name, multiplier)

    def _prepare_to_fit(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        _, cat_features = _get_cat_features(X)
        num_features = [col for col in X.columns if col not in cat_features]
        X[num_features] = X[num_features].fillna(-999)
        X[cat_features] = X[cat_features].fillna("")
        # TODO use one-hot encoding if cardinality is less 50
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes
        return X, {}


def _get_scorer(target_type: ModelTaskType, scoring: Union[Callable, str, None]) -> Tuple[Callable, str, int]:
    if scoring is None:
        if target_type == ModelTaskType.BINARY:
            scoring = "roc_auc"
        elif target_type == ModelTaskType.MULTICLASS:
            scoring = "accuracy"
        elif target_type == ModelTaskType.REGRESSION:
            scoring = "mean_squared_error"
        else:
            raise Exception(f"Unsupported type of target: {target_type}")

    multiplier = 1
    if isinstance(scoring, str):
        metric_name = scoring
        if scoring in SCORERS.keys():
            scoring = get_scorer(scoring)
        elif ("neg_" + scoring) in SCORERS.keys():
            scoring = get_scorer("neg_" + scoring)
            multiplier = -1
        else:
            raise ValueError(
                f"{scoring} is not a valid scoring value. " f"Use {sorted(SCORERS.keys())} " "to get valid options."
            )
    elif hasattr(scoring, "__name__"):
        metric_name = scoring.__name__
    else:
        metric_name = str(scoring)

    return scoring, metric_name, multiplier


def _get_cat_features(X: pd.DataFrame) -> Tuple[List[int], List[str]]:
    zipped = [(i, c) for i, c in enumerate(X.columns) if not is_numeric_dtype(X[c])]
    if len(zipped) == 0:
        return ([], [])
    unzipped = list(zip(*zipped))
    return list(unzipped[0]), list(unzipped[1])
