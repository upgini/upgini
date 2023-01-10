import logging
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from numpy import log1p
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import SCORERS, check_scoring, get_scorer, make_scorer
from sklearn.metrics._regression import (
    _check_reg_targets,
    check_consistent_length,
    mean_squared_error,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    cross_validate,
)

from upgini.errors import ValidationError
from upgini.metadata import ModelTaskType
from upgini.resource_bundle import bundle
from upgini.utils.target_utils import correct_string_target

CATBOOST_PARAMS = {
    "iterations": 250,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "max_depth": 5,
    "early_stopping_rounds": 20,
    "one_hot_max_size": 100,
    "verbose": False,
    "random_state": 42,
    "allow_writing_files": False,
}

N_FOLDS = 5
BLOCKED_TS_TEST_SIZE = 0.2


class EstimatorWrapper:
    def __init__(
        self,
        estimator,
        scorer: Callable,
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
    ):
        self.estimator = estimator
        self.scorer = scorer
        self.metric_name = metric_name
        self.multiplier = multiplier
        self.cv = cv
        self.target_type = target_type
        self.cv_estimators = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs):
        X, y, fit_params = self._prepare_to_fit(X, y)
        kwargs.update(fit_params)
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, **kwargs):
        return self.estimator.predict(**kwargs)

    def _prepare_to_fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, dict]:
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                X[c] = X[c].astype(float)
            else:
                X[c] = X[c].astype(str)

        if not isinstance(y, pd.Series):
            raise Exception(bundle.get("metrics_unsupported_target_type").format(type(y)))

        joined = pd.concat([X, y], axis=1)
        joined = joined[joined[y.name].notna()]
        joined = joined.reset_index(drop=True)
        X = joined.drop(columns=y.name)
        y = joined[y.name].values
        return X, y, {}

    def cross_val_predict(self, X: pd.DataFrame, y: np.ndarray):
        X, y, fit_params = self._prepare_to_fit(X, y)

        if X.shape[1] == 0:
            return None

        scorer = check_scoring(self.estimator, scoring=self.scorer)

        cv_results = cross_validate(
            estimator=self.estimator,
            X=X,
            y=y,
            scoring={"score": scorer},
            cv=self.cv,
            fit_params=fit_params,
            return_estimator=True,
        )
        metrics_by_fold = cv_results["test_score"]
        self.cv_estimators = cv_results["estimator"]

        return np.mean(metrics_by_fold) * self.multiplier

    def calculate_metric(self, X: pd.DataFrame, y: np.ndarray) -> float:
        X, y, _ = self._prepare_to_fit(X, y)
        metrics = []
        for est in self.cv_estimators:
            metrics.append(self.scorer(est, X, y))

        return np.mean(metrics) * self.multiplier

    @staticmethod
    def create(
        estimator,
        logger: logging.Logger,
        target_type: ModelTaskType,
        cv: BaseCrossValidator,
        scoring: Union[Callable, str, None] = None,
    ) -> "EstimatorWrapper":
        scorer, metric_name, multiplier = _get_scorer(target_type, scoring)
        kwargs = {
            "scorer": scorer,
            "metric_name": metric_name,
            "multiplier": multiplier,
            "cv": cv,
            "target_type": target_type,
        }
        if estimator is None:
            if target_type in [ModelTaskType.MULTICLASS, ModelTaskType.BINARY]:
                estimator = CatBoostWrapper(CatBoostClassifier(**CATBOOST_PARAMS), **kwargs)
            elif target_type == ModelTaskType.REGRESSION:
                estimator = CatBoostWrapper(CatBoostRegressor(**CATBOOST_PARAMS), **kwargs)
            else:
                raise Exception(bundle.get("metrics_unsupported_target_type").format(target_type))
        else:
            kwargs["estimator"] = estimator
            if isinstance(estimator, CatBoostClassifier) or isinstance(estimator, CatBoostRegressor):
                estimator = CatBoostWrapper(**kwargs)
            else:
                try:
                    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

                    if isinstance(estimator, LGBMClassifier) or isinstance(estimator, LGBMRegressor):
                        estimator = LightGBMWrapper(**kwargs)
                    else:
                        logger.warning(
                            f"Unexpected estimator is used for metrics: {estimator}. "
                            "Default strategy for category features will be used"
                        )
                        estimator = OtherEstimatorWrapper(**kwargs)
                except ModuleNotFoundError:
                    logger.warning(
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
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
    ):
        super(CatBoostWrapper, self).__init__(estimator, scorer, metric_name, multiplier, cv, target_type)

    def _prepare_to_fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, dict]:
        X, y, params = super()._prepare_to_fit(X, y)
        cat_features = _get_cat_features(X)
        X[cat_features] = X[cat_features].astype("string").fillna("").astype(str)
        unique_cat_features = []
        # TODO try to remove this condition because now we remove constant features earlier
        for name in cat_features:
            # Remove constant categorical features
            if X[name].nunique() > 1:
                unique_cat_features.append(name)
            else:
                X = X.drop(columns=name)
        cat_features_idx = [X.columns.get_loc(c) for c in unique_cat_features]

        params.update({"cat_features": cat_features_idx})
        return X, y, params


class LightGBMWrapper(EstimatorWrapper):
    def __init__(
        self,
        estimator,
        scorer: Callable,
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
    ):
        super(LightGBMWrapper, self).__init__(estimator, scorer, metric_name, multiplier, cv, target_type)

    def _prepare_to_fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, dict]:
        X, y, params = super()._prepare_to_fit(X, y)
        cat_features = _get_cat_features(X)
        X[cat_features] = X[cat_features].astype("string").fillna("").astype(str)
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes
        if not is_numeric_dtype(y):
            y = correct_string_target(y)

        return X, y, params


class OtherEstimatorWrapper(EstimatorWrapper):
    def __init__(
        self,
        estimator,
        scorer: Callable,
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
    ):
        super(OtherEstimatorWrapper, self).__init__(estimator, scorer, metric_name, multiplier, cv, target_type)

    def _prepare_to_fit(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, dict]:
        X, y, params = super()._prepare_to_fit(X, y)
        cat_features = _get_cat_features(X)
        num_features = [col for col in X.columns if col not in cat_features]
        X[num_features] = X[num_features].fillna(-999)
        X[cat_features] = X[cat_features].astype("string").fillna("").astype(str)
        # TODO use one-hot encoding if cardinality is less 50
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes
        if not is_numeric_dtype(y):
            y = correct_string_target(y)
        return X, y, params


def _get_scorer(target_type: ModelTaskType, scoring: Union[Callable, str, None]) -> Tuple[Callable, str, int]:
    if scoring is None:
        if target_type == ModelTaskType.BINARY:
            scoring = "roc_auc"
        elif target_type == ModelTaskType.MULTICLASS:
            scoring = "accuracy"
        elif target_type == ModelTaskType.REGRESSION:
            scoring = "mean_squared_error"
        else:
            raise Exception(bundle.get("metrics_unsupported_target_type").format(target_type))

    multiplier = 1
    if isinstance(scoring, str):
        metric_name = scoring
        if "mean_squared_log_error" == metric_name or "MSLE" == metric_name or "msle" == metric_name:
            scoring = make_scorer(_ext_mean_squared_log_error, greater_is_better=False)
            multiplier = -1
        elif "root_mean_squared_log_error" in metric_name or "RMSLE" == metric_name or "rmsle" == metric_name:
            scoring = make_scorer(_ext_root_mean_squared_log_error, greater_is_better=False)
            multiplier = -1
        elif "root_mean_squared_error" == metric_name or "RMSE" == metric_name or "rmse" == metric_name:
            scoring = get_scorer("neg_root_mean_squared_error")
            multiplier = -1
        elif scoring in SCORERS.keys():
            scoring = get_scorer(scoring)
        elif ("neg_" + scoring) in SCORERS.keys():
            scoring = get_scorer("neg_" + scoring)
            multiplier = -1
        else:
            raise ValidationError(bundle.get("metrics_invalid_scoring").format(scoring, sorted(SCORERS.keys())))
    elif hasattr(scoring, "__name__"):
        metric_name = scoring.__name__
    else:
        metric_name = str(scoring)

    return scoring, metric_name, multiplier


def _get_cat_features(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if not is_numeric_dtype(X[c])]


def _ext_root_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    return _ext_mean_squared_log_error(
        y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=False
    )


def _ext_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
    """Mean squared logarithmic error regression loss.

    Extended version with clip(0) for y_pred

    Read more in the :ref:`User Guide <mean_squared_log_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'

        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors when the input is of multioutput
            format.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    squared : bool, default=True
        If True returns MSLE (mean squared log error) value.
        If False returns RMSLE (root mean squared log error) value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_log_error
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> mean_squared_log_error(y_true, y_pred)
    0.039...
    >>> mean_squared_log_error(y_true, y_pred, squared=False)
    0.199...
    >>> y_true = [[0.5, 1], [1, 2], [7, 6]]
    >>> y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
    >>> mean_squared_log_error(y_true, y_pred)
    0.044...
    >>> mean_squared_log_error(y_true, y_pred, multioutput='raw_values')
    array([0.00462428, 0.08377444])
    >>> mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.060...
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if (y_true < 0).any():
        raise ValidationError(bundle.get("metrics_msle_negative_target"))

    return mean_squared_error(
        log1p(y_true),
        log1p(y_pred.clip(0)),
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )
