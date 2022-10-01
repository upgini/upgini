import logging
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from numpy import log1p
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import SCORERS, get_scorer, make_scorer
from sklearn.metrics._regression import (
    _check_reg_targets,
    check_consistent_length,
    mean_squared_error,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)

from upgini.metadata import CVType, ModelTaskType
from upgini.utils.blocked_time_series import BlockedTimeSeriesSplit

CATBOOST_PARAMS = {
    "iterations": 300,
    "early_stopping_rounds": 100,
    "one_hot_max_size": 100,
    "verbose": False,
    "random_state": 42,
    "allow_writing_files": False,
}

N_FOLDS = 5
BLOCKED_TS_TEST_SIZE = 0.2


class EstimatorWrapper:
    def __init__(self, estimator, scorer: Callable, metric_name: str, multiplier: int, cv: BaseCrossValidator):
        self.estimator = estimator
        self.scorer = scorer
        self.metric_name = metric_name
        self.multiplier = multiplier
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        X, y, fit_params = self._prepare_to_fit(X.copy(), y.copy())
        kwargs.update(fit_params)
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, **kwargs):
        return self.estimator.predict(**kwargs)

    def _prepare_to_fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, dict]:
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                X[c] = X[c].astype(float)
        joined = X
        if isinstance(y, pd.Series):
            pass
        elif isinstance(y, np.ndarray) or isinstance(y, list):
            y = pd.Series(y, name="target")
        else:
            msg = f"Unsupported type of y: {type(y)}"
            raise Exception(msg)

        joined = pd.concat([joined, y.to_frame(name=y.name)], axis=1)
        # joined[y.name] = y.to_frame(name=y.name)
        joined = joined[joined[y.name].notna()]
        X = joined.drop(columns=y.name)
        y = joined[y.name]  # type: ignore
        return X, y, {}

    def cross_val_predict(self, X: pd.DataFrame, y: pd.Series):
        X, y, fit_params = self._prepare_to_fit(X.copy(), y.copy())

        if X.shape[1] == 0:
            return None

        metrics_by_fold = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scorer, fit_params=fit_params)

        return np.mean(metrics_by_fold) * self.multiplier

    def calculate_metric(self, X: pd.DataFrame, y) -> float:
        X, y, _ = self._prepare_to_fit(X, y)
        return self.scorer(self.estimator, X, y) * self.multiplier

    @staticmethod
    def _create_cv(
        cv: Union[BaseCrossValidator, CVType, None], target_type: ModelTaskType, shuffle: bool, random_state: int
    ) -> BaseCrossValidator:
        if isinstance(cv, BaseCrossValidator):
            return cv

        if cv == CVType.time_series:
            return TimeSeriesSplit(n_splits=N_FOLDS)
        elif cv == CVType.blocked_time_series:
            return BlockedTimeSeriesSplit(n_splits=N_FOLDS, test_size=BLOCKED_TS_TEST_SIZE)
        elif target_type == ModelTaskType.REGRESSION:
            if shuffle:
                return KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
            else:
                return KFold(n_splits=N_FOLDS, shuffle=False)
        else:
            if shuffle:
                return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
            else:
                return StratifiedKFold(n_splits=N_FOLDS, shuffle=False)

    @staticmethod
    def create(
        estimator,
        logger: logging.Logger,
        target_type: ModelTaskType,
        cv: Union[BaseCrossValidator, CVType, None],
        scoring: Union[Callable, str, None] = None,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> "EstimatorWrapper":
        scorer, metric_name, multiplier = _get_scorer(target_type, scoring)
        cv = EstimatorWrapper._create_cv(cv, target_type, shuffle, random_state)
        kwargs = {"scorer": scorer, "metric_name": metric_name, "multiplier": multiplier, "cv": cv}
        if estimator is None:
            if target_type in [ModelTaskType.MULTICLASS, ModelTaskType.BINARY]:
                estimator = CatBoostWrapper(CatBoostClassifier(**CATBOOST_PARAMS), **kwargs)
            elif target_type == ModelTaskType.REGRESSION:
                estimator = CatBoostWrapper(CatBoostRegressor(**CATBOOST_PARAMS), **kwargs)
            else:
                raise Exception(f"Unsupported type of target: {target_type}")
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
    ):
        super(CatBoostWrapper, self).__init__(estimator, scorer, metric_name, multiplier, cv)

    def _prepare_to_fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, dict]:
        X, y, params = super()._prepare_to_fit(X, y)
        cat_features = _get_cat_features(X)
        X[cat_features] = X[cat_features].astype(str).fillna("")
        unique_cat_features = []
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
    ):
        super(LightGBMWrapper, self).__init__(estimator, scorer, metric_name, multiplier, cv)

    def _prepare_to_fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, dict]:
        X, y, params = super()._prepare_to_fit(X, y)
        cat_features = _get_cat_features(X)
        X[cat_features] = X[cat_features].astype(str).fillna("")
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes

        return X, y, params


class OtherEstimatorWrapper(EstimatorWrapper):
    def __init__(
        self,
        estimator,
        scorer: Callable,
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
    ):
        super(OtherEstimatorWrapper, self).__init__(estimator, scorer, metric_name, multiplier, cv)

    def _prepare_to_fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, dict]:
        X, y, params = super()._prepare_to_fit(X, y)
        cat_features = _get_cat_features(X)
        num_features = [col for col in X.columns if col not in cat_features]
        X[num_features] = X[num_features].fillna(-999)
        X[cat_features] = X[cat_features].astype(str).fillna("")
        # TODO use one-hot encoding if cardinality is less 50
        for feature in cat_features:
            X[feature] = X[feature].astype("category").cat.codes
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
            raise Exception(f"Unsupported type of target: {target_type}")

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
            raise ValueError(
                f"{scoring} is not a valid scoring value. " f"Use {sorted(SCORERS.keys())} " "to get valid options."
            )
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
        raise ValueError("Mean Squared Logarithmic Error cannot be used when " "targets contain negative values.")

    return mean_squared_error(
        log1p(y_true),
        log1p(y_pred.clip(0)),
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )
