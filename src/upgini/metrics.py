from __future__ import annotations

import inspect
import logging
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from category_encoders.cat_boost import CatBoostEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy import log1p
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype
from sklearn.metrics import check_scoring, get_scorer, make_scorer, roc_auc_score

from upgini.utils.blocked_time_series import BlockedTimeSeriesSplit
from upgini.utils.features_validator import FeaturesValidator
from upgini.utils.sklearn_ext import cross_validate

try:
    from sklearn.metrics import get_scorer_names

    available_scorers = get_scorer_names()
except ImportError:
    from sklearn.metrics._scorer import SCORERS

    available_scorers = SCORERS
from sklearn.metrics import mean_squared_error
from sklearn.metrics._regression import _check_reg_targets, check_consistent_length
from sklearn.model_selection import (  # , TimeSeriesSplit
    BaseCrossValidator,
    TimeSeriesSplit,
)

from upgini.errors import ValidationError
from upgini.metadata import ModelTaskType
from upgini.resource_bundle import bundle
from upgini.utils.target_utils import prepare_target

DEFAULT_RANDOM_STATE = 42

CATBOOST_REGRESSION_PARAMS = {
    "iterations": 250,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "max_depth": 5,
    "early_stopping_rounds": 20,
    "use_best_model": True,
    "one_hot_max_size": 100,
    "verbose": False,
    "random_state": DEFAULT_RANDOM_STATE,
    "allow_writing_files": False,
}

CATBOOST_BINARY_PARAMS = {
    "iterations": 250,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "max_depth": 5,
    "early_stopping_rounds": 20,
    "use_best_model": True,
    "one_hot_max_size": 100,
    "verbose": False,
    "random_state": DEFAULT_RANDOM_STATE,
    "allow_writing_files": False,
    "auto_class_weights": "Balanced",
}

CATBOOST_MULTICLASS_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.25,
    "max_depth": 3,
    "border_count": 15,
    "max_ctr_complexity": 1,
    "loss_function": "MultiClass",
    "subsample": 0.5,
    "bootstrap_type": "Bernoulli",
    "early_stopping_rounds": 20,
    "use_best_model": True,
    "rsm": 0.1,
    "verbose": False,
    "random_state": DEFAULT_RANDOM_STATE,
    "allow_writing_files": False,
    "auto_class_weights": "Balanced",
}

LIGHTGBM_REGRESSION_PARAMS = {
    "random_state": DEFAULT_RANDOM_STATE,
    "n_estimators": 275,
    "feature_fraction": 1.0,
    "deterministic": "true",
    "verbosity": -1,
}

LIGHTGBM_MULTICLASS_PARAMS = {
    "random_state": DEFAULT_RANDOM_STATE,
    "n_estimators": 275,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_gain_to_split": 0.001,
    "max_cat_threshold": 80,
    "min_data_per_group": 20,
    "cat_smooth": 18,
    "cat_l2": 8,
    "objective": "multiclass",
    "use_quantized_grad": "true",
    "num_grad_quant_bins": "8",
    "stochastic_rounding": "true",
    "deterministic": "true",
    "verbosity": -1,
}

LIGHTGBM_BINARY_PARAMS = {
    "random_state": DEFAULT_RANDOM_STATE,
    "min_gain_to_split": 0.001,
    "n_estimators": 275,
    "max_depth": 5,
    "learning_rate": 0.05,
    "objective": "binary",
    "max_cat_threshold": 80,
    "min_data_per_group": 20,
    "cat_smooth": 18,
    "cat_l2": 8,
    "deterministic": "true",
    "verbosity": -1,
}

LIGHTGBM_EARLY_STOPPING_ROUNDS = 20

N_FOLDS = 5
BLOCKED_TS_TEST_SIZE = 0.2

SUPPORTED_CATBOOST_METRICS = {
    s.upper(): s
    for s in (
        "Logloss",
        "CrossEntropy",
        "CtrFactor",
        "Focal",
        "RMSE",
        "LogCosh",
        "Lq",
        "MAE",
        "Quantile",
        "MultiQuantile",
        "Expectile",
        "LogLinQuantile",
        "MAPE",
        "Poisson",
        "MSLE",
        "MedianAbsoluteError",
        "SMAPE",
        "Huber",
        "Tweedie",
        "Cox",
        "RMSEWithUncertainty",
        "MultiClass",
        "MultiClassOneVsAll",
        "PairLogit",
        "PairLogitPairwise",
        "YetiRank",
        "YetiRankPairwise",
        "QueryRMSE",
        "QuerySoftMax",
        "QueryCrossEntropy",
        "StochasticFilter",
        "LambdaMart",
        "StochasticRank",
        "PythonUserDefinedPerObject",
        "PythonUserDefinedMultiTarget",
        "UserPerObjMetric",
        "UserQuerywiseMetric",
        "R2",
        "NumErrors",
        "FairLoss",
        "AUC",
        "Accuracy",
        "BalancedAccuracy",
        "BalancedErrorRate",
        "BrierScore",
        "Precision",
        "Recall",
        "F1",
        "TotalF1",
        "F",
        "MCC",
        "ZeroOneLoss",
        "HammingLoss",
        "HingeLoss",
        "Kappa",
        "WKappa",
        "LogLikelihoodOfPrediction",
        "NormalizedGini",
        "PRAUC",
        "PairAccuracy",
        "AverageGain",
        "QueryAverage",
        "QueryAUC",
        "PFound",
        "PrecisionAt",
        "RecallAt",
        "MAP",
        "NDCG",
        "DCG",
        "FilteredDCG",
        "MRR",
        "ERR",
        "SurvivalAft",
        "MultiRMSE",
        "MultiRMSEWithMissingValues",
        "MultiLogloss",
        "MultiCrossEntropy",
        "Combination",
    )
}


def is_catboost_estimator(estimator):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        return isinstance(estimator, (CatBoostClassifier, CatBoostRegressor))
    except ImportError:
        return False


@dataclass
class _CrossValResults:
    metric: Optional[float]
    metric_std: Optional[float]
    shap_values: Optional[Dict[str, float]]

    def get_display_metric(self) -> Optional[str]:
        if self.metric is None:
            return None
        elif self.metric_std is None:
            return f"{self.metric:.3f}"
        else:
            return f"{self.metric:.3f} ± {self.metric_std:.3f}"


def is_numeric_object(x: pd.Series) -> bool:
    try:
        pd.to_numeric(x, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def is_valid_numeric_array_data(data: pd.Series) -> bool:
    data_without_na = data.dropna()
    if data_without_na.empty:
        return False

    first_element = data_without_na.iloc[0]

    # numpy.ndarray with numeric types
    if isinstance(first_element, np.ndarray):
        return np.issubdtype(first_element.dtype, np.number)

    # DataFrame with all numeric columns
    elif isinstance(first_element, pd.DataFrame):
        return all(np.issubdtype(dtype, np.number) for dtype in first_element.dtypes)

    # list or list of lists with numeric types
    elif isinstance(first_element, list):
        try:
            # flat list
            if all(isinstance(x, (int, float, np.number)) or pd.isna(x) for x in first_element):
                return True
            # list of lists
            elif all(
                isinstance(x, list) and all(isinstance(y, (int, float, np.number)) or pd.isna(y) for y in x)
                for x in first_element
            ):
                return True
        except Exception:
            return False

    return False


class EstimatorWrapper:
    default_estimator: Literal["catboost", "lightgbm"] = "catboost"

    def __init__(
        self,
        estimator,
        scorer: Callable,
        cat_features: Optional[List[str]],
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
        add_params: Optional[Dict[str, Any]] = None,
        groups: Optional[np.ndarray] = None,
        text_features: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.estimator = estimator
        self.scorer = scorer
        self.cat_features = cat_features
        self.metric_name = metric_name
        self.multiplier = multiplier
        self.cv = cv
        self.target_type = target_type
        self.add_params = add_params
        self.cv_estimators = None
        self.groups = groups
        self.text_features = text_features
        self.logger = logger or logging.getLogger()
        self.droped_features = []
        self.converted_to_int = []
        self.converted_to_str = []
        self.converted_to_numeric = []

    def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        x, y, _, fit_params = self._prepare_to_fit(x, y)
        kwargs.update(fit_params)
        self.estimator.fit(x, y, **kwargs)
        return self

    def predict(self, x: pd.DataFrame, **kwargs):
        x, _, _ = self._prepare_to_calculate(x, None)
        return self.estimator.predict(x, **kwargs)

    def _prepare_data(
        self, x: pd.DataFrame, y: pd.Series, groups: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

        if not isinstance(y, pd.Series):
            raise Exception(bundle.get("metrics_unsupported_target_type").format(type(y)))

        if groups is not None:
            x = x.copy()
            x["__groups"] = groups
            x, y = self._remove_empty_target_rows(x, y)
            groups = x["__groups"]
            x = x.drop(columns="__groups")
        else:
            x, y = self._remove_empty_target_rows(x, y)

        y = prepare_target(y, self.target_type)

        self.logger.info(f"After preparing data columns: {x.columns.to_list()}")
        return x, y, groups

    def _remove_empty_target_rows(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray]:
        joined = pd.concat([x, y], axis=1)
        joined = joined[joined[y.name].notna()]
        joined = joined.reset_index(drop=True)
        x = joined.drop(columns=y.name)
        y = np.array(list(joined[y.name].values))

        return x, y

    def _prepare_to_fit(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
        x, y, groups = self._prepare_data(x, y, groups=self.groups)

        self.logger.info(f"Before preparing data columns: {x.columns.to_list()}")
        self.droped_features = []
        self.converted_to_int = []
        self.converted_to_str = []
        self.converted_to_numeric = []
        for c in x.columns:

            if _get_unique_count(x[c]) < 2:
                self.logger.warning(f"Remove feature {c} because it has less than 2 unique values")
                if c in self.cat_features:
                    self.cat_features.remove(c)
                x.drop(columns=[c], inplace=True)
                self.droped_features.append(c)
            elif self.text_features is not None and c in self.text_features:
                x[c] = x[c].astype(str)
                self.converted_to_str.append(c)
            elif c in self.cat_features:
                if x[c].dtype == "bool" or (x[c].dtype == "category" and x[c].cat.categories.dtype == "bool"):
                    x[c] = x[c].astype(np.int64)
                    self.converted_to_int.append(c)
                elif x[c].dtype == "category" and is_integer_dtype(x[c].cat.categories):
                    self.logger.info(
                        f"Convert categorical feature {c} with integer categories"
                        " to int64 and remove from cat_features"
                    )
                    x[c] = x[c].astype(np.int64)
                    self.converted_to_int.append(c)
                    self.cat_features.remove(c)
                elif is_float_dtype(x[c]) or (x[c].dtype == "category" and is_float_dtype(x[c].cat.categories)):
                    self.logger.info(
                        f"Convert float cat feature {c} to string"
                    )
                    x[c] = x[c].astype(str)
                    self.converted_to_str.append(c)
                elif x[c].dtype not in ["category", "int64"]:
                    x[c] = x[c].astype(str)
                    self.converted_to_str.append(c)
            else:
                if x[c].dtype == "bool" or (x[c].dtype == "category" and x[c].cat.categories.dtype == "bool"):
                    self.logger.info(f"Convert bool feature {c} to int64")
                    x[c] = x[c].astype(np.int64)
                    self.converted_to_int.append(c)
                elif not is_valid_numeric_array_data(x[c]) and not is_numeric_dtype(x[c]):
                    try:
                        x[c] = pd.to_numeric(x[c], errors="raise")
                        self.converted_to_numeric.append(c)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Remove feature {c} because it is not numeric and not in cat_features")
                        x.drop(columns=[c], inplace=True)
                        self.droped_features.append(c)

        return x, y, groups, {}

    def _prepare_to_calculate(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, dict]:
        x, y, _ = self._prepare_data(x, y)

        if self.droped_features:
            self.logger.info(f"Drop features on calculate metrics: {self.droped_features}")
            x = x.drop(columns=self.droped_features)

        if self.converted_to_int:
            self.logger.info(f"Convert to int features on calculate metrics: {self.converted_to_int}")
            for c in self.converted_to_int:
                x[c] = x[c].astype(np.int64)

        if self.converted_to_str:
            self.logger.info(f"Convert to str features on calculate metrics: {self.converted_to_str}")
            for c in self.converted_to_str:
                x[c] = x[c].astype(str)

        if self.converted_to_numeric:
            self.logger.info(f"Convert to numeric features on calculate metrics: {self.converted_to_numeric}")
            for c in self.converted_to_numeric:
                x[c] = pd.to_numeric(x[c], errors="coerce")

        return x, y, {}

    def calculate_shap(self, x: pd.DataFrame, y: pd.Series, estimator) -> Optional[Dict[str, float]]:
        return None

    def cross_val_predict(
        self, x: pd.DataFrame, y: np.ndarray, baseline_score_column: Optional[Any] = None
    ) -> _CrossValResults:
        x, y, groups, fit_params = self._prepare_to_fit(x, y)

        if x.shape[1] == 0:
            return _CrossValResults(metric=None, metric_std=None, shap_values=None)

        scorer = check_scoring(self.estimator, scoring=self.scorer)

        shap_values_all_folds = defaultdict(list)
        if baseline_score_column is not None and self.metric_name == "GINI":
            self.logger.info("Calculate baseline GINI on passed baseline_score_column and target")
            metric = roc_auc_score(y, x[baseline_score_column])
            metric_std = None
            average_shap_values = None
        else:
            self.logger.info(f"Cross validate with estimeator: {self.estimator}")
            cv_results = cross_validate(
                estimator=self.estimator,
                x=x,
                y=y,
                scoring=scorer,
                cv=self.cv,
                groups=groups,
                fit_params=fit_params,
                return_estimator=True,
                error_score="raise",
            )
            metrics_by_fold = cv_results["test_score"]
            self.cv_estimators = cv_results["estimator"]

            self.check_fold_metrics(metrics_by_fold)

            metric, metric_std = self._calculate_metric_from_folds(metrics_by_fold)

            splits = self.cv.split(x, y, groups)

            for estimator, split in zip(self.cv_estimators, splits):
                _, validation_idx = split
                cv_x = x.iloc[validation_idx]
                if isinstance(y, pd.Series):
                    cv_y = y.iloc[validation_idx]
                else:
                    cv_y = y[validation_idx]
                shaps = self.calculate_shap(cv_x, cv_y, estimator)
                if shaps is not None:
                    for feature, shap_value in shaps.items():
                        shap_values_all_folds[feature].append(shap_value)

        if shap_values_all_folds:
            average_shap_values = {
                feature: np.mean(np.array(shaps)) for feature, shaps in shap_values_all_folds.items() if len(shaps) > 0
            }
            if len(average_shap_values) == 0:
                average_shap_values = None
            else:
                average_shap_values = self.process_shap_values(average_shap_values)
        else:
            average_shap_values = None

        return _CrossValResults(metric=metric, metric_std=metric_std, shap_values=average_shap_values)

    def process_shap_values(self, shap_values: Dict[str, float]) -> Dict[str, float]:
        return shap_values

    def check_fold_metrics(self, metrics_by_fold: List[float]):
        first_metric_sign = 1 if metrics_by_fold[0] >= 0 else -1
        for metric in metrics_by_fold[1:]:
            if first_metric_sign * metric < 0:
                self.logger.warning(f"Sign of metrics differs between folds: {metrics_by_fold}")

    def post_process_metric(self, metric: float) -> float:
        if self.metric_name == "GINI":
            metric = 2 * metric - 1
        return metric

    def calculate_metric(
        self, x: pd.DataFrame, y: np.ndarray, baseline_score_column: Optional[Any] = None
    ) -> _CrossValResults:
        x, y, _ = self._prepare_to_calculate(x, y)
        if baseline_score_column is not None and self.metric_name == "GINI":
            metric, metric_std = roc_auc_score(y, x[baseline_score_column]), None
        else:
            metrics = []
            for est in self.cv_estimators:
                metrics.append(self.scorer(est, x, y))

            metric, metric_std = self._calculate_metric_from_folds(metrics)
        return _CrossValResults(metric=metric, metric_std=metric_std, shap_values=None)

    def _calculate_metric_from_folds(self, metrics_by_fold: List[float]) -> Tuple[float, float]:
        metrics_by_fold = [self.post_process_metric(m) for m in metrics_by_fold]
        metric = np.mean(metrics_by_fold) * self.multiplier
        metric_std = np.std(metrics_by_fold) * np.abs(self.multiplier)
        return metric, metric_std

    @staticmethod
    def create(
        estimator,
        logger: logging.Logger,
        target_type: ModelTaskType,
        cv: BaseCrossValidator,
        *,
        scoring: Union[Callable, str, None] = None,
        cat_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        add_params: Optional[Dict[str, Any]] = None,
        groups: Optional[List[str]] = None,
        has_date: Optional[bool] = None,
    ) -> EstimatorWrapper:
        scorer, metric_name, multiplier = define_scorer(target_type, scoring)
        kwargs = {
            "scorer": scorer,
            "cat_features": cat_features,
            "metric_name": metric_name,
            "multiplier": multiplier,
            "cv": cv,
            "target_type": target_type,
            "groups": groups,
            "text_features": text_features,
            "logger": logger,
        }
        if estimator is None:
            if EstimatorWrapper.default_estimator == "catboost":
                logger.info("Using CatBoost as default estimator")
                params = {"has_time": has_date}
                if target_type == ModelTaskType.MULTICLASS:
                    params = _get_add_params(params, CATBOOST_MULTICLASS_PARAMS)
                    params = _get_add_params(params, add_params)
                    estimator = CatBoostWrapper(CatBoostClassifier(**params), **kwargs)
                elif target_type == ModelTaskType.BINARY:
                    params = _get_add_params(params, CATBOOST_BINARY_PARAMS)
                    params = _get_add_params(params, add_params)
                    estimator = CatBoostWrapper(CatBoostClassifier(**params), **kwargs)
                elif target_type == ModelTaskType.REGRESSION:
                    params = _get_add_params(params, CATBOOST_REGRESSION_PARAMS)
                    params = _get_add_params(params, add_params)
                    estimator = CatBoostWrapper(CatBoostRegressor(**params), **kwargs)
                else:
                    raise Exception(bundle.get("metrics_unsupported_target_type").format(target_type))
            elif EstimatorWrapper.default_estimator == "lightgbm":
                logger.info("Using LightGBM as default estimator")
                params = {"random_state": DEFAULT_RANDOM_STATE, "verbose": -1}
                if target_type == ModelTaskType.MULTICLASS:
                    params = _get_add_params(params, LIGHTGBM_MULTICLASS_PARAMS)
                    params = _get_add_params(params, add_params)
                    estimator = LightGBMWrapper(LGBMClassifier(**params), **kwargs)
                elif target_type == ModelTaskType.BINARY:
                    params = _get_add_params(params, LIGHTGBM_BINARY_PARAMS)
                    params = _get_add_params(params, add_params)
                    estimator = LightGBMWrapper(LGBMClassifier(**params), **kwargs)
                elif target_type == ModelTaskType.REGRESSION:
                    if not isinstance(cv, TimeSeriesSplit) and not isinstance(cv, BlockedTimeSeriesSplit):
                        params = _get_add_params(params, LIGHTGBM_REGRESSION_PARAMS)
                    params = _get_add_params(params, add_params)
                    estimator = LightGBMWrapper(LGBMRegressor(**params), **kwargs)
                else:
                    raise Exception(bundle.get("metrics_unsupported_target_type").format(target_type))
            else:
                raise Exception("Unsupported default_estimator. Available: catboost, lightgbm")
        else:
            if hasattr(estimator, "copy"):
                estimator_copy = estimator.copy()
            else:
                estimator_copy = deepcopy(estimator)
            kwargs["estimator"] = estimator_copy
            if is_catboost_estimator(estimator):
                if has_date is not None:
                    estimator_copy.set_params(has_time=has_date)
                estimator = CatBoostWrapper(**kwargs)
            else:
                if isinstance(estimator, (LGBMClassifier, LGBMRegressor)):
                    estimator = LightGBMWrapper(**kwargs)
                else:
                    logger.warning(
                        f"Unexpected estimator is used for metrics: {estimator}. "
                        "Default strategy for category features will be used"
                    )
                    estimator = OtherEstimatorWrapper(**kwargs)

        return estimator


class CatBoostWrapper(EstimatorWrapper):
    def __init__(
        self,
        estimator,
        scorer: Callable,
        cat_features: Optional[List[str]],
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
        groups: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(CatBoostWrapper, self).__init__(
            estimator,
            scorer,
            cat_features,
            metric_name,
            multiplier,
            cv,
            target_type,
            groups=groups,
            text_features=text_features,
            logger=logger,
        )
        self.emb_features = None
        self.grouped_embedding_features = None

    def _prepare_to_fit(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
        x, y, groups, params = super()._prepare_to_fit(x, y)

        # Find embeddings
        import catboost
        from catboost import CatBoostClassifier

        if not hasattr(CatBoostClassifier, "get_embedding_feature_indices"):
            self.logger.warning(f"Embedding features are not supported by Catboost version {catboost.__version__}")
        else:
            emb_pattern = r"(.+)_emb\d+"
            self.emb_features = [c for c in x.columns if re.match(emb_pattern, c) and is_numeric_dtype(x[c])]
            x, self.grouped_embedding_features = self.group_embeddings(x)
            if len(self.grouped_embedding_features) > 0:
                params["embedding_features"] = self.grouped_embedding_features

        # Find text features from passed in generate_features
        if not hasattr(CatBoostClassifier, "get_text_feature_indices"):
            self.text_features = None
            self.logger.warning(f"Text features are not supported by this Catboost version {catboost.__version__}")
        else:
            if self.text_features is not None:
                self.logger.info(f"Passed text features for CatBoost: {self.text_features}")
                self.text_features = [f for f in self.text_features if f in x.columns and not is_numeric_dtype(x[f])]
                self.logger.info(f"Rest text features after checks: {self.text_features}")
                params["text_features"] = self.text_features

        # Find rest categorical features
        self.cat_features = [
            f
            for f in self.cat_features
            if f not in (self.text_features or []) and f not in (self.grouped_embedding_features or [])
        ]
        if self.cat_features:
            for c in self.cat_features:
                if is_numeric_dtype(x[c]):
                    x[c] = x[c].fillna(np.nan)
                elif x[c].dtype != "category":
                    x[c] = x[c].fillna("NA")
            params["cat_features"] = self.cat_features

        return x, y, groups, params

    def group_embeddings(self, df: pd.DataFrame):
        embeddings_columns = []
        if len(self.emb_features) > 3:
            self.logger.info(
                "Embedding features count more than 3, so group them into one vector for CatBoost: "
                f"{self.emb_features}"
            )
            emb_name = "__grouped_embeddings"
            df = df.copy()
            df[self.emb_features] = df[self.emb_features].fillna(0.0)
            embeddings_series = pd.Series(df[self.emb_features].values.tolist(), index=df.index)
            df = pd.concat([df.drop(columns=self.emb_features), pd.DataFrame({emb_name: embeddings_series})], axis=1)
            embeddings_columns.append(emb_name)
        for c in df.columns:
            if is_valid_numeric_array_data(df[c]):
                embeddings_columns.append(c)

        return df, embeddings_columns

    def process_shap_values(self, shap_values: Dict[str, float]) -> Dict[str, float]:
        if "__grouped_embeddings" in shap_values:
            for emb_feature in self.emb_features:
                shap_values[emb_feature] = shap_values["__grouped_embeddings"]
            del shap_values["__grouped_embeddings"]
        return shap_values

    def _prepare_to_calculate(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, dict]:
        x, y, params = super()._prepare_to_calculate(x, y)
        if self.text_features:
            params["text_features"] = self.text_features
        if self.grouped_embedding_features:
            x, emb_columns = self.group_embeddings(x)
            params["embedding_features"] = emb_columns

        if self.cat_features:
            for c in self.cat_features:
                if is_numeric_dtype(x[c]):
                    x[c] = x[c].fillna(np.nan)
                elif x[c].dtype != "category":
                    x[c] = x[c].fillna("NA")
            params["cat_features"] = self.cat_features

        return x, y, params

    def cross_val_predict(
        self, x: pd.DataFrame, y: np.ndarray, baseline_score_column: Optional[Any] = None
    ) -> _CrossValResults:
        try:
            return super().cross_val_predict(x, y, baseline_score_column)
        except Exception as e:
            if "Dictionary size is 0" in e.args[0] and self.text_features:
                high_cardinality_features = FeaturesValidator.find_high_cardinality(x[self.text_features])
                if len(high_cardinality_features) == 0:
                    high_cardinality_features = self.text_features
                    self.logger.warning(
                        "Calculate metrics has problem with CatBoost text features. High cardinality features not found"
                        f". Try to remove all text features {high_cardinality_features} and retry"
                    )
                else:
                    self.logger.warning(
                        "Calculate metrics has problem with CatBoost text features. Try to remove high cardinality"
                        f" text features {high_cardinality_features} and retry"
                    )
                for f in high_cardinality_features:
                    self.text_features.remove(f)
                    self.droped_features.append(f)
                    x = x.drop(columns=f, errors="ignore")
                return super().cross_val_predict(x, y, baseline_score_column)
            else:
                raise e

    def calculate_shap(self, x: pd.DataFrame, y: pd.Series, estimator) -> Optional[Dict[str, float]]:
        try:
            from catboost import Pool

            # Create Pool for fold data, if need (for example, when categorical features are present)
            fold_pool = Pool(
                x,
                y,
                cat_features=self.cat_features,
                text_features=self.text_features,
                embedding_features=self.grouped_embedding_features,
            )

            shap_values = estimator.get_feature_importance(data=fold_pool, type="ShapValues")

            if self.target_type == ModelTaskType.MULTICLASS:
                # For multiclass, shap_values has shape (n_samples, n_classes, n_features + 1)
                # Last column is bias term
                shap_values = shap_values[:, :, :-1]  # Remove bias term
                # Average SHAP values across classes
                shap_values = np.mean(np.abs(shap_values), axis=1)
            else:
                # For binary/regression, shap_values has shape (n_samples, n_features + 1)
                # Last column is bias term
                shap_values = shap_values[:, :-1]  # Remove bias term
                # Take absolute values
                shap_values = np.abs(shap_values)

            feature_importance = {}
            for i, col in enumerate(x.columns):
                feature_importance[col] = np.mean(np.abs(shap_values[:, i]))

            return feature_importance

        except Exception as e:
            self.logger.exception(f"Failed to recalculate new SHAP values: {str(e)}")
            return None


class LightGBMWrapper(EstimatorWrapper):
    def __init__(
        self,
        estimator,
        scorer: Callable,
        cat_features: Optional[List[str]],
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
        groups: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(LightGBMWrapper, self).__init__(
            estimator,
            scorer,
            cat_features,
            metric_name,
            multiplier,
            cv,
            target_type,
            groups=groups,
            text_features=text_features,
            logger=logger,
        )
        self.cat_encoder = None
        self.n_classes = None

    def _prepare_to_fit(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, dict]:
        x, y_numpy, groups, params = super()._prepare_to_fit(x, y)
        if self.target_type in [ModelTaskType.BINARY, ModelTaskType.MULTICLASS]:
            self.n_classes = len(np.unique(y_numpy))
        if LIGHTGBM_EARLY_STOPPING_ROUNDS is not None:
            if self.target_type == ModelTaskType.BINARY:
                params["eval_metric"] = "auc"
            params["callbacks"] = [lgb.early_stopping(stopping_rounds=LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=False)]
        if self.cat_features:
            encoder = CatBoostEncoder(random_state=DEFAULT_RANDOM_STATE, cols=self.cat_features, return_df=True)
            encoded = encoder.fit_transform(x[self.cat_features].astype("object"), y_numpy).astype("category")
            x[self.cat_features] = encoded
            self.cat_encoder = encoder
        for c in x.columns:
            if x[c].dtype not in ["category", "int64", "float64", "bool"]:
                self.logger.warning(f"Feature {c} is not numeric and will be dropped")
                self.droped_features.append(c)
                x = x.drop(columns=c, errors="ignore")
        return x, y_numpy, groups, params

    def _prepare_to_calculate(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, dict]:
        x, y_numpy, params = super()._prepare_to_calculate(x, y)
        if self.cat_features is not None and self.cat_encoder is not None:
            encoded = self.cat_encoder.transform(x[self.cat_features].astype("object"), y_numpy).astype("category")
            x[self.cat_features] = encoded
        return x, y_numpy, params

    def calculate_shap(self, x: pd.DataFrame, y: pd.Series, estimator) -> Optional[Dict[str, float]]:
        try:
            shap_matrix = estimator.predict(
                x,
                predict_disable_shape_check=True,
                raw_score=True,
                pred_leaf=False,
                pred_early_stop=True,
                pred_contrib=True,
            )

            if self.target_type == ModelTaskType.MULTICLASS:
                n_feat = x.shape[1]
                shap_matrix.shape = (shap_matrix.shape[0], self.n_classes, n_feat + 1)
                shap_matrix = np.mean(np.abs(shap_matrix), axis=1)

            # exclude base value
            shap_matrix = shap_matrix[:, :-1]

            feature_importance = {}
            for i, col in enumerate(x.columns):
                feature_importance[col] = np.mean(np.abs(shap_matrix[:, i]))

            return feature_importance

        except Exception as e:
            self.logger.warning(f"Failed to calculate SHAP values: {str(e)}")
            return None


class OtherEstimatorWrapper(EstimatorWrapper):
    def __init__(
        self,
        estimator,
        scorer: Callable,
        cat_features: Optional[List[str]],
        metric_name: str,
        multiplier: int,
        cv: BaseCrossValidator,
        target_type: ModelTaskType,
        groups: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(OtherEstimatorWrapper, self).__init__(
            estimator,
            scorer,
            cat_features,
            metric_name,
            multiplier,
            cv,
            target_type,
            groups=groups,
            text_features=text_features,
            logger=logger,
        )

    def _prepare_to_fit(self, x: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
        x, y_numpy, groups, params = super()._prepare_to_fit(x, y)
        num_features = [col for col in x.columns if col not in self.cat_features]
        x[num_features] = x[num_features].fillna(-999)
        if self.cat_features:
            encoder = CatBoostEncoder(random_state=DEFAULT_RANDOM_STATE, return_df=True)
            encoded = encoder.fit_transform(x[self.cat_features].astype("object"), y_numpy).astype("category")
            x[self.cat_features] = encoded
            self.cat_encoder = encoder
        for c in x.columns:
            if x[c].dtype not in ["category", "int64", "float64", "bool"]:
                self.logger.warning(f"Feature {c} is not numeric and will be dropped")
                self.droped_features.append(c)
                x = x.drop(columns=c, errors="ignore")
        return x, y_numpy, groups, params

    def _prepare_to_calculate(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, dict]:
        x, y_numpy, params = super()._prepare_to_calculate(x, y)
        if self.cat_features is not None:
            num_features = [col for col in x.columns if col not in self.cat_features]
            x[num_features] = x[num_features].fillna(-999)
            if self.cat_features and self.cat_encoder is not None:
                x[self.cat_features] = self.cat_encoder.transform(
                    x[self.cat_features].astype("object"), y_numpy
                ).astype("category")
        return x, y_numpy, params


def validate_scoring_argument(scoring: Union[Callable, str, None]):
    if scoring is None:
        return

    if isinstance(scoring, str):
        _get_scorer_by_name(scoring)
        return

    if not isinstance(scoring, Callable):
        raise ValidationError(
            f"Invalid scoring argument passed {scoring}. It should be string with scoring name or function"
            " that accepts 3 input arguments: estimator, x, y"
        )

    spec = inspect.getfullargspec(scoring)
    if len(spec.args) < 3:
        raise ValidationError(
            f"Invalid scoring function passed {scoring}. It should accept 3 input arguments: estimator, x, y"
        )


def _get_scorer_by_name(scoring: str) -> Tuple[Callable, str, int]:
    metric_name = scoring
    multiplier = 1
    if metric_name == "mean_squared_log_error" or metric_name == "MSLE" or metric_name == "msle":
        scoring = make_scorer(_ext_mean_squared_log_error, greater_is_better=False)
        multiplier = -1
    elif "root_mean_squared_log_error" in metric_name or metric_name == "RMSLE" or metric_name == "rmsle":
        scoring = make_scorer(_ext_root_mean_squared_log_error, greater_is_better=False)
        multiplier = -1
    elif metric_name == "root_mean_squared_error" or metric_name == "RMSE" or metric_name == "rmse":
        scoring = get_scorer("neg_root_mean_squared_error")
        multiplier = -1
    elif scoring in available_scorers:
        scoring = get_scorer(scoring)
    elif ("neg_" + scoring) in available_scorers:
        scoring = get_scorer("neg_" + scoring)
        multiplier = -1
    else:
        supported_metrics = set(available_scorers)
        neg_metrics = [m[4:] for m in supported_metrics if m.startswith("neg_")]
        supported_metrics.update(neg_metrics)
        supported_metrics.update(
            [
                "mean_squared_log_error",
                "MSLE",
                "msle",
                "root_mean_squared_log_error",
                "RMSLE",
                "rmsle",
                "root_mean_squared_error",
                "RMSE",
                "rmse",
            ]
        )
        raise ValidationError(bundle.get("metrics_invalid_scoring").format(scoring, sorted(supported_metrics)))
    return scoring, metric_name, multiplier


def define_scorer(target_type: ModelTaskType, scoring: Union[Callable, str, None]) -> Tuple[Callable, str, int]:
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
        scoring, metric_name, multiplier = _get_scorer_by_name(scoring)
    elif hasattr(scoring, "__name__"):
        metric_name = scoring.__name__
    else:
        metric_name = str(scoring)

    metric_name = "GINI" if metric_name.upper() == "ROC_AUC" and target_type == ModelTaskType.BINARY else metric_name

    return scoring, metric_name, multiplier


def _get_add_params(input_params, add_params):
    output_params = dict(input_params)
    if add_params is not None:
        output_params.update(add_params)

    return output_params


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

    mse = mean_squared_error(
        log1p(y_true),
        log1p(y_pred.clip(0)),
        sample_weight=sample_weight,
        multioutput=multioutput,
    )
    return mse if squared else np.sqrt(mse)


def _get_unique_count(series: pd.Series) -> int:
    try:
        return series.nunique(dropna=False)
    except TypeError:
        return series.astype(str).nunique(dropna=False)
