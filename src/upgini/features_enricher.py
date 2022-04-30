import hashlib
import logging
import sys
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pandas.api.types import is_string_dtype

try:
    from sklearn.base import TransformerMixin  # type: ignore
    from sklearn.exceptions import NotFittedError  # type: ignore
except ImportError:
    TransformerMixin = object
    NotFittedError = Exception

import itertools
import uuid
from copy import deepcopy

import numpy as np
import pandas as pd
from yaspin import yaspin
from yaspin.spinners import Spinners

from upgini.dataset import Dataset
from upgini.http import init_logging
from upgini.metadata import (
    EVAL_SET_INDEX,
    ISO_CODE,
    SYSTEM_FAKE_DATE,
    SYSTEM_RECORD_ID,
    CVType,
    FileColumnMeaningType,
    ModelTaskType,
    RuntimeParameters,
    SearchKey,
)
from upgini.metrics import EstimatorWrapper
from upgini.search_task import SearchTask
from upgini.utils.format import Format
from upgini.utils.target_utils import define_task


class FeaturesEnricher(TransformerMixin):  # type: ignore
    """Retrieve external features via Upgini that are most relevant to predict your target.

    Parameters
    ----------
    search_keys: dict of str->SearchKey or int->SearchKey
        Dictionary with column names or indices mapping to key types.
        Each of this columns will be used as a search key to find features.

    iso_code: str, optional (default=None)
        ISO-3166 COUNTRY code of country for all rows in dataset.

    keep_input: bool, optional (default=False)
        If True, copy original input columns to the output dataframe.

    model_task_type: ModelTaskType, optional (default=None)
        If defined, used as type of training model, else autdefined type will be used

    importance_threshold: float, optional (default=None)
        Minimum importance shap value for selected features. By default minimum importance is 0.0

    max_features: int, optional (default=None)
        Maximum count of selected most important features. By default it is unlimited

    api_key: str, optional (default=None)
        Token to authorize search requests. You can get it on https://profile.upgini.com/.
        If not specified then read the value from the environment variable UPGINI_API_KEY.

    endpoint: str, optional (default=None)
        URL of Upgini API where search requests are submitted.
        If not specified then used the default value.
        Please don't overwrite it if you are unsure.

    search_id: str, optional (default=None)
        Identifier of fitted enricher.
        If not specified transform could be called only after fit or fit_transform call

    runtime_parameters: dict of str->str, optional (default None).
        Not for public use. Ignore it. It's a way to argument requests with extra parameters.
        Used to trigger experimental features at backend. Used by backend team.

    date_format: str, optional (default=None)
        Format for date column with string type. For example: %Y-%m-%d

    scoring: string, callable or None, optional, default: None
        A string or a scorer callable object / function with signature scorer(estimator, X, y).
        If None, the estimator's score method is used.

    cv: CVType, optional (default=None)
        Type of cross validation: CVType.k_fold, CVType.time_series, CVType.blocked_time_series
    """

    TARGET_NAME = "target"

    RANDOM_STATE = 42

    _search_task: Optional[SearchTask] = None
    passed_features: List[str] = []
    importance_threshold: Optional[float]
    max_features: Optional[int]
    features_info: pd.DataFrame = pd.DataFrame(columns=["feature_name", "shap_value", "match_percent"])
    enriched_X: Optional[pd.DataFrame] = None
    enriched_eval_set: Optional[pd.DataFrame] = None

    def __init__(
        self,
        search_keys: Dict[str, SearchKey],
        iso_code: Optional[str] = None,
        keep_input: bool = False,
        model_task_type: Optional[ModelTaskType] = None,
        importance_threshold: Optional[float] = 0,
        max_features: Optional[int] = 400,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        search_id: Optional[str] = None,
        runtime_parameters: Optional[RuntimeParameters] = None,
        date_format: Optional[str] = None,
        random_state: int = 42,
        scoring: Optional[Callable] = None,
        cv: Optional[CVType] = None,
    ):
        init_logging(endpoint, api_key)
        self.__validate_search_keys(search_keys, search_id)
        self.search_keys = search_keys
        self.iso_code = iso_code
        self.keep_input = keep_input
        self.model_task_type = model_task_type
        self.scoring = scoring
        if importance_threshold is not None:
            try:
                self.importance_threshold = float(importance_threshold)
            except ValueError:
                logging.exception(f"Invalid importance_threshold provided: {importance_threshold}")
                raise ValueError("importance_threshold should be float")
        if max_features is not None:
            try:
                self.max_features = int(max_features)
            except ValueError:
                logging.exception(f"Invalid max_features provided: {max_features}")
                raise ValueError("max_features should be int")
        self.endpoint = endpoint
        self.api_key = api_key
        if search_id:
            search_task = SearchTask(
                search_id,
                endpoint=self.endpoint,
                api_key=self.api_key,
            )
            print("Checking existing search")
            try:
                self._search_task = search_task.poll_result(quiet=True)
                file_metadata = self._search_task.get_file_metadata()
                x_columns = [c.originalName or c.name for c in file_metadata.columns]
                self.__prepare_feature_importances(x_columns)
                # TODO validate search_keys with search_keys from file_metadata
                print("Search found. Now you can use transform")
            except Exception as e:
                logging.exception("Failed to check existing search")
                raise e
        self.runtime_parameters = runtime_parameters
        self.date_format = date_format
        self.random_state = random_state
        self.cv = cv
        if cv is not None:
            if self.runtime_parameters is None:
                self.runtime_parameters = RuntimeParameters()
            if self.runtime_parameters.properties is None:
                self.runtime_parameters.properties = {}
            self.runtime_parameters.properties["cv_type"] = cv.name

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[List[tuple]] = None,
        **fit_params,
    ):
        """Fit to data.

        Fits transformer to `X` and `y` with optional parameters `fit_params`.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) default=None
            Target values.

        eval_set : List[tuple], optional (default=None)
            List of pairs like (X, y) for validation

        **fit_params : dict
            Additional fit parameters.
        """
        logging.info(f"Start fit. X shape: {X.shape}. y shape: {len(y)}")
        try:
            self.__inner_fit(X, y, eval_set, True, **fit_params)
        except Exception as e:
            logging.exception("Failed inner fit")
            raise e

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[List[tuple]] = None,
        **fit_params,
    ) -> pd.DataFrame:
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.
        If keep_input is True, then all input columns are present in output dataframe.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) default=None
            Target values.

        eval_set : List[tuple], optional (default=None)
            List of pairs like (X, y) for validation

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : pandas dataframe of shape (n_samples, n_features_new)
            Transformed dataframe, enriched with important features.
        """

        logging.info(f"Start fit_transform. X shape: {X.shape}. y shape: {len(y)}")
        try:
            result = self.__inner_fit(X, y, eval_set, extract_features=True, **fit_params)
        except Exception as e:
            logging.exception("Failed in inner_fit")
            raise e

        if self.keep_input:
            return result
        else:
            return result.drop(columns=[c for c in X.columns if c in result.columns])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform `X`.

        Returns a transformed version of `X`.
        If keep_input is True, then all input columns are present in output dataframe.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_new : pandas dataframe of shape (n_samples, n_features_new)
            Transformed dataframe, enriched with important features.
        """
        logging.info(f"Start transform. X shape: {X.shape}")
        try:
            result, _ = self.__inner_transform(X)
        except Exception as e:
            logging.exception("Failed to inner transform")
            raise e

        if self.keep_input:
            return result
        else:
            return result.drop(columns=[c for c in X.columns if c in result.columns])

    def __inner_transform(
        self, X: pd.DataFrame, silent_mode: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if self._search_task is None:
            msg = "`fit` or `fit_transform` should be called before `transform`."
            logging.error(msg)
            raise NotFittedError(msg)
        if not isinstance(X, pd.DataFrame):
            msg = f"Only pandas.DataFrame supported for X, but {type(X)} was passed."
            logging.error(msg)
            raise TypeError(msg)

        self.__prepare_search_keys(X)
        meaning_types = {col: key.value for col, key in self.search_keys.items()}
        feature_columns = [column for column in X.columns if column not in self.search_keys.keys()]

        self.__check_string_dates(X)

        df = X.copy()

        df = df.reset_index(drop=True)

        self.__add_fake_date(df, meaning_types)

        self.__add_iso_code(df, meaning_types)

        df[SYSTEM_RECORD_ID] = df.apply(lambda row: self._hash_row(row[self.search_keys.keys()]), axis=1)
        meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID

        combined_search_keys = []
        for L in range(1, len(self.search_keys.keys()) + 1):
            for subset in itertools.combinations(self.search_keys.keys(), L):
                combined_search_keys.append(subset)

        # Don't pass features in backend on transform
        if len(feature_columns) > 0:
            df_without_features = df.drop(columns=feature_columns)
        else:
            df_without_features = df

        dataset = Dataset(
            "sample_" + str(uuid.uuid4()),
            df=df_without_features,  # type: ignore
            endpoint=self.endpoint,  # type: ignore
            api_key=self.api_key,  # type: ignore
            date_format=self.date_format,  # type: ignore
        )
        dataset.meaning_types = meaning_types
        dataset.search_keys = combined_search_keys
        validation_task = self._search_task.validation(
            dataset, extract_features=True, runtime_parameters=self.runtime_parameters, silent_mode=silent_mode
        )

        if not silent_mode:
            print("Executing transform step")
            with yaspin(Spinners.material) as sp:
                result = self.__enrich(df, validation_task.get_all_validation_raw_features(), X.index)
                sp.ok("Done                         ")
        else:
            result = self.__enrich(df, validation_task.get_all_validation_raw_features(), X.index)

        return result

    def get_search_id(self) -> Optional[str]:
        """Returns search_id of fitted enricher. It's present only after fit completed"""
        return self._search_task.search_task_id if self._search_task else None

    def get_features_info(self) -> pd.DataFrame:
        """Returns pandas dataframe with importances for each feature"""
        if self._search_task is None or self._search_task.summary is None:
            msg = "Run fit or pass search_id before get features info."
            logging.info(msg)
            raise NotFittedError(msg)

        return self.features_info

    def get_metrics(self) -> Optional[pd.DataFrame]:
        """Returns pandas dataframe with quality metrics for main dataset and eval_set"""
        if self._search_task is None or self._search_task.summary is None:
            msg = "Run fit or pass search_id before get metrics."
            logging.error(msg)
            raise NotFittedError(msg)

        metrics = []
        initial_metrics = {}

        initial_hit_rate = self._search_task.initial_max_hit_rate()
        if initial_hit_rate is None:
            logging.warning("Get metrics called, but initial search information is empty")
            return None

        initial_metrics["segment"] = "train"
        initial_metrics["match rate"] = initial_hit_rate["value"]
        metrics.append(initial_metrics)
        initial_auc = self._search_task.initial_max_auc()
        if initial_auc is not None:
            initial_metrics["auc"] = initial_auc["value"]
        initial_accuracy = self._search_task.initial_max_accuracy()
        if initial_accuracy is not None:
            initial_metrics["accuracy"] = initial_accuracy["value"]
        initial_rmse = self._search_task.initial_max_rmse()
        if initial_rmse is not None:
            initial_metrics["rmse"] = initial_rmse["value"]
        initial_uplift = self._search_task.initial_max_uplift()
        if len(self.passed_features) > 0 and initial_uplift is not None:
            initial_metrics["uplift"] = initial_uplift["value"]
        max_initial_eval_set_metrics = self._search_task.get_max_initial_eval_set_metrics()

        if max_initial_eval_set_metrics is not None:
            for eval_set_metrics in max_initial_eval_set_metrics:
                if "gini" in eval_set_metrics:
                    del eval_set_metrics["gini"]
                eval_set_index = eval_set_metrics["eval_set_index"]
                eval_set_metrics["match rate"] = eval_set_metrics["hit_rate"]
                eval_set_metrics["segment"] = f"eval {eval_set_index}"
                del eval_set_metrics["hit_rate"]
                del eval_set_metrics["eval_set_index"]
                metrics.append(eval_set_metrics)

        metrics_df = pd.DataFrame(metrics)
        metrics_df.set_index("segment", inplace=True)
        metrics_df.rename_axis("", inplace=True)
        return metrics_df

    @staticmethod
    def __validate_search_keys(search_keys: Dict[str, SearchKey], search_id: Optional[str]):
        if len(search_keys) == 0:
            if search_id:
                logging.error(f"search_id {search_id} provided without search_keys")
                raise ValueError("To transform with search_id please set search_keys to the value used for fitting.")
            else:
                logging.error("search_keys not provided")
                raise ValueError("Key columns should be marked up by search_keys.")

        key_types = search_keys.values()

        if SearchKey.DATE in key_types and SearchKey.DATETIME in key_types:
            msg = "Date and datetime search keys are presented simultaniously. Select only one of them"
            logging.error(msg)
            raise Exception(msg)

        if SearchKey.EMAIL in key_types and SearchKey.HEM in key_types:
            msg = "Email and HEM search keys are presented simultaniously. Select only one of them"
            logging.error(msg)
            raise Exception(msg)

        if SearchKey.POSTAL_CODE in key_types and SearchKey.COUNTRY not in key_types:
            msg = "COUNTRY search key should be provided if POSTAL_CODE is presented"
            logging.error(msg)
            raise Exception(msg)

        for key_type in SearchKey.__members__.values():
            if len(list(filter(lambda x: x == key_type, key_types))) > 1:
                msg = f"Search key {key_type} presented multiple times"
                logging.error(msg)
                raise Exception(msg)

    def __inner_fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list, None] = None,
        eval_set: Optional[List[tuple]] = None,
        extract_features: bool = False,
        **fit_params,
    ) -> pd.DataFrame:
        self.enriched_X = None
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Only pandas.DataFrame supported for X, but {type(X)} was passed.")
        if not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not isinstance(y, list):
            raise TypeError(f"Only pandas.Series or numpy.ndarray or list supported for y, but {type(y)} was passed.")

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        if X.shape[0] != len(y_array):
            raise ValueError("X and y should be the same size")

        self.__prepare_search_keys(X)

        meaning_types = {
            **{col: key.value for col, key in self.search_keys.items()},
            **{str(c): FileColumnMeaningType.FEATURE for c in X.columns if c not in self.search_keys.keys()},
        }

        self.__check_string_dates(X)

        df: pd.DataFrame = X.copy()  # type: ignore
        df[self.TARGET_NAME] = y_array

        df.reset_index(drop=True, inplace=True)

        meaning_types[self.TARGET_NAME] = FileColumnMeaningType.TARGET

        model_task_type = self.model_task_type or define_task(df[self.TARGET_NAME])

        if eval_set is not None and len(eval_set) > 0:
            df[EVAL_SET_INDEX] = 0
            meaning_types[EVAL_SET_INDEX] = FileColumnMeaningType.EVAL_SET_INDEX
            for idx, eval_pair in enumerate(eval_set):
                if len(eval_pair) != 2:
                    raise TypeError(
                        f"Invalid size of eval_set pair: {len(eval_pair)}. "
                        "It should contain tuples of 2 elements: X and y."
                    )
                eval_X = eval_pair[0]
                eval_y = eval_pair[1]
                if not isinstance(eval_X, pd.DataFrame):
                    raise TypeError(
                        f"Only pandas.DataFrame supported for X in eval_set, but {type(eval_X)} was passed."
                    )
                if (
                    not isinstance(eval_y, pd.Series)
                    and not isinstance(eval_y, np.ndarray)
                    and not isinstance(eval_y, list)
                ):
                    raise TypeError(
                        "pandas.Series or numpy.ndarray or list supported for y in eval_set, "
                        f"but {type(eval_y)} was passed."
                    )
                eval_df: pd.DataFrame = eval_X.copy()  # type: ignore
                eval_df[self.TARGET_NAME] = pd.Series(eval_y)
                eval_df[EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_df], ignore_index=True)

        self.__add_fit_system_record_id(df, meaning_types)

        self.__add_fake_date(df, meaning_types)

        self.__add_iso_code(df, meaning_types)

        combined_search_keys = []
        for L in range(1, len(self.search_keys.keys()) + 1):
            for subset in itertools.combinations(self.search_keys.keys(), L):
                combined_search_keys.append(subset)

        dataset = Dataset(
            "tds_" + str(uuid.uuid4()),
            df=df,  # type: ignore
            model_task_type=model_task_type,  # type: ignore
            endpoint=self.endpoint,  # type: ignore
            api_key=self.api_key,  # type: ignore
            date_format=self.date_format,  # type: ignore
            random_state=self.random_state,  # type: ignore
        )
        dataset.meaning_types = meaning_types
        dataset.search_keys = combined_search_keys

        self.passed_features = [
            column for column, meaning_type in meaning_types.items() if meaning_type == FileColumnMeaningType.FEATURE
        ]

        self._search_task = dataset.search(
            extract_features=extract_features,
            importance_threshold=self.importance_threshold,
            max_features=self.max_features,
            runtime_parameters=self.runtime_parameters,
        )

        self.__show_metrics()

        self.__prepare_feature_importances(list(X.columns))

        self.__show_selected_features()

        if dataset.sampled:
            logging.info(
                "Train dataset has size more than fit threshold. Transform will be executed in separate action"
            )
            try:
                self.enriched_X, self.enriched_eval_set = self.__inner_transform(X, silent_mode=True)
            except Exception as e:
                logging.exception("Failed to transform")
                raise e
        else:
            try:
                self.enriched_X, self.enriched_eval_set = self.__enrich(
                    df, self._search_task.get_all_initial_raw_features(), X.index
                )
            except Exception as e:
                logging.exception("Failed to download features")
                raise e

        return self.enriched_X

    def __is_date_key_present(self) -> bool:
        return len({SearchKey.DATE, SearchKey.DATETIME}.intersection(self.search_keys.values())) != 0

    def __add_fit_system_record_id(self, df: pd.DataFrame, meaning_types: Dict[str, FileColumnMeaningType]):
        if (self.cv is None or self.cv == CVType.k_fold) and self.__is_date_key_present():
            date_column = [
                col
                for col, t in meaning_types.items()
                if t in [FileColumnMeaningType.DATE, FileColumnMeaningType.DATETIME]
            ]
            df.sort_values(by=date_column, kind="mergesort")
            pass
        df.reset_index(drop=True, inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={"index": SYSTEM_RECORD_ID}, inplace=True)
        meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID

    def __check_string_dates(self, df: pd.DataFrame):
        for column, search_key in self.search_keys.items():
            if search_key in [SearchKey.DATE, SearchKey.DATETIME] and is_string_dtype(df[column]):
                if self.date_format is None or len(self.date_format) == 0:
                    msg = (
                        f"Date column `{column}` has string type, but constructor argument `date_format` is empty.\n"
                        "Please, convert column to datetime type or pass date format implicitly"
                    )
                    logging.error(msg)
                    raise Exception(msg)

    # temporary while statistic on date will not be removed
    def __add_fake_date(self, df: pd.DataFrame, meaning_types: Dict[str, FileColumnMeaningType]):
        if not self.__is_date_key_present():
            logging.info("Fake date column added with 2200-01-01 value")
            df[SYSTEM_FAKE_DATE] = date(2200, 1, 1)  # remove when statistics by date will be deleted
            self.search_keys[SYSTEM_FAKE_DATE] = SearchKey.DATE
            meaning_types[SYSTEM_FAKE_DATE] = FileColumnMeaningType.DATE

    def __add_iso_code(self, df: pd.DataFrame, meaning_types: Dict[str, FileColumnMeaningType]):
        if self.iso_code is not None and SearchKey.COUNTRY not in self.search_keys.values():
            logging.info(f"Add COUNTRY column with {self.iso_code} value")
            df[ISO_CODE] = self.iso_code
            self.search_keys[ISO_CODE] = SearchKey.COUNTRY
            meaning_types[ISO_CODE] = FileColumnMeaningType.COUNTRY

    def calculate_metrics(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list],
        eval_set: Optional[List[Tuple[pd.DataFrame, Any]]] = None,
        scoring: Union[Callable, str, None] = None,
        estimator=None,
    ) -> pd.DataFrame:
        if (
            (self.enriched_X is None)
            or (self._search_task is None)
            or (self._search_task.initial_max_hit_rate() is None)
        ):
            raise Exception("Fit wasn't completed successfully")

        if scoring is None:
            scoring = self.scoring

        fitting_X = X.drop(columns=[col for col in self.search_keys.keys() if col in X.columns])
        fitting_enriched_X = self.enriched_X.drop(
            columns=[col for col in self.search_keys.keys() if col in self.enriched_X.columns]
        )

        model_task_type = self.model_task_type or define_task(pd.Series(y), silent=True)

        # 1 If client features are presented - fit and predict with KFold CatBoost model on etalon features
        # and calculate baseline metric
        etalon_metric = None
        if fitting_X.shape[1] > 0:
            etalon_metric = EstimatorWrapper.create(estimator, model_task_type, self.cv, scoring).cross_val_predict(
                fitting_X, y
            )

        # 2 Fit and predict with KFold Catboost model on enriched tds and calculate final metric (and uplift)
        wrapper = EstimatorWrapper.create(estimator, model_task_type, self.cv, scoring)
        enriched_metric = wrapper.cross_val_predict(fitting_enriched_X, y)
        metric = wrapper.metric_name

        uplift = None
        if etalon_metric is not None:
            uplift = (enriched_metric - etalon_metric) * wrapper.multiplier

        metrics = [
            {
                "segment": "train",
                "match_rate": self._search_task.initial_max_hit_rate()["value"],  # type: ignore
                f"baseline {metric}": etalon_metric,
                f"enriched {metric}": enriched_metric,
                "uplift": uplift,
            }
        ]

        # 3 If eval_set is presented - fit final model on train enriched data and score each validation dataset
        # and calculate final metric (and uplift)
        max_initial_eval_set_metrics = self._search_task.get_max_initial_eval_set_metrics()
        if eval_set is not None and self.enriched_eval_set is not None:
            # Fit models
            etalon_model = None
            if fitting_X.shape[1] > 0:
                etalon_model = EstimatorWrapper.create(deepcopy(estimator), model_task_type, self.cv, scoring)
                etalon_model.fit(fitting_X, y)
            enriched_model = EstimatorWrapper.create(deepcopy(estimator), model_task_type, self.cv, scoring)
            enriched_model.fit(fitting_enriched_X, y)

            for idx, eval_pair in enumerate(eval_set):
                eval_hit_rate = max_initial_eval_set_metrics[idx]["hit_rate"] if max_initial_eval_set_metrics else None
                eval_X = eval_pair[0]
                eval_X = eval_X.drop(columns=[col for col in self.search_keys.keys() if col in eval_X.columns])
                enriched_eval_X = self.enriched_eval_set[self.enriched_eval_set[EVAL_SET_INDEX] == idx + 1]
                enriched_eval_X = enriched_eval_X.drop(
                    columns=[col for col in self.search_keys.keys() if col in enriched_eval_X.columns]
                )
                enriched_eval_X = enriched_eval_X.drop(columns=EVAL_SET_INDEX)
                eval_y = eval_pair[1]

                etalon_eval_metric = None
                if etalon_model is not None:
                    etalon_eval_metric = etalon_model.calculate_metric(eval_X, eval_y)

                enriched_eval_metric = enriched_model.calculate_metric(enriched_eval_X, eval_y)

                eval_uplift = None
                if etalon_eval_metric is not None:
                    eval_uplift = (enriched_eval_metric - etalon_eval_metric) * enriched_model.multiplier

                metrics.append(
                    {
                        "segment": f"eval {idx + 1}",
                        "match_rate": eval_hit_rate,
                        f"baseline {metric}": etalon_eval_metric,
                        f"enriched {metric}": enriched_eval_metric,
                        "uplift": eval_uplift,
                    }
                )

        metrics_df = pd.DataFrame(metrics)
        metrics_df.set_index("segment", inplace=True)
        metrics_df.rename_axis("", inplace=True)
        return metrics_df

    def __enrich(
        self,
        df: pd.DataFrame,
        result_features: Optional[pd.DataFrame],
        original_index: pd.Index,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if result_features is None:
            logging.error(f"result features not found by search_task_id: {self.get_search_id()}")
            raise RuntimeError("Search engine crashed on this request.")

        sorted_result_columns = [name for name in self.__filtered_importance_names() if name in result_features.columns]
        result_features = result_features[[SYSTEM_RECORD_ID] + sorted_result_columns]

        df_without_target = df.drop(columns=self.TARGET_NAME) if self.TARGET_NAME in df.columns else df
        result = pd.merge(
            df_without_target,
            result_features,
            left_on=SYSTEM_RECORD_ID,
            right_on=SYSTEM_RECORD_ID,
            how="left",
        )

        if EVAL_SET_INDEX in result.columns:
            result_train = result[result[EVAL_SET_INDEX] == 0]
            result_eval_set = result[result[EVAL_SET_INDEX] != 0]
            result_train.drop(columns=EVAL_SET_INDEX, inplace=True)
        else:
            result_train = result
            result_eval_set = None

        result_train.index = original_index
        if SYSTEM_RECORD_ID in result.columns:
            result_train.drop(columns=SYSTEM_RECORD_ID, inplace=True)
            if result_eval_set is not None:
                result_eval_set.drop(columns=SYSTEM_RECORD_ID, inplace=True)
        if SYSTEM_FAKE_DATE in result.columns:
            result_train.drop(columns=SYSTEM_FAKE_DATE, inplace=True)
            if result_eval_set is not None:
                result_eval_set.drop(columns=SYSTEM_FAKE_DATE, inplace=True)

        return result_train, result_eval_set

    def __prepare_feature_importances(self, x_columns: List[str]):
        if self._search_task is None:
            raise NotFittedError("`fit` or `fit_transform` should be called before `transform`.")
        importances = self._search_task.initial_features()

        def feature_metadata_by_name(name: str):
            for f in importances:
                if f["feature_name"] == name:
                    return f

        self.feature_names_ = []
        self.feature_importances_ = []
        features_info = []

        service_columns = [SYSTEM_RECORD_ID, EVAL_SET_INDEX, self.TARGET_NAME]

        importances.sort(key=lambda m: -m["shap_value"])
        for feature_metadata in importances:
            if feature_metadata["feature_name"] not in x_columns:
                self.feature_names_.append(feature_metadata["feature_name"])
                self.feature_importances_.append(feature_metadata["shap_value"])
            features_info.append(feature_metadata)

        for x_column in x_columns:
            if x_column in (list(self.search_keys.keys()) + service_columns):
                continue
            feature_metadata = feature_metadata_by_name(x_column)
            if feature_metadata is None:
                features_info.append(
                    {
                        "feature_name": x_column,
                        "shap_value": 0.0,
                        "coverage %": None,  # TODO fill from X
                    }
                )

        if len(features_info) > 0:
            self.features_info = pd.DataFrame(features_info)

    def __filtered_importance_names(self) -> List[str]:
        if len(self.feature_names_) == 0:
            return []

        filtered_importances = zip(self.feature_names_, self.feature_importances_)
        if self.importance_threshold is not None:
            filtered_importances = [
                (name, importance)
                for name, importance in filtered_importances
                if importance > self.importance_threshold
            ]
        if self.max_features is not None:
            filtered_importances = list(filtered_importances)[: self.max_features]
        filtered_importance_names, _ = zip(*filtered_importances)
        return list(filtered_importance_names)

    def __prepare_search_keys(self, x: pd.DataFrame):
        valid_search_keys = {}
        for column_id, meaning_type in self.search_keys.items():
            if isinstance(column_id, str):
                valid_search_keys[column_id] = meaning_type
            elif isinstance(column_id, int):
                valid_search_keys[x.columns[column_id]] = meaning_type
            else:
                msg = f"Unsupported type of key in search_keys: {type(column_id)}."
                logging.error(msg)
                raise ValueError(msg)
            if meaning_type == SearchKey.CUSTOM_KEY:
                msg = "SearchKey.CUSTOM_KEY is not supported for FeaturesEnricher."
                logging.error(msg)
                raise ValueError(msg)
            if meaning_type == SearchKey.COUNTRY and self.iso_code is not None:
                msg = "SearchKey.COUNTRY cannot be used together with a iso_code property at the same time"
                logging.error(msg)
                raise ValueError(msg)

        self.search_keys = valid_search_keys

    def __show_metrics(self):
        metrics = self.get_metrics()
        if metrics is not None:
            print(Format.GREEN + Format.BOLD + "\nQuality metrics" + Format.END)
            try:
                from IPython.display import display

                display(metrics)
            except ImportError:
                print(metrics)

            if self.__is_uplift_present_in_metrics():
                print(
                    "\nFollowing features was used for accuracy uplift estimation:",
                    ", ".join(self.passed_features),
                )

    def __show_selected_features(self):
        print(
            Format.GREEN + Format.BOLD + f"\nWe found {len(self.feature_names_)} useful feature(s) for you" + Format.END
        )
        try:
            from IPython.display import display

            display(self.features_info)
        except ImportError:
            print(self.features_info)

    def __is_uplift_present_in_metrics(self):
        uplift_presented = False

        if self._search_task is not None and self._search_task.summary is not None:
            if len(self.passed_features) > 0 and self._search_task.initial_max_uplift() is not None:
                uplift_presented = True

            max_initial_eval_set_metrics = self._search_task.get_max_initial_eval_set_metrics()

            if max_initial_eval_set_metrics is not None:
                for eval_set_metrics in max_initial_eval_set_metrics:
                    if "uplift" in eval_set_metrics:
                        uplift_presented = True

            if self._search_task.summary.status == "VALIDATION_COMPLETED":
                if len(self.passed_features) > 0 and self._search_task.validation_max_uplift() is not None:
                    uplift_presented = True

        return uplift_presented

    @staticmethod
    def _hash_row(row) -> int:
        t = tuple(row)
        m = hashlib.md5()
        for i in t:
            m.update(str(i).encode())
        return FeaturesEnricher._hex_to_int(m.hexdigest())

    @staticmethod
    def _hex_to_int(s: str) -> int:
        chars = []
        for ch in s:
            if not ch.isdecimal():
                ch = str(ord(ch) - 97)
            chars.append(ch)
        result = int("".join(chars))
        if result > sys.maxsize:
            result = result % sys.maxsize
        return result

    def __is_quality_by_metrics_low(self) -> bool:
        if self._search_task is None:
            return False
        if len(self.passed_features) > 0 and self._search_task.task_type is not None:
            max_uplift = self._search_task.initial_max_uplift()
            if self._search_task.task_type == ModelTaskType.BINARY:
                threshold = 0.002
            elif self._search_task.task_type == ModelTaskType.MULTICLASS:
                threshold = 3.0
            elif self._search_task.task_type == ModelTaskType.REGRESSION:
                threshold = 0.0
            else:
                return False
            if max_uplift is not None and max_uplift["value"] < threshold:
                return True
        elif self._search_task.task_type is not None:
            max_auc = self._search_task.initial_max_auc()
            if self._search_task.task_type == ModelTaskType.BINARY and max_auc is not None:
                if max_auc["value"] < 0.55:
                    return True
        return False

    def __check_quality(self, no_data_found: bool):
        if no_data_found or self.__is_quality_by_metrics_low():
            try:
                from IPython.display import HTML, display  # type: ignore

                display(
                    HTML(
                        "<h9>Oops, looks like we're not able to find data which gives a strong uplift for your ML "
                        "algorithm.<br>If you have ANY data which you might consider as royalty and "
                        "license-free and potentially valuable for supervised ML applications,<br> we shall be "
                        "happy to give you free individual access in exchange for sharing this data with "
                        "community.<br>Just upload your data sample right from Jupyter. We will check your data "
                        "sharing proposal and get back to you ASAP."
                    )
                )
                display(
                    HTML(
                        "<a href='https://github.com/upgini/upgini/blob/main/README.md"
                        "#share-license-free-data-with-community' "
                        "target='_blank'>How to upload your data sample from Jupyter</a>"
                    )
                )
            except ImportError:
                print("Oops, looks like we're not able to find data which gives a strong uplift for your ML algorithm.")
                print(
                    "If you have ANY data which you might consider as royalty and license-free and potentially "
                    "valuable for supervised ML applications,"
                )
                print(
                    "we shall be happy to give you free individual access in exchange for sharing this data with "
                    "community."
                )
                print(
                    "Just upload your data sample right from Jupyter. We will check your data sharing proposal and "
                    "get back to you ASAP."
                )
                print("https://github.com/upgini/upgini/blob/main/README.md#share-license-free-data-with-community")
