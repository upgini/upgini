import itertools
import logging
import os
import time
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import BaseCrossValidator

from upgini.dataset import Dataset
from upgini.http import UPGINI_API_KEY, LoggerFactory
from upgini.mdc import MDC
from upgini.metadata import (
    COUNTRY,
    DEFAULT_INDEX,
    EVAL_SET_INDEX,
    RENAMED_INDEX,
    SYSTEM_RECORD_ID,
    CVType,
    FileColumnMeaningType,
    ModelTaskType,
    RuntimeParameters,
    SearchKey,
)
from upgini.metrics import EstimatorWrapper
from upgini.search_task import SearchTask
from upgini.spinner import Spinner
from upgini.utils.country_utils import CountrySearchKeyDetector
from upgini.utils.email_utils import EmailSearchKeyDetector
from upgini.utils.format import Format
from upgini.utils.phone_utils import PhoneSearchKeyDetector
from upgini.utils.postal_code_utils import PostalCodeSearchKeyDetector
from upgini.utils.target_utils import define_task
from upgini.version_validator import validate_version


class FeaturesEnricher(TransformerMixin):
    """Retrieve external features via Upgini that are most relevant to predict your target.

    Parameters
    ----------
    search_keys: dict of str->SearchKey or int->SearchKey
        Dictionary with column names or indices mapping to key types.
        Each of this columns will be used as a search key to find features.

    country_code: str, optional (default=None)
        If defined, set this ISO-3166 country code for all rows in the dataset.

    model_task_type: ModelTaskType, optional (default=None)
        Type of training model. If not specified, the type will be autodetected.

    api_key: str, optional (default=None)
        Token to authorize search requests. You can get it on https://profile.upgini.com/.
        If not specified, the value will be read from the environment variable UPGINI_API_KEY.

    endpoint: str, optional (default=None)
        URL of Upgini API where search requests are submitted.
        If not specified, use the default value. Don't overwrite it if you are unsure.

    search_id: str, optional (default=None)
        Identifier of a previously fitted enricher to continue using it without refitting.
        If not specified, you must fit the enricher before calling transform.

    date_format: str, optional (default=None)
        Format of dates if they are represented by strings. For example: %Y-%m-%d.

    cv: CVType, optional (default=None)
        Type of cross validation: CVType.k_fold, CVType.time_series, CVType.blocked_time_series.
        Default cross validation is k-fold for regressions and stratified k-fold for classifications.
    """

    TARGET_NAME = "target"
    RANDOM_STATE = 42

    def __init__(
        self,
        search_keys: Dict[str, SearchKey],
        country_code: Optional[str] = None,
        model_task_type: Optional[ModelTaskType] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        search_id: Optional[str] = None,
        runtime_parameters: Optional[RuntimeParameters] = None,
        date_format: Optional[str] = None,
        random_state: int = 42,
        cv: Optional[CVType] = None,
        detect_missing_search_keys: bool = True,
        logs_enabled: bool = True,
    ):
        self.api_key = api_key or os.environ.get(UPGINI_API_KEY)
        if logs_enabled:
            self.logger = LoggerFactory().get_logger(endpoint, self.api_key)
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")

        validate_version(self.logger)

        self.search_keys = search_keys
        self.country_code = country_code
        self.__validate_search_keys(search_keys, search_id)
        self.model_task_type = model_task_type
        self.endpoint = endpoint
        self._search_task: Optional[SearchTask] = None
        if search_id:
            search_task = SearchTask(
                search_id,
                endpoint=self.endpoint,
                api_key=self.api_key,
            )

            print("Retrieving the specified search...")
            trace_id = str(uuid.uuid4())
            with MDC(trace_id=trace_id):
                try:
                    self.logger.info(f"FeaturesEnricher created from existing search: {search_id}")
                    self._search_task = search_task.poll_result(trace_id, quiet=True)
                    file_metadata = self._search_task.get_file_metadata(trace_id)
                    x_columns = [c.originalName or c.name for c in file_metadata.columns]
                    self.__prepare_feature_importances(trace_id, x_columns)
                    # TODO validate search_keys with search_keys from file_metadata
                    print("Search found. Now you can use transform.")
                    self.logger.info(f"Successfully initialized with search_id: {search_id}")
                except Exception as e:
                    print("Failed to retrieve the specified search.")
                    self.logger.exception(f"Failed to find search_id: {search_id}")
                    raise e

        self.runtime_parameters = runtime_parameters
        self.date_format = date_format
        self.random_state = random_state
        self.detect_missing_search_keys = detect_missing_search_keys
        self.cv = cv
        if cv is not None:
            if self.runtime_parameters is None:
                self.runtime_parameters = RuntimeParameters()
            if self.runtime_parameters.properties is None:
                self.runtime_parameters.properties = {}
            self.runtime_parameters.properties["cv_type"] = cv.name

        self.passed_features: List[str] = []
        self.features_info: pd.DataFrame = pd.DataFrame(columns=["feature_name", "shap_value", "match_percent"])
        self.feature_names_ = []
        self.feature_importances_ = []
        self.enriched_X: Optional[pd.DataFrame] = None
        self.enriched_eval_set: Optional[pd.DataFrame] = None
        self.country_added = False
        self.index_renamed = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[List[tuple]] = None,
        *,
        calculate_metrics: bool = False,
        estimator: Optional[Any] = None,
        scoring: Union[Callable, str, None] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
    ):
        """Fit to data.

        Fits transformer to `X` and `y`.

        Parameters
        ----------
        X: pandas.DataFrame of shape (n_samples, n_features)
            Input samples.

        y: array-like of shape (n_samples,)
            Target values.

        eval_set: List[tuple], optional (default=None)
            List of pairs (X, y) for validation.

        keep_input: bool, optional (default=False)
            If True, copy original input columns to the output dataframe.

        importance_threshold: float, optional (default=None)
            Minimum SHAP value to select a feature. Default value is 0.0.

        max_features: int, optional (default=None)
            Maximum number of most important features to select. If None, the number is unlimited.

        calculate_metrics: bool (default=False)
            Whether to calculate and show metrics.

        estimator: sklearn-compatible estimator, optional (default=None)
            Custom estimator for metrics calculation.

        scoring: string or callable, optional (default=None)
            A string or a scorer callable object / function with signature scorer(estimator, X, y).
            If None, the estimator's score method is used.
        """
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        with MDC(trace_id=trace_id):
            self.logger.info(f"Start fit. X shape: {X.shape}. y shape: {len(y)}")
            if eval_set:
                self.logger.info(
                    [
                        f"Eval {i} X shape: {eval_X.shape}, y shape: {len(eval_y)}"
                        for i, (eval_X, eval_y) in enumerate(eval_set)
                    ]
                )

            try:
                self.__inner_fit(
                    trace_id,
                    X,
                    y,
                    eval_set,
                    calculate_metrics=calculate_metrics,
                    estimator=estimator,
                    scoring=scoring,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                )
                self.logger.info("Fit finished successfully")
            except Exception as e:
                self.logger.exception("Failed inner fit")
                raise e
            finally:
                self.logger.info(f"Fit elapsed time: {time.time() - start_time}")
                if self.country_added and COUNTRY in self.search_keys.keys():
                    del self.search_keys[COUNTRY]
                if self.index_renamed and RENAMED_INDEX in self.search_keys.keys():
                    index_key = self.search_keys[RENAMED_INDEX]
                    self.search_keys[DEFAULT_INDEX] = index_key
                    del self.search_keys[RENAMED_INDEX]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[List[tuple]] = None,
        *,
        keep_input: bool = False,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        calculate_metrics: bool = False,
        scoring: Union[Callable, str, None] = None,
        estimator: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` and returns a transformed version of `X`.
        If keep_input is True, then all input columns are copied to the output dataframe.

        Parameters
        ----------
        X: pandas.DataFrame of shape (n_samples, n_features)
            Input samples.

        y: array-like of shape (n_samples,)
            Target values.

        eval_set: List[tuple], optional (default=None)
            List of pairs (X, y) for validation.

        keep_input: bool, optional (default=False)
            If True, copy original input columns to the output dataframe.

        importance_threshold: float, optional (default=None)
            Minimum SHAP value to select a feature. Default value is 0.0.

        max_features: int, optional (default=None)
            Maximum number of most important features to select. If None, the number is unlimited.

        calculate_metrics: bool (default=False)
            Whether to calculate and show metrics.

        estimator: sklearn-compatible estimator, optional (default=None)
            Custom estimator for metrics calculation.

        scoring: string or callable, optional (default=None)
            A string or a scorer callable object / function with signature scorer(estimator, X, y).
            If None, the estimator's score method is used.

        Returns
        -------
        X_new: pandas.DataFrame of shape (n_samples, n_features_new)
            Transformed dataframe, enriched with valuable features.
        """

        trace_id = str(uuid.uuid4())
        start_time = time.time()
        with MDC(trace_id=trace_id):
            self.logger.info(f"Start fit_transform. X shape: {X.shape}. y shape: {len(y)}")
            if eval_set:
                self.logger.info(
                    [
                        f"Eval {i} X shape: {eval_X.shape}, y shape: {len(eval_y)}"
                        for i, (eval_X, eval_y) in enumerate(eval_set)
                    ]
                )
            try:
                self.__inner_fit(
                    trace_id,
                    X,
                    y,
                    eval_set,
                    calculate_metrics=calculate_metrics,
                    scoring=scoring,
                    estimator=estimator,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                )
                self.logger.info("Fit_transform finished successfully")
            except Exception as e:
                self.logger.exception("Failed in inner_fit")
                raise e
            finally:
                self.logger.info(f"Fit elapsed time: {time.time() - start_time}")
                if self.country_added and COUNTRY in self.search_keys.keys():
                    del self.search_keys[COUNTRY]
                if self.index_renamed and RENAMED_INDEX in self.search_keys.keys():
                    index_key = self.search_keys[RENAMED_INDEX]
                    self.search_keys[DEFAULT_INDEX] = index_key
                    del self.search_keys[RENAMED_INDEX]

            return self.transform(
                X, keep_input=keep_input, importance_threshold=importance_threshold, max_features=max_features
            )

    def transform(
        self,
        X: pd.DataFrame,
        *,
        keep_input: bool = False,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
    ) -> pd.DataFrame:
        """Transform `X`.

        Returns a transformed version of `X`.
        If keep_input is True, then all input columns are copied to the output dataframe.

        Parameters
        ----------
        X: pandas.DataFrame of shape (n_samples, n_features)
            Input samples.

        keep_input: bool, optional (default=False)
            If True, copy original input columns to the output dataframe.

        importance_threshold: float, optional (default=None)
            Minimum SHAP value to select a feature. Default value is 0.0.

        max_features: int, optional (default=None)
            Maximum number of most important features to select. If None, the number is unlimited.

        Returns
        -------
        X_new: pandas.DataFrame of shape (n_samples, n_features_new)
            Transformed dataframe, enriched with valuable features.
        """

        trace_id = str(uuid.uuid4())
        start_time = time.time()
        with MDC(trace_id=trace_id):
            self.logger.info(f"Start transform. X shape: {X.shape}")
            try:
                result = self.__inner_transform(
                    trace_id, X, importance_threshold=importance_threshold, max_features=max_features
                )
                self.logger.info("Transform finished successfully")
            except Exception as e:
                self.logger.exception("Failed to inner transform")
                raise e
            finally:
                self.logger.info(f"Transform elapsed time: {time.time() - start_time}")
                if self.country_added and COUNTRY in self.search_keys.keys():
                    del self.search_keys[COUNTRY]
                if self.index_renamed and RENAMED_INDEX in self.search_keys.keys():
                    index_key = self.search_keys[RENAMED_INDEX]
                    self.search_keys["index"] = index_key
                    del self.search_keys[RENAMED_INDEX]

            if self.country_added and COUNTRY in result.columns:
                result = result.drop(columns=COUNTRY)

            if keep_input:
                return result
            else:
                return result.drop(columns=[c for c in X.columns if c in result.columns])

    def calculate_metrics(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list],
        eval_set: Optional[List[Tuple[pd.DataFrame, Any]]] = None,
        scoring: Union[Callable, str, None] = None,
        cv: Optional[BaseCrossValidator] = None,
        estimator=None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Calculate metrics

        Parameters
        ----------
        X: pandas.DataFrame of shape (n_samples, n_features)
            The same input samples that have been used for fit.

        y: array-like of shape (n_samples,)
            The same target values that have been used for fit.

        eval_set: List[tuple], optional (default=None)
            The same list of validation pairs (X, y) that have been used for fit.

        scoring: string or callable, optional (default=None)
            A string or a scorer callable object / function with signature scorer(estimator, X, y).
            If None, the estimator's score method is used.

        cv: sklearn.model_selection.BaseCrossValidator, optional (default=None)
            Custom cross validator to calculate metric on train.

        estimator: sklearn-compatible estimator, optional (default=None)
            Custom estimator for metrics calculation. If not passed then CatBoost will be used.

        importance_threshold: float, optional (default=None)
            Minimum SHAP value to select a feature. Default value is 0.0.

        max_features: int, optional (default=None)
            Maximum number of most important features to select. If None, the number is unlimited.

        Returns
        -------
        metrics: pandas.DataFrame
            Dataframe with metrics calculated on train and validation datasets.
        """

        trace_id = trace_id or str(uuid.uuid4())
        start_time = time.time()
        with MDC(trace_id=trace_id):
            try:
                if self._search_task is None or self._search_task.initial_max_hit_rate() is None:
                    raise Exception("Fit the enricher before calling calculate_metrics.")
                if self.enriched_X is None:
                    raise Exception("Metrics calculation isn't possible after restart. Please fit the enricher again.")

                if not isinstance(X, pd.DataFrame):
                    raise Exception(f"Unsupported type of X: {type(X)}. Use pandas.DataFrame.")

                if isinstance(y, np.ndarray) or isinstance(y, list):
                    y = pd.Series(y, name="target")
                elif not isinstance(y, pd.Series):
                    raise Exception(f"Unsupported type of y: {type(y)}. Use pandas.Series, numpy.ndarray or list.")

                Xy = X.copy()
                Xy["target"] = y
                self.__log_debug_information(Xy)

                filtered_columns = self.__filtered_columns(
                    X.columns.to_list(), importance_threshold, max_features, only_features=True
                )

                fitting_X = X.drop(columns=[col for col in self.search_keys.keys() if col in X.columns])
                fitting_enriched_X = self.enriched_X[filtered_columns]

                if fitting_X.shape[1] == 0 and fitting_enriched_X.shape[1] == 0:
                    print("WARN: No features to calculate metrics.")
                    self.logger.warning("No client or relevant ADS features found to calculate metrics")
                    return None

                model_task_type = self.model_task_type or define_task(pd.Series(y), self.logger, silent=True)

                # shuffle Kfold for case when date/datetime keys are not presented
                key_types = self.search_keys.values()
                shuffle = True
                if SearchKey.DATE in key_types or SearchKey.DATETIME in key_types:
                    shuffle = False

                _cv = cv or self.cv

                self.logger.info("Start calculating metrics")
                print("Calculating metrics...")

                with Spinner():
                    # 1 If client features are presented - fit and predict with KFold CatBoost model
                    # on etalon features and calculate baseline metric
                    etalon_metric = None
                    if fitting_X.shape[1] > 0:
                        etalon_metric = EstimatorWrapper.create(
                            estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                        ).cross_val_predict(fitting_X, y)

                    # 2 Fit and predict with KFold Catboost model on enriched tds
                    # and calculate final metric (and uplift)
                    if set(fitting_X.columns) != set(fitting_enriched_X.columns):
                        wrapper = EstimatorWrapper.create(
                            estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                        )
                        enriched_metric = wrapper.cross_val_predict(fitting_enriched_X, y)
                        metric = wrapper.metric_name
                        uplift = None
                        if etalon_metric is not None:
                            uplift = (enriched_metric - etalon_metric) * wrapper.multiplier
                    else:
                        enriched_metric = etalon_metric
                        metric = EstimatorWrapper.create(
                            estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                        ).metric_name
                        uplift = 0.0

                    metrics = [
                        {
                            "segment": "train",
                            "match_rate": self._search_task.initial_max_hit_rate()["value"],  # type: ignore
                            f"baseline {metric}": etalon_metric,
                            f"enriched {metric}": enriched_metric,
                            "uplift": uplift,
                        }
                    ]

                    # 3 If eval_set is presented - fit final model on train enriched data and score each
                    # validation dataset and calculate final metric (and uplift)
                    max_initial_eval_set_metrics = self._search_task.get_max_initial_eval_set_metrics()
                    if eval_set is not None and self.enriched_eval_set is not None:
                        # Fit models
                        etalon_model = None
                        if fitting_X.shape[1] > 0:
                            etalon_model = EstimatorWrapper.create(
                                deepcopy(estimator),
                                self.logger,
                                model_task_type,
                                _cv,
                                scoring,
                                shuffle,
                                self.random_state,
                            )
                            etalon_model.fit(fitting_X, y)

                        if set(fitting_X.columns) != set(fitting_enriched_X.columns):
                            enriched_model = EstimatorWrapper.create(
                                deepcopy(estimator),
                                self.logger,
                                model_task_type,
                                _cv,
                                scoring,
                                shuffle,
                                self.random_state,
                            )
                            enriched_model.fit(fitting_enriched_X, y)
                        elif etalon_model is None:
                            self.logger.error("No client or ADS features, but first validation didn't work")
                            print("WARN: No features to calculate metrics.")
                            return None
                        else:
                            enriched_model = etalon_model

                        for idx, eval_pair in enumerate(eval_set):
                            eval_hit_rate = (
                                max_initial_eval_set_metrics[idx]["hit_rate"] * 100.0
                                if max_initial_eval_set_metrics
                                else None
                            )
                            eval_X = eval_pair[0]
                            eval_X = eval_X.drop(
                                columns=[col for col in self.search_keys.keys() if col in eval_X.columns]
                            )
                            enriched_eval_X = self.enriched_eval_set[self.enriched_eval_set[EVAL_SET_INDEX] == idx + 1]
                            enriched_eval_X = enriched_eval_X[filtered_columns]
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
                    self.logger.info("Metrics calculation finished successfully")
                    return pd.DataFrame(metrics).set_index("segment").rename_axis("")
            except Exception as e:
                self.logger.exception("Failed to calculate metrics")
                raise e
            finally:
                self.logger.info(f"Calculating metrics elapsed time: {time.time() - start_time}")

    def get_search_id(self) -> Optional[str]:
        """Returns search_id of the fitted enricher. Not available before a successful fit."""
        return self._search_task.search_task_id if self._search_task else None

    def get_features_info(self) -> pd.DataFrame:
        """Returns pandas.DataFrame with SHAP values and other info for each feature."""
        if self._search_task is None or self._search_task.summary is None:
            msg = "Fit the enricher or pass search_id before calling get_features_info."
            self.logger.warning(msg)
            raise NotFittedError(msg)

        return self.features_info

    def __inner_transform(
        self,
        trace_id,
        X: pd.DataFrame,
        *,
        importance_threshold: Optional[float],
        max_features: Optional[int],
        silent_mode: bool = False,
    ) -> pd.DataFrame:
        with MDC(trace_id=trace_id):
            if self._search_task is None:
                msg = "Fit the enricher or pass search_id before calling transform."
                self.logger.error(msg)
                raise NotFittedError(msg)
            if not isinstance(X, pd.DataFrame):
                msg = f"Unsupported type of X: {type(X)}. Use pandas.DataFrame."
                self.logger.error(msg)
                raise TypeError(msg)

            if len(set(X.columns)) != len(X.columns):
                raise ValueError("X contains duplicate column names. Please rename or drop them.")

            self.__prepare_search_keys(X)

            df = X.copy()

            self.__log_debug_information(df)

            df = self.__handle_index_search_keys(df)

            self.__check_string_dates(X)
            df = self.__add_country_code(df)

            meaning_types = {col: key.value for col, key in self.search_keys.items()}
            search_keys = self.__using_search_keys()
            feature_columns = [column for column in df.columns if column not in self.search_keys.keys()]

            df[SYSTEM_RECORD_ID] = [hash(tuple(row)) for row in df[search_keys.keys()].values]  # type: ignore
            meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID

            combined_search_keys = []
            for L in range(1, len(search_keys.keys()) + 1):
                for subset in itertools.combinations(search_keys.keys(), L):
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
                logger=self.logger,
            )
            dataset.meaning_types = meaning_types
            dataset.search_keys = combined_search_keys
            validation_task = self._search_task.validation(
                trace_id,
                dataset,
                extract_features=True,
                runtime_parameters=self.runtime_parameters,
                silent_mode=silent_mode,
            )

            if not silent_mode:
                print("Collecting selected features...")
                with Spinner():
                    result, _ = self.__enrich(df, validation_task.get_all_validation_raw_features(trace_id), X.index)
            else:
                result, _ = self.__enrich(df, validation_task.get_all_validation_raw_features(trace_id), X.index)

            input_columns = [c for c in X.columns if c in result.columns]
            filtered_columns = self.__filtered_columns(input_columns, importance_threshold, max_features)

            return result[filtered_columns]

    def __validate_search_keys(self, search_keys: Dict[str, SearchKey], search_id: Optional[str]):
        if len(search_keys) == 0:
            if search_id:
                self.logger.error(f"search_id {search_id} provided without search_keys")
                raise ValueError(
                    "When search_id is passed, search_keys must be set to the same value that have been used for fit."
                )
            else:
                self.logger.error("search_keys not provided")
                raise ValueError("At least one column must be provided in search_keys.")

        key_types = search_keys.values()

        if SearchKey.DATE in key_types and SearchKey.DATETIME in key_types:
            msg = "DATE and DATETIME search keys cannot be used simultaneously. Choose one to keep."
            self.logger.error(msg)
            raise Exception(msg)

        if SearchKey.EMAIL in key_types and SearchKey.HEM in key_types:
            msg = "EMAIL and HEM search keys cannot be used simultaneously. Choose one to keep."
            self.logger.error(msg)
            raise Exception(msg)

        if SearchKey.POSTAL_CODE in key_types and SearchKey.COUNTRY not in key_types and self.country_code is None:
            msg = "COUNTRY search key must be provided if POSTAL_CODE is present."
            self.logger.error(msg)
            raise Exception(msg)

        for key_type in SearchKey.__members__.values():
            if key_type != SearchKey.CUSTOM_KEY and list(key_types).count(key_type) > 1:
                msg = f"Search key {key_type} is presented multiple times."
                self.logger.error(msg)
                raise Exception(msg)

        non_personal_keys = set(SearchKey.__members__.values()) - set(SearchKey.personal_keys())
        if not self.__is_registered and len(set(key_types).intersection(non_personal_keys)) == 0:
            msg = (
                "No API key found and all search keys require registration. "
                "You can use DATE, COUNTRY and POSTAL_CODE keys for free search without registration. "
                "Or provide the API key either directly or via the environment variable UPGINI_API_KEY."
            )
            self.logger.error(msg + f" Provided search keys: {key_types}")
            raise Exception(msg)

    @property
    def __is_registered(self) -> bool:
        return self.api_key is not None and self.api_key != ""

    def __inner_fit(
        self,
        trace_id: str,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list, None],
        eval_set: Optional[List[tuple]],
        *,
        calculate_metrics: bool,
        scoring: Union[Callable, str, None],
        estimator: Optional[Any],
        importance_threshold: Optional[float],
        max_features: Optional[int],
    ) -> pd.DataFrame:
        self.enriched_X = None
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Unsupported type of X: {type(X)}. Use pandas.DataFrame.")
        if not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not isinstance(y, list):
            raise TypeError(f"Unsupported type of y: {type(y)}. Use pandas.Series, numpy.ndarray or list.")

        if isinstance(y, pd.Series):
            y_array = y.values
        elif isinstance(y, np.ndarray):
            y_array = y
        else:
            y_array = np.array(y)

        if len(np.unique(y_array)) < 2:
            raise ValueError("y is a constant. Finding relevant features requires a non-constant y.")

        if X.shape[0] != len(y_array):
            raise ValueError(f"X and y contain different number of samples: {X.shape[0]}, {len(y_array)}.")

        if len(set(X.columns)) != len(X.columns):
            raise ValueError("X contains duplicate column names. Please rename or drop them.")

        self.__prepare_search_keys(X)

        df: pd.DataFrame = X.copy()  # type: ignore
        df[self.TARGET_NAME] = y_array

        self.__log_debug_information(df)

        df = self.__handle_index_search_keys(df)

        self.__check_string_dates(df)

        df = self.__correct_target(df)

        model_task_type = self.model_task_type or define_task(df[self.TARGET_NAME], self.logger)

        if eval_set is not None and len(eval_set) > 0:
            df[EVAL_SET_INDEX] = 0
            for idx, eval_pair in enumerate(eval_set):
                if len(eval_pair) != 2:
                    raise TypeError(
                        f"eval_set contains a tuple of size {len(eval_pair)}. It should contain only pairs of X and y."
                    )
                eval_X = eval_pair[0]
                eval_y = eval_pair[1]
                if not isinstance(eval_X, pd.DataFrame):
                    raise TypeError(f"Unsupported type of X in eval_set: {type(X)}. Use pandas.DataFrame.")
                if eval_X.columns.to_list() != X.columns.to_list():
                    raise Exception("The columns in eval_set are different from the columns in X.")
                if (
                    not isinstance(eval_y, pd.Series)
                    and not isinstance(eval_y, np.ndarray)
                    and not isinstance(eval_y, list)
                ):
                    raise TypeError(
                        f"Unsupported type of y in eval_set: {type(y)}. Use pandas.Series, numpy.ndarray or list."
                    )
                eval_df: pd.DataFrame = eval_X.copy()  # type: ignore
                eval_df[self.TARGET_NAME] = pd.Series(eval_y)
                eval_df[EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_df], ignore_index=True)

        df = self.__add_country_code(df)

        non_feature_columns = [self.TARGET_NAME, EVAL_SET_INDEX] + list(self.search_keys.keys())
        meaning_types = {
            **{col: key.value for col, key in self.search_keys.items()},
            **{str(c): FileColumnMeaningType.FEATURE for c in df.columns if c not in non_feature_columns},
        }
        meaning_types[self.TARGET_NAME] = FileColumnMeaningType.TARGET
        if eval_set is not None and len(eval_set) > 0:
            meaning_types[EVAL_SET_INDEX] = FileColumnMeaningType.EVAL_SET_INDEX

        search_keys = self.__using_search_keys()

        df = self.__add_fit_system_record_id(df, meaning_types)

        combined_search_keys = []
        for L in range(1, len(search_keys.keys()) + 1):
            for subset in itertools.combinations(search_keys.keys(), L):
                combined_search_keys.append(subset)

        dataset = Dataset(
            "tds_" + str(uuid.uuid4()),
            df=df,  # type: ignore
            model_task_type=model_task_type,  # type: ignore
            endpoint=self.endpoint,  # type: ignore
            api_key=self.api_key,  # type: ignore
            date_format=self.date_format,  # type: ignore
            random_state=self.random_state,  # type: ignore
            logger=self.logger,
        )
        dataset.meaning_types = meaning_types
        dataset.search_keys = combined_search_keys

        self.passed_features = [
            column for column, meaning_type in meaning_types.items() if meaning_type == FileColumnMeaningType.FEATURE
        ]

        self._search_task = dataset.search(
            trace_id,
            extract_features=True,
            runtime_parameters=self.runtime_parameters,
        )

        self.__prepare_feature_importances(trace_id, list(X.columns))

        self.__show_selected_features()

        try:
            self.enriched_X, self.enriched_eval_set = self.__enrich(
                df, self._search_task.get_all_initial_raw_features(trace_id), X.index
            )
        except Exception as e:
            self.logger.exception("Failed to download features")
            raise e

        if calculate_metrics:
            self.__show_metrics(X, y, eval_set, scoring, estimator, importance_threshold, max_features, trace_id)

        filtered_columns = self.__filtered_columns(X.columns.to_list(), importance_threshold, max_features)

        return self.enriched_X[filtered_columns]

    def __log_debug_information(self, df: pd.DataFrame):
        self.logger.info(f"Search keys: {self.search_keys}")
        self.logger.info(f"Country code: {self.country_code}")
        self.logger.info(f"Model task type: {self.model_task_type}")
        resolved_api_key = self.api_key or os.environ.get(UPGINI_API_KEY)
        self.logger.info(f"Api key presented?: {resolved_api_key is not None and resolved_api_key != ''}")
        self.logger.info(f"Endpoint: {self.endpoint}")
        self.logger.info(f"Runtime parameters: {self.runtime_parameters}")
        self.logger.info(f"Date format: {self.date_format}")
        self.logger.info(f"CV: {self.cv}")
        self.logger.info(f"Random state: {self.random_state}")
        self.logger.info(f"First 10 rows of the dataset:\n{df.head(10)}")

    def __handle_index_search_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        index_names = df.index.names if df.index.names != [None] else [DEFAULT_INDEX]
        index_search_keys = set(index_names).intersection(self.search_keys.keys())
        if len(index_search_keys) > 0:
            for index_name in index_search_keys:
                if index_name not in df.columns:
                    if df.index.names == [None]:
                        df[index_name] = df.index
                    else:
                        df[index_name] = df.index.get_level_values(index_name)
            df = df.reset_index(drop=True)
            if DEFAULT_INDEX in index_names:
                df = df.rename(columns={DEFAULT_INDEX: RENAMED_INDEX})
                self.search_keys[RENAMED_INDEX] = self.search_keys[DEFAULT_INDEX]
                del self.search_keys[DEFAULT_INDEX]
                self.index_renamed = True
        else:
            if DEFAULT_INDEX in df.columns:
                print(f"Column name `{DEFAULT_INDEX}` is reserved and will be renamed to `{RENAMED_INDEX}`")
                df = df.rename(columns={DEFAULT_INDEX: RENAMED_INDEX})
                self.index_renamed = True
            df = df.reset_index(drop=True)
        return df

    def __using_search_keys(self) -> Dict[str, SearchKey]:
        return {col: key for col, key in self.search_keys.items() if key != SearchKey.CUSTOM_KEY}

    def __is_date_key_present(self) -> bool:
        return len({SearchKey.DATE, SearchKey.DATETIME}.intersection(self.search_keys.values())) != 0

    def __add_fit_system_record_id(
        self, df: pd.DataFrame, meaning_types: Dict[str, FileColumnMeaningType]
    ) -> pd.DataFrame:
        if (self.cv is None or self.cv == CVType.k_fold) and self.__is_date_key_present():
            date_column = [
                col
                for col, t in meaning_types.items()
                if t in [FileColumnMeaningType.DATE, FileColumnMeaningType.DATETIME]
            ]
            df.sort_values(by=date_column, kind="mergesort")
            pass
        df = df.reset_index(drop=True)
        df = df.reset_index()
        df = df.rename(columns={DEFAULT_INDEX: SYSTEM_RECORD_ID})
        meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID
        return df

    def __check_string_dates(self, df: pd.DataFrame):
        for column, search_key in self.search_keys.items():
            if search_key in [SearchKey.DATE, SearchKey.DATETIME] and is_string_dtype(df[column]):
                if self.date_format is None or len(self.date_format) == 0:
                    msg = (
                        f"Date column `{column}` is of string type, but date_format is not specified. "
                        "Please convert column to datetime type or pass date_format."
                    )
                    self.logger.error(msg)
                    raise Exception(msg)

    def __correct_target(self, df: pd.DataFrame) -> pd.DataFrame:
        target = df[self.TARGET_NAME]
        if is_string_dtype(target):
            maybe_numeric_target = pd.to_numeric(target, errors="coerce")
            # If less than 5% is non numeric then leave this rows with NaN target and later it will be dropped
            if maybe_numeric_target.isna().sum() <= len(df) * 0.05:
                self.logger.info("Target column has less than 5% non numeric values. Change non numeric values to NaN")
                df[self.TARGET_NAME] = maybe_numeric_target
            else:
                # Suppose that target is multiclass and mark rows with unique target with NaN for later dropping
                self.logger.info("Target has more than 5% non numeric values. Change unique values to NaN")
                vc = target.value_counts()
                uniq_values = vc[vc == 1].index.to_list()
                for uniq_val in uniq_values:
                    df[self.TARGET_NAME] = np.where(df[self.TARGET_NAME] == uniq_val, np.nan, df[self.TARGET_NAME])

        return df

    def __add_country_code(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.country_code and SearchKey.COUNTRY not in self.search_keys.values():
            self.logger.info(f"Add COUNTRY column with {self.country_code} value")
            df[COUNTRY] = self.country_code
            self.search_keys[COUNTRY] = SearchKey.COUNTRY
            self.country_added = True

        if SearchKey.COUNTRY in self.search_keys.values():
            country_column = list(self.search_keys.keys())[list(self.search_keys.values()).index(SearchKey.COUNTRY)]
            df = CountrySearchKeyDetector.convert_country_to_iso_code(df, country_column)

        return df

    def __enrich(
        self,
        df: pd.DataFrame,
        result_features: Optional[pd.DataFrame],
        original_index: pd.Index,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if result_features is None:
            self.logger.error(f"result features not found by search_task_id: {self.get_search_id()}")
            raise RuntimeError("Search engine crashed on this request.")
        result_features = (
            result_features.drop(columns=EVAL_SET_INDEX)
            if EVAL_SET_INDEX in result_features.columns
            else result_features
        )
        df_without_target = df.drop(columns=self.TARGET_NAME) if self.TARGET_NAME in df.columns else df

        dup_features = [c for c in df_without_target.columns if c in result_features.columns and c != SYSTEM_RECORD_ID]
        if len(dup_features) > 0:
            self.logger.error(f"X contain columns with same name as returned from backend: {dup_features}")
            raise Exception(
                "Columns set for transform method should be the same as for fit method, please check input dataframe. "
                f"These columns are different: {dup_features}"
            )

        result = pd.merge(
            df_without_target,
            result_features,
            left_on=SYSTEM_RECORD_ID,
            right_on=SYSTEM_RECORD_ID,
            how="left",
            # suffixes=("_ads", "")
        )

        if EVAL_SET_INDEX in result.columns:
            result_train = result[result[EVAL_SET_INDEX] == 0]
            result_eval_set = result[result[EVAL_SET_INDEX] != 0]
            result_train = result_train.drop(columns=EVAL_SET_INDEX)
        else:
            result_train = result
            result_eval_set = None

        result_train.index = original_index
        if SYSTEM_RECORD_ID in result.columns:
            result_train = result_train.drop(columns=SYSTEM_RECORD_ID)
            if result_eval_set is not None:
                result_eval_set = result_eval_set.drop(columns=SYSTEM_RECORD_ID)

        return result_train, result_eval_set

    def __prepare_feature_importances(self, trace_id: str, x_columns: List[str]):
        if self._search_task is None:
            raise NotFittedError("Fit the enricher or pass search_id before calling transform.")
        importances = self._search_task.initial_features(trace_id)

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

    def __filtered_importance_names(
        self, importance_threshold: Optional[float], max_features: Optional[int]
    ) -> List[str]:
        if len(self.feature_names_) == 0:
            return []

        filtered_importances = list(zip(self.feature_names_, self.feature_importances_))
        # temporary workaround. generate this column later
        filtered_importances = [
            (name, importance) for name, importance in filtered_importances if name != "email_domain"
        ]
        if importance_threshold is not None:
            filtered_importances = [
                (name, importance) for name, importance in filtered_importances if importance > importance_threshold
            ]
        if max_features is not None:
            filtered_importances = list(filtered_importances)[:max_features]
        if len(filtered_importances) == 0:
            return []
        filtered_importance_names, _ = zip(*filtered_importances)
        return list(filtered_importance_names)

    def __prepare_search_keys(self, x: pd.DataFrame):
        valid_search_keys = {}
        api_key = self.api_key or os.environ.get(UPGINI_API_KEY)
        is_registered = api_key is not None and api_key != ""
        for column_id, meaning_type in self.search_keys.items():
            column_name = None
            if isinstance(column_id, str):
                if column_id not in x.columns:
                    raise ValueError(f"Key `{column_id}` in search_keys was not found in X: {list(x.columns)}.")
                column_name = column_id
                valid_search_keys[column_name] = meaning_type
            elif isinstance(column_id, int):
                if column_id >= x.shape[1]:
                    raise ValueError(
                        f"Index {column_id} in search_keys is out of bounds for {x.shape[1]} columns of X."
                    )
                column_name = x.columns[column_id]
                valid_search_keys[column_name] = meaning_type
            else:
                raise ValueError(f"Unsupported type of key in search_keys: {type(column_id)}.")

            if meaning_type == SearchKey.COUNTRY and self.country_code is not None:
                msg = "SearchKey.COUNTRY and iso_code cannot be used simultaneously."
                # self.logger.error(msg)
                raise ValueError(msg)

            if not is_registered and meaning_type in SearchKey.personal_keys():
                msg = f"Search key {meaning_type} cannot be used without API key. It will be ignored."
                self.logger.warning(msg)
                print("WARNING: " + msg)
                valid_search_keys[column_name] = SearchKey.CUSTOM_KEY

        if self.detect_missing_search_keys:
            valid_search_keys = self.__detect_missing_search_keys(x, valid_search_keys)

        self.search_keys = valid_search_keys

        using_keys = self.__using_search_keys()
        if (
            len(using_keys.values()) == 1
            and self.country_code is None
            and next(iter(using_keys.values())) == SearchKey.DATE
        ):
            msg = (
                "WARNING: You have started the search with the DATE key only. "
                "Try to add the COUNTRY and/or POSTAL_CODE keys to your dataset so that the search engine gets access "
                "to the additional data sources. Get details on "
                "https://github.com/upgini/upgini#readme"
            )
            print(msg)

        maybe_date = [k for k, v in using_keys.items() if v in [SearchKey.DATE, SearchKey.DATETIME]]
        if (self.cv is None or self.cv == CVType.k_fold) and len(maybe_date) > 0:
            date_column = next(iter(maybe_date))
            if x[date_column].nunique() > 0.9 * len(x):
                msg = (
                    "WARNING: Looks like your training dataset is a time series. "
                    "We recommend to set `cv=CVType.time_series` for the best results."
                )
                print(msg)

    def __show_metrics(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list],
        eval_set: Optional[List[Tuple[pd.DataFrame, Any]]],
        scoring: Union[Callable, str, None],
        estimator: Optional[Any],
        importance_threshold: Optional[float],
        max_features: Optional[int],
        trace_id: str,
    ):
        metrics = self.calculate_metrics(
            X,
            y,
            eval_set,
            scoring=scoring,
            estimator=estimator,
            importance_threshold=importance_threshold,
            max_features=max_features,
            trace_id=trace_id,
        )
        if metrics is not None:
            msg = "\nQuality metrics"

            try:
                from IPython.display import display

                print(Format.GREEN + Format.BOLD + msg + Format.END)
                display(metrics)
            except ImportError:
                print(msg)
                print(metrics)

    def __show_selected_features(self):
        search_keys = self.__using_search_keys().keys()
        msg = f"\n{len(self.feature_names_)} relevant feature(s) found with the search keys: {list(search_keys)}."

        try:
            from IPython.display import display

            print(Format.GREEN + Format.BOLD + msg + Format.END)
            display(self.features_info.head(60))
        except ImportError:
            print(msg)
            print(self.features_info.head(60))

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

    def __validate_importance_threshold(self, importance_threshold: Optional[float]) -> float:
        try:
            return float(importance_threshold) if importance_threshold is not None else 0.0
        except ValueError:
            self.logger.exception(f"Invalid importance_threshold provided: {importance_threshold}")
            raise ValueError("importance_threshold must be float.")

    def __validate_max_features(self, max_features: Optional[int]) -> int:
        try:
            return int(max_features) if max_features is not None else 400
        except ValueError:
            self.logger.exception(f"Invalid max_features provided: {max_features}")
            raise ValueError("max_features must be int.")

    def __filtered_columns(
        self,
        x_columns: List[str],
        importance_threshold: Optional[float],
        max_features: Optional[int],
        only_features: bool = False,
    ) -> List[str]:
        importance_threshold = self.__validate_importance_threshold(importance_threshold)
        max_features = self.__validate_max_features(max_features)

        exclude_columns = list(self.search_keys.keys()) if only_features else []

        return sorted(
            list(
                set(
                    [col for col in x_columns if col not in exclude_columns]
                    + self.__filtered_importance_names(importance_threshold, max_features)
                )
            )
        )

    def __detect_missing_search_keys(self, df: pd.DataFrame, search_keys: Dict[str, SearchKey]) -> Dict[str, SearchKey]:
        sample = df.head(100)

        if SearchKey.POSTAL_CODE not in search_keys.values():
            maybe_key = PostalCodeSearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                search_keys[maybe_key] = SearchKey.POSTAL_CODE
                self.logger.info(f"Autodetected search key POSTAL_CODE in column {maybe_key}")
                msg = (
                    f"Postal codes detected in column `{maybe_key}`. It will be used as a search key. "
                    "Read how to turn off the automatic detection of search keys: "
                    "https://github.com/upgini/upgini#readme"
                )
                print(msg)

        if SearchKey.COUNTRY not in search_keys.values() and self.country_code is None:
            maybe_key = CountrySearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                search_keys[maybe_key] = SearchKey.COUNTRY
                self.logger.info(f"Autodetected search key COUNTRY in column {maybe_key}")
                msg = (
                    f"Countries detected in column `{maybe_key}`. It will be used as a search key. "
                    "Read how to turn off the automatic detection of search keys: "
                    "https://github.com/upgini/upgini#readme"
                )
                print(msg)

        if SearchKey.EMAIL not in search_keys.values() and SearchKey.HEM not in search_keys.values():
            maybe_key = EmailSearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                if self.__is_registered:
                    search_keys[maybe_key] = SearchKey.EMAIL
                    self.logger.info(f"Autodetected search key EMAIL in column {maybe_key}")
                else:
                    self.logger.info(
                        f"Autodetected search key EMAIL in column {maybe_key}. But not used because not registered user"
                    )
                msg = (
                    f"Emails detected in column `{maybe_key}`. It will be used as a search key. "
                    "Read how to turn off the automatic detection of search keys: "
                    "https://github.com/upgini/upgini#readme"
                )
                print(msg)

        if SearchKey.PHONE not in search_keys.values():
            maybe_key = PhoneSearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                if self.__is_registered:
                    search_keys[maybe_key] = SearchKey.PHONE
                    self.logger.info(f"Autodetected search key PHONE in column {maybe_key}")
                else:
                    self.logger.info(
                        f"Autodetected search key PHONE in column {maybe_key}. But not used because not registered user"
                    )
                msg = (
                    f"Phone numbers detected in column `{maybe_key}`. It will be used as a search key. "
                    "Read how to turn off the automatic detection of search keys: "
                    "https://github.com/upgini/upgini#readme"
                )
                print(msg)

        return search_keys
