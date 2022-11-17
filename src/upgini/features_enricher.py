import itertools
import logging
import os
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import BaseCrossValidator

from upgini.dataset import Dataset
from upgini.errors import ValidationError
from upgini.http import UPGINI_API_KEY, LoggerFactory
from upgini.mdc import MDC
from upgini.metadata import (
    COUNTRY,
    DEFAULT_INDEX,
    EVAL_SET_INDEX,
    ORIGINAL_INDEX,
    RENAMED_INDEX,
    SYSTEM_RECORD_ID,
    TARGET,
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
from upgini.utils.features_validator import FeaturesValidator
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

    shared_datasets: list of str, optional (default=None)
        List of private shared dataset ids for custom search
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
        shared_datasets: Optional[List[str]] = None,
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
        self.features_info: pd.DataFrame = pd.DataFrame(
            columns=["provider", "source", "feature name", "shap value", "coverage %", "type", "feature type"]
        )
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
        if shared_datasets is not None:
            if self.runtime_parameters is None:
                self.runtime_parameters = RuntimeParameters()
            if self.runtime_parameters.properties is None:
                self.runtime_parameters.properties = {}
            self.runtime_parameters.properties["shared_datasets"] = ",".join(shared_datasets)

        self.passed_features: List[str] = []
        self.feature_names_ = []
        self.feature_importances_ = []
        self.enriched_X: Optional[pd.DataFrame] = None
        self.enriched_eval_sets: Dict[int, pd.DataFrame] = dict()
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
                error_message = "Failed on inner fit" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                self._dump_python_libs()
                self.__display_slack_community_link()
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
                error_message = "Failed on inner fit" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                self._dump_python_libs()
                self.__display_slack_community_link()
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
                error_message = "Failed on inner transform" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                self._dump_python_libs()
                self.__display_slack_community_link()
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
        silent: bool = False,
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
                if self._search_task is None or self._search_task.initial_max_hit_rate_v2() is None:
                    raise ValidationError("Fit the enricher before calling calculate_metrics.")
                if self.enriched_X is None:
                    raise ValidationError(
                        "Metrics calculation isn't possible after restart. Please fit the enricher again."
                    )

                self._validate_X(X)
                y_array = self._validate_y(X, y)

                # TODO check that X and y are the same as on the fit

                self.logger.info("Start calculating metrics")
                print("Calculating metrics...")

                self.__log_debug_information(X, y, eval_set)

                X_sampled, y_sampled = self._sample_X_and_y(X, y_array, self.enriched_X)
                self.logger.info(f"Shape of enriched_X: {self.enriched_X.shape}")
                self.logger.info(f"Shape of X after sampling: {X_sampled.shape}")
                self.logger.info(f"Shape of y after sampling: {len(y_sampled)}")
                X_sorted, y_sorted = self._sort_by_date(X_sampled, y_sampled)
                enriched_X_sorted, enriched_y_sorted = self._sort_by_date(self.enriched_X, y_sampled)

                client_features = [c for c in X.columns if c not in self.search_keys.keys()]

                filtered_client_features = self.__filtered_client_features(client_features)

                filtered_enriched_features = self.__filtered_enriched_features(
                    importance_threshold,
                    max_features,
                )

                fitting_X = X_sorted[filtered_client_features].copy()
                fitting_enriched_X = enriched_X_sorted[filtered_client_features + filtered_enriched_features].copy()

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

                wrapper = EstimatorWrapper.create(
                    estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                )
                metric = wrapper.metric_name
                multiplier = wrapper.multiplier

                with Spinner():
                    # 1 If client features are presented - fit and predict with KFold CatBoost model
                    # on etalon features and calculate baseline metric
                    etalon_metric = None
                    baseline_estimator = None
                    if fitting_X.shape[1] > 0:
                        baseline_estimator = EstimatorWrapper.create(
                            estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                        )
                        etalon_metric = baseline_estimator.cross_val_predict(fitting_X, y_sorted)

                    # 2 Fit and predict with KFold Catboost model on enriched tds
                    # and calculate final metric (and uplift)
                    enriched_estimator = None
                    if set(fitting_X.columns) != set(fitting_enriched_X.columns):
                        enriched_estimator = EstimatorWrapper.create(
                            estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                        )
                        enriched_metric = enriched_estimator.cross_val_predict(fitting_enriched_X, enriched_y_sorted)
                        if etalon_metric is not None:
                            uplift = (enriched_metric - etalon_metric) * multiplier
                        else:
                            uplift = None
                    else:
                        enriched_metric = None
                        uplift = None

                    train_metrics = {
                        "segment": "train",
                        "match_rate": self._search_task.initial_max_hit_rate_v2(),
                    }
                    if etalon_metric is not None:
                        train_metrics[f"baseline {metric}"] = etalon_metric
                    if enriched_metric is not None:
                        train_metrics[f"enriched {metric}"] = enriched_metric
                    if uplift is not None:
                        train_metrics["uplift"] = uplift
                    metrics = [train_metrics]

                    # 3 If eval_set is presented - fit final model on train enriched data and score each
                    # validation dataset and calculate final metric (and uplift)
                    max_initial_eval_set_hit_rate = self._search_task.get_max_initial_eval_set_hit_rate_v2()
                    if eval_set is not None:
                        if len(self.enriched_eval_sets) != len(eval_set):
                            raise ValidationError(
                                "Count of eval_set datasets on fit and on calculation metrics differs: "
                                f"fit: {len(self.enriched_eval_sets)}, calculation metrics: {len(eval_set)}"
                            )
                        # TODO check that eval_set is the same as on the fit

                        for idx, eval_pair in enumerate(eval_set):
                            eval_hit_rate = max_initial_eval_set_hit_rate[idx + 1]

                            eval_X, eval_y_array = self._validate_eval_set_pair(X, eval_pair)
                            enriched_eval_X = self.enriched_eval_sets[idx + 1]

                            sampled_eval_X, sampled_eval_y = self._sample_X_and_y(eval_X, eval_y_array, enriched_eval_X)
                            self.logger.info(f"Shape of enriched_eval_X: {enriched_eval_X.shape}")
                            self.logger.info(f"Shape of eval_X_{idx} after sampling: {sampled_eval_X.shape}")
                            self.logger.info(f"Shape of eval_y_{idx} after sampling: {len(sampled_eval_y)}")
                            eval_X_sorted, eval_y_sorted = self._sort_by_date(sampled_eval_X, sampled_eval_y)
                            eval_X_sorted = eval_X_sorted[filtered_client_features].copy()

                            enriched_eval_X_sorted, enriched_y_sorted = self._sort_by_date(
                                enriched_eval_X, sampled_eval_y
                            )
                            enriched_eval_X_sorted = enriched_eval_X_sorted[
                                filtered_client_features + filtered_enriched_features
                            ].copy()

                            if baseline_estimator is not None:
                                etalon_eval_metric = baseline_estimator.calculate_metric(eval_X_sorted, eval_y_sorted)
                            else:
                                etalon_eval_metric = None

                            if enriched_estimator is not None:
                                enriched_eval_metric = enriched_estimator.calculate_metric(
                                    enriched_eval_X_sorted, enriched_y_sorted
                                )
                            else:
                                enriched_eval_metric = None

                            if etalon_eval_metric is not None and enriched_eval_metric is not None:
                                eval_uplift = (enriched_eval_metric - etalon_eval_metric) * multiplier
                            else:
                                eval_uplift = None

                            eval_metrics = {
                                "segment": f"eval {idx + 1}",
                                "match_rate": eval_hit_rate,
                            }
                            if etalon_eval_metric is not None:
                                eval_metrics[f"baseline {metric}"] = etalon_eval_metric
                            if enriched_eval_metric is not None:
                                eval_metrics[f"enriched {metric}"] = enriched_eval_metric
                            if eval_uplift is not None:
                                eval_metrics["uplift"] = eval_uplift

                            metrics.append(eval_metrics)

                    self.logger.info("Metrics calculation finished successfully")
                    return pd.DataFrame(metrics).set_index("segment").rename_axis("")
            except Exception as e:
                error_message = "Failed to calculate metrics" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                self._dump_python_libs()
                if not silent:
                    self.__display_slack_community_link()
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
                raise NotFittedError(msg)
            self._validate_X(X)

            self.__log_debug_information(X)

            self.__prepare_search_keys(X)

            df = X.copy()

            df = self.__handle_index_search_keys(df)

            self.__check_string_dates(X)
            df = self.__add_country_code(df)

            meaning_types = {col: key.value for col, key in self.search_keys.items()}
            search_keys = self.__using_search_keys()
            feature_columns = [column for column in df.columns if column not in self.search_keys.keys()]

            df[SYSTEM_RECORD_ID] = [hash(tuple(row)) for row in df[search_keys.keys()].values]  # type: ignore
            meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID
            index_name = df.index.name or DEFAULT_INDEX
            df = df.reset_index()
            df = df.rename(columns={index_name: ORIGINAL_INDEX})
            system_columns_with_original_index = [SYSTEM_RECORD_ID, ORIGINAL_INDEX]
            df_with_original_index = df[system_columns_with_original_index].copy()
            df = df.drop(columns=ORIGINAL_INDEX)

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
                    result, _ = self.__enrich(
                        df_with_original_index, validation_task.get_all_validation_raw_features(trace_id), X, {}
                    )
            else:
                result, _ = self.__enrich(
                    df_with_original_index, validation_task.get_all_validation_raw_features(trace_id), X, {}
                )

            filtered_columns = self.__filtered_enriched_features(importance_threshold, max_features)

            return result[X.columns.tolist() + filtered_columns]  # TODO check it twice

    def __validate_search_keys(self, search_keys: Dict[str, SearchKey], search_id: Optional[str]):
        if len(search_keys) == 0:
            if search_id:
                self.logger.warning(f"search_id {search_id} provided without search_keys")
                raise ValidationError(
                    "When search_id is passed, search_keys must be set to the same value that have been used for fit."
                )
            else:
                self.logger.warning("search_keys not provided")
                raise ValidationError("At least one column must be provided in search_keys.")

        key_types = search_keys.values()

        if SearchKey.DATE in key_types and SearchKey.DATETIME in key_types:
            msg = "DATE and DATETIME search keys cannot be used simultaneously. Choose one to keep."
            self.logger.warning(msg)
            raise ValidationError(msg)

        if SearchKey.EMAIL in key_types and SearchKey.HEM in key_types:
            msg = "EMAIL and HEM search keys cannot be used simultaneously. Choose one to keep."
            self.logger.warning(msg)
            raise ValidationError(msg)

        if SearchKey.POSTAL_CODE in key_types and SearchKey.COUNTRY not in key_types and self.country_code is None:
            msg = "COUNTRY search key must be provided if POSTAL_CODE is present."
            self.logger.warning(msg)
            raise ValidationError(msg)

        for key_type in SearchKey.__members__.values():
            if key_type != SearchKey.CUSTOM_KEY and list(key_types).count(key_type) > 1:
                msg = f"Search key {key_type} is presented multiple times."
                self.logger.warning(msg)
                raise ValidationError(msg)

        non_personal_keys = set(SearchKey.__members__.values()) - set(SearchKey.personal_keys())
        if not self.__is_registered and len(set(key_types).intersection(non_personal_keys)) == 0:
            msg = (
                "No API key found and all search keys require registration. "
                "You can use DATE, COUNTRY and POSTAL_CODE keys for free search without registration. "
                "Or provide the API key either directly or via the environment variable UPGINI_API_KEY."
            )
            self.logger.warning(msg + f" Provided search keys: {key_types}")
            raise ValidationError(msg)

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
    ):
        self.enriched_X = None
        self._validate_X(X)
        y_array = self._validate_y(X, y)

        self.__log_debug_information(X, y, eval_set)

        self.__prepare_search_keys(X)

        df: pd.DataFrame = X.copy()  # type: ignore
        df[self.TARGET_NAME] = y_array

        df = self.__handle_index_search_keys(df)

        self.__check_string_dates(df)

        df = self.__correct_target(df)

        model_task_type = self.model_task_type or define_task(df[self.TARGET_NAME], self.logger)

        eval_X_by_id = dict()
        if eval_set is not None and len(eval_set) > 0:
            df[EVAL_SET_INDEX] = 0
            for idx, eval_pair in enumerate(eval_set):
                eval_X, eval_y_array = self._validate_eval_set_pair(X, eval_pair)
                eval_df: pd.DataFrame = eval_X.copy()
                eval_df[self.TARGET_NAME] = eval_y_array
                eval_df[EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_df])
                eval_X_by_id[idx + 1] = eval_X

        df = self.__add_country_code(df)

        non_feature_columns = [self.TARGET_NAME, EVAL_SET_INDEX] + list(self.search_keys.keys())

        features_columns = [c for c in df.columns if c not in non_feature_columns]

        features_to_drop = FeaturesValidator(self.logger).validate(df, features_columns)
        df = df.drop(columns=features_to_drop)

        meaning_types = {
            **{col: key.value for col, key in self.search_keys.items()},
            **{str(c): FileColumnMeaningType.FEATURE for c in df.columns if c not in non_feature_columns},
        }
        meaning_types[self.TARGET_NAME] = FileColumnMeaningType.TARGET
        if eval_set is not None and len(eval_set) > 0:
            meaning_types[EVAL_SET_INDEX] = FileColumnMeaningType.EVAL_SET_INDEX

        search_keys = self.__using_search_keys()

        df = self.__add_fit_system_record_id(df, meaning_types)

        system_columns_with_original_index = [SYSTEM_RECORD_ID, ORIGINAL_INDEX]
        if EVAL_SET_INDEX in df.columns:
            system_columns_with_original_index.append(EVAL_SET_INDEX)
        df_with_original_index = df[system_columns_with_original_index].copy()
        df = df.drop(columns=ORIGINAL_INDEX)

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
            self.enriched_X, self.enriched_eval_sets = self.__enrich(
                df_with_original_index,
                self._search_task.get_all_initial_raw_features(trace_id),
                X,
                eval_X_by_id,
                "inner",
            )
        except Exception as e:
            self.logger.exception("Failed to download features")
            raise e

        if calculate_metrics:
            self.__show_metrics(X, y, eval_set, scoring, estimator, importance_threshold, max_features, trace_id)

    def _validate_X(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Unsupported type of X: {type(X)}. Use pandas.DataFrame.")
        if len(set(X.columns)) != len(X.columns):
            raise ValidationError("X contains duplicate column names. Please rename or drop them.")

    def _validate_y(self, X: pd.DataFrame, y) -> np.ndarray:
        if not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not isinstance(y, list):
            raise ValidationError(f"Unsupported type of y: {type(y)}. Use pandas.Series, numpy.ndarray or list.")

        if isinstance(y, pd.Series):
            y_array = y.values
        elif isinstance(y, np.ndarray):
            y_array = y
        else:
            y_array = np.array(y)

        if len(np.unique(y_array)) < 2:
            raise ValidationError("y is a constant. Finding relevant features requires a non-constant y.")

        if X.shape[0] != len(y_array):
            raise ValidationError(f"X and y contain different number of samples: {X.shape[0]}, {len(y_array)}.")

        return y_array

    def _validate_eval_set_pair(self, X: pd.DataFrame, eval_pair: Tuple) -> Tuple[pd.DataFrame, np.ndarray]:
        if len(eval_pair) != 2:
            raise ValidationError(
                f"eval_set contains a tuple of size {len(eval_pair)}. It should contain only pairs of X and y."
            )
        eval_X = eval_pair[0]
        eval_y = eval_pair[1]
        if not isinstance(eval_X, pd.DataFrame):
            raise ValidationError(f"Unsupported type of X in eval_set: {type(eval_X)}. Use pandas.DataFrame.")
        if eval_X.columns.to_list() != X.columns.to_list():
            raise ValidationError("The columns in eval_set are different from the columns in X.")
        if not isinstance(eval_y, pd.Series) and not isinstance(eval_y, np.ndarray) and not isinstance(eval_y, list):
            raise ValidationError(
                f"Unsupported type of y in eval_set: {type(eval_y)}. Use pandas.Series, numpy.ndarray or list."
            )

        if isinstance(eval_y, pd.Series):
            eval_y_array = eval_y.values
        elif isinstance(eval_y, np.ndarray):
            eval_y_array = eval_y
        else:
            eval_y_array = np.array(eval_y)

        if len(np.unique(eval_y_array)) < 2:
            raise ValidationError("y in eval_set is a constant. Finding relevant features requires a non-constant y.")

        if eval_X.shape[0] != len(eval_y_array):
            raise ValidationError(
                f"X and y in eval_set contain different number of samples: {eval_X.shape[0]}, {len(eval_y_array)}."
            )

        return eval_X, eval_y_array

    def _sample_X_and_y(
        self, X: pd.DataFrame, y: np.ndarray, enriched_X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        Xy = X.copy()
        Xy[TARGET] = y
        Xy = pd.merge(Xy, enriched_X, left_index=True, right_index=True, how="inner", suffixes=("", "enriched"))
        return Xy[X.columns].copy(), Xy[TARGET].values

    def _sort_by_date(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        if self.__is_date_key_present():
            date_column = [col for col, t in self.search_keys.items() if t in [SearchKey.DATE, SearchKey.DATETIME]]
            Xy = X.copy()
            Xy[TARGET] = y
            Xy = Xy.sort_values(by=date_column).reset_index(drop=True)
            X = Xy.drop(columns=TARGET)
            y = Xy[TARGET].values

        return X, y

    def __log_debug_information(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list, None] = None,
        eval_set: Optional[List[tuple]] = None,
    ):
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
        self.logger.info(f"First 10 rows of the X with shape {X.shape}:\n{X.head(10)}")
        if y is not None:
            self.logger.info(f"First 10 rows of the y with shape {len(y)}:\n{y[:10]}")
        if eval_set is not None:
            for idx, eval_pair in enumerate(eval_set):
                eval_X: pd.DataFrame = eval_pair[0]
                eval_y = eval_pair[1]
                self.logger.info(f"First 10 rows of the eval_X_{idx} with shape {eval_X.shape}:\n{eval_X.head(10)}")
                self.logger.info(f"First 10 rows of the eval_y_{idx} with shape {len(eval_y)}:\n{eval_y[:10]}")

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
        elif DEFAULT_INDEX in df.columns:
            raise ValidationError(
                "Delete or rename the column with the name 'index' please. "
                "This system name cannot be used in the enricher"
            )
        return df

    def __using_search_keys(self) -> Dict[str, SearchKey]:
        return {col: key for col, key in self.search_keys.items() if key != SearchKey.CUSTOM_KEY}

    def __is_date_key_present(self) -> bool:
        return len({SearchKey.DATE, SearchKey.DATETIME}.intersection(self.search_keys.values())) != 0

    def __add_fit_system_record_id(
        self, df: pd.DataFrame, meaning_types: Dict[str, FileColumnMeaningType]
    ) -> pd.DataFrame:
        index_name = df.index.name or DEFAULT_INDEX
        df = df.reset_index()
        df = df.rename(columns={index_name: ORIGINAL_INDEX})

        if (self.cv is None or self.cv == CVType.k_fold) and self.__is_date_key_present():
            date_column = [
                col
                for col, t in meaning_types.items()
                if t in [FileColumnMeaningType.DATE, FileColumnMeaningType.DATETIME]
            ]
            df = df.sort_values(by=date_column)

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
                    self.logger.warning(msg)
                    raise ValidationError(msg)

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
        df_with_original_index: pd.DataFrame,
        result_features: Optional[pd.DataFrame],
        X: pd.DataFrame,
        eval_set_by_id: Dict[int, pd.DataFrame],
        join_type: str = "left",
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        if result_features is None:
            self.logger.error(f"result features not found by search_task_id: {self.get_search_id()}")
            raise RuntimeError("Search engine crashed on this request.")
        result_features = (
            result_features.drop(columns=EVAL_SET_INDEX)
            if EVAL_SET_INDEX in result_features.columns
            else result_features
        )

        dup_features = [c for c in X.columns if c in result_features.columns]
        if len(dup_features) > 0:
            self.logger.warning(f"X contain columns with same name as returned from backend: {dup_features}")
            raise ValidationError(
                "Columns set for transform method should be the same as for fit method, please check input dataframe. "
                f"These columns are different: {dup_features}"
            )

        result = pd.merge(
            df_with_original_index,
            result_features,
            left_on=SYSTEM_RECORD_ID,
            right_on=SYSTEM_RECORD_ID,
            how=join_type,
        )

        result_eval_sets = dict()
        if EVAL_SET_INDEX in result.columns:
            result_train = result.loc[result[EVAL_SET_INDEX] == 0].copy()
            result_eval_set = result[result[EVAL_SET_INDEX] != 0]
            for eval_set_index in result_eval_set[EVAL_SET_INDEX].unique().tolist():
                result_eval = result.loc[result[EVAL_SET_INDEX] == eval_set_index].copy()
                if eval_set_index in eval_set_by_id.keys():
                    eval_X = eval_set_by_id[eval_set_index]
                    result_eval = result_eval.set_index(ORIGINAL_INDEX)
                    result_eval = pd.merge(left=eval_X, right=result_eval, left_index=True, right_index=True)
                else:
                    raise RuntimeError(
                        f"Eval_set index {eval_set_index} from enriched result not found in original eval_set"
                    )
                result_eval_sets[eval_set_index] = result_eval
            result_train = result_train.drop(columns=EVAL_SET_INDEX)
        else:
            result_train = result

        result_train = result_train.set_index(ORIGINAL_INDEX)
        result_train = pd.merge(left=X, right=result_train, left_index=True, right_index=True, how=join_type)
        if SYSTEM_RECORD_ID in result.columns:
            result_train = result_train.drop(columns=SYSTEM_RECORD_ID)
            for eval_set_index in result_eval_sets.keys():
                result_eval_sets[eval_set_index] = result_eval_sets[eval_set_index].drop(columns=SYSTEM_RECORD_ID)

        return result_train, result_eval_sets

    def __prepare_feature_importances(self, trace_id: str, x_columns: List[str]):
        if self._search_task is None:
            raise NotFittedError("Fit the enricher or pass search_id before calling transform.")
        features_meta = self._search_task.get_all_features_metadata_v2()
        if features_meta is None:
            raise Exception("Internal error. There is no features metadata")

        original_names_dict = {c.name: c.originalName for c in self._search_task.get_file_metadata(trace_id).columns}

        self.feature_names_ = []
        self.feature_importances_ = []
        features_info = []

        features_meta.sort(key=lambda m: -m.shap_value)
        for feature_meta in features_meta:
            if feature_meta.name in original_names_dict.keys():
                feature_meta.name = original_names_dict[feature_meta.name]
            if feature_meta.name not in x_columns:
                self.feature_names_.append(feature_meta.name)
                self.feature_importances_.append(feature_meta.shap_value)
            features_info.append(
                {
                    "provider": f"<a href='{feature_meta.data_provider_link}' "
                    "target='_blank' rel='noopener noreferrer'>"
                    f"{feature_meta.data_provider}</a>"
                    if feature_meta.data_provider
                    else "",
                    "source": f"<a href='{feature_meta.data_source_link}' "
                    "target='_blank' rel='noopener noreferrer'>"
                    f"{feature_meta.data_source}</a>"
                    if feature_meta.data_source
                    else "",
                    "feature name": feature_meta.name,
                    "shap value": feature_meta.shap_value,
                    "coverage %": feature_meta.hit_rate,
                    "type": feature_meta.type,
                    "feature type": feature_meta.commercial_schema or "",
                }
            )

        if len(features_info) > 0:
            self.features_info = pd.DataFrame(features_info)

    def __filtered_client_features(self, client_features: List[str]) -> List[str]:
        return self.features_info.loc[
            self.features_info["feature name"].isin(client_features) & self.features_info["shap value"] > 0,
            "feature name",
        ].values.tolist()

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
                    raise ValidationError(f"Key `{column_id}` in search_keys was not found in X: {list(x.columns)}.")
                column_name = column_id
                valid_search_keys[column_name] = meaning_type
            elif isinstance(column_id, int):
                if column_id >= x.shape[1]:
                    raise ValidationError(
                        f"Index {column_id} in search_keys is out of bounds for {x.shape[1]} columns of X."
                    )
                column_name = x.columns[column_id]
                valid_search_keys[column_name] = meaning_type
            else:
                raise ValidationError(f"Unsupported type of key in search_keys: {type(column_id)}.")

            if meaning_type == SearchKey.COUNTRY and self.country_code is not None:
                msg = "SearchKey.COUNTRY and iso_code cannot be used simultaneously."
                raise ValidationError(msg)

            if not is_registered and meaning_type in SearchKey.personal_keys():
                msg = f"Search key {meaning_type} cannot be used without API key. It will be ignored."
                self.logger.warning(msg)
                print("WARNING: " + msg)
                valid_search_keys[column_name] = SearchKey.CUSTOM_KEY
            else:
                if x[column_name].isnull().all() or (
                    is_string_dtype(x[column_name]) and (x[column_name].str.strip() == "").all()
                ):
                    msg = (
                        f"Search key {column_name} is empty. "
                        "Please fill values or remove this search key and try again."
                    )
                    raise ValidationError(msg)

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
                "Try to add the COUNTRY and/or POSTAL_CODE and/or PHONE NUMBER and/or EMAIL/HEM and/or IP address "
                "keys to your dataset so that the search engine gets access to the additional data sources. "
                "Get details on https://github.com/upgini/upgini#readme"
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

        if len(using_keys) == 1:
            for k, v in using_keys.items():
                if x[k].nunique() == 1:
                    msg = (
                        f"WARNING: Constant value detected for the {v} search key in the X dataframe: {x.loc[0, k]}.\n"
                        "That search key will add same constant features for different values from y. "
                        "Please add extra search keys with non constant values, like postal code, date, phone number, "
                        "hashed email or IP address."
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
            silent=True,
        )
        if metrics is not None:
            msg = "\nQuality metrics"

            try:
                from IPython.display import display

                _ = get_ipython()  # type: ignore

                print(Format.GREEN + Format.BOLD + msg + Format.END)
                display(metrics)
            except (ImportError, NameError):
                print(msg)
                print(metrics)

    def __show_selected_features(self):
        search_keys = self.__using_search_keys().keys()
        msg = f"\n{len(self.feature_names_)} relevant feature(s) found with the search keys: {list(search_keys)}."

        try:
            from IPython.display import display

            _ = get_ipython()  # type: ignore

            print(Format.GREEN + Format.BOLD + msg + Format.END)
            display(self.features_info.head(60).style.hide_index())
        except (ImportError, NameError):
            print(msg)
            print(self.features_info.head(60))

    def __validate_importance_threshold(self, importance_threshold: Optional[float]) -> float:
        try:
            return float(importance_threshold) if importance_threshold is not None else 0.0
        except ValueError:
            self.logger.exception(f"Invalid importance_threshold provided: {importance_threshold}")
            raise ValidationError("importance_threshold must be float.")

    def __validate_max_features(self, max_features: Optional[int]) -> int:
        try:
            return int(max_features) if max_features is not None else 400
        except ValueError:
            self.logger.exception(f"Invalid max_features provided: {max_features}")
            raise ValidationError("max_features must be int.")

    def __filtered_enriched_features(
        self,
        importance_threshold: Optional[float],
        max_features: Optional[int],
    ) -> List[str]:
        importance_threshold = self.__validate_importance_threshold(importance_threshold)
        max_features = self.__validate_max_features(max_features)

        return self.__filtered_importance_names(importance_threshold, max_features)

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

    def _dump_python_libs(self):
        result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
        libs = result.stdout.decode("utf-8")
        self.logger.warning(f"User python libs versions: {libs}")

    def __display_slack_community_link(self):
        slack_community_link = "https://4mlg.short.gy/join-upgini-community"
        link_text = "WARNING: Looks like you've run into some kind of error. For help write us in the Upgini community"
        badge = "https://img.shields.io/badge/slack-@upgini-orange.svg?logo=slack"
        try:
            from IPython.display import HTML, display

            _ = get_ipython()  # type: ignore
            display(
                HTML(
                    f"""<p>{link_text}</p><a href='{slack_community_link}' target='_blank' rel='noopener noreferrer'>
                    <img alt='Upgini slack community' src='{badge}'></a>
                    """
                )
            )
        except (ImportError, NameError):
            print(f"{link_text} at {slack_community_link}")
