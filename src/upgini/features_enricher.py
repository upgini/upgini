import itertools
import logging
import numbers
import os
import pickle
import subprocess
import tempfile
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
from upgini.errors import UpginiConnectionError, ValidationError
from upgini.http import UPGINI_API_KEY, LoggerFactory, get_rest_client
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
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask
from upgini.spinner import Spinner
from upgini.utils.country_utils import CountrySearchKeyDetector
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter, is_time_series
from upgini.utils.email_utils import EmailSearchKeyConverter, EmailSearchKeyDetector
from upgini.utils.features_validator import FeaturesValidator
from upgini.utils.format import Format
from upgini.utils.phone_utils import PhoneSearchKeyDetector
from upgini.utils.postal_code_utils import PostalCodeSearchKeyDetector
from upgini.utils.target_utils import define_task
from upgini.utils.warning_counter import WarningCounter
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
    EMPTY_FEATURES_INFO = pd.DataFrame(
        columns=[
            bundle.get("features_info_provider"),
            bundle.get("features_info_source"),
            bundle.get("features_info_name"),
            bundle.get("features_info_shap"),
            bundle.get("features_info_hitrate"),
            bundle.get("features_info_type"),
            bundle.get("features_info_commercial_schema"),
        ]
    )

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
        try:
            self.rest_client = get_rest_client(endpoint, self.api_key)
        except UpginiConnectionError as e:
            print(e)
            return

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
        self.features_info: pd.DataFrame = self.EMPTY_FEATURES_INFO
        self.search_id = search_id
        if search_id:
            search_task = SearchTask(
                search_id,
                endpoint=self.endpoint,
                api_key=self.api_key,
            )

            print(bundle.get("search_by_task_id_start"))
            trace_id = str(uuid.uuid4())
            with MDC(trace_id=trace_id):
                try:
                    self.logger.info(f"FeaturesEnricher created from existing search: {search_id}")
                    self._search_task = search_task.poll_result(trace_id, quiet=True)
                    file_metadata = self._search_task.get_file_metadata(trace_id)
                    x_columns = [c.originalName or c.name for c in file_metadata.columns]
                    self.__prepare_feature_importances(trace_id, x_columns)
                    # TODO validate search_keys with search_keys from file_metadata
                    print(bundle.get("search_by_task_id_finish"))
                    self.logger.info(f"Successfully initialized with search_id: {search_id}")
                except Exception as e:
                    print(bundle.get("failed_search_by_task_id"))
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
        self.shared_datasets = shared_datasets
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
        self.fit_generated_features: List[str] = []
        self.warning_counter = WarningCounter()
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.eval_set: Optional[List[Tuple]] = None
        self.autodetected_search_keys: Dict[str, SearchKey] = {}

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
            self.logger.info("Start fit")

            try:
                self.X = X
                self.y = y
                self.eval_set = eval_set
                self.dump_input(trace_id, X, y, eval_set)
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
                if "File doesn't intersect with any ADS" in str(e.args[0]) or "Empty intersection" in str(e.args[0]):
                    self.__display_slack_community_link(bundle.get("features_info_zero_important_features"))
                else:
                    self._dump_python_libs()
                    self.__display_slack_community_link()
                    raise e
            finally:
                self.logger.info(f"Fit elapsed time: {time.time() - start_time}")

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[List[tuple]] = None,
        *,
        keep_input: bool = True,
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

        keep_input: bool, optional (default=True)
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
            self.logger.info("Start fit_transform")
            try:
                self.X = X
                self.y = y
                self.eval_set = eval_set
                self.dump_input(trace_id, X, y, eval_set)
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
                self.logger.info("Inner fit finished successfully")
            except Exception as e:
                error_message = "Failed on inner fit" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                if e.args[0] == {"userMessage": "File doesn't intersect with any ADS"}:
                    self.__display_slack_community_link(bundle.get("features_info_zero_important_features"))
                else:
                    self._dump_python_libs()
                    self.__display_slack_community_link()
                    raise e
            finally:
                self.logger.info(f"Fit elapsed time: {time.time() - start_time}")

            result = self.transform(
                X,
                keep_input=keep_input,
                importance_threshold=importance_threshold,
                max_features=max_features,
                silent_mode=True,
            )
            self.logger.info("Fit_transform finished successfully")
            return result

    def transform(
        self,
        X: pd.DataFrame,
        *,
        keep_input: bool = True,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        silent_mode=False,
    ) -> pd.DataFrame:
        """Transform `X`.

        Returns a transformed version of `X`.
        If keep_input is True, then all input columns are copied to the output dataframe.

        Parameters
        ----------
        X: pandas.DataFrame of shape (n_samples, n_features)
            Input samples.

        keep_input: bool, optional (default=True)
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
            self.logger.info("Start transform")
            try:
                self.dump_input(trace_id, X)
                result = self.__inner_transform(
                    trace_id,
                    X,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                    silent_mode=silent_mode,
                )
                self.logger.info("Transform finished successfully")
            except Exception as e:
                error_message = "Failed on inner transform" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                if e.args[0] == {"userMessage": "File doesn't intersect with any ADS"}:
                    self.__display_slack_community_link(bundle.get("features_info_zero_important_features"))
                else:
                    self._dump_python_libs()
                    self.__display_slack_community_link()
                    raise e
            finally:
                self.logger.info(f"Transform elapsed time: {time.time() - start_time}")

            if self.country_added and COUNTRY in result.columns:
                result = result.drop(columns=COUNTRY)

            if keep_input:
                return result
            else:
                return result.drop(columns=[c for c in X.columns if c in result.columns])

    def calculate_metrics(
        self,
        *,
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
                self.logger.info(
                    f"Start calculating metrics\nscoring: {scoring}\n"
                    f"cv: {cv}\n"
                    f"estimator: {estimator}\n"
                    f"importance_threshold: {importance_threshold}\n"
                    f"max_features: {max_features}"
                )

                if (
                    self._search_task is None
                    or self._search_task.initial_max_hit_rate_v2() is None
                    or self.X is None
                    or self.y is None
                ):
                    raise ValidationError(bundle.get("metrics_unfitted_enricher"))
                if self.enriched_X is None:
                    raise ValidationError(bundle.get("metrics_empty_enriched_features"))

                if self._has_important_paid_features():
                    self.logger.warning("Metrics will be calculated on free features only")
                    self.__display_slack_community_link(bundle.get("metrics_exclude_paid_features"))
                    self.warning_counter.increment()

                validated_X = self._validate_X(self.X)
                validated_y = self._validate_y(validated_X, self.y)

                # TODO check that X and y are the same as on the fit

                self.__log_debug_information(self.X, self.y, self.eval_set)

                search_keys = self.search_keys.copy()
                search_keys = self.__prepare_search_keys(validated_X, search_keys, silent_mode=True)

                extended_X = validated_X.copy()
                generated_features = []
                date_column = self.__get_date_column(search_keys)
                if date_column is not None:
                    converter = DateTimeSearchKeyConverter(date_column, self.date_format, self.logger)
                    extended_X = converter.convert(extended_X)

                    generated_features.extend(converter.generated_features)
                email_column = self.__get_email_column(search_keys)
                hem_column = self.__get_hem_column(search_keys)
                if email_column:
                    converter = EmailSearchKeyConverter(email_column, hem_column, search_keys, self.logger)
                    extended_X = converter.convert(extended_X)
                    generated_features.extend(converter.generated_features)
                generated_features = [f for f in generated_features if f in self.fit_generated_features]

                X_sampled, y_sampled = self._sample_X_and_y(extended_X, validated_y, self.enriched_X)
                self.logger.info(f"Shape of enriched_X: {self.enriched_X.shape}")
                self.logger.info(f"Shape of X after sampling: {X_sampled.shape}")
                self.logger.info(f"Shape of y after sampling: {len(y_sampled)}")
                X_sorted, y_sorted = self._sort_by_date(X_sampled, y_sampled, date_column)
                enriched_X_sorted, enriched_y_sorted = self._sort_by_date(self.enriched_X, y_sampled, date_column)

                client_features = [
                    c for c in (validated_X.columns.to_list() + generated_features) if c not in search_keys.keys()
                ]

                filtered_client_features = self.__filtered_client_features(client_features)

                filtered_enriched_features = self.__filtered_enriched_features(
                    importance_threshold,
                    max_features,
                )

                existing_filtered_enriched_features = [
                    c for c in filtered_enriched_features if c in enriched_X_sorted.columns
                ]

                fitting_X = X_sorted[filtered_client_features].copy()
                fitting_enriched_X = enriched_X_sorted[
                    filtered_client_features + existing_filtered_enriched_features
                ].copy()

                if fitting_X.shape[1] == 0 and fitting_enriched_X.shape[1] == 0:
                    if self._has_important_paid_features():
                        print(bundle.get("metrics_no_important_free_features"))
                        self.logger.warning("No client or free relevant ADS features found to calculate metrics")
                    else:
                        print(bundle.get("metrics_no_important_features"))
                        self.logger.warning("No client or relevant ADS features found to calculate metrics")
                    self.warning_counter.increment()
                    return None

                model_task_type = self.model_task_type or define_task(validated_y, self.logger, silent=True)

                # shuffle Kfold for case when date/datetime keys are not presented
                key_types = search_keys.values()
                shuffle = True
                if SearchKey.DATE in key_types or SearchKey.DATETIME in key_types:
                    shuffle = False

                _cv = cv or self.cv

                wrapper = EstimatorWrapper.create(
                    estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                )
                metric = wrapper.metric_name
                multiplier = wrapper.multiplier

                print(bundle.get("metrics_start"))

                with Spinner():
                    # 1 If client features are presented - fit and predict with KFold CatBoost model
                    # on etalon features and calculate baseline metric
                    etalon_metric = None
                    baseline_estimator = None
                    if fitting_X.shape[1] > 0:
                        self.logger.info(
                            f"Calculate baseline {metric} on client features: {fitting_X.columns.to_list()}"
                        )
                        baseline_estimator = EstimatorWrapper.create(
                            estimator, self.logger, model_task_type, _cv, scoring, shuffle, self.random_state
                        )
                        etalon_metric = baseline_estimator.cross_val_predict(fitting_X, y_sorted)

                    # 2 Fit and predict with KFold Catboost model on enriched tds
                    # and calculate final metric (and uplift)
                    enriched_estimator = None
                    if set(fitting_X.columns) != set(fitting_enriched_X.columns):
                        self.logger.info(
                            f"Calculate enriched {metric} on combined features: {fitting_enriched_X.columns.to_list()}"
                        )
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
                        bundle.get("quality_metrics_segment_header"): bundle.get("quality_metrics_train_segment"),
                        bundle.get("quality_metrics_match_rate_header"): self._search_task.initial_max_hit_rate_v2(),
                    }
                    if etalon_metric is not None:
                        train_metrics[bundle.get("quality_metrics_baseline_header").format(metric)] = etalon_metric
                    if enriched_metric is not None:
                        train_metrics[bundle.get("quality_metrics_enriched_header").format(metric)] = enriched_metric
                    if uplift is not None:
                        train_metrics[bundle.get("quality_metrics_uplift_header")] = uplift
                    metrics = [train_metrics]

                    # 3 If eval_set is presented - fit final model on train enriched data and score each
                    # validation dataset and calculate final metric (and uplift)
                    max_initial_eval_set_hit_rate = self._search_task.get_max_initial_eval_set_hit_rate_v2()
                    if self.eval_set is not None:
                        if len(self.enriched_eval_sets) != len(self.eval_set):
                            raise ValidationError(
                                bundle.get("metrics_eval_set_count_diff").format(
                                    len(self.enriched_eval_sets), len(self.eval_set)
                                )
                            )
                        # TODO check that eval_set is the same as on the fit

                        for idx, eval_pair in enumerate(self.eval_set):
                            eval_hit_rate = max_initial_eval_set_hit_rate[idx + 1]

                            eval_X, validated_eval_y = self._validate_eval_set_pair(validated_X, eval_pair)
                            enriched_eval_X = self.enriched_eval_sets[idx + 1]

                            search_keys = self.search_keys.copy()
                            search_keys = self.__prepare_search_keys(eval_X, search_keys, silent_mode=True)

                            extended_eval_X = eval_X.copy()
                            generated_features = []
                            date_column = self.__get_date_column(search_keys)
                            if date_column is not None:
                                converter = DateTimeSearchKeyConverter(date_column, self.date_format, self.logger)
                                extended_eval_X = converter.convert(extended_eval_X)
                                generated_features.extend(converter.generated_features)
                            email_column = self.__get_email_column(search_keys)
                            hem_column = self.__get_hem_column(search_keys)
                            if email_column:
                                converter = EmailSearchKeyConverter(email_column, hem_column, search_keys, self.logger)
                                extended_eval_X = converter.convert(extended_eval_X)
                                generated_features.extend(converter.generated_features)
                            generated_features = [f for f in generated_features if f in self.fit_generated_features]

                            sampled_eval_X, sampled_eval_y = self._sample_X_and_y(
                                extended_eval_X, validated_eval_y, enriched_eval_X
                            )
                            self.logger.info(f"Shape of enriched_eval_X: {enriched_eval_X.shape}")
                            self.logger.info(f"Shape of eval_X_{idx} after sampling: {sampled_eval_X.shape}")
                            self.logger.info(f"Shape of eval_y_{idx} after sampling: {len(sampled_eval_y)}")
                            eval_X_sorted, eval_y_sorted = self._sort_by_date(
                                sampled_eval_X, sampled_eval_y, date_column
                            )
                            eval_X_sorted = eval_X_sorted[filtered_client_features].copy()

                            enriched_eval_X_sorted, enriched_y_sorted = self._sort_by_date(
                                enriched_eval_X, sampled_eval_y, date_column
                            )
                            enriched_eval_X_sorted = enriched_eval_X_sorted[
                                filtered_client_features + existing_filtered_enriched_features
                            ].copy()

                            if baseline_estimator is not None:
                                self.logger.info(
                                    f"Calculate baseline {metric} on eval set {idx + 1} "
                                    f"on client features: {eval_X_sorted.columns.to_list()}"
                                )
                                etalon_eval_metric = baseline_estimator.calculate_metric(eval_X_sorted, eval_y_sorted)
                            else:
                                etalon_eval_metric = None

                            if enriched_estimator is not None:
                                self.logger.info(
                                    f"Calculate enriched {metric} on eval set {idx + 1} "
                                    f"on client features: {enriched_eval_X_sorted.columns.to_list()}"
                                )
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
                                bundle.get("quality_metrics_segment_header"): bundle.get(
                                    "quality_metrics_eval_segment"
                                ).format(idx + 1),
                                bundle.get("quality_metrics_match_rate_header"): eval_hit_rate,
                            }
                            if etalon_eval_metric is not None:
                                eval_metrics[
                                    bundle.get("quality_metrics_baseline_header").format(metric)
                                ] = etalon_eval_metric
                            if enriched_eval_metric is not None:
                                eval_metrics[
                                    bundle.get("quality_metrics_enriched_header").format(metric)
                                ] = enriched_eval_metric
                            if eval_uplift is not None:
                                eval_metrics[bundle.get("quality_metrics_uplift_header")] = eval_uplift

                            metrics.append(eval_metrics)

                    metrics_df = (
                        pd.DataFrame(metrics).set_index(bundle.get("quality_metrics_segment_header")).rename_axis("")
                    )
                    do_without_pandas_limits(
                        lambda: self.logger.info(f"Metrics calculation finished successfully:\n{metrics_df}")
                    )

                    uplift_col = bundle.get("quality_metrics_uplift_header")
                    if (
                        uplift_col in metrics_df.columns
                        and (metrics_df[uplift_col] < 0).any()
                        and model_task_type == ModelTaskType.REGRESSION
                        and self.cv not in [CVType.time_series, CVType.blocked_time_series]
                        and self.__get_date_column(self.search_keys) is not None
                        and is_time_series(validated_X, self.__get_date_column(self.search_keys))
                    ):
                        msg = bundle.get("metrics_negative_uplift_without_cv")
                        self.logger.warning(msg)
                        self.__display_slack_community_link(msg)
                    elif uplift_col in metrics_df.columns and (metrics_df[uplift_col] < 0).any():
                        self.logger.warning("Uplift is negative")

                    return metrics_df
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
            msg = bundle.get("features_unfitted_enricher")
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
                raise NotFittedError(bundle.get("transform_unfitted_enricher"))

            validated_X = self._validate_X(X)

            self.__log_debug_information(X)

            search_keys = self.search_keys.copy()
            search_keys = self.__prepare_search_keys(validated_X, search_keys, silent_mode=silent_mode)

            df = validated_X.copy()

            df = self.__handle_index_search_keys(df, search_keys)

            self.__check_string_dates(validated_X, search_keys)
            df = self.__add_country_code(df, search_keys)

            generated_features = []
            date_column = self.__get_date_column(search_keys)
            if date_column is not None:
                converter = DateTimeSearchKeyConverter(date_column, self.date_format, self.logger)
                df = converter.convert(df)
                generated_features.extend(converter.generated_features)
            email_column = self.__get_email_column(search_keys)
            hem_column = self.__get_hem_column(search_keys)
            if email_column:
                converter = EmailSearchKeyConverter(email_column, hem_column, search_keys, self.logger)
                df = converter.convert(df)
                generated_features.extend(converter.generated_features)
            generated_features = [f for f in generated_features if f in self.fit_generated_features]

            meaning_types = {col: key.value for col, key in search_keys.items()}
            search_keys = self.__using_search_keys(search_keys)
            feature_columns = [column for column in df.columns if column not in search_keys.keys()]

            df[SYSTEM_RECORD_ID] = [hash(tuple(row)) for row in df[search_keys.keys()].values]  # type: ignore
            meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID
            index_name = df.index.name or DEFAULT_INDEX
            df = df.reset_index()
            df = df.rename(columns={index_name: ORIGINAL_INDEX})
            system_columns_with_original_index = [SYSTEM_RECORD_ID, ORIGINAL_INDEX] + generated_features
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
                print(bundle.get("transform_start"))
                with Spinner():
                    result, _ = self.__enrich(
                        df_with_original_index,
                        validation_task.get_all_validation_raw_features(trace_id),
                        validated_X,
                        {},
                    )
            else:
                result, _ = self.__enrich(
                    df_with_original_index, validation_task.get_all_validation_raw_features(trace_id), validated_X, {}
                )

            filtered_columns = self.__filtered_enriched_features(importance_threshold, max_features)

            existing_filtered_columns = [c for c in filtered_columns if c in result.columns]

            return result[validated_X.columns.tolist() + generated_features + existing_filtered_columns]

    def __validate_search_keys(self, search_keys: Dict[str, SearchKey], search_id: Optional[str]):
        if len(search_keys) == 0:
            if search_id:
                self.logger.warning(f"search_id {search_id} provided without search_keys")
                raise ValidationError(bundle.get("search_key_differ_from_fit"))
            else:
                self.logger.warning("search_keys not provided")
                raise ValidationError(bundle.get("empty_search_keys"))

        key_types = search_keys.values()

        if SearchKey.DATE in key_types and SearchKey.DATETIME in key_types:
            msg = bundle.get("date_and_datetime_simultanious")
            self.logger.warning(msg)
            raise ValidationError(msg)

        if SearchKey.EMAIL in key_types and SearchKey.HEM in key_types:
            msg = bundle.get("email_and_hem_simultanious")
            self.logger.warning(msg)
            raise ValidationError(msg)

        if SearchKey.POSTAL_CODE in key_types and SearchKey.COUNTRY not in key_types and self.country_code is None:
            msg = bundle.get("postal_code_without_country")
            self.logger.warning(msg)
            raise ValidationError(msg)

        for key_type in SearchKey.__members__.values():
            if key_type != SearchKey.CUSTOM_KEY and list(key_types).count(key_type) > 1:
                msg = bundle.get("multiple_search_key").format(key_type)
                self.logger.warning(msg)
                raise ValidationError(msg)

        non_personal_keys = set(SearchKey.__members__.values()) - set(SearchKey.personal_keys())
        if not self.__is_registered and len(set(key_types).intersection(non_personal_keys)) == 0:
            msg = bundle.get("unregistered_only_personal_keys")
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
        self.warning_counter.reset()
        self.enriched_X = None
        validated_X = self._validate_X(X)
        validated_y = self._validate_y(validated_X, y)

        self.__log_debug_information(X, y, eval_set)

        search_keys = self.search_keys.copy()
        search_keys = self.__prepare_search_keys(validated_X, search_keys)

        df = pd.concat([validated_X, validated_y], axis=1)

        df = self.__handle_index_search_keys(df, search_keys)

        self.__check_string_dates(df, search_keys)

        df = self.__correct_target(df)

        model_task_type = self.model_task_type or define_task(df[self.TARGET_NAME], self.logger)

        eval_X_by_id = dict()
        if eval_set is not None and len(eval_set) > 0:
            df[EVAL_SET_INDEX] = 0
            for idx, eval_pair in enumerate(eval_set):
                eval_X, eval_y = self._validate_eval_set_pair(validated_X, eval_pair)
                eval_df = pd.concat([eval_X, eval_y], axis=1)
                eval_df[EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_df])
                eval_X_by_id[idx + 1] = eval_X

        df = self.__add_country_code(df, search_keys)

        self.fit_generated_features = []
        date_column = self.__get_date_column(search_keys)
        if date_column is not None:
            converter = DateTimeSearchKeyConverter(date_column, self.date_format, self.logger)
            df = converter.convert(df)
            self.fit_generated_features.extend(converter.generated_features)
        email_column = self.__get_email_column(search_keys)
        hem_column = self.__get_hem_column(search_keys)
        if email_column:
            converter = EmailSearchKeyConverter(email_column, hem_column, search_keys, self.logger)
            df = converter.convert(df)
            self.fit_generated_features.extend(converter.generated_features)

        non_feature_columns = [self.TARGET_NAME, EVAL_SET_INDEX] + list(search_keys.keys())

        features_columns = [c for c in df.columns if c not in non_feature_columns]

        features_to_drop = FeaturesValidator(self.logger).validate(df, features_columns)
        df = df.drop(columns=features_to_drop)

        self.fit_generated_features = [f for f in self.fit_generated_features if f not in features_to_drop]

        meaning_types = {
            **{col: key.value for col, key in search_keys.items()},
            **{str(c): FileColumnMeaningType.FEATURE for c in df.columns if c not in non_feature_columns},
        }
        meaning_types[self.TARGET_NAME] = FileColumnMeaningType.TARGET
        if eval_set is not None and len(eval_set) > 0:
            meaning_types[EVAL_SET_INDEX] = FileColumnMeaningType.EVAL_SET_INDEX

        search_keys = self.__using_search_keys(search_keys)

        df = self.__add_fit_system_record_id(df, meaning_types, search_keys)

        system_columns_with_original_index = [SYSTEM_RECORD_ID, ORIGINAL_INDEX] + self.fit_generated_features
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

        zero_hit_search_keys = self._search_task.get_zero_hit_rate_search_keys()
        if zero_hit_search_keys:
            self.logger.warning(
                f"Intersections with this search keys are empty for all datasets: {zero_hit_search_keys}"
            )
            zero_hit_columns = self.get_columns_by_search_keys(zero_hit_search_keys)
            msg = bundle.get("features_info_zero_hit_rate_search_keys").format(zero_hit_columns)
            self.logger.warning(msg)
            self.__display_slack_community_link(msg)
            self.warning_counter.increment()

        self.__prepare_feature_importances(trace_id, validated_X.columns.to_list() + self.fit_generated_features)

        self.__show_selected_features(search_keys)

        if not self.warning_counter.has_warnings():
            self.__display_slack_community_link(bundle.get("all_ok_community_invite"))

        try:
            self.enriched_X, self.enriched_eval_sets = self.__enrich(
                df_with_original_index,
                self._search_task.get_all_initial_raw_features(trace_id),
                validated_X,
                eval_X_by_id,
                "inner",
            )
        except Exception as e:
            self.logger.exception("Failed to download features")
            raise e

        if calculate_metrics:
            self.__show_metrics(scoring, estimator, importance_threshold, max_features, trace_id)

    def get_columns_by_search_keys(self, keys: List[str]):
        if "HEM" in keys:
            keys.append("EMAIL")
        if "DATE" in keys:
            keys.append("DATETIME")
        search_keys_with_autodetection = {**self.search_keys, **self.autodetected_search_keys}
        return [c for c, v in search_keys_with_autodetection.items() if v.value.value in keys]

    def _validate_X(self, X) -> pd.DataFrame:
        if _num_samples(X) == 0:
            raise ValidationError(bundle.get("x_is_empty"))

        if isinstance(X, pd.DataFrame):
            if isinstance(X.columns, pd.MultiIndex) or isinstance(X.index, pd.MultiIndex):
                raise ValidationError(bundle.get("x_multiindex_unsupported"))
            validated_X = X
        elif isinstance(X, pd.Series):
            validated_X = X.to_frame()
        elif isinstance(X, np.ndarray) or isinstance(X, list):
            validated_X = pd.DataFrame(X)
            renaming = {c: str(c) for c in validated_X.columns}
            validated_X = validated_X.rename(columns=renaming)
        else:
            raise ValidationError(bundle.get("unsupported_x_type").format(type(X)))

        if len(set(validated_X.columns)) != len(validated_X.columns):
            raise ValidationError(bundle.get("x_contains_dup_columns"))
        if not validated_X.index.is_unique:
            raise ValidationError(bundle.get("x_non_unique_index"))

        if TARGET in validated_X.columns:
            raise ValidationError(bundle.get("x_contains_reserved_column_name").format(TARGET))
        if EVAL_SET_INDEX in validated_X.columns:
            raise ValidationError(bundle.get("x_contains_reserved_column_name").format(EVAL_SET_INDEX))
        if SYSTEM_RECORD_ID in validated_X.columns:
            raise ValidationError(bundle.get("x_contains_reserved_column_name").format(SYSTEM_RECORD_ID))

        return validated_X

    def _validate_y(self, X: pd.DataFrame, y) -> pd.Series:
        if _num_samples(y) == 0:
            raise ValidationError(bundle.get("y_is_empty"))

        if (
            not isinstance(y, pd.Series)
            and not isinstance(y, pd.DataFrame)
            and not isinstance(y, np.ndarray)
            and not isinstance(y, list)
        ):
            raise ValidationError(bundle.get("unsupported_y_type").format(type(y)))

        if _num_samples(X) != _num_samples(y):
            raise ValidationError(bundle.get("x_and_y_diff_size").format(_num_samples(X), _num_samples(y)))

        if isinstance(y, pd.DataFrame):
            if len(y.columns) != 1:
                raise ValidationError(bundle.get("y_invalid_dimension_dataframe"))
            if isinstance(y.columns, pd.MultiIndex) or isinstance(y.index, pd.MultiIndex):
                raise ValidationError(bundle.get("y_multiindex_unsupported"))
            y = y[y.columns[0]]

        if isinstance(y, pd.Series):
            if (y.index != X.index).any():
                raise ValidationError(bundle.get("x_and_y_diff_index"))
            validated_y = y.copy()
            validated_y.rename(TARGET, inplace=True)
        elif isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValidationError(bundle.get("y_invalid_dimension_array"))
            Xy = X.copy()
            Xy[TARGET] = y
            validated_y = Xy[TARGET].copy()
        else:
            Xy = X.copy()
            Xy[TARGET] = y
            validated_y = Xy[TARGET].copy()

        if validated_y.nunique() < 2:
            raise ValidationError(bundle.get("y_is_constant"))

        return validated_y

    def _validate_eval_set_pair(self, X: pd.DataFrame, eval_pair: Tuple) -> Tuple[pd.DataFrame, pd.Series]:
        if len(eval_pair) != 2:
            raise ValidationError(bundle.get("eval_set_invalid_tuple_size").format(len(eval_pair)))
        eval_X = eval_pair[0]
        eval_y = eval_pair[1]

        if _num_samples(eval_X) == 0:
            raise ValidationError(bundle.get("eval_x_is_empty"))
        if _num_samples(eval_X) == 0:
            raise ValidationError(bundle.get("eval_y_is_empty"))

        if isinstance(eval_X, pd.DataFrame):
            if isinstance(eval_X.columns, pd.MultiIndex) or isinstance(eval_X.index, pd.MultiIndex):
                raise ValidationError(bundle.get("eval_x_multiindex_unsupported"))
            validated_eval_X = eval_X
        elif isinstance(eval_X, pd.Series):
            validated_eval_X = eval_X.to_frame()
        elif isinstance(eval_X, np.ndarray) or isinstance(eval_X, list):
            validated_eval_X = pd.DataFrame(eval_X)
            renaming = {c: str(c) for c in validated_eval_X.columns}
            validated_eval_X = validated_eval_X.rename(columns=renaming)
        else:
            raise ValidationError(bundle.get("unsupported_x_type_eval_set").format(type(eval_X)))

        if not validated_eval_X.index.is_unique:
            raise ValidationError(bundle.get("x_non_unique_index_eval_set"))
        if validated_eval_X.columns.to_list() != X.columns.to_list():
            raise ValidationError(bundle.get("eval_x_and_x_diff_shape"))

        if _num_samples(validated_eval_X) != _num_samples(eval_y):
            raise ValidationError(
                bundle.get("x_and_y_diff_size_eval_set").format(_num_samples(validated_eval_X), _num_samples(eval_y))
            )

        if isinstance(eval_y, pd.DataFrame):
            if len(eval_y.columns) != 1:
                raise ValidationError(bundle.get("y_invalid_dimension_dataframe_eval_set"))
            if isinstance(eval_y.columns, pd.MultiIndex) or isinstance(eval_y.index, pd.MultiIndex):
                raise ValidationError(bundle.get("eval_y_multiindex_unsupported"))
            eval_y = eval_y[eval_y.columns[0]]

        if isinstance(eval_y, pd.Series):
            if (eval_y.index != validated_eval_X.index).any():
                raise ValidationError(bundle.get("x_and_y_diff_index_eval_set"))
            validated_eval_y = eval_y.copy()
            validated_eval_y.rename(TARGET, inplace=True)
        elif isinstance(eval_y, np.ndarray):
            if eval_y.ndim != 1:
                raise ValidationError(bundle.get("y_invalid_dimension_array_eval_set"))
            Xy = validated_eval_X.copy()
            Xy[TARGET] = eval_y
            validated_eval_y = Xy[TARGET].copy()
        elif isinstance(eval_y, list):
            Xy = validated_eval_X.copy()
            Xy[TARGET] = eval_y
            validated_eval_y = Xy[TARGET].copy()
        else:
            raise ValidationError(bundle.get("unsupported_y_type_eval_set").format(type(eval_y)))

        if validated_eval_y.nunique() < 2:
            raise ValidationError(bundle.get("y_is_constant_eval_set"))

        return validated_eval_X, validated_eval_y

    @staticmethod
    def _sample_X_and_y(X: pd.DataFrame, y: pd.Series, enriched_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        Xy = pd.concat([X, y], axis=1)
        Xy = pd.merge(Xy, enriched_X, left_index=True, right_index=True, how="inner", suffixes=("", "enriched"))
        return Xy[X.columns].copy(), Xy[TARGET].copy()

    @staticmethod
    def _sort_by_date(X: pd.DataFrame, y: pd.Series, date_column: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        if date_column is not None:
            Xy = pd.concat([X, y], axis=1)
            Xy = Xy.sort_values(by=date_column).reset_index(drop=True)
            X = Xy.drop(columns=TARGET)
            y = Xy[TARGET].copy()

        return X, y

    def __log_debug_information(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list, None] = None,
        eval_set: Optional[List[tuple]] = None,
    ):
        resolved_api_key = self.api_key or os.environ.get(UPGINI_API_KEY)
        self.logger.info(
            f"Search keys: {self.search_keys}\n"
            f"Country code: {self.country_code}\n"
            f"Model task type: {self.model_task_type}\n"
            f"Api key presented?: {resolved_api_key is not None and resolved_api_key != ''}\n"
            f"Endpoint: {self.endpoint}\n"
            f"Runtime parameters: {self.runtime_parameters}\n"
            f"Date format: {self.date_format}\n"
            f"CV: {self.cv}\n"
            f"Shared datasets: {self.shared_datasets}\n"
            f"Random state: {self.random_state}\n"
            f"Search id: {self.search_id}\n"
        )

        def sample(df):
            if isinstance(df, pd.Series) or isinstance(df, pd.DataFrame):
                return df.head(10)
            else:
                return df[:10]

        def print_datasets_sample():

            self.logger.info(f"First 10 rows of the X with shape {X.shape}:\n{sample(X)}")
            if y is not None:
                self.logger.info(f"First 10 rows of the y with shape {_num_samples(y)}:\n{sample(y)}")
            if eval_set is not None:
                for idx, eval_pair in enumerate(eval_set):
                    eval_X: pd.DataFrame = eval_pair[0]
                    eval_y = eval_pair[1]
                    self.logger.info(f"First 10 rows of the eval_X_{idx} with shape {eval_X.shape}:\n{sample(eval_X)}")
                    self.logger.info(
                        f"First 10 rows of the eval_y_{idx} with shape {_num_samples(eval_y)}:\n{sample(eval_y)}"
                    )

        do_without_pandas_limits(print_datasets_sample)

    @staticmethod
    def __handle_index_search_keys(df: pd.DataFrame, search_keys: Dict[str, SearchKey]) -> pd.DataFrame:
        index_names = df.index.names if df.index.names != [None] else [DEFAULT_INDEX]
        index_search_keys = set(index_names).intersection(search_keys.keys())
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
                search_keys[RENAMED_INDEX] = search_keys[DEFAULT_INDEX]
                del search_keys[DEFAULT_INDEX]
        elif DEFAULT_INDEX in df.columns:
            raise ValidationError(bundle.get("unsupported_index_column"))
        return df

    @staticmethod
    def __using_search_keys(search_keys: Dict[str, SearchKey]) -> Dict[str, SearchKey]:
        return {col: key for col, key in search_keys.items() if key != SearchKey.CUSTOM_KEY}

    @staticmethod
    def __get_date_column(search_keys: Dict[str, SearchKey]) -> Optional[str]:
        date_columns = [col for col, t in search_keys.items() if t in [SearchKey.DATE, SearchKey.DATETIME]]
        if len(date_columns) > 0:
            return date_columns[0]

    @staticmethod
    def __get_email_column(search_keys: Dict[str, SearchKey]) -> Optional[str]:
        email_columns = [col for col, t in search_keys.items() if t == SearchKey.EMAIL]
        if len(email_columns) > 0:
            return email_columns[0]

    @staticmethod
    def __get_hem_column(search_keys: Dict[str, SearchKey]) -> Optional[str]:
        hem_columns = [col for col, t in search_keys.items() if t == SearchKey.HEM]
        if len(hem_columns) > 0:
            return hem_columns[0]

    def __add_fit_system_record_id(
        self, df: pd.DataFrame, meaning_types: Dict[str, FileColumnMeaningType], search_keys: Dict[str, SearchKey]
    ) -> pd.DataFrame:
        index_name = df.index.name or DEFAULT_INDEX
        df = df.reset_index()
        df = df.rename(columns={index_name: ORIGINAL_INDEX})

        date_column = self.__get_date_column(search_keys)
        if (self.cv is None or self.cv == CVType.k_fold) and date_column is not None:
            other_search_keys = sorted([sk for sk in search_keys.keys() if sk != date_column])
            df = df.sort_values(by=[date_column] + other_search_keys)

        df = df.reset_index(drop=True)
        df = df.reset_index()
        df = df.rename(columns={DEFAULT_INDEX: SYSTEM_RECORD_ID})
        meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID
        return df

    def __check_string_dates(self, df: pd.DataFrame, search_keys: Dict[str, SearchKey]):
        for column, search_key in search_keys.items():
            if search_key in [SearchKey.DATE, SearchKey.DATETIME] and is_string_dtype(df[column]):
                if self.date_format is None or len(self.date_format) == 0:
                    msg = bundle.get("date_string_without_format").format(column)
                    self.logger.warning(msg)
                    raise ValidationError(msg)

    def __correct_target(self, df: pd.DataFrame) -> pd.DataFrame:
        target = df[self.TARGET_NAME]
        if is_string_dtype(target):
            maybe_numeric_target = pd.to_numeric(target, errors="coerce")
            # If less than 5% is non numeric then leave this rows with NaN target and later it will be dropped
            if maybe_numeric_target.isna().sum() <= _num_samples(df) * 0.05:
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

    def __add_country_code(self, df: pd.DataFrame, search_keys: Dict[str, SearchKey]) -> pd.DataFrame:
        self.country_added = False

        if self.country_code and SearchKey.COUNTRY not in search_keys.values():
            self.logger.info(f"Add COUNTRY column with {self.country_code} value")
            df[COUNTRY] = self.country_code
            search_keys[COUNTRY] = SearchKey.COUNTRY
            self.country_added = True

        if SearchKey.COUNTRY in search_keys.values():
            country_column = list(search_keys.keys())[list(search_keys.values()).index(SearchKey.COUNTRY)]
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
            raise RuntimeError(bundle.get("features_wasnt_returned"))
        result_features = (
            result_features.drop(columns=EVAL_SET_INDEX)
            if EVAL_SET_INDEX in result_features.columns
            else result_features
        )

        dup_features = [c for c in X.columns if c in result_features.columns]
        if len(dup_features) > 0:
            self.logger.warning(f"X contain columns with same name as returned from backend: {dup_features}")
            raise ValidationError(bundle.get("returned_features_same_as_passed").format(dup_features))

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
                    raise RuntimeError(bundle.get("missing_eval_set_for_enrichment").format(eval_set_index))
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
            raise NotFittedError(bundle.get("transform_unfitted_enricher"))
        features_meta = self._search_task.get_all_features_metadata_v2()
        if features_meta is None:
            raise Exception(bundle.get("missing_features_meta"))

        original_names_dict = {c.name: c.originalName for c in self._search_task.get_file_metadata(trace_id).columns}

        self.feature_names_ = []
        self.feature_importances_ = []
        features_info = []

        def round_shap_value(shap: float) -> float:
            if shap > 0.0 and shap < 0.000001:
                return 0.000001
            else:
                return shap

        features_meta.sort(key=lambda m: -m.shap_value)
        for feature_meta in features_meta:
            if feature_meta.name in original_names_dict.keys():
                feature_meta.name = original_names_dict[feature_meta.name]
            if feature_meta.name not in x_columns:
                self.feature_names_.append(feature_meta.name)
                self.feature_importances_.append(round_shap_value(feature_meta.shap_value))

            if feature_meta.data_provider and ipython_available():
                provider = (
                    f"<a href='{feature_meta.data_provider_link}' "
                    "target='_blank' rel='noopener noreferrer'>"
                    f"{feature_meta.data_provider}</a>"
                )
            else:
                provider = feature_meta.data_provider or ""

            if feature_meta.data_source and ipython_available():
                source = (
                    f"<a href='{feature_meta.data_source_link}' "
                    "target='_blank' rel='noopener noreferrer'>"
                    f"{feature_meta.data_source}</a>"
                )
            else:
                source = feature_meta.data_source or ""

            features_info.append(
                {
                    bundle.get("features_info_provider"): provider,
                    bundle.get("features_info_source"): source,
                    bundle.get("features_info_name"): feature_meta.name,
                    bundle.get("features_info_shap"): round_shap_value(feature_meta.shap_value),
                    bundle.get("features_info_hitrate"): feature_meta.hit_rate,
                    bundle.get("features_info_type"): feature_meta.type,
                    bundle.get("features_info_commercial_schema"): feature_meta.commercial_schema or "",
                }
            )

        if len(features_info) > 0:
            self.features_info = pd.DataFrame(features_info)
            do_without_pandas_limits(lambda: self.logger.info(f"Features info:\n{self.features_info}"))
        else:
            self.features_info = self.EMPTY_FEATURES_INFO
            self.logger.warning("Empty features info")

    def __filtered_client_features(self, client_features: List[str]) -> List[str]:
        return self.features_info.loc[
            self.features_info[bundle.get("features_info_name")].isin(client_features)
            & self.features_info[bundle.get("features_info_shap")]
            > 0,
            bundle.get("features_info_name"),
        ].values.tolist()

    def __filtered_importance_names(
        self, importance_threshold: Optional[float], max_features: Optional[int]
    ) -> List[str]:
        if len(self.feature_names_) == 0:
            return []

        filtered_importances = list(zip(self.feature_names_, self.feature_importances_))

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

    def __prepare_search_keys(self, x: pd.DataFrame, search_keys: Dict[str, SearchKey], silent_mode=False):
        valid_search_keys = {}
        api_key = self.api_key or os.environ.get(UPGINI_API_KEY)
        is_registered = api_key is not None and api_key != ""
        for column_id, meaning_type in search_keys.items():
            column_name = None
            if isinstance(column_id, str):
                if column_id not in x.columns:
                    raise ValidationError(bundle.get("search_key_not_found").format(column_id, list(x.columns)))
                column_name = column_id
                valid_search_keys[column_name] = meaning_type
            elif isinstance(column_id, int):
                if column_id >= x.shape[1]:
                    raise ValidationError(bundle.get("numeric_search_key_not_found").format(column_id, x.shape[1]))
                column_name = x.columns[column_id]
                valid_search_keys[column_name] = meaning_type
            else:
                raise ValidationError(bundle.get("unsupported_search_key_type").format(type(column_id)))

            if meaning_type == SearchKey.COUNTRY and self.country_code is not None:
                raise ValidationError(bundle.get("search_key_country_and_country_code"))

            if not is_registered and meaning_type in SearchKey.personal_keys():
                msg = bundle.get("unregistered_with_personal_keys").format(meaning_type)
                self.logger.warning(msg)
                if not silent_mode:
                    self.warning_counter.increment()
                    print(msg)
                valid_search_keys[column_name] = SearchKey.CUSTOM_KEY
            else:
                if x[column_name].isnull().all() or (
                    is_string_dtype(x[column_name]) and (x[column_name].astype("string").str.strip() == "").all()
                ):
                    raise ValidationError(bundle.get("empty_search_key").format(column_name))

        if self.detect_missing_search_keys:
            valid_search_keys = self.__detect_missing_search_keys(x, valid_search_keys, silent_mode)

        using_keys = self.__using_search_keys(search_keys)
        if (
            len(using_keys.values()) == 1
            and self.country_code is None
            and next(iter(using_keys.values())) == SearchKey.DATE
            and not silent_mode
        ):
            msg = bundle.get("date_only_search")
            print(msg)
            self.logger.warning(msg)
            self.warning_counter.increment()

        maybe_date = [k for k, v in using_keys.items() if v in [SearchKey.DATE, SearchKey.DATETIME]]
        if (self.cv is None or self.cv == CVType.k_fold) and len(maybe_date) > 0 and not silent_mode:
            date_column = next(iter(maybe_date))
            if x[date_column].nunique() > 0.9 * _num_samples(x):
                msg = bundle.get("date_search_without_time_series")
                print(msg)
                self.logger.warning(msg)
                self.warning_counter.increment()

        if len(using_keys) == 1:
            for k, v in using_keys.items():
                # Show warning for country only if country is the only key
                if x[k].nunique() == 1 and (v != SearchKey.COUNTRY or len(using_keys) == 1):
                    msg = bundle.get("single_constant_search_key").format(v, x.loc[0, k])
                    print(msg)
                    self.logger.warning(msg)
                    self.warning_counter.increment()

        return valid_search_keys

    def __show_metrics(
        self,
        scoring: Union[Callable, str, None],
        estimator: Optional[Any],
        importance_threshold: Optional[float],
        max_features: Optional[int],
        trace_id: str,
    ):
        metrics = self.calculate_metrics(
            scoring=scoring,
            estimator=estimator,
            importance_threshold=importance_threshold,
            max_features=max_features,
            trace_id=trace_id,
            silent=True,
        )
        if metrics is not None:
            msg = bundle.get("quality_metrics_header")

            try:
                from IPython.display import display

                _ = get_ipython()  # type: ignore

                print(Format.GREEN + Format.BOLD + msg + Format.END)
                display(metrics)
            except (ImportError, NameError):
                print(msg)
                print(metrics)

    def _has_important_paid_features(self) -> bool:
        return (self.features_info[bundle.get("features_info_commercial_schema")] == "Paid").any()

    def __show_selected_features(self, search_keys: Dict[str, SearchKey]):
        msg = bundle.get("features_info_header").format(len(self.feature_names_), list(search_keys.keys()))

        try:
            _ = get_ipython()  # type: ignore

            print(Format.GREEN + Format.BOLD + msg + Format.END)
            display_html_dataframe(self.features_info.head(60))

            if len(self.feature_names_) == 0:
                msg = bundle.get("features_info_zero_important_features")
                self.logger.warning(msg)
                self.__display_slack_community_link(msg)
                self.warning_counter.increment()
        except (ImportError, NameError):
            print(msg)
            print(self.features_info.head(60))

    def __validate_importance_threshold(self, importance_threshold: Optional[float]) -> float:
        try:
            return float(importance_threshold) if importance_threshold is not None else 0.0
        except ValueError:
            self.logger.exception(f"Invalid importance_threshold provided: {importance_threshold}")
            raise ValidationError(bundle.get("invalid_importance_threshold"))

    def __validate_max_features(self, max_features: Optional[int]) -> int:
        try:
            return int(max_features) if max_features is not None else 400
        except ValueError:
            self.logger.exception(f"Invalid max_features provided: {max_features}")
            raise ValidationError(bundle.get("invalid_max_features"))

    def __filtered_enriched_features(
        self,
        importance_threshold: Optional[float],
        max_features: Optional[int],
    ) -> List[str]:
        importance_threshold = self.__validate_importance_threshold(importance_threshold)
        max_features = self.__validate_max_features(max_features)

        return self.__filtered_importance_names(importance_threshold, max_features)

    def __detect_missing_search_keys(
        self, df: pd.DataFrame, search_keys: Dict[str, SearchKey], silent_mode=False
    ) -> Dict[str, SearchKey]:
        sample = df.head(100)

        if SearchKey.POSTAL_CODE not in search_keys.values():
            maybe_key = PostalCodeSearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                search_keys[maybe_key] = SearchKey.POSTAL_CODE
                self.autodetected_search_keys[maybe_key] = SearchKey.POSTAL_CODE
                self.logger.info(f"Autodetected search key POSTAL_CODE in column {maybe_key}")
                if not silent_mode:
                    print(bundle.get("postal_code_detected").format(maybe_key))

        if SearchKey.COUNTRY not in search_keys.values() and self.country_code is None:
            maybe_key = CountrySearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                search_keys[maybe_key] = SearchKey.COUNTRY
                self.autodetected_search_keys[maybe_key] = SearchKey.COUNTRY
                self.logger.info(f"Autodetected search key COUNTRY in column {maybe_key}")
                if not silent_mode:
                    print(bundle.get("country_detected").format(maybe_key))

        if SearchKey.EMAIL not in search_keys.values() and SearchKey.HEM not in search_keys.values():
            maybe_key = EmailSearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                if self.__is_registered:
                    search_keys[maybe_key] = SearchKey.EMAIL
                    self.autodetected_search_keys[maybe_key] = SearchKey.EMAIL
                    self.logger.info(f"Autodetected search key EMAIL in column {maybe_key}")
                    if not silent_mode:
                        print(bundle.get("email_detected").format(maybe_key))
                else:
                    self.logger.warning(
                        f"Autodetected search key EMAIL in column {maybe_key}. But not used because not registered user"
                    )
                    if not silent_mode:
                        print(bundle.get("email_detected_not_registered").format(maybe_key))
                    self.warning_counter.increment()

        if SearchKey.PHONE not in search_keys.values():
            maybe_key = PhoneSearchKeyDetector().get_search_key_column(sample)
            if maybe_key is not None:
                if self.__is_registered:
                    search_keys[maybe_key] = SearchKey.PHONE
                    self.autodetected_search_keys[maybe_key] = SearchKey.PHONE
                    self.logger.info(f"Autodetected search key PHONE in column {maybe_key}")
                    if not silent_mode:
                        print(bundle.get("phone_detected").format(maybe_key))
                else:
                    self.logger.warning(
                        f"Autodetected search key PHONE in column {maybe_key}. But not used because not registered user"
                    )
                    if not silent_mode:
                        print(bundle.get("phone_detected_not_registered"))
                    self.warning_counter.increment()

        return search_keys

    def _dump_python_libs(self):
        python_version_result = subprocess.run(["python", "-V"], stdout=subprocess.PIPE)
        python_version = python_version_result.stdout.decode("utf-8")
        result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
        libs = result.stdout.decode("utf-8")
        self.logger.warning(f"User python {python_version} libs versions:\n{libs}")

    def __display_slack_community_link(self, link_text: Optional[str] = None):
        slack_community_link = bundle.get("slack_community_link")
        link_text = link_text or bundle.get("slack_community_text")
        badge = bundle.get("slack_community_bage")
        alt = bundle.get("slack_community_alt")
        try:
            from IPython.display import HTML, display

            _ = get_ipython()  # type: ignore
            print(link_text)
            display(
                HTML(
                    f"""<a href='{slack_community_link}' target='_blank' rel='noopener noreferrer'>
                    <img alt='{alt}' src='{badge}'></a>
                    """
                )
            )
        except (ImportError, NameError):
            print(f"{link_text} at {slack_community_link}")

    def dump_input(
        self,
        trace_id: str,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series, None] = None,
        eval_set: Union[Tuple, None] = None,
    ):
        try:
            random_state = 42
            rnd = np.random.RandomState(random_state)
            if _num_samples(X) > 0:
                xy_sample_index = rnd.randint(0, _num_samples(X), size=1000)
            else:
                xy_sample_index = []

            def sample(inp, sample_index):
                if _num_samples(inp) <= 1000:
                    return inp
                if isinstance(inp, pd.DataFrame) or isinstance(inp, pd.Series):
                    return inp.sample(n=1000, random_state=random_state)
                if isinstance(inp, np.ndarray):
                    return inp[sample_index]
                if isinstance(inp, list):
                    return inp[sample_index]

            with tempfile.TemporaryDirectory() as tmp_dir:
                with open(f"{tmp_dir}/x.pickle", "wb") as x_file:
                    pickle.dump(sample(X, xy_sample_index), x_file)
                if y is not None:
                    with open(f"{tmp_dir}/y.pickle", "wb") as y_file:
                        pickle.dump(sample(y, xy_sample_index), y_file)
                    if eval_set is not None:
                        eval_xy_sample_index = rnd.randint(0, _num_samples(eval_set[0][0]), size=1000)
                        with open(f"{tmp_dir}/eval_x.pickle", "wb") as eval_x_file:
                            pickle.dump(sample(eval_set[0][0], eval_xy_sample_index), eval_x_file)
                        with open(f"{tmp_dir}/eval_y.pickle", "wb") as eval_y_file:
                            pickle.dump(sample(eval_set[0][1], eval_xy_sample_index), eval_y_file)
                        get_rest_client(self.endpoint, self.api_key).dump_input_files(
                            trace_id,
                            f"{tmp_dir}/x.pickle",
                            f"{tmp_dir}/y.pickle",
                            f"{tmp_dir}/eval_x.pickle",
                            f"{tmp_dir}/eval_y.pickle",
                        )
                    else:
                        get_rest_client(self.endpoint, self.api_key).dump_input_files(
                            trace_id,
                            f"{tmp_dir}/x.pickle",
                            f"{tmp_dir}/y.pickle",
                        )
                else:
                    get_rest_client(self.endpoint, self.api_key).dump_input_files(
                        trace_id,
                        f"{tmp_dir}/x.pickle",
                    )
        except Exception:
            self.logger.warning("Failed to dump input files", exc_info=True)


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def do_without_pandas_limits(func: Callable):
    prev_max_rows = pd.options.display.max_rows
    prev_max_columns = pd.options.display.max_columns
    prev_max_colwidth = pd.options.display.max_colwidth

    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None

    try:
        func()
    finally:
        pd.options.display.max_rows = prev_max_rows
        pd.options.display.max_columns = prev_max_columns
        pd.options.display.max_colwidth = prev_max_colwidth


def ipython_available() -> bool:
    try:
        _ = get_ipython()  # type: ignore
        return True
    except NameError:
        return False


def display_html_dataframe(df: pd.DataFrame):
    from IPython.display import HTML, display

    def map_to_td(value) -> str:
        if isinstance(value, float):
            return f"<td class='upgini-number'>{value:.6f}</td>"
        else:
            return f"<td class='upgini-text'>{value}</td>"

    table_str = (
        """<style>
            .upgini-df thead th {
                font-weight:bold;
                text-align: right;
                padding: 0.5em;
            }

            .upgini-df td {
                padding: 0.5em;
            }

            .upgini-text {
                text-align: right;
            }

            .upgini-number {
                text-align: center;
            }
        </style>"""
        + "<table class='upgini-df'>"
        + "<thead>"
        + "".join(f"<th>{col}</th>" for col in df.columns)
        + "</thead>"
        + "<tbody>"
        + "".join("<tr>" + "".join(map(map_to_td, row[1:])) + "</tr>" for row in df.itertuples())
        + "</tbody>"
        + "</table>"
    )
    display(HTML(table_str))
