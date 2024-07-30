import dataclasses
import datetime
import gc
import hashlib
import itertools
import logging
import numbers
import os
import pickle
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_period_dtype,
    is_string_dtype,
)
from scipy.stats import ks_2samp
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import BaseCrossValidator

from upgini.autofe.feature import Feature
from upgini.data_source.data_source_publisher import CommercialSchema
from upgini.dataset import Dataset
from upgini.errors import HttpError, ValidationError
from upgini.http import (
    UPGINI_API_KEY,
    LoggerFactory,
    ProgressStage,
    SearchProgress,
    get_rest_client,
)
from upgini.mdc import MDC
from upgini.metadata import (
    COUNTRY,
    DEFAULT_INDEX,
    ENTITY_SYSTEM_RECORD_ID,
    EVAL_SET_INDEX,
    ORIGINAL_INDEX,
    RENAMED_INDEX,
    SEARCH_KEY_UNNEST,
    SORT_ID,
    SYSTEM_RECORD_ID,
    TARGET,
    CVType,
    FileColumnMeaningType,
    ModelTaskType,
    RuntimeParameters,
    SearchKey,
)
from upgini.metrics import EstimatorWrapper, validate_scoring_argument
from upgini.resource_bundle import ResourceBundle, bundle, get_custom_bundle
from upgini.search_task import SearchTask
from upgini.spinner import Spinner
from upgini.utils import combine_search_keys, find_numbers_with_decimal_comma
from upgini.utils.country_utils import CountrySearchKeyDetector
from upgini.utils.custom_loss_utils import (
    get_additional_params_custom_loss,
    get_runtime_params_custom_loss,
)
from upgini.utils.cv_utils import CVConfig, get_groups
from upgini.utils.datetime_utils import (
    DateTimeSearchKeyConverter,
    is_blocked_time_series,
    is_time_series,
    validate_dates_distribution,
)
from upgini.utils.deduplicate_utils import (
    clean_full_duplicates,
    remove_fintech_duplicates,
)
from upgini.utils.display_utils import (
    display_html_dataframe,
    do_without_pandas_limits,
    prepare_and_show_report,
    show_request_quote_button,
)
from upgini.utils.email_utils import EmailSearchKeyConverter, EmailSearchKeyDetector
from upgini.utils.features_validator import FeaturesValidator
from upgini.utils.format import Format
from upgini.utils.ip_utils import IpToCountrySearchKeyConverter
from upgini.utils.phone_utils import PhoneSearchKeyDetector
from upgini.utils.postal_code_utils import PostalCodeSearchKeyDetector

try:
    from upgini.utils.progress_bar import CustomProgressBar as ProgressBar
except Exception:
    from upgini.utils.fallback_progress_bar import CustomFallbackProgressBar as ProgressBar

from upgini.utils.target_utils import calculate_psi, define_task
from upgini.utils.warning_counter import WarningCounter
from upgini.version_validator import validate_version

DEMO_DATASET_HASHES = [
    "7c354d1b1794c53ac7d7e5a2f2574568b660ca9159bc0d2aca9c7127ebcea2f7",  # demo_salary fit
    "2519c9077c559f8975fdcdb5c50e9daae8d50b1d8a3ec72296c65ea7276f8812",  # demo_salary transform
]


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

    loss: str, optional (default=None)
        Custom loss function to use for feature selection and metrics calculation.

    shared_datasets: list of str, optional (default=None)
        List of private shared dataset ids for custom search
    """

    TARGET_NAME = "target"
    CURRENT_DATE = "current_date"
    RANDOM_STATE = 42
    CALCULATE_METRICS_THRESHOLD = 50_000_000
    CALCULATE_METRICS_MIN_THRESHOLD = 500
    GENERATE_FEATURES_LIMIT = 10
    EMPTY_FEATURES_INFO = pd.DataFrame(
        columns=[
            bundle.get("features_info_name"),
            bundle.get("features_info_shap"),
            bundle.get("features_info_hitrate"),
            bundle.get("features_info_value_preview"),
            bundle.get("features_info_provider"),
            bundle.get("features_info_source"),
            # bundle.get("features_info_commercial_schema"),
            bundle.get("features_info_update_frequency"),
        ]
    )
    EMPTY_INTERNAL_FEATURES_INFO = pd.DataFrame(
        columns=[
            bundle.get("features_info_name"),
            bundle.get("features_info_shap"),
            bundle.get("features_info_hitrate"),
            bundle.get("features_info_value_preview"),
            bundle.get("features_info_provider"),
            bundle.get("features_info_source"),
            bundle.get("features_info_commercial_schema"),
            bundle.get("features_info_update_frequency"),
        ]
    )
    EMPTY_DATA_SOURCES = pd.DataFrame(
        columns=[
            bundle.get("features_info_provider"),
            bundle.get("features_info_source"),
            bundle.get("relevant_data_sources_all_shap"),
            bundle.get("relevant_data_sources_number"),
        ]
    )

    def __init__(
        self,
        search_keys: Optional[Dict[str, SearchKey]] = None,
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
        loss: Optional[str] = None,
        detect_missing_search_keys: bool = True,
        generate_features: Optional[List[str]] = None,
        round_embeddings: Optional[int] = None,
        logs_enabled: bool = True,
        raise_validation_error: bool = True,
        exclude_columns: Optional[List[str]] = None,
        baseline_score_column: Optional[Any] = None,
        client_ip: Optional[str] = None,
        client_visitorid: Optional[str] = None,
        custom_bundle_config: Optional[str] = None,
        add_date_if_missing: bool = True,
        **kwargs,
    ):
        self.bundle = get_custom_bundle(custom_bundle_config)
        self._api_key = api_key or os.environ.get(UPGINI_API_KEY)
        if api_key is not None and not isinstance(api_key, str):
            raise ValidationError(f"api_key should be `string`, but passed: `{api_key}`")
        self.rest_client = get_rest_client(endpoint, self._api_key, client_ip, client_visitorid)
        self.client_ip = client_ip
        self.client_visitorid = client_visitorid

        self.logs_enabled = logs_enabled
        if logs_enabled:
            self.logger = LoggerFactory().get_logger(endpoint, self._api_key, client_ip, client_visitorid)
        else:
            self.logger = logging.getLogger("muted_logger")
            self.logger.setLevel("FATAL")

        if len(kwargs) > 0:
            msg = f"WARNING: Unsupported arguments: {kwargs}"
            self.logger.warning(msg)
            print(msg)

        self.passed_features: List[str] = []
        self.df_with_original_index: Optional[pd.DataFrame] = None
        self.country_added = False
        self.fit_generated_features: List[str] = []
        self.fit_dropped_features: Set[str] = set()
        self.fit_search_keys = search_keys
        self.warning_counter = WarningCounter()
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.eval_set: Optional[List[Tuple]] = None
        self.autodetected_search_keys: Dict[str, SearchKey] = {}
        self.imbalanced = False
        self.__cached_sampled_datasets: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict, Dict]] = None

        validate_version(self.logger)
        self.search_keys = search_keys or {}
        self.country_code = country_code
        self.__validate_search_keys(search_keys, search_id)
        self.model_task_type = model_task_type
        self.endpoint = endpoint
        self._search_task: Optional[SearchTask] = None
        self.features_info: pd.DataFrame = self.EMPTY_FEATURES_INFO
        self._features_info_without_links: pd.DataFrame = self.EMPTY_FEATURES_INFO
        self._internal_features_info: pd.DataFrame = self.EMPTY_INTERNAL_FEATURES_INFO
        self.relevant_data_sources: pd.DataFrame = self.EMPTY_DATA_SOURCES
        self._relevant_data_sources_wo_links: pd.DataFrame = self.EMPTY_DATA_SOURCES
        self.metrics: Optional[pd.DataFrame] = None
        self.feature_names_ = []
        self.feature_importances_ = []
        self.search_id = search_id
        if search_id:
            search_task = SearchTask(search_id, rest_client=self.rest_client, logger=self.logger)

            print(self.bundle.get("search_by_task_id_start"))
            trace_id = str(uuid.uuid4())
            with MDC(trace_id=trace_id):
                try:
                    self.logger.debug(f"FeaturesEnricher created from existing search: {search_id}")
                    self._search_task = search_task.poll_result(trace_id, quiet=True, check_fit=True)
                    file_metadata = self._search_task.get_file_metadata(trace_id)
                    x_columns = [c.originalName or c.name for c in file_metadata.columns]
                    self.__prepare_feature_importances(trace_id, x_columns, silent=True)
                    # TODO validate search_keys with search_keys from file_metadata
                    print(self.bundle.get("search_by_task_id_finish"))
                    self.logger.debug(f"Successfully initialized with search_id: {search_id}")
                except HttpError as e:
                    if "Interrupted by client" in e.args[0]:
                        raise ValidationError("Search was cancelled")
                except Exception as e:
                    print(self.bundle.get("failed_search_by_task_id"))
                    self.logger.exception(f"Failed to find search_id: {search_id}")
                    raise e

        self.runtime_parameters = runtime_parameters or RuntimeParameters()
        self.runtime_parameters.properties["feature_generation_params.hash_index"] = True
        self.date_format = date_format
        self.random_state = random_state
        self.detect_missing_search_keys = detect_missing_search_keys
        self.cv = cv
        if cv is not None:
            self.runtime_parameters.properties["cv_type"] = cv.name
        self.loss = loss.lower() if loss is not None else None

        self.shared_datasets = shared_datasets
        if shared_datasets is not None:
            self.runtime_parameters.properties["shared_datasets"] = ",".join(shared_datasets)
        self.generate_features = generate_features
        self.round_embeddings = round_embeddings
        if generate_features is not None:
            if len(generate_features) > self.GENERATE_FEATURES_LIMIT:
                msg = self.bundle.get("too_many_generate_features").format(self.GENERATE_FEATURES_LIMIT)
                self.logger.error(msg)
                raise ValidationError(msg)
            self.runtime_parameters.properties["generate_features"] = ",".join(generate_features)
            if round_embeddings is not None:
                if not isinstance(round_embeddings, int) or round_embeddings < 0:
                    msg = self.bundle.get("invalid_round_embeddings")
                    self.logger.error(msg)
                    raise ValidationError(msg)
                self.runtime_parameters.properties["round_embeddings"] = round_embeddings
        maybe_downsampling_limit = self.runtime_parameters.properties.get("downsampling_limit")
        if maybe_downsampling_limit is not None:
            Dataset.FIT_SAMPLE_THRESHOLD = int(maybe_downsampling_limit)
            Dataset.FIT_SAMPLE_ROWS = int(maybe_downsampling_limit)

        self.raise_validation_error = raise_validation_error
        self.exclude_columns = exclude_columns
        self.baseline_score_column = baseline_score_column
        self.add_date_if_missing = add_date_if_missing

    def _get_api_key(self):
        return self._api_key

    def _set_api_key(self, api_key: str):
        self._api_key = api_key
        if self.logs_enabled:
            self.logger = LoggerFactory().get_logger(
                self.endpoint, self._api_key, self.client_ip, self.client_visitorid
            )

    api_key = property(_get_api_key, _set_api_key)

    @staticmethod
    def _check_eval_set(eval_set, X, bundle: ResourceBundle):
        checked_eval_set = []
        if eval_set is not None and isinstance(eval_set, tuple):
            eval_set = [eval_set]
        if eval_set is not None and not isinstance(eval_set, list):
            raise ValidationError(bundle.get("unsupported_type_eval_set").format(type(eval_set)))
        for eval_pair in eval_set or []:
            if not isinstance(eval_pair, tuple) or len(eval_pair) != 2:
                raise ValidationError(bundle.get("eval_set_invalid_tuple_size").format(len(eval_pair)))
            if not is_frames_equal(X, eval_pair[0], bundle):
                checked_eval_set.append(eval_pair)
        return checked_eval_set

    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[Union[List[tuple], tuple]] = None,
        *args,
        exclude_features_sources: Optional[List[str]] = None,
        calculate_metrics: Optional[bool] = None,
        estimator: Optional[Any] = None,
        scoring: Union[Callable, str, None] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        remove_outliers_calc_metrics: Optional[bool] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        search_id_callback: Optional[Callable[[str], Any]] = None,
        **kwargs,
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

        calculate_metrics: bool, optional (default=None)
            Whether to calculate and show metrics.

        estimator: sklearn-compatible estimator, optional (default=None)
            Custom estimator for metrics calculation.

        scoring: string or callable, optional (default=None)
            A string or a scorer callable object / function with signature scorer(estimator, X, y).
            If None, the estimator's score method is used.

        remove_outliers_calc_metrics, optional (default=True)
            If True then rows with target ouliers will be dropped on metrics calculation
        """
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        search_progress = SearchProgress(0.0, ProgressStage.START_FIT)
        if progress_callback is not None:
            progress_callback(search_progress)
            progress_bar = None
        else:
            progress_bar = ProgressBar()
            progress_bar.progress = search_progress.to_progress_bar()
            progress_bar.display()

        with MDC(trace_id=trace_id):
            if len(args) > 0:
                msg = f"WARNING: Unsupported positional arguments for fit: {args}"
                self.logger.warning(msg)
                print(msg)
            if len(kwargs) > 0:
                msg = f"WARNING: Unsupported named arguments for fit: {kwargs}"
                self.logger.warning(msg)
                print(msg)

            self.logger.info("Start fit")

            self.__validate_search_keys(self.search_keys)

            # Validate client estimator params
            self._get_client_cat_features(estimator, X, self.search_keys)

            try:
                self.X = X
                self.y = y
                self.eval_set = self._check_eval_set(eval_set, X, self.bundle)
                self.dump_input(trace_id, X, y, self.eval_set)
                self.__inner_fit(
                    trace_id,
                    X,
                    y,
                    self.eval_set,
                    progress_bar,
                    start_time=start_time,
                    exclude_features_sources=exclude_features_sources,
                    calculate_metrics=calculate_metrics,
                    estimator=estimator,
                    scoring=scoring,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                    remove_outliers_calc_metrics=remove_outliers_calc_metrics,
                    progress_callback=progress_callback,
                    search_id_callback=search_id_callback,
                )
                self.logger.info("Fit finished successfully")
                search_progress = SearchProgress(100.0, ProgressStage.FINISHED)
                if progress_bar is not None:
                    progress_bar.progress = search_progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(search_progress)
            except Exception as e:
                search_progress = SearchProgress(100.0, ProgressStage.FAILED)
                if progress_bar is not None:
                    progress_bar.progress = search_progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(search_progress)
                error_message = "Failed on inner fit" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                if len(e.args) > 0 and (
                    "File doesn't intersect with any ADS" in str(e.args[0]) or "Empty intersection" in str(e.args[0])
                ):
                    self.__display_support_link(self.bundle.get("features_info_zero_important_features"))
                elif isinstance(e, ValidationError):
                    self._dump_python_libs()
                    self._show_error(str(e))
                    if self.raise_validation_error:
                        raise e
                else:
                    self._dump_python_libs()
                    self.__display_support_link()
                    raise e
            finally:
                self.logger.info(f"Fit elapsed time: {time.time() - start_time}")

    def fit_transform(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List],
        eval_set: Optional[Union[List[tuple], tuple]] = None,
        *args,
        exclude_features_sources: Optional[List[str]] = None,
        keep_input: bool = True,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        calculate_metrics: Optional[bool] = None,
        scoring: Union[Callable, str, None] = None,
        estimator: Optional[Any] = None,
        remove_outliers_calc_metrics: Optional[bool] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        **kwargs,
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

        calculate_metrics: bool, optional (default=None)
            Whether to calculate and show metrics.

        estimator: sklearn-compatible estimator, optional (default=None)
            Custom estimator for metrics calculation.

        scoring: string or callable, optional (default=None)
            A string or a scorer callable object / function with signature scorer(estimator, X, y).
            If None, the estimator's score method is used.

        remove_outliers_calc_metrics, optional (default=True)
            If True then rows with target ouliers will be dropped on metrics calculation

        Returns
        -------
        X_new: pandas.DataFrame of shape (n_samples, n_features_new)
            Transformed dataframe, enriched with valuable features.
        """

        trace_id = str(uuid.uuid4())
        start_time = time.time()
        with MDC(trace_id=trace_id):
            if len(args) > 0:
                msg = f"WARNING: Unsupported positional arguments for fit_transform: {args}"
                self.logger.warning(msg)
                print(msg)
            if len(kwargs) > 0:
                msg = f"WARNING: Unsupported named arguments for fit_transform: {kwargs}"
                self.logger.warning(msg)
                print(msg)

            self.logger.info("Start fit_transform")

            self.__validate_search_keys(self.search_keys)

            search_progress = SearchProgress(0.0, ProgressStage.START_FIT)
            if progress_callback is not None:
                progress_callback(search_progress)
                progress_bar = None
            else:
                progress_bar = ProgressBar()
                progress_bar.progress = search_progress.to_progress_bar()
                progress_bar.display()
            try:
                self.X = X
                self.y = y
                self.eval_set = self._check_eval_set(eval_set, X, self.bundle)
                self.dump_input(trace_id, X, y, self.eval_set)

                if _num_samples(drop_duplicates(X)) > Dataset.MAX_ROWS:
                    raise ValidationError(self.bundle.get("dataset_too_many_rows_registered").format(Dataset.MAX_ROWS))

                self.__inner_fit(
                    trace_id,
                    X,
                    y,
                    self.eval_set,
                    progress_bar,
                    start_time=start_time,
                    exclude_features_sources=exclude_features_sources,
                    calculate_metrics=calculate_metrics,
                    scoring=scoring,
                    estimator=estimator,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                    remove_outliers_calc_metrics=remove_outliers_calc_metrics,
                    progress_callback=progress_callback,
                )
                self.logger.info("Inner fit finished successfully")
                search_progress = SearchProgress(100.0, ProgressStage.FINISHED)
                if progress_bar is not None:
                    progress_bar.progress = search_progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(search_progress)
            except Exception as e:
                search_progress = SearchProgress(100.0, ProgressStage.FAILED)
                if progress_bar is not None:
                    progress_bar.progress = search_progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(search_progress)
                error_message = "Failed on inner fit" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                if len(e.args) > 0 and (
                    "File doesn't intersect with any ADS" in str(e.args[0]) or "Empty intersection" in str(e.args[0])
                ):
                    self.__display_support_link(self.bundle.get("features_info_zero_important_features"))
                    return None
                elif isinstance(e, ValidationError):
                    self._dump_python_libs()
                    self._show_error(str(e))
                    if self.raise_validation_error:
                        raise e
                    return None
                else:
                    self._dump_python_libs()
                    self.__display_support_link()
                    raise e
            finally:
                self.logger.info(f"Fit elapsed time: {time.time() - start_time}")

            result = self.transform(
                X,
                exclude_features_sources=exclude_features_sources,
                keep_input=keep_input,
                importance_threshold=importance_threshold,
                max_features=max_features,
                trace_id=trace_id,
                silent_mode=True,
                progress_bar=progress_bar,
                progress_callback=progress_callback,
            )
            self.logger.info("Fit_transform finished successfully")
            return result

    def transform(
        self,
        X: pd.DataFrame,
        *args,
        exclude_features_sources: Optional[List[str]] = None,
        keep_input: bool = True,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        trace_id: Optional[str] = None,
        metrics_calculation: bool = False,
        silent_mode=False,
        progress_bar: Optional[ProgressBar] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        **kwargs,
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

        search_progress = SearchProgress(0.0, ProgressStage.START_TRANSFORM)
        if progress_callback is not None:
            progress_callback(search_progress)
            progress_bar = None
        else:
            new_progress = progress_bar is None
            progress_bar = progress_bar or ProgressBar()
            progress_bar.progress = search_progress.to_progress_bar()
            if new_progress:
                progress_bar.display()
        trace_id = trace_id or str(uuid.uuid4())
        with MDC(trace_id=trace_id):
            self.dump_input(trace_id, X)
            if len(args) > 0:
                msg = f"WARNING: Unsupported positional arguments for transform: {args}"
                self.logger.warning(msg)
                print(msg)
            if len(kwargs) > 0:
                msg = f"WARNING: Unsupported named arguments for transform: {kwargs}"
                self.logger.warning(msg)
                print(msg)

            start_time = time.time()
            try:
                result = self.__inner_transform(
                    trace_id,
                    X,
                    exclude_features_sources=exclude_features_sources,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                    metrics_calculation=metrics_calculation,
                    silent_mode=silent_mode,
                    progress_bar=progress_bar,
                )
                self.logger.info("Transform finished successfully")
                search_progress = SearchProgress(100.0, ProgressStage.FINISHED)
                if progress_bar is not None:
                    progress_bar.progress = search_progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(search_progress)
            except Exception as e:
                search_progress = SearchProgress(100.0, ProgressStage.FINISHED)
                if progress_bar is not None:
                    progress_bar.progress = search_progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(search_progress)
                error_message = "Failed on inner transform" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                if len(e.args) > 0 and (
                    "File doesn't intersect with any ADS" in str(e.args[0]) or "Empty intersection" in str(e.args[0])
                ):
                    self.__display_support_link(self.bundle.get("features_info_zero_important_features"))
                    return None
                elif len(e.args) > 0 and (
                    "You have reached the quota limit of trial data usage" in str(e.args[0])
                    or "Current user hasn't access to trial features" in str(e.args[0])
                ):
                    self.__display_support_link(self.bundle.get("trial_quota_limit_riched"))
                    return None
                elif isinstance(e, ValidationError):
                    self._dump_python_libs()
                    self._show_error(str(e))
                    if self.raise_validation_error:
                        raise e
                    return None
                else:
                    if not silent_mode:
                        self._dump_python_libs()
                        self.__display_support_link()
                    raise e
            finally:
                self.logger.info(f"Transform elapsed time: {time.time() - start_time}")

            if result is not None:
                if keep_input:
                    return result
                else:
                    return result.drop(columns=X.columns, errors="ignore")

    def calculate_metrics(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None,
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List, None] = None,
        eval_set: Optional[Union[List[tuple], tuple]] = None,
        *args,
        scoring: Union[Callable, str, None] = None,
        cv: Union[BaseCrossValidator, CVType, None] = None,
        estimator=None,
        exclude_features_sources: Optional[List[str]] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        remove_outliers_calc_metrics: Optional[bool] = None,
        trace_id: Optional[str] = None,
        silent: bool = False,
        progress_bar: Optional[ProgressBar] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Calculate metrics

        Parameters
        ----------
        X: pandas.DataFrame of shape (n_samples, n_features), optional (default=None)
            Input samples. If not passed then X from fit will be used

        y: array-like of shape (n_samples,), optional (default=None)
            Target values. If X not passed then y from fit will be used

        eval_set: List[tuple], optional (default=None)
            List of pairs (X, y) for validation. If X not passed then eval_set from fit will be used

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

        remove_outliers_calc_metrics, optional (default=True)
            If True then rows with target ouliers will be dropped on metrics calculation

        Returns
        -------
        metrics: pandas.DataFrame
            Dataframe with metrics calculated on train and validation datasets.
        """

        trace_id = trace_id or str(uuid.uuid4())
        start_time = time.time()
        with MDC(trace_id=trace_id):
            self.logger.info("Start calculate metrics")
            if len(args) > 0:
                msg = f"WARNING: Unsupported positional arguments for calculate_metrics: {args}"
                self.logger.warning(msg)
                print(msg)
            if len(kwargs) > 0:
                msg = f"WARNING: Unsupported named arguments for calculate_metrics: {kwargs}"
                self.logger.warning(msg)
                print(msg)

            self.__validate_search_keys(self.search_keys, self.search_id)
            effective_X = X if X is not None else self.X
            effective_y = y if y is not None else self.y
            effective_eval_set = eval_set if eval_set is not None else self.eval_set
            effective_eval_set = self._check_eval_set(effective_eval_set, effective_X, self.bundle)

            try:
                self.__log_debug_information(
                    effective_X,
                    effective_y,
                    effective_eval_set,
                    exclude_features_sources=exclude_features_sources,
                    cv=cv if cv is not None else self.cv,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                    scoring=scoring,
                    estimator=estimator,
                    remove_outliers_calc_metrics=remove_outliers_calc_metrics,
                )

                if (
                    self._search_task is None
                    or self._search_task.provider_metadata_v2 is None
                    or len(self._search_task.provider_metadata_v2) == 0
                    or effective_X is None
                    or effective_y is None
                ):
                    raise ValidationError(self.bundle.get("metrics_unfitted_enricher"))

                if X is not None and y is None:
                    raise ValidationError("X passed without y")

                if self.X is None:
                    self.X = X
                if self.y is None:
                    self.y = y
                if self.eval_set is None:
                    self.eval_set = effective_eval_set

                validate_scoring_argument(scoring)

                self._validate_baseline_score(effective_X, effective_eval_set)

                if self._has_paid_features(exclude_features_sources):
                    msg = self.bundle.get("metrics_with_paid_features")
                    self.logger.warning(msg)
                    self.__display_support_link(msg)
                    return None

                cat_features, search_keys_for_metrics = self._get_client_cat_features(
                    estimator, effective_X, self.search_keys
                )

                prepared_data = self._prepare_data_for_metrics(
                    trace_id=trace_id,
                    X=effective_X,
                    y=effective_y,
                    eval_set=effective_eval_set,
                    exclude_features_sources=exclude_features_sources,
                    importance_threshold=importance_threshold,
                    max_features=max_features,
                    remove_outliers_calc_metrics=remove_outliers_calc_metrics,
                    cv_override=cv,
                    search_keys_for_metrics=search_keys_for_metrics,
                    progress_bar=progress_bar,
                    progress_callback=progress_callback,
                    cat_features=cat_features,
                )
                if prepared_data is None:
                    return None
                (
                    validated_X,
                    fitting_X,
                    y_sorted,
                    fitting_enriched_X,
                    enriched_y_sorted,
                    fitting_eval_set_dict,
                    search_keys,
                    groups,
                    _cv,
                ) = prepared_data

                gc.collect()

                print(self.bundle.get("metrics_start"))
                with Spinner():
                    if fitting_X.shape[1] == 0 and fitting_enriched_X.shape[1] == 0:
                        print(self.bundle.get("metrics_no_important_free_features"))
                        self.logger.warning("No client or free relevant ADS features found to calculate metrics")
                        self.warning_counter.increment()
                        return None

                    self._check_train_and_eval_target_distribution(y_sorted, fitting_eval_set_dict)

                    has_date = self._get_date_column(search_keys) is not None
                    model_task_type = self.model_task_type or define_task(y_sorted, has_date, self.logger, silent=True)

                    wrapper = EstimatorWrapper.create(
                        estimator,
                        self.logger,
                        model_task_type,
                        _cv,
                        fitting_enriched_X,
                        scoring,
                        groups=groups,
                        text_features=self.generate_features,
                        has_date=has_date,
                    )
                    metric = wrapper.metric_name
                    multiplier = wrapper.multiplier

                    # 1 If client features are presented - fit and predict with KFold estimator
                    # on etalon features and calculate baseline metric
                    etalon_metric = None
                    baseline_estimator = None
                    custom_loss_add_params = get_additional_params_custom_loss(
                        self.loss, model_task_type, logger=self.logger
                    )
                    if fitting_X.shape[1] > 0:
                        self.logger.info(
                            f"Calculate baseline {metric} on train client features: {fitting_X.columns.to_list()}"
                        )
                        baseline_estimator = EstimatorWrapper.create(
                            estimator,
                            self.logger,
                            model_task_type,
                            _cv,
                            fitting_enriched_X,
                            scoring,
                            cat_features,
                            add_params=custom_loss_add_params,
                            groups=groups,
                            text_features=self.generate_features,
                            has_date=has_date,
                        )
                        etalon_metric = baseline_estimator.cross_val_predict(
                            fitting_X, y_sorted, self.baseline_score_column
                        )
                        if etalon_metric is None:
                            self.logger.info(
                                f"Baseline {metric} on train client features is None (maybe all features was removed)"
                            )
                            baseline_estimator = None
                        else:
                            self.logger.info(f"Baseline {metric} on train client features: {etalon_metric}")

                    # 2 Fit and predict with KFold estimator on enriched tds
                    # and calculate final metric (and uplift)
                    enriched_metric = None
                    uplift = None
                    enriched_estimator = None
                    if set(fitting_X.columns) != set(fitting_enriched_X.columns):
                        self.logger.info(
                            f"Calculate enriched {metric} on train combined "
                            f"features: {fitting_enriched_X.columns.to_list()}"
                        )
                        enriched_estimator = EstimatorWrapper.create(
                            estimator,
                            self.logger,
                            model_task_type,
                            _cv,
                            fitting_enriched_X,
                            scoring,
                            cat_features,
                            add_params=custom_loss_add_params,
                            groups=groups,
                            text_features=self.generate_features,
                            has_date=has_date,
                        )
                        enriched_metric = enriched_estimator.cross_val_predict(fitting_enriched_X, enriched_y_sorted)
                        if enriched_metric is None:
                            self.logger.warning(
                                f"Enriched {metric} on train combined features is None (maybe all features was removed)"
                            )
                            enriched_estimator = None
                        else:
                            self.logger.info(f"Enriched {metric} on train combined features: {enriched_metric}")
                        if etalon_metric is not None and enriched_metric is not None:
                            uplift = (enriched_metric - etalon_metric) * multiplier

                    train_metrics = {
                        self.bundle.get("quality_metrics_segment_header"): self.bundle.get(
                            "quality_metrics_train_segment"
                        ),
                        self.bundle.get("quality_metrics_rows_header"): _num_samples(effective_X),
                    }
                    if model_task_type in [ModelTaskType.BINARY, ModelTaskType.REGRESSION] and is_numeric_dtype(
                        y_sorted
                    ):
                        train_metrics[self.bundle.get("quality_metrics_mean_target_header")] = round(
                            np.mean(effective_y), 4
                        )
                    if etalon_metric is not None:
                        train_metrics[self.bundle.get("quality_metrics_baseline_header").format(metric)] = etalon_metric
                    if enriched_metric is not None:
                        train_metrics[self.bundle.get("quality_metrics_enriched_header").format(metric)] = (
                            enriched_metric
                        )
                    if uplift is not None:
                        train_metrics[self.bundle.get("quality_metrics_uplift_header")] = uplift
                    metrics = [train_metrics]

                    # 3 If eval_set is presented - fit final model on train enriched data and score each
                    # validation dataset and calculate final metric (and uplift)
                    # max_initial_eval_set_hit_rate = self._search_task.get_max_initial_eval_set_hit_rate_v2()
                    if len(fitting_eval_set_dict) > 0:
                        for idx in fitting_eval_set_dict.keys():
                            # eval_hit_rate = max_initial_eval_set_hit_rate[idx + 1]

                            (
                                eval_X_sorted,
                                eval_y_sorted,
                                enriched_eval_X_sorted,
                                enriched_eval_y_sorted,
                            ) = fitting_eval_set_dict[idx]

                            if baseline_estimator is not None:
                                self.logger.info(
                                    f"Calculate baseline {metric} on eval set {idx + 1} "
                                    f"on client features: {eval_X_sorted.columns.to_list()}"
                                )
                                etalon_eval_metric = baseline_estimator.calculate_metric(
                                    eval_X_sorted, eval_y_sorted, self.baseline_score_column
                                )
                                self.logger.info(
                                    f"Baseline {metric} on eval set {idx + 1} client features: {etalon_eval_metric}"
                                )
                            else:
                                etalon_eval_metric = None

                            if enriched_estimator is not None:
                                self.logger.info(
                                    f"Calculate enriched {metric} on eval set {idx + 1} "
                                    f"on combined features: {enriched_eval_X_sorted.columns.to_list()}"
                                )
                                enriched_eval_metric = enriched_estimator.calculate_metric(
                                    enriched_eval_X_sorted, enriched_eval_y_sorted
                                )
                                self.logger.info(
                                    f"Enriched {metric} on eval set {idx + 1} combined features: {enriched_eval_metric}"
                                )
                            else:
                                enriched_eval_metric = None

                            if etalon_eval_metric is not None and enriched_eval_metric is not None:
                                eval_uplift = (enriched_eval_metric - etalon_eval_metric) * multiplier
                            else:
                                eval_uplift = None

                            effective_eval_set = eval_set if eval_set is not None else self.eval_set
                            eval_metrics = {
                                self.bundle.get("quality_metrics_segment_header"): self.bundle.get(
                                    "quality_metrics_eval_segment"
                                ).format(idx + 1),
                                self.bundle.get("quality_metrics_rows_header"): _num_samples(
                                    effective_eval_set[idx][0]
                                ),
                                # self.bundle.get("quality_metrics_match_rate_header"): eval_hit_rate,
                            }
                            if model_task_type in [ModelTaskType.BINARY, ModelTaskType.REGRESSION] and is_numeric_dtype(
                                eval_y_sorted
                            ):
                                eval_metrics[self.bundle.get("quality_metrics_mean_target_header")] = round(
                                    np.mean(effective_eval_set[idx][1]), 4
                                )
                            if etalon_eval_metric is not None:
                                eval_metrics[self.bundle.get("quality_metrics_baseline_header").format(metric)] = (
                                    etalon_eval_metric
                                )
                            if enriched_eval_metric is not None:
                                eval_metrics[self.bundle.get("quality_metrics_enriched_header").format(metric)] = (
                                    enriched_eval_metric
                                )
                            if eval_uplift is not None:
                                eval_metrics[self.bundle.get("quality_metrics_uplift_header")] = eval_uplift

                            metrics.append(eval_metrics)

                    metrics_df = pd.DataFrame(metrics)
                    mean_target_hdr = self.bundle.get("quality_metrics_mean_target_header")
                    if mean_target_hdr in metrics_df.columns:
                        metrics_df[mean_target_hdr] = metrics_df[mean_target_hdr].astype("float64")
                    do_without_pandas_limits(
                        lambda: self.logger.info(f"Metrics calculation finished successfully:\n{metrics_df}")
                    )

                    uplift_col = self.bundle.get("quality_metrics_uplift_header")
                    date_column = self._get_date_column(search_keys)
                    if (
                        uplift_col in metrics_df.columns
                        and (metrics_df[uplift_col] < 0).any()
                        and model_task_type == ModelTaskType.REGRESSION
                        and self.cv not in [CVType.time_series, CVType.blocked_time_series]
                        and date_column is not None
                        and is_time_series(validated_X, date_column)
                    ):
                        msg = self.bundle.get("metrics_negative_uplift_without_cv")
                        self.logger.warning(msg)
                        self.__display_support_link(msg)
                    elif uplift_col in metrics_df.columns and (metrics_df[uplift_col] < 0).any():
                        self.logger.warning("Uplift is negative")

                    return metrics_df
            except Exception as e:
                error_message = "Failed to calculate metrics" + (
                    " with validation error" if isinstance(e, ValidationError) else ""
                )
                self.logger.exception(error_message)
                if len(e.args) > 0 and (
                    "You have reached the quota limit of trial data usage" in str(e.args[0])
                    or "Current user hasn't access to trial features" in str(e.args[0])
                ):
                    self.__display_support_link(self.bundle.get("trial_quota_limit_riched"))
                elif isinstance(e, ValidationError):
                    self._dump_python_libs()
                    self._show_error(str(e))
                    if self.raise_validation_error:
                        raise e
                else:
                    if not silent:
                        self._dump_python_libs()
                        self.__display_support_link()
                    raise e
            finally:
                self.logger.info(f"Calculating metrics elapsed time: {time.time() - start_time}")

    def _check_train_and_eval_target_distribution(self, y, eval_set_dict):
        uneven_distribution = False
        for eval_set in eval_set_dict.values():
            _, eval_y, _, _ = eval_set
            res = ks_2samp(y, eval_y)
            if res[1] < 0.05:
                uneven_distribution = True
        if uneven_distribution:
            msg = self.bundle.get("uneven_eval_target_distribution")
            print(msg)
            self.logger.warning(msg)

    def _has_features_with_commercial_schema(
        self, commercial_schema: str, exclude_features_sources: Optional[List[str]]
    ) -> bool:
        return len(self._get_features_with_commercial_schema(commercial_schema, exclude_features_sources)) > 0

    def _get_features_with_commercial_schema(
        self, commercial_schema: str, exclude_features_sources: Optional[List[str]]
    ) -> List[str]:
        if exclude_features_sources:
            filtered_features_info = self._internal_features_info[
                ~self._internal_features_info[self.bundle.get("features_info_name")].isin(exclude_features_sources)
            ]
        else:
            filtered_features_info = self._internal_features_info
        return list(
            filtered_features_info.loc[
                filtered_features_info[self.bundle.get("features_info_commercial_schema")] == commercial_schema,
                self.bundle.get("features_info_name"),
            ].values
        )

    def _has_paid_features(self, exclude_features_sources: Optional[List[str]]) -> bool:
        return self._has_features_with_commercial_schema(CommercialSchema.PAID.value, exclude_features_sources)

    def _extend_x(self, x: pd.DataFrame, is_demo_dataset: bool) -> Tuple[pd.DataFrame, Dict[str, SearchKey]]:
        search_keys = self.search_keys.copy()
        search_keys = self.__prepare_search_keys(x, search_keys, is_demo_dataset, is_transform=True, silent_mode=True)

        extended_X = x.copy()
        generated_features = []
        date_column = self._get_date_column(search_keys)
        if date_column is not None:
            converter = DateTimeSearchKeyConverter(date_column, self.date_format, self.logger, self.bundle)
            extended_X = converter.convert(extended_X, keep_time=True)
            generated_features.extend(converter.generated_features)
        email_column = self._get_email_column(search_keys)
        hem_column = self._get_hem_column(search_keys)
        if email_column:
            converter = EmailSearchKeyConverter(email_column, hem_column, search_keys, [], self.logger)
            extended_X = converter.convert(extended_X)
            generated_features.extend(converter.generated_features)
        if (
            self.detect_missing_search_keys
            and list(search_keys.values()) == [SearchKey.DATE]
            and self.country_code is None
        ):
            converter = IpToCountrySearchKeyConverter(search_keys, self.logger)
            extended_X = converter.convert(extended_X)
        generated_features = [f for f in generated_features if f in self.fit_generated_features]

        return extended_X, search_keys

    def _is_input_same_as_fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None,
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List, None] = None,
        eval_set: Optional[List[tuple]] = None,
    ) -> Tuple:
        if X is None:
            return True, self.X, self.y, self.eval_set

        checked_eval_set = self._check_eval_set(eval_set, X, self.bundle)

        if (
            X is self.X
            and y is self.y
            and (
                (checked_eval_set == [] and self.eval_set == [])
                or (
                    len(checked_eval_set) == len(self.eval_set)
                    and all(
                        [
                            eval_x is self_eval_x and eval_y is self_eval_y
                            for ((eval_x, eval_y), (self_eval_x, self_eval_y)) in zip(checked_eval_set, self.eval_set)
                        ]
                    )
                )
            )
        ):
            return True, self.X, self.y, self.eval_set
        else:
            self.logger.info("Passed X, y and eval_set that differs from passed on fit. Transform will be used")
            return False, X, y, checked_eval_set

    def _get_cv_and_groups(
        self,
        X: pd.DataFrame,
        cv_override: Union[BaseCrossValidator, CVType, str, None],
        search_keys: Dict[str, SearchKey],
    ) -> Tuple[BaseCrossValidator, Optional[np.ndarray]]:
        _cv = cv_override or self.cv
        group_columns = sorted(self._get_group_columns(X, search_keys))
        groups = None

        if not isinstance(_cv, BaseCrossValidator):
            date_column = self._get_date_column(search_keys)
            date_series = X[date_column] if date_column is not None else None
            _cv, groups = CVConfig(
                _cv, date_series, self.random_state, self._search_task.get_shuffle_kfold(), group_columns=group_columns
            ).get_cv_and_groups(X)
        else:
            from sklearn import __version__ as sklearn_version

            try:
                from sklearn.model_selection._split import GroupsConsumerMixin

                if isinstance(_cv, GroupsConsumerMixin):
                    groups = get_groups(X, group_columns)
            except ImportError:
                print(f"WARNING: Unsupported scikit-learn version {sklearn_version}. Restart kernel and try again")
                self.logger.exception(
                    f"Failed to import GroupsConsumerMixin to check CV. Version of sklearn: {sklearn_version}"
                )

        return _cv, groups

    def _get_client_cat_features(
        self, estimator: Optional[Any], X: pd.DataFrame, search_keys: Dict[str, SearchKey]
    ) -> Optional[List[str]]:
        cat_features = None
        search_keys_for_metrics = []
        if (
            estimator is not None
            and hasattr(estimator, "get_param")
            and estimator.get_param("cat_features") is not None
        ):
            cat_features = estimator.get_param("cat_features")
            if len(cat_features) > 0:
                if all([isinstance(f, int) for f in cat_features]):
                    cat_features = [X.columns[i] for i in cat_features]
                self.logger.info(f"Collected categorical features {cat_features} from user estimator")
                for cat_feature in cat_features:
                    if cat_feature in search_keys:
                        if search_keys[cat_feature] in [SearchKey.COUNTRY, SearchKey.POSTAL_CODE]:
                            search_keys_for_metrics.append(cat_feature)
                        else:
                            raise ValidationError(self.bundle.get("cat_feature_search_key").format(cat_feature))
        return cat_features, search_keys_for_metrics

    def _prepare_data_for_metrics(
        self,
        trace_id: str,
        X: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None,
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List, None] = None,
        eval_set: Optional[Union[List[tuple], tuple]] = None,
        exclude_features_sources: Optional[List[str]] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        remove_outliers_calc_metrics: Optional[bool] = None,
        cv_override: Union[BaseCrossValidator, CVType, str, None] = None,
        search_keys_for_metrics: Optional[List[str]] = None,
        progress_bar: Optional[ProgressBar] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        cat_features: Optional[List[str]] = None,
    ):
        is_input_same_as_fit, X, y, eval_set = self._is_input_same_as_fit(X, y, eval_set)
        is_demo_dataset = hash_input(X, y, eval_set) in DEMO_DATASET_HASHES
        validated_X = self._validate_X(X)
        validated_y = self._validate_y(validated_X, y)
        checked_eval_set = self._check_eval_set(eval_set, X, self.bundle)
        validated_eval_set = (
            [self._validate_eval_set_pair(validated_X, eval_set_pair) for eval_set_pair in checked_eval_set]
            if checked_eval_set
            else None
        )

        sampled_data = self._sample_data_for_metrics(
            trace_id,
            validated_X,
            validated_y,
            validated_eval_set,
            exclude_features_sources,
            is_input_same_as_fit,
            is_demo_dataset,
            remove_outliers_calc_metrics,
            progress_bar,
            progress_callback,
        )
        X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, search_keys = dataclasses.astuple(sampled_data)

        excluding_search_keys = list(search_keys.keys())
        if search_keys_for_metrics is not None and len(search_keys_for_metrics) > 0:
            excluding_search_keys = [sk for sk in excluding_search_keys if sk not in search_keys_for_metrics]

        client_features = [
            c
            for c in X_sampled.columns.to_list()
            if c
            not in (
                excluding_search_keys
                + list(self.fit_dropped_features)
                + [DateTimeSearchKeyConverter.DATETIME_COL, SYSTEM_RECORD_ID, ENTITY_SYSTEM_RECORD_ID]
            )
        ]

        filtered_enriched_features = self.__filtered_enriched_features(
            importance_threshold,
            max_features,
        )

        X_sorted, y_sorted = self._sort_by_system_record_id(X_sampled, y_sampled, self.cv)
        enriched_X_sorted, enriched_y_sorted = self._sort_by_system_record_id(enriched_X, y_sampled, self.cv)

        cv, groups = self._get_cv_and_groups(enriched_X_sorted, cv_override, search_keys)

        existing_filtered_enriched_features = [c for c in filtered_enriched_features if c in enriched_X_sorted.columns]

        fitting_X = X_sorted[client_features].copy()
        fitting_enriched_X = enriched_X_sorted[client_features + existing_filtered_enriched_features].copy()

        # Detect and drop high cardinality columns in train
        columns_with_high_cardinality = FeaturesValidator.find_high_cardinality(fitting_X)
        non_excluding_columns = (self.generate_features or []) + (cat_features or [])
        columns_with_high_cardinality = [c for c in columns_with_high_cardinality if c not in non_excluding_columns]
        if len(columns_with_high_cardinality) > 0:
            self.logger.warning(
                f"High cardinality columns {columns_with_high_cardinality} will be dropped for metrics calculation"
            )
            fitting_X = fitting_X.drop(columns=columns_with_high_cardinality, errors="ignore")
            fitting_enriched_X = fitting_enriched_X.drop(columns=columns_with_high_cardinality, errors="ignore")

        # Detect and drop constant columns
        constant_columns = FeaturesValidator.find_constant_features(fitting_X)
        if len(constant_columns) > 0:
            self.logger.warning(f"Constant columns {constant_columns} will be dropped for metrics calculation")
            fitting_X = fitting_X.drop(columns=constant_columns, errors="ignore")
            fitting_enriched_X = fitting_enriched_X.drop(columns=constant_columns, errors="ignore")

        # Remove datetime features
        datetime_features = [
            f for f in fitting_X.columns if is_datetime64_any_dtype(fitting_X[f]) or is_period_dtype(fitting_X[f])
        ]
        if len(datetime_features) > 0:
            self.logger.warning(self.bundle.get("dataset_date_features").format(datetime_features))
            fitting_X = fitting_X.drop(columns=datetime_features, errors="ignore")
            fitting_enriched_X = fitting_enriched_X.drop(columns=datetime_features, errors="ignore")

        bool_columns = []
        for col in fitting_X.columns:
            if is_bool(fitting_X[col]):
                bool_columns.append(col)
                fitting_X[col] = fitting_X[col].astype(str)
                fitting_enriched_X[col] = fitting_enriched_X[col].astype(str)
        if len(bool_columns) > 0:
            self.logger.warning(f"Bool columns {bool_columns} was converted to string for metrics calculation")

        decimal_columns_to_fix = find_numbers_with_decimal_comma(fitting_X)
        if len(decimal_columns_to_fix) > 0:
            self.logger.warning(f"Convert strings with decimal comma to float: {decimal_columns_to_fix}")
            for col in decimal_columns_to_fix:
                fitting_X[col] = fitting_X[col].astype("string").str.replace(",", ".", regex=False).astype(np.float64)
                fitting_enriched_X[col] = (
                    fitting_enriched_X[col].astype("string").str.replace(",", ".", regex=False).astype(np.float64)
                )

        fitting_eval_set_dict = {}
        for idx, eval_tuple in eval_set_sampled_dict.items():
            eval_X_sampled, enriched_eval_X, eval_y_sampled = eval_tuple
            eval_X_sorted, eval_y_sorted = self._sort_by_system_record_id(eval_X_sampled, eval_y_sampled, self.cv)
            enriched_eval_X_sorted, enriched_eval_y_sorted = self._sort_by_system_record_id(
                enriched_eval_X, eval_y_sampled, self.cv
            )
            fitting_eval_X = eval_X_sorted[client_features].copy()
            fitting_enriched_eval_X = enriched_eval_X_sorted[
                client_features + existing_filtered_enriched_features
            ].copy()

            # # Drop high cardinality features in eval set
            if len(columns_with_high_cardinality) > 0:
                fitting_eval_X = fitting_eval_X.drop(columns=columns_with_high_cardinality, errors="ignore")
                fitting_enriched_eval_X = fitting_enriched_eval_X.drop(
                    columns=columns_with_high_cardinality, errors="ignore"
                )
            # Drop constant features in eval_set
            if len(constant_columns) > 0:
                fitting_eval_X = fitting_eval_X.drop(columns=constant_columns, errors="ignore")
                fitting_enriched_eval_X = fitting_enriched_eval_X.drop(columns=constant_columns, errors="ignore")
            # Drop datetime features in eval_set
            if len(datetime_features) > 0:
                fitting_eval_X = fitting_eval_X.drop(columns=datetime_features, errors="ignore")
                fitting_enriched_eval_X = fitting_enriched_eval_X.drop(columns=datetime_features, errors="ignore")
            # Convert bool to string in eval_set
            if len(bool_columns) > 0:
                fitting_eval_X[col] = fitting_eval_X[col].astype(str)
                fitting_enriched_eval_X[col] = fitting_enriched_eval_X[col].astype(str)
            # Correct string features with decimal commas
            if len(decimal_columns_to_fix) > 0:
                for col in decimal_columns_to_fix:
                    fitting_eval_X[col] = (
                        fitting_eval_X[col].astype("string").str.replace(",", ".", regex=False).astype(np.float64)
                    )
                    fitting_enriched_eval_X[col] = (
                        fitting_enriched_eval_X[col]
                        .astype("string")
                        .str.replace(",", ".", regex=False)
                        .astype(np.float64)
                    )

            fitting_eval_set_dict[idx] = (
                fitting_eval_X,
                eval_y_sorted,
                fitting_enriched_eval_X,
                enriched_eval_y_sorted,
            )

        return (
            validated_X,
            fitting_X,
            y_sorted,
            fitting_enriched_X,
            enriched_y_sorted,
            fitting_eval_set_dict,
            search_keys,
            groups,
            cv,
        )

    @dataclass
    class _SampledDataForMetrics:
        X_sampled: pd.DataFrame
        y_sampled: pd.Series
        enriched_X: pd.DataFrame
        eval_set_sampled_dict: Dict[int, Tuple[pd.DataFrame, pd.Series]]
        search_keys: Dict[str, SearchKey]

    def _sample_data_for_metrics(
        self,
        trace_id: str,
        validated_X: Union[pd.DataFrame, pd.Series, np.ndarray, None],
        validated_y: Union[pd.DataFrame, pd.Series, np.ndarray, List, None],
        eval_set: Optional[List[tuple]],
        exclude_features_sources: Optional[List[str]],
        is_input_same_as_fit: bool,
        is_demo_dataset: bool,
        remove_outliers_calc_metrics: Optional[bool],
        progress_bar: Optional[ProgressBar],
        progress_callback: Optional[Callable[[SearchProgress], Any]],
    ) -> _SampledDataForMetrics:
        if self.__cached_sampled_datasets is not None and is_input_same_as_fit and remove_outliers_calc_metrics is None:
            self.logger.info("Cached enriched dataset found - use it")
            return self.__get_sampled_cached_enriched(exclude_features_sources)
        elif len(self.feature_importances_) == 0:
            self.logger.info("No external features selected. So use only input datasets for metrics calculation")
            return self.__sample_only_input(validated_X, validated_y, eval_set, is_demo_dataset)
        # TODO save and check if dataset was deduplicated - use imbalance branch for such case
        elif not self.imbalanced and not exclude_features_sources and is_input_same_as_fit:
            self.logger.info("Dataset is not imbalanced, so use enriched_X from fit")
            return self.__sample_balanced(eval_set, trace_id, remove_outliers_calc_metrics)
        else:
            self.logger.info("Dataset is imbalanced or exclude_features_sources or X was passed. Run transform")
            print(self.bundle.get("prepare_data_for_metrics"))
            return self.__sample_imbalanced(
                validated_X,
                validated_y,
                eval_set,
                is_demo_dataset,
                exclude_features_sources,
                trace_id,
                progress_bar,
                progress_callback,
            )

    def __get_sampled_cached_enriched(self, exclude_features_sources: Optional[List[str]]) -> _SampledDataForMetrics:
        X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, search_keys = self.__cached_sampled_datasets
        if exclude_features_sources:
            enriched_X = enriched_X.drop(columns=exclude_features_sources, errors="ignore")

        return self.__mk_sampled_data_tuple(X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, search_keys)

    def __sample_only_input(
        self, validated_X: pd.DataFrame, validated_y: pd.Series, eval_set: Optional[List[tuple]], is_demo_dataset: bool
    ) -> _SampledDataForMetrics:
        eval_set_sampled_dict = {}

        df = validated_X.copy()
        df[TARGET] = validated_y

        if eval_set is not None:
            df[EVAL_SET_INDEX] = 0
            for idx, (eval_x, eval_y) in enumerate(eval_set):
                eval_xy = eval_x.copy()
                eval_xy[TARGET] = eval_y
                eval_xy[EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_xy])

        df = clean_full_duplicates(df, logger=self.logger, silent=True, bundle=self.bundle)

        num_samples = _num_samples(df)
        sample_threshold, sample_rows = (
            (Dataset.FIT_SAMPLE_WITH_EVAL_SET_THRESHOLD, Dataset.FIT_SAMPLE_WITH_EVAL_SET_ROWS)
            if eval_set is not None
            else (Dataset.FIT_SAMPLE_THRESHOLD, Dataset.FIT_SAMPLE_ROWS)
        )
        if num_samples > sample_threshold:
            self.logger.info(f"Downsampling from {num_samples} to {sample_rows}")
            df = df.sample(n=sample_rows, random_state=self.random_state)

        df_extended, search_keys = self._extend_x(df, is_demo_dataset)
        df_extended = self.__add_fit_system_record_id(df_extended, {}, search_keys, SYSTEM_RECORD_ID)

        train_df = df_extended.query(f"{EVAL_SET_INDEX} == 0") if eval_set is not None else df_extended
        X_sampled = train_df.drop(columns=[TARGET, EVAL_SET_INDEX], errors="ignore")
        y_sampled = train_df[TARGET].copy()
        enriched_X = X_sampled

        if eval_set is not None:
            for idx in range(len(eval_set)):
                eval_xy_sampled = df_extended.query(f"{EVAL_SET_INDEX} == {idx + 1}")
                eval_X_sampled = eval_xy_sampled.drop(columns=[TARGET, EVAL_SET_INDEX], errors="ignore")
                eval_y_sampled = eval_xy_sampled[TARGET].copy()
                enriched_eval_X = eval_X_sampled
                eval_set_sampled_dict[idx] = (eval_X_sampled, enriched_eval_X, eval_y_sampled)
        self.__cached_sampled_datasets = (X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, search_keys)

        return self.__mk_sampled_data_tuple(X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, search_keys)

    def __sample_balanced(
        self,
        eval_set: Optional[List[tuple]],
        trace_id: str,
        remove_outliers_calc_metrics: Optional[bool],
    ) -> _SampledDataForMetrics:
        eval_set_sampled_dict = {}
        search_keys = self.fit_search_keys

        rows_to_drop = None
        has_date = self._get_date_column(search_keys) is not None
        task_type = self.model_task_type or define_task(
            self.df_with_original_index[TARGET], has_date, self.logger, silent=True
        )
        if task_type == ModelTaskType.REGRESSION:
            target_outliers_df = self._search_task.get_target_outliers(trace_id)
            if target_outliers_df is not None and len(target_outliers_df) > 0:
                outliers = pd.merge(
                    self.df_with_original_index,
                    target_outliers_df,
                    on=ENTITY_SYSTEM_RECORD_ID,
                    how="inner",
                )
                top_outliers = outliers.sort_values(by=TARGET, ascending=False)[TARGET].head(3)
                if remove_outliers_calc_metrics is None or remove_outliers_calc_metrics is True:
                    rows_to_drop = outliers
                    not_msg = ""
                else:
                    not_msg = "not "
                msg = self.bundle.get("target_outliers_warning").format(len(target_outliers_df), top_outliers, not_msg)
                print(msg)
                self.logger.warning(msg)

        # index in each dataset (X, eval set) may be reordered and non unique, but index in validated datasets
        # can differs from it
        fit_features = self._search_task.get_all_initial_raw_features(trace_id, metrics_calculation=True)
        enriched_Xy, enriched_eval_sets = self.__enrich(
            self.df_with_original_index,
            fit_features,
            rows_to_drop=rows_to_drop,
            drop_system_record_id=False,
        )

        x_columns = [c for c in self.df_with_original_index.columns if c not in [EVAL_SET_INDEX, TARGET]]
        X_sampled = enriched_Xy[x_columns].copy()
        y_sampled = enriched_Xy[TARGET].copy()
        enriched_X = enriched_Xy.drop(columns=[TARGET, EVAL_SET_INDEX], errors="ignore")

        self.logger.info(f"Shape of enriched_X: {enriched_X.shape}")
        self.logger.info(f"Shape of X after sampling: {X_sampled.shape}")
        self.logger.info(f"Shape of y after sampling: {len(y_sampled)}")

        if eval_set is not None:
            if len(enriched_eval_sets) != len(eval_set):
                raise ValidationError(
                    self.bundle.get("metrics_eval_set_count_diff").format(len(enriched_eval_sets), len(eval_set))
                )

            for idx in range(len(eval_set)):
                eval_X_sampled = enriched_eval_sets[idx + 1][x_columns].copy()
                eval_y_sampled = enriched_eval_sets[idx + 1][TARGET].copy()
                enriched_eval_X = enriched_eval_sets[idx + 1].drop(columns=[TARGET, EVAL_SET_INDEX])
                eval_set_sampled_dict[idx] = (eval_X_sampled, enriched_eval_X, eval_y_sampled)

        self.__cached_sampled_datasets = (X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, search_keys)

        return self.__mk_sampled_data_tuple(X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, search_keys)

    def __sample_imbalanced(
        self,
        validated_X: pd.DataFrame,
        validated_y: pd.Series,
        eval_set: Optional[List[tuple]],
        is_demo_dataset: bool,
        exclude_features_sources: Optional[List[str]],
        trace_id: str,
        progress_bar: Optional[ProgressBar],
        progress_callback: Optional[Callable[[SearchProgress], Any]],
    ) -> _SampledDataForMetrics:
        eval_set_sampled_dict = {}
        if eval_set is not None:
            self.logger.info("Transform with eval_set")
            # concatenate X and eval_set with eval_set_index
            df = validated_X.copy()
            df[TARGET] = validated_y
            df[EVAL_SET_INDEX] = 0
            for idx, eval_pair in enumerate(eval_set):
                eval_x, eval_y = self._validate_eval_set_pair(validated_X, eval_pair)
                eval_df_with_index = eval_x.copy()
                eval_df_with_index[TARGET] = eval_y
                eval_df_with_index[EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_df_with_index])

            df = clean_full_duplicates(df, logger=self.logger, silent=True, bundle=self.bundle)

            # downsample if need to eval_set threshold
            num_samples = _num_samples(df)
            if num_samples > Dataset.FIT_SAMPLE_WITH_EVAL_SET_THRESHOLD:
                self.logger.info(f"Downsampling from {num_samples} to {Dataset.FIT_SAMPLE_WITH_EVAL_SET_ROWS}")
                df = df.sample(n=Dataset.FIT_SAMPLE_WITH_EVAL_SET_ROWS, random_state=self.random_state)

            eval_set_sampled_dict = {}

            tmp_target_name = "__target"
            df = df.rename(columns={TARGET: tmp_target_name})

            enriched_df = self.__inner_transform(
                trace_id,
                df,
                exclude_features_sources=exclude_features_sources,
                silent_mode=True,
                metrics_calculation=True,
                progress_bar=progress_bar,
                progress_callback=progress_callback,
                add_fit_system_record_id=True,
            )
            if enriched_df is None:
                return None

            enriched_df = enriched_df.rename(columns={tmp_target_name: TARGET})

            x_columns = [
                c
                for c in (validated_X.columns.tolist() + self.fit_generated_features + [SYSTEM_RECORD_ID])
                if c in enriched_df.columns
            ]

            enriched_Xy = enriched_df.query(f"{EVAL_SET_INDEX} == 0")
            X_sampled = enriched_Xy[x_columns].copy()
            y_sampled = enriched_Xy[TARGET].copy()
            enriched_X = enriched_Xy.drop(columns=[TARGET, EVAL_SET_INDEX])

            for idx in range(len(eval_set)):
                enriched_eval_xy = enriched_df.query(f"{EVAL_SET_INDEX} == {idx + 1}")
                eval_x_sampled = enriched_eval_xy[x_columns].copy()
                eval_y_sampled = enriched_eval_xy[TARGET].copy()
                enriched_eval_x = enriched_eval_xy.drop(columns=[TARGET, EVAL_SET_INDEX])
                eval_set_sampled_dict[idx] = (eval_x_sampled, enriched_eval_x, eval_y_sampled)
        else:
            self.logger.info("Transform without eval_set")
            df = validated_X.copy()

            df[TARGET] = validated_y

            df = clean_full_duplicates(df, logger=self.logger, silent=True, bundle=self.bundle)

            num_samples = _num_samples(df)
            if num_samples > Dataset.FIT_SAMPLE_THRESHOLD:
                self.logger.info(f"Downsampling from {num_samples} to {Dataset.FIT_SAMPLE_ROWS}")
                df = df.sample(n=Dataset.FIT_SAMPLE_ROWS, random_state=self.random_state)

            tmp_target_name = "__target"
            df = df.rename(columns={TARGET: tmp_target_name})

            enriched_Xy = self.__inner_transform(
                trace_id,
                df,
                exclude_features_sources=exclude_features_sources,
                silent_mode=True,
                metrics_calculation=True,
                progress_bar=progress_bar,
                progress_callback=progress_callback,
                add_fit_system_record_id=True,
            )
            if enriched_Xy is None:
                return None

            enriched_Xy = enriched_Xy.rename(columns={tmp_target_name: TARGET})

            x_columns = [
                c
                for c in (validated_X.columns.tolist() + self.fit_generated_features + [SYSTEM_RECORD_ID])
                if c in enriched_Xy.columns
            ]

            X_sampled = enriched_Xy[x_columns].copy()
            y_sampled = enriched_Xy[TARGET].copy()
            enriched_X = enriched_Xy.drop(columns=TARGET)

        self.__cached_sampled_datasets = (X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, self.search_keys)

        return self.__mk_sampled_data_tuple(X_sampled, y_sampled, enriched_X, eval_set_sampled_dict, self.search_keys)

    def __mk_sampled_data_tuple(
        self,
        X_sampled: pd.DataFrame,
        y_sampled: pd.Series,
        enriched_X: pd.DataFrame,
        eval_set_sampled_dict: Dict,
        search_keys: Dict,
    ):
        search_keys = {k: v for k, v in search_keys.items() if k in X_sampled.columns.to_list()}
        return FeaturesEnricher._SampledDataForMetrics(
            X_sampled=X_sampled,
            y_sampled=y_sampled,
            enriched_X=enriched_X,
            eval_set_sampled_dict=eval_set_sampled_dict,
            search_keys=search_keys,
        )

    def get_search_id(self) -> Optional[str]:
        """Returns search_id of the fitted enricher. Not available before a successful fit."""
        return self._search_task.search_task_id if self._search_task else None

    def get_features_info(self) -> pd.DataFrame:
        """Returns pandas.DataFrame with SHAP values and other info for each feature."""
        if self._search_task is None or self._search_task.summary is None:
            msg = self.bundle.get("features_unfitted_enricher")
            self.logger.warning(msg)
            raise NotFittedError(msg)

        return self.features_info

    def get_progress(self, trace_id: Optional[str] = None, search_task: Optional[SearchTask] = None) -> SearchProgress:
        search_task = search_task or self._search_task
        if search_task is not None:
            trace_id = trace_id or uuid.uuid4()
            return search_task.get_progress(trace_id)

    def get_transactional_transform_api(self):
        if self.api_key is None:
            raise ValidationError(self.bundle.get("transactional_transform_unregistered"))
        if self._search_task is None:
            raise ValidationError(self.bundle.get("transactional_transform_unfited"))

        def key_example(key: SearchKey):
            if key == SearchKey.COUNTRY:
                return "US"
            if key == SearchKey.DATE:
                return "2020-01-01"
            if key == SearchKey.DATETIME:
                return "2020-01-01T00:12:00"
            if key == SearchKey.EMAIL:
                return "test@email.com"
            if key == SearchKey.HEM:
                return "test_hem"
            if key == SearchKey.IP:
                return "127.0.0.1"
            if key == SearchKey.PHONE:
                return "1029384756"
            if key == SearchKey.POSTAL_CODE:
                return "12345678"
            return "test_value"

        file_metadata = self._search_task.get_file_metadata(str(uuid.uuid4()))
        search_keys = file_metadata.search_types()
        if SearchKey.IPV6_ADDRESS in search_keys:
            search_keys.remove(SearchKey.IPV6_ADDRESS)

        keys = "{" + ", ".join([f'"{key.name}": "{key_example(key)}"' for key in search_keys]) + "}"
        features_for_transform = self._search_task.get_features_for_transform()
        if features_for_transform:
            original_features_for_transform = [
                c.originalName or c.name for c in file_metadata.columns if c.name in features_for_transform
            ]
            features_section = (
                ', "features": {'
                + ", ".join([f'"{feature}": "test_value"' for feature in original_features_for_transform])
                + "}"
            )
        else:
            features_section = ""

        search_id = self._search_task.search_task_id
        api_example = f"""curl 'https://search.upgini.com/online/api/http_inference_trigger?search_id={search_id}' \\
    -H 'Authorization: {self.api_key}' \\
    -H 'Content-Type: application/json' \\
    -d '{{"search_keys": {keys}{features_section}}}'"""
        return api_example

    def _get_copy_of_runtime_parameters(self) -> RuntimeParameters:
        return RuntimeParameters(properties=self.runtime_parameters.properties.copy())

    def __inner_transform(
        self,
        trace_id: str,
        X: pd.DataFrame,
        *,
        exclude_features_sources: Optional[List[str]] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        metrics_calculation: bool = False,
        silent_mode: bool = False,
        progress_bar: Optional[ProgressBar] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        add_fit_system_record_id: bool = False,
    ) -> pd.DataFrame:
        if self._search_task is None:
            raise NotFittedError(self.bundle.get("transform_unfitted_enricher"))

        start_time = time.time()
        with MDC(trace_id=trace_id):
            self.logger.info("Start transform")
            self.__log_debug_information(X, exclude_features_sources=exclude_features_sources)

            self.__validate_search_keys(self.search_keys, self.search_id)

            if len(self.feature_names_) == 0:
                self.logger.warning(self.bundle.get("no_important_features_for_transform"))
                return X

            if self._has_paid_features(exclude_features_sources):
                msg = self.bundle.get("transform_with_paid_features")
                self.logger.warning(msg)
                self.__display_support_link(msg)
                return None

            if not metrics_calculation:
                transform_usage = self.rest_client.get_current_transform_usage(trace_id)
                self.logger.info(f"Current transform usage: {transform_usage}. Transforming {len(X)} rows")
                if transform_usage.has_limit:
                    if len(X) > transform_usage.rest_rows:
                        msg = self.bundle.get("transform_usage_warning").format(len(X), transform_usage.rest_rows)
                        self.logger.warning(msg)
                        print(msg)
                        show_request_quote_button()
                        return None
                    else:
                        msg = self.bundle.get("transform_usage_info").format(
                            transform_usage.limit, transform_usage.transformed_rows
                        )
                        self.logger.info(msg)
                        print(msg)

            validated_X = self._validate_X(X, is_transform=True)

            is_demo_dataset = hash_input(validated_X) in DEMO_DATASET_HASHES

            columns_to_drop = [c for c in validated_X.columns if c in self.feature_names_]
            if len(columns_to_drop) > 0:
                msg = self.bundle.get("x_contains_enriching_columns").format(columns_to_drop)
                self.logger.warning(msg)
                print(msg)
                validated_X = validated_X.drop(columns=columns_to_drop)

            search_keys = self.search_keys.copy()
            search_keys = self.__prepare_search_keys(
                validated_X, search_keys, is_demo_dataset, is_transform=True, silent_mode=silent_mode
            )

            df = validated_X.copy()

            df = self.__handle_index_search_keys(df, search_keys)

            if DEFAULT_INDEX in df.columns:
                msg = self.bundle.get("unsupported_index_column")
                self.logger.info(msg)
                print(msg)
                df.drop(columns=DEFAULT_INDEX, inplace=True)
                validated_X.drop(columns=DEFAULT_INDEX, inplace=True)

            df = self.__add_country_code(df, search_keys)

            generated_features = []
            date_column = self._get_date_column(search_keys)
            if date_column is not None:
                converter = DateTimeSearchKeyConverter(date_column, self.date_format, self.logger, bundle=self.bundle)
                df = converter.convert(df)
                self.logger.info(f"Date column after convertion: {df[date_column]}")
                generated_features.extend(converter.generated_features)
            else:
                self.logger.info("Input dataset hasn't date column")
                if self.add_date_if_missing:
                    df = self._add_current_date_as_key(df, search_keys, self.logger, self.bundle)

            # Don't pass all features in backend on transform
            original_features_for_transform = []
            runtime_parameters = self._get_copy_of_runtime_parameters()
            features_not_to_pass = [column for column in df.columns if column not in search_keys.keys()]
            if len(features_not_to_pass) > 0:
                # Pass only features that need for transform
                features_for_transform = self._search_task.get_features_for_transform()
                if features_for_transform is not None and len(features_for_transform) > 0:
                    file_metadata = self._search_task.get_file_metadata(trace_id)
                    original_features_for_transform = [
                        c.originalName or c.name for c in file_metadata.columns if c.name in features_for_transform
                    ]

                    runtime_parameters.properties["features_for_embeddings"] = ",".join(features_for_transform)

            columns_for_system_record_id = sorted(list(search_keys.keys()) + (original_features_for_transform))

            df[ENTITY_SYSTEM_RECORD_ID] = pd.util.hash_pandas_object(
                df[columns_for_system_record_id], index=False
            ).astype("Float64")

            # Explode multiple search keys
            df, unnest_search_keys = self._explode_multiple_search_keys(df, search_keys)

            email_column = self._get_email_column(search_keys)
            hem_column = self._get_hem_column(search_keys)
            email_converted_to_hem = False
            if email_column:
                converter = EmailSearchKeyConverter(
                    email_column, hem_column, search_keys, list(unnest_search_keys.keys()), self.logger
                )
                df = converter.convert(df)
                generated_features.extend(converter.generated_features)
                email_converted_to_hem = converter.email_converted_to_hem
            if (
                self.detect_missing_search_keys
                and list(search_keys.values()) == [SearchKey.DATE]
                and self.country_code is None
            ):
                converter = IpToCountrySearchKeyConverter(search_keys, self.logger)
                df = converter.convert(df)
            generated_features = [f for f in generated_features if f in self.fit_generated_features]

            meaning_types = {col: key.value for col, key in search_keys.items()}
            # non_keys_columns = [column for column in df.columns if column not in search_keys.keys()]
            for col in original_features_for_transform:
                meaning_types[col] = FileColumnMeaningType.FEATURE
            features_not_to_pass = [column for column in features_not_to_pass if column not in search_keys.keys()]

            if email_converted_to_hem:
                features_not_to_pass.append(email_column)

            features_not_to_pass = [c for c in features_not_to_pass if c not in original_features_for_transform]
            columns_for_system_record_id = sorted(list(search_keys.keys()) + (original_features_for_transform))

            if add_fit_system_record_id:
                df = self.__add_fit_system_record_id(df, {}, search_keys, SYSTEM_RECORD_ID)
                df = df.rename(columns={SYSTEM_RECORD_ID: SORT_ID})
                features_not_to_pass.append(SORT_ID)

            columns_for_system_record_id = sorted(list(search_keys.keys()) + (original_features_for_transform or []))

            df[SYSTEM_RECORD_ID] = pd.util.hash_pandas_object(df[columns_for_system_record_id], index=False).astype(
                "Float64"
            )
            meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID
            meaning_types[ENTITY_SYSTEM_RECORD_ID] = FileColumnMeaningType.ENTITY_SYSTEM_RECORD_ID
            if SEARCH_KEY_UNNEST in df.columns:
                meaning_types[SEARCH_KEY_UNNEST] = FileColumnMeaningType.UNNEST_KEY

            df = df.reset_index(drop=True)
            system_columns_with_original_index = [SYSTEM_RECORD_ID, ENTITY_SYSTEM_RECORD_ID] + generated_features
            if add_fit_system_record_id:
                system_columns_with_original_index.append(SORT_ID)
            df_with_original_index = df[system_columns_with_original_index].copy()

            combined_search_keys = combine_search_keys(search_keys.keys())

            df_without_features = df.drop(columns=features_not_to_pass)

            df_without_features = clean_full_duplicates(
                df_without_features, self.logger, silent=silent_mode, bundle=self.bundle
            )

            del df
            gc.collect()

            dataset = Dataset(
                "sample_" + str(uuid.uuid4()),
                df=df_without_features,
                meaning_types=meaning_types,
                search_keys=combined_search_keys,
                unnest_search_keys=unnest_search_keys,
                date_format=self.date_format,
                rest_client=self.rest_client,
                logger=self.logger,
            )
            if email_converted_to_hem:
                dataset.ignore_columns = [email_column]

            if max_features is not None or importance_threshold is not None:
                exclude_features_sources = list(
                    set(
                        (exclude_features_sources or [])
                        + self._get_excluded_features(max_features, importance_threshold)
                    )
                )
                if len(exclude_features_sources) == 0:
                    exclude_features_sources = None

            validation_task = self._search_task.validation(
                trace_id,
                dataset,
                start_time=start_time,
                extract_features=True,
                runtime_parameters=runtime_parameters,
                exclude_features_sources=exclude_features_sources,
                metrics_calculation=metrics_calculation,
                silent_mode=silent_mode,
                progress_bar=progress_bar,
                progress_callback=progress_callback,
            )

            del df_without_features, dataset
            gc.collect()

            if not silent_mode:
                print(self.bundle.get("polling_search_task").format(validation_task.search_task_id))
                if not self.__is_registered:
                    print(self.bundle.get("polling_unregister_information"))

            progress = self.get_progress(trace_id, validation_task)
            progress.recalculate_eta(time.time() - start_time)
            if progress_bar is not None:
                progress_bar.progress = progress.to_progress_bar()
            if progress_callback is not None:
                progress_callback(progress)
            prev_progress: Optional[SearchProgress] = None
            polling_period_seconds = 1
            try:
                while progress.stage != ProgressStage.DOWNLOADING.value:
                    if prev_progress is None or prev_progress.percent != progress.percent:
                        progress.recalculate_eta(time.time() - start_time)
                    else:
                        progress.update_eta(prev_progress.eta - polling_period_seconds)
                    prev_progress = progress
                    if progress_bar is not None:
                        progress_bar.progress = progress.to_progress_bar()
                    if progress_callback is not None:
                        progress_callback(progress)
                    if progress.stage == ProgressStage.FAILED.value:
                        raise Exception(progress.error_message)
                    time.sleep(polling_period_seconds)
                    progress = self.get_progress(trace_id, validation_task)
            except KeyboardInterrupt as e:
                print(self.bundle.get("search_stopping"))
                self.rest_client.stop_search_task_v2(trace_id, validation_task.search_task_id)
                self.logger.warning(f"Search {validation_task.search_task_id} stopped by user")
                print(self.bundle.get("search_stopped"))
                raise e

            validation_task.poll_result(trace_id, quiet=True)

            seconds_left = time.time() - start_time
            progress = SearchProgress(97.0, ProgressStage.DOWNLOADING, seconds_left)
            if progress_bar is not None:
                progress_bar.progress = progress.to_progress_bar()
            if progress_callback is not None:
                progress_callback(progress)

            def enrich():
                res, _ = self.__enrich(
                    df_with_original_index,
                    validation_task.get_all_validation_raw_features(trace_id, metrics_calculation),
                    validated_X,
                    is_transform=True,
                )
                return res

            if not silent_mode:
                print(self.bundle.get("transform_start"))
                # with Spinner():
                result = enrich()
            else:
                result = enrich()

            filtered_columns = self.__filtered_enriched_features(importance_threshold, max_features)
            existing_filtered_columns = [c for c in filtered_columns if c in result.columns]
            selecting_columns = validated_X.columns.tolist() + generated_features + existing_filtered_columns
            if add_fit_system_record_id:
                selecting_columns.append(SORT_ID)

            result = result[selecting_columns]

            if self.country_added:
                result = result.drop(columns=COUNTRY, errors="ignore")

            if add_fit_system_record_id:
                result = result.rename(columns={SORT_ID: SYSTEM_RECORD_ID})

            return result

    def _get_excluded_features(self, max_features: Optional[int], importance_threshold: Optional[float]) -> List[str]:
        features_info = self._internal_features_info
        comm_schema_header = self.bundle.get("features_info_commercial_schema")
        shap_value_header = self.bundle.get("features_info_shap")
        feature_name_header = self.bundle.get("features_info_name")
        external_features = features_info[features_info[comm_schema_header].str.len() > 0]
        filtered_features = external_features
        if importance_threshold is not None:
            filtered_features = filtered_features[filtered_features[shap_value_header] >= importance_threshold]
        if max_features is not None and len(filtered_features) > max_features:
            filtered_features = filtered_features.iloc[:max_features, :]
        if len(filtered_features) == len(external_features):
            return []
        else:
            if len(filtered_features[filtered_features[comm_schema_header].isin([CommercialSchema.PAID.value])]):
                return []
            excluded_features = external_features[~external_features.index.isin(filtered_features.index)].copy()
            excluded_features = excluded_features[
                excluded_features[comm_schema_header].isin([CommercialSchema.PAID.value])
            ]
            return excluded_features[feature_name_header].values.tolist()

    def __validate_search_keys(self, search_keys: Dict[str, SearchKey], search_id: Optional[str] = None):
        if (search_keys is None or len(search_keys) == 0) and self.country_code is None:
            if search_id:
                self.logger.debug(f"search_id {search_id} provided without search_keys")
                return
            else:
                self.logger.warning("search_keys not provided")
                raise ValidationError(self.bundle.get("empty_search_keys"))

        key_types = search_keys.values()

        # Multiple search keys allowed only for PHONE, IP, POSTAL_CODE, EMAIL, HEM
        multi_keys = [key for key, count in Counter(key_types).items() if count > 1]
        for multi_key in multi_keys:
            if multi_key not in [SearchKey.PHONE, SearchKey.IP, SearchKey.POSTAL_CODE, SearchKey.EMAIL, SearchKey.HEM]:
                msg = self.bundle.get("unsupported_multi_key").format(multi_key)
                self.logger.warning(msg)
                raise ValidationError(msg)

        if SearchKey.DATE in key_types and SearchKey.DATETIME in key_types:
            msg = self.bundle.get("date_and_datetime_simultanious")
            self.logger.warning(msg)
            raise ValidationError(msg)

        if SearchKey.EMAIL in key_types and SearchKey.HEM in key_types:
            msg = self.bundle.get("email_and_hem_simultanious")
            self.logger.warning(msg)
            raise ValidationError(msg)

        if SearchKey.POSTAL_CODE in key_types and SearchKey.COUNTRY not in key_types and self.country_code is None:
            msg = self.bundle.get("postal_code_without_country")
            self.logger.warning(msg)
            raise ValidationError(msg)

        # for key_type in SearchKey.__members__.values():
        #     if key_type != SearchKey.CUSTOM_KEY and list(key_types).count(key_type) > 1:
        #         msg = self.bundle.get("multiple_search_key").format(key_type)
        #         self.logger.warning(msg)
        #         raise ValidationError(msg)

        # non_personal_keys = set(SearchKey.__members__.values()) - set(SearchKey.personal_keys())
        # if (
        #     not self.__is_registered
        #     and not is_demo_dataset
        #     and len(set(key_types).intersection(non_personal_keys)) == 0
        # ):
        #     msg = self.bundle.get("unregistered_only_personal_keys")
        #     self.logger.warning(msg + f" Provided search keys: {key_types}")
        #     raise ValidationError(msg)

    @property
    def __is_registered(self) -> bool:
        return self.api_key is not None and self.api_key != ""

    def __inner_fit(
        self,
        trace_id: str,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List, None],
        eval_set: Optional[List[tuple]],
        progress_bar: Optional[ProgressBar],
        start_time: int,
        *,
        exclude_features_sources: Optional[List[str]] = None,
        calculate_metrics: Optional[bool],
        scoring: Union[Callable, str, None],
        estimator: Optional[Any],
        importance_threshold: Optional[float],
        max_features: Optional[int],
        remove_outliers_calc_metrics: Optional[bool],
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        search_id_callback: Optional[Callable[[str], Any]] = None,
    ):
        self.warning_counter.reset()
        self.df_with_original_index = None
        self.__cached_sampled_datasets = None
        self.metrics = None

        validated_X = self._validate_X(X)
        validated_y = self._validate_y(validated_X, y)
        validated_eval_set = (
            [self._validate_eval_set_pair(validated_X, eval_pair) for eval_pair in eval_set]
            if eval_set is not None
            else None
        )
        is_demo_dataset = hash_input(validated_X, validated_y, validated_eval_set) in DEMO_DATASET_HASHES
        if is_demo_dataset:
            msg = self.bundle.get("demo_dataset_info")
            self.logger.info(msg)
            if not self.__is_registered:
                print(msg)

        if self.generate_features is not None and len(self.generate_features) > 0:
            x_columns = list(validated_X.columns)
            checked_generate_features = []
            for gen_feature in self.generate_features:
                if gen_feature not in x_columns:
                    msg = self.bundle.get("missing_generate_feature").format(gen_feature, x_columns)
                    print(msg)
                    self.logger.warning(msg)
                else:
                    checked_generate_features.append(gen_feature)
            self.generate_features = checked_generate_features
            self.runtime_parameters.properties["generate_features"] = ",".join(self.generate_features)

        validate_scoring_argument(scoring)

        self.__log_debug_information(
            X,
            y,
            eval_set,
            exclude_features_sources=exclude_features_sources,
            calculate_metrics=calculate_metrics,
            scoring=scoring,
            estimator=estimator,
            remove_outliers_calc_metrics=remove_outliers_calc_metrics,
        )

        df = pd.concat([validated_X, validated_y], axis=1)

        self.fit_search_keys = self.search_keys.copy()
        self.fit_search_keys = self.__prepare_search_keys(validated_X, self.fit_search_keys, is_demo_dataset)

        validate_dates_distribution(validated_X, self.fit_search_keys, self.logger, self.bundle, self.warning_counter)

        maybe_date_column = self._get_date_column(self.fit_search_keys)
        has_date = maybe_date_column is not None
        model_task_type = self.model_task_type or define_task(validated_y, has_date, self.logger)
        self._validate_binary_observations(validated_y, model_task_type)

        self.runtime_parameters = get_runtime_params_custom_loss(
            self.loss, model_task_type, self.runtime_parameters, self.logger
        )

        if validated_eval_set is not None and len(validated_eval_set) > 0:
            df[EVAL_SET_INDEX] = 0
            for idx, (eval_X, eval_y) in enumerate(validated_eval_set):
                eval_df = pd.concat([eval_X, eval_y], axis=1)
                eval_df[EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_df])

        df = self.__correct_target(df)

        df = self.__handle_index_search_keys(df, self.fit_search_keys)

        if is_numeric_dtype(df[self.TARGET_NAME]) and has_date:
            self._validate_PSI(df.sort_values(by=maybe_date_column))

        if DEFAULT_INDEX in df.columns:
            msg = self.bundle.get("unsupported_index_column")
            self.logger.info(msg)
            print(msg)
            self.fit_dropped_features.add(DEFAULT_INDEX)
            df.drop(columns=DEFAULT_INDEX, inplace=True)

        df = self.__add_country_code(df, self.fit_search_keys)

        df = remove_fintech_duplicates(
            df, self.fit_search_keys, date_format=self.date_format, logger=self.logger, bundle=self.bundle
        )
        df = clean_full_duplicates(df, self.logger, bundle=self.bundle)

        date_column = self._get_date_column(self.fit_search_keys)
        self.__adjust_cv(df, date_column, model_task_type)

        self.fit_generated_features = []

        if date_column is not None:
            converter = DateTimeSearchKeyConverter(date_column, self.date_format, self.logger, bundle=self.bundle)
            df = converter.convert(df, keep_time=True)
            self.logger.info(f"Date column after convertion: {df[date_column]}")
            self.fit_generated_features.extend(converter.generated_features)
        else:
            self.logger.info("Input dataset hasn't date column")
            if self.add_date_if_missing:
                df = self._add_current_date_as_key(df, self.fit_search_keys, self.logger, self.bundle)

        if (
            self.detect_missing_search_keys
            and list(self.fit_search_keys.values()) == [SearchKey.DATE]
            and self.country_code is None
        ):
            converter = IpToCountrySearchKeyConverter(self.fit_search_keys, self.logger)
            df = converter.convert(df)

        # Explode multiple search keys
        non_feature_columns = [self.TARGET_NAME, EVAL_SET_INDEX] + list(self.fit_search_keys.keys())
        meaning_types = {
            **{col: key.value for col, key in self.fit_search_keys.items()},
            **{str(c): FileColumnMeaningType.FEATURE for c in df.columns if c not in non_feature_columns},
        }
        meaning_types[self.TARGET_NAME] = FileColumnMeaningType.TARGET
        if eval_set is not None and len(eval_set) > 0:
            meaning_types[EVAL_SET_INDEX] = FileColumnMeaningType.EVAL_SET_INDEX
        df = self.__add_fit_system_record_id(df, meaning_types, self.fit_search_keys, ENTITY_SYSTEM_RECORD_ID)

        # TODO check that this is correct for enrichment
        self.df_with_original_index = df.copy()

        df, unnest_search_keys = self._explode_multiple_search_keys(df, self.fit_search_keys)

        # Convert EMAIL to HEM after unnesting to do it only with one column
        email_column = self._get_email_column(self.fit_search_keys)
        hem_column = self._get_hem_column(self.fit_search_keys)
        email_converted_to_hem = False
        if email_column:
            converter = EmailSearchKeyConverter(
                email_column, hem_column, self.fit_search_keys, list(unnest_search_keys.keys()), self.logger
            )
            df = converter.convert(df)
            self.fit_generated_features.extend(converter.generated_features)
            email_converted_to_hem = converter.email_converted_to_hem

        non_feature_columns = [self.TARGET_NAME, EVAL_SET_INDEX, ENTITY_SYSTEM_RECORD_ID, SEARCH_KEY_UNNEST] + list(
            self.fit_search_keys.keys()
        )
        if email_converted_to_hem:
            non_feature_columns.append(email_column)
        if DateTimeSearchKeyConverter.DATETIME_COL in df.columns:
            non_feature_columns.append(DateTimeSearchKeyConverter.DATETIME_COL)

        features_columns = [c for c in df.columns if c not in non_feature_columns]

        features_to_drop = FeaturesValidator(self.logger).validate(
            df, features_columns, self.generate_features, self.warning_counter
        )
        self.fit_dropped_features.update(features_to_drop)
        df = df.drop(columns=features_to_drop)

        if email_converted_to_hem:
            self.fit_dropped_features.add(email_column)

        self.fit_generated_features = [f for f in self.fit_generated_features if f not in self.fit_dropped_features]

        meaning_types = {
            **{col: key.value for col, key in self.fit_search_keys.items()},
            **{str(c): FileColumnMeaningType.FEATURE for c in df.columns if c not in non_feature_columns},
        }
        meaning_types[self.TARGET_NAME] = FileColumnMeaningType.TARGET
        meaning_types[ENTITY_SYSTEM_RECORD_ID] = FileColumnMeaningType.ENTITY_SYSTEM_RECORD_ID
        if SEARCH_KEY_UNNEST in df.columns:
            meaning_types[SEARCH_KEY_UNNEST] = FileColumnMeaningType.UNNEST_KEY
        if eval_set is not None and len(eval_set) > 0:
            meaning_types[EVAL_SET_INDEX] = FileColumnMeaningType.EVAL_SET_INDEX

        df = self.__add_fit_system_record_id(df, meaning_types, self.fit_search_keys, SYSTEM_RECORD_ID)

        df = df.reset_index(drop=True).sort_values(by=SYSTEM_RECORD_ID).reset_index(drop=True)

        combined_search_keys = combine_search_keys(self.fit_search_keys.keys())

        dataset = Dataset(
            "tds_" + str(uuid.uuid4()),
            df=df,
            meaning_types=meaning_types,
            search_keys=combined_search_keys,
            unnest_search_keys=unnest_search_keys,
            model_task_type=model_task_type,
            date_format=self.date_format,
            random_state=self.random_state,
            rest_client=self.rest_client,
            logger=self.logger,
        )
        if email_converted_to_hem:
            dataset.ignore_columns = [email_column]

        self.passed_features = [
            column for column, meaning_type in meaning_types.items() if meaning_type == FileColumnMeaningType.FEATURE
        ]

        self._search_task = dataset.search(
            trace_id=trace_id,
            progress_bar=progress_bar,
            start_time=start_time,
            progress_callback=progress_callback,
            extract_features=True,
            runtime_parameters=self._get_copy_of_runtime_parameters(),
            exclude_features_sources=exclude_features_sources,
        )

        if search_id_callback is not None:
            search_id_callback(self._search_task.search_task_id)

        print(self.bundle.get("polling_search_task").format(self._search_task.search_task_id))
        if not self.__is_registered:
            print(self.bundle.get("polling_unregister_information"))

        progress = self.get_progress(trace_id)
        prev_progress = None
        progress.recalculate_eta(time.time() - start_time)
        if progress_bar is not None:
            progress_bar.progress = progress.to_progress_bar()
        if progress_callback is not None:
            progress_callback(progress)
        poll_period_seconds = 1
        try:
            while progress.stage != ProgressStage.GENERATING_REPORT.value:
                if prev_progress is None or prev_progress.percent != progress.percent:
                    progress.recalculate_eta(time.time() - start_time)
                else:
                    progress.update_eta(prev_progress.eta - poll_period_seconds)
                prev_progress = progress
                if progress_bar is not None:
                    progress_bar.progress = progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(progress)
                if progress.stage == ProgressStage.FAILED.value:
                    self.logger.error(
                        f"Search {self._search_task.search_task_id} failed with error {progress.error}"
                        f" and message {progress.error_message}"
                    )
                    raise RuntimeError(self.bundle.get("search_task_failed_status"))
                time.sleep(poll_period_seconds)
                progress = self.get_progress(trace_id)
        except KeyboardInterrupt as e:
            print(self.bundle.get("search_stopping"))
            self.rest_client.stop_search_task_v2(trace_id, self._search_task.search_task_id)
            self.logger.warning(f"Search {self._search_task.search_task_id} stopped by user")
            print(self.bundle.get("search_stopped"))
            raise e

        self._search_task.poll_result(trace_id, quiet=True)

        seconds_left = time.time() - start_time
        progress = SearchProgress(97.0, ProgressStage.GENERATING_REPORT, seconds_left)
        if progress_bar is not None:
            progress_bar.progress = progress.to_progress_bar()
        if progress_callback is not None:
            progress_callback(progress)

        self.imbalanced = dataset.imbalanced

        zero_hit_search_keys = self._search_task.get_zero_hit_rate_search_keys()
        if zero_hit_search_keys:
            self.logger.warning(
                f"Intersections with this search keys are empty for all datasets: {zero_hit_search_keys}"
            )
            zero_hit_columns = self.get_columns_by_search_keys(zero_hit_search_keys)
            if zero_hit_columns:
                msg = self.bundle.get("features_info_zero_hit_rate_search_keys").format(zero_hit_columns)
                self.logger.warning(msg)
                self.__display_support_link(msg)
                self.warning_counter.increment()

        if (
            self._search_task.unused_features_for_generation is not None
            and len(self._search_task.unused_features_for_generation) > 0
        ):
            unused_features_for_generation = [
                dataset.columns_renaming.get(col) or col for col in self._search_task.unused_features_for_generation
            ]
            msg = self.bundle.get("features_not_generated").format(unused_features_for_generation)
            self.logger.warning(msg)
            print(msg)
            self.warning_counter.increment()

        self.__prepare_feature_importances(trace_id, validated_X.columns.to_list() + self.fit_generated_features)

        self.__show_selected_features(self.fit_search_keys)

        autofe_description = self.get_autofe_features_description()
        if autofe_description is not None:
            display_html_dataframe(autofe_description, autofe_description, "*Description of AutoFE feature names")

        if self._has_paid_features(exclude_features_sources):
            if calculate_metrics is not None and calculate_metrics:
                msg = self.bundle.get("metrics_with_paid_features")
                self.logger.warning(msg)
                self.__display_support_link(msg)
        else:
            if (scoring is not None or estimator is not None) and calculate_metrics is None:
                calculate_metrics = True

            if calculate_metrics is None:
                if len(validated_X) < self.CALCULATE_METRICS_MIN_THRESHOLD or any(
                    [len(eval_X) < self.CALCULATE_METRICS_MIN_THRESHOLD for eval_X, _ in validated_eval_set]
                ):
                    msg = self.bundle.get("too_small_for_metrics")
                    self.logger.warning(msg)
                    calculate_metrics = False
                elif len(dataset) * len(dataset.columns) > self.CALCULATE_METRICS_THRESHOLD:
                    self.logger.warning("Too big dataset for automatic metrics calculation")
                    calculate_metrics = False
                else:
                    calculate_metrics = True

            del df, validated_X, validated_y, dataset
            gc.collect()

            if calculate_metrics:
                try:
                    self.__show_metrics(
                        scoring,
                        estimator,
                        importance_threshold,
                        max_features,
                        remove_outliers_calc_metrics,
                        trace_id,
                        progress_bar,
                        progress_callback,
                    )
                except Exception:
                    self.__show_report_button()
                    raise

        self.__show_report_button()

        if not self.warning_counter.has_warnings():
            self.__display_support_link(self.bundle.get("all_ok_community_invite"))

    def __adjust_cv(self, df: pd.DataFrame, date_column: pd.Series, model_task_type: ModelTaskType):
        # Check Multivariate time series
        if (
            self.cv is None
            and date_column
            and model_task_type == ModelTaskType.REGRESSION
            and len({SearchKey.PHONE, SearchKey.EMAIL, SearchKey.HEM}.intersection(self.fit_search_keys.keys())) == 0
            and is_blocked_time_series(df, date_column, list(self.fit_search_keys.keys()) + [TARGET])
        ):
            msg = self.bundle.get("multivariate_timeseries_detected")
            self.__override_cv(CVType.blocked_time_series, msg, print_warning=False)
        elif (
            self.cv is None
            and model_task_type != ModelTaskType.REGRESSION
            and self._get_group_columns(df, self.fit_search_keys)
        ):
            msg = self.bundle.get("group_k_fold_in_classification")
            self.__override_cv(CVType.group_k_fold, msg, print_warning=self.cv is not None)

    def __override_cv(self, cv: CVType, msg: str, print_warning: bool = True):
        if print_warning:
            print(msg)
        self.logger.warning(msg)
        self.cv = cv
        self.runtime_parameters.properties["cv_type"] = self.cv.name

    def get_columns_by_search_keys(self, keys: List[str]):
        if "HEM" in keys:
            keys.append("EMAIL")
        if "DATE" in keys:
            keys.append("DATETIME")
        search_keys_with_autodetection = {**self.search_keys, **self.autodetected_search_keys}
        return [c for c, v in search_keys_with_autodetection.items() if v.value.value in keys]

    def _validate_X(self, X, is_transform=False) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if isinstance(X.columns, pd.MultiIndex) or isinstance(X.index, pd.MultiIndex):
                raise ValidationError(self.bundle.get("x_multiindex_unsupported"))
            validated_X = X.copy()
        elif isinstance(X, pd.Series):
            validated_X = X.to_frame()
        elif isinstance(X, (list, np.ndarray)):
            validated_X = pd.DataFrame(X)
            renaming = {c: str(c) for c in validated_X.columns}
            validated_X = validated_X.rename(columns=renaming)
        else:
            raise ValidationError(self.bundle.get("unsupported_x_type").format(type(X)))

        if _num_samples(X) == 0:
            raise ValidationError(self.bundle.get("x_is_empty"))

        if len(set(validated_X.columns)) != len(validated_X.columns):
            raise ValidationError(self.bundle.get("x_contains_dup_columns"))
        if not is_transform and not validated_X.index.is_unique:
            raise ValidationError(self.bundle.get("x_non_unique_index"))

        if self.exclude_columns is not None:
            validated_X = validated_X.drop(columns=self.exclude_columns, errors="ignore")

        if self.baseline_score_column:
            validated_X[self.baseline_score_column] = validated_X[self.baseline_score_column].astype(
                "float64", errors="ignore"
            )

        if TARGET in validated_X.columns:
            raise ValidationError(self.bundle.get("x_contains_reserved_column_name").format(TARGET))
        if not is_transform and EVAL_SET_INDEX in validated_X.columns:
            raise ValidationError(self.bundle.get("x_contains_reserved_column_name").format(EVAL_SET_INDEX))
        if SYSTEM_RECORD_ID in validated_X.columns:
            raise ValidationError(self.bundle.get("x_contains_reserved_column_name").format(SYSTEM_RECORD_ID))

        return validated_X

    def _validate_y(self, X: pd.DataFrame, y) -> pd.Series:
        if (
            not isinstance(y, pd.Series)
            and not isinstance(y, pd.DataFrame)
            and not isinstance(y, np.ndarray)
            and not isinstance(y, list)
        ):
            raise ValidationError(self.bundle.get("unsupported_y_type").format(type(y)))

        if _num_samples(y) == 0:
            raise ValidationError(self.bundle.get("y_is_empty"))

        if _num_samples(X) != _num_samples(y):
            raise ValidationError(self.bundle.get("x_and_y_diff_size").format(_num_samples(X), _num_samples(y)))

        if isinstance(y, pd.DataFrame):
            if len(y.columns) != 1:
                raise ValidationError(self.bundle.get("y_invalid_dimension_dataframe"))
            if isinstance(y.columns, pd.MultiIndex) or isinstance(y.index, pd.MultiIndex):
                raise ValidationError(self.bundle.get("y_multiindex_unsupported"))
            y = y[y.columns[0]]

        if isinstance(y, pd.Series):
            if (y.index != X.index).any():
                raise ValidationError(self.bundle.get("x_and_y_diff_index"))
            validated_y = y.copy()
            validated_y.rename(TARGET, inplace=True)
        elif isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValidationError(self.bundle.get("y_invalid_dimension_array"))
            Xy = X.copy()
            Xy[TARGET] = y
            validated_y = Xy[TARGET].copy()
        else:
            Xy = X.copy()
            Xy[TARGET] = y
            validated_y = Xy[TARGET].copy()

        if validated_y.nunique() < 2:
            raise ValidationError(self.bundle.get("y_is_constant"))

        return validated_y

    def _validate_eval_set_pair(self, X: pd.DataFrame, eval_pair: Tuple) -> Tuple[pd.DataFrame, pd.Series]:
        if len(eval_pair) != 2:
            raise ValidationError(self.bundle.get("eval_set_invalid_tuple_size").format(len(eval_pair)))
        eval_X = eval_pair[0]
        eval_y = eval_pair[1]

        if _num_samples(eval_X) == 0:
            raise ValidationError(self.bundle.get("eval_x_is_empty"))
        if _num_samples(eval_y) == 0:
            raise ValidationError(self.bundle.get("eval_y_is_empty"))

        if isinstance(eval_X, pd.DataFrame):
            if isinstance(eval_X.columns, pd.MultiIndex) or isinstance(eval_X.index, pd.MultiIndex):
                raise ValidationError(self.bundle.get("eval_x_multiindex_unsupported"))
            validated_eval_X = eval_X.copy()
        elif isinstance(eval_X, pd.Series):
            validated_eval_X = eval_X.to_frame()
        elif isinstance(eval_X, (list, np.ndarray)):
            validated_eval_X = pd.DataFrame(eval_X)
            renaming = {c: str(c) for c in validated_eval_X.columns}
            validated_eval_X = validated_eval_X.rename(columns=renaming)
        else:
            raise ValidationError(self.bundle.get("unsupported_x_type_eval_set").format(type(eval_X)))

        if not validated_eval_X.index.is_unique:
            raise ValidationError(self.bundle.get("x_non_unique_index_eval_set"))

        if self.exclude_columns is not None:
            validated_eval_X = validated_eval_X.drop(columns=self.exclude_columns, errors="ignore")

        if self.baseline_score_column:
            validated_eval_X[self.baseline_score_column] = validated_eval_X[self.baseline_score_column].astype(
                "float64", errors="ignore"
            )

        if validated_eval_X.columns.to_list() != X.columns.to_list():
            if set(validated_eval_X.columns.to_list()) == set(X.columns.to_list()):
                validated_eval_X = validated_eval_X[X.columns.to_list()]
            else:
                raise ValidationError(self.bundle.get("eval_x_and_x_diff_shape"))

        if _num_samples(validated_eval_X) != _num_samples(eval_y):
            raise ValidationError(
                self.bundle.get("x_and_y_diff_size_eval_set").format(
                    _num_samples(validated_eval_X), _num_samples(eval_y)
                )
            )

        if isinstance(eval_y, pd.DataFrame):
            if len(eval_y.columns) != 1:
                raise ValidationError(self.bundle.get("y_invalid_dimension_dataframe_eval_set"))
            if isinstance(eval_y.columns, pd.MultiIndex) or isinstance(eval_y.index, pd.MultiIndex):
                raise ValidationError(self.bundle.get("eval_y_multiindex_unsupported"))
            eval_y = eval_y[eval_y.columns[0]]

        if isinstance(eval_y, pd.Series):
            if (eval_y.index != validated_eval_X.index).any():
                raise ValidationError(self.bundle.get("x_and_y_diff_index_eval_set"))
            validated_eval_y = eval_y.copy()
            validated_eval_y.rename(TARGET, inplace=True)
        elif isinstance(eval_y, np.ndarray):
            if eval_y.ndim != 1:
                raise ValidationError(self.bundle.get("y_invalid_dimension_array_eval_set"))
            Xy = validated_eval_X.copy()
            Xy[TARGET] = eval_y
            validated_eval_y = Xy[TARGET].copy()
        elif isinstance(eval_y, list):
            Xy = validated_eval_X.copy()
            Xy[TARGET] = eval_y
            validated_eval_y = Xy[TARGET].copy()
        else:
            raise ValidationError(self.bundle.get("unsupported_y_type_eval_set").format(type(eval_y)))

        if validated_eval_y.nunique() < 2:
            raise ValidationError(self.bundle.get("y_is_constant_eval_set"))

        return validated_eval_X, validated_eval_y

    def _validate_baseline_score(self, X: pd.DataFrame, eval_set: Optional[List[Tuple]]):
        if self.baseline_score_column is not None:
            if self.baseline_score_column not in X.columns:
                raise ValidationError(
                    self.bundle.get("baseline_score_column_not_exists").format(self.baseline_score_column)
                )
            if X[self.baseline_score_column].isna().any():
                raise ValidationError(self.bundle.get("baseline_score_column_has_na"))
            if eval_set is not None:
                if isinstance(eval_set, tuple):
                    eval_set = [eval_set]
                for eval in eval_set:
                    if self.baseline_score_column not in eval[0].columns:
                        raise ValidationError(self.bundle.get("baseline_score_column_not_exists"))
                    if eval[0][self.baseline_score_column].isna().any():
                        raise ValidationError(self.bundle.get("baseline_score_column_has_na"))

    @staticmethod
    def _sample_X_and_y(X: pd.DataFrame, y: pd.Series, enriched_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        Xy = pd.concat([X, y], axis=1)
        Xy = pd.merge(Xy, enriched_X, left_index=True, right_index=True, how="inner", suffixes=("", "enriched"))
        return Xy[X.columns].copy(), Xy[TARGET].copy()

    @staticmethod
    def _sort_by_system_record_id(
        X: pd.DataFrame, y: pd.Series, cv: Optional[CVType]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if cv not in [CVType.time_series, CVType.blocked_time_series]:
            record_id_column = ENTITY_SYSTEM_RECORD_ID if ENTITY_SYSTEM_RECORD_ID in X else SYSTEM_RECORD_ID
            Xy = X.copy()
            Xy[TARGET] = y
            Xy = Xy.sort_values(by=record_id_column).reset_index(drop=True)
            X = Xy.drop(columns=TARGET)
            y = Xy[TARGET].copy()

        if DateTimeSearchKeyConverter.DATETIME_COL in X.columns:
            X.drop(columns=DateTimeSearchKeyConverter.DATETIME_COL, inplace=True)

        return X, y

    # Deprecated
    @staticmethod
    def _sort_by_keys(
        X: pd.DataFrame, y: pd.Series, search_keys: Dict[str, SearchKey], cv: Optional[CVType]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if cv not in [CVType.time_series, CVType.blocked_time_series]:
            if DateTimeSearchKeyConverter.DATETIME_COL in X.columns:
                date_column = DateTimeSearchKeyConverter.DATETIME_COL
            else:
                date_column = FeaturesEnricher._get_date_column(search_keys)
            sort_columns = [date_column] if date_column is not None else []

            # Xy = pd.concat([X, y], axis=1)
            Xy = X.copy()
            Xy[TARGET] = y

            other_search_keys = sorted([sk for sk in search_keys.keys() if sk != date_column and sk in Xy.columns])
            search_keys_hash = "search_keys_hash"

            if len(other_search_keys) > 0:
                sort_columns.append(search_keys_hash)
                Xy[search_keys_hash] = pd.util.hash_pandas_object(Xy[sorted(other_search_keys)], index=False)

            if len(sort_columns) > 0:
                Xy = Xy.sort_values(by=sort_columns).reset_index(drop=True)
            else:
                Xy = Xy.sort_index()

            drop_columns = [TARGET]
            if search_keys_hash in Xy.columns:
                drop_columns.append(search_keys_hash)
            X = Xy.drop(columns=drop_columns)

            y = Xy[TARGET].copy()

        if DateTimeSearchKeyConverter.DATETIME_COL in X.columns:
            X.drop(columns=DateTimeSearchKeyConverter.DATETIME_COL, inplace=True)

        return X, y

    def __log_debug_information(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list, None] = None,
        eval_set: Optional[List[tuple]] = None,
        exclude_features_sources: Optional[List[str]] = None,
        calculate_metrics: Optional[bool] = None,
        cv: Optional[Any] = None,
        importance_threshold: Optional[Any] = None,
        max_features: Optional[Any] = None,
        scoring: Optional[Any] = None,
        estimator: Optional[Any] = None,
        remove_outliers_calc_metrics: Optional[bool] = None,
    ):
        try:
            resolved_api_key = self.api_key or os.environ.get(UPGINI_API_KEY)
            self.logger.info(
                f"Search keys: {self.search_keys}\n"
                f"Country code: {self.country_code}\n"
                f"Model task type: {self.model_task_type}\n"
                f"Api key presented?: {resolved_api_key is not None and resolved_api_key != ''}\n"
                f"Endpoint: {self.endpoint}\n"
                f"Runtime parameters: {self.runtime_parameters}\n"
                f"Date format: {self.date_format}\n"
                f"CV: {cv}\n"
                f"importance_threshold: {importance_threshold}\n"
                f"max_features: {max_features}\n"
                f"Shared datasets: {self.shared_datasets}\n"
                f"Random state: {self.random_state}\n"
                f"Generate features: {self.generate_features}\n"
                f"Round embeddings: {self.round_embeddings}\n"
                f"Detect missing search keys: {self.detect_missing_search_keys}\n"
                f"Exclude features sources: {exclude_features_sources}\n"
                f"Calculate metrics: {calculate_metrics}\n"
                f"Scoring: {scoring}\n"
                f"Estimator: {estimator}\n"
                f"Remove target outliers: {remove_outliers_calc_metrics}\n"
                f"Exclude columns: {self.exclude_columns}\n"
                f"Search id: {self.search_id}\n"
            )

            def sample(df):
                if isinstance(df, (pd.DataFrame, pd.Series)):
                    return df.head(10)
                else:
                    return df[:10]

            def print_datasets_sample():
                if X is not None:
                    self.logger.info(f"First 10 rows of the X with shape {X.shape}:\n{sample(X)}")
                if y is not None:
                    self.logger.info(f"First 10 rows of the y with shape {_num_samples(y)}:\n{sample(y)}")
                if eval_set is not None:
                    for idx, eval_pair in enumerate(eval_set):
                        eval_X: pd.DataFrame = eval_pair[0]
                        eval_y = eval_pair[1]
                        self.logger.info(
                            f"First 10 rows of the eval_X_{idx} with shape {eval_X.shape}:\n{sample(eval_X)}"
                        )
                        self.logger.info(
                            f"First 10 rows of the eval_y_{idx} with shape {_num_samples(eval_y)}:\n{sample(eval_y)}"
                        )

            do_without_pandas_limits(print_datasets_sample)

            maybe_date_col = self._get_date_column(self.search_keys)
            if X is not None and maybe_date_col is not None and maybe_date_col in X.columns:
                # TODO cast date column to single dtype
                date_converter = DateTimeSearchKeyConverter(maybe_date_col, self.date_format)
                converted_X = date_converter.convert(X)
                min_date = converted_X[maybe_date_col].min()
                max_date = converted_X[maybe_date_col].max()
                self.logger.info(f"Dates interval is ({min_date}, {max_date})")

        except Exception:
            self.logger.warning("Failed to log debug information", exc_info=True)

    def __handle_index_search_keys(self, df: pd.DataFrame, search_keys: Dict[str, SearchKey]) -> pd.DataFrame:
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

        return df

    @staticmethod
    def _get_date_column(search_keys: Dict[str, SearchKey]) -> Optional[str]:
        for col, t in search_keys.items():
            if t in [SearchKey.DATE, SearchKey.DATETIME]:
                return col

    @staticmethod
    def _add_current_date_as_key(
        df: pd.DataFrame, search_keys: Dict[str, SearchKey], logger: logging.Logger, bundle: ResourceBundle
    ) -> pd.DataFrame:
        if (
            set(search_keys.values()) == {SearchKey.PHONE}
            or set(search_keys.values()) == {SearchKey.EMAIL}
            or set(search_keys.values()) == {SearchKey.HEM}
            or set(search_keys.values()) == {SearchKey.COUNTRY, SearchKey.POSTAL_CODE}
        ):
            msg = bundle.get("current_date_added")
            print(msg)
            logger.warning(msg)
            df[FeaturesEnricher.CURRENT_DATE] = datetime.date.today()
            search_keys[FeaturesEnricher.CURRENT_DATE] = SearchKey.DATE
            converter = DateTimeSearchKeyConverter(FeaturesEnricher.CURRENT_DATE, None, logger, bundle)
            df = converter.convert(df)
        return df

    @staticmethod
    def _get_group_columns(df: pd.DataFrame, search_keys: Dict[str, SearchKey]) -> List[str]:
        return [
            col
            for col, t in search_keys.items()
            if t not in [SearchKey.DATE, SearchKey.DATETIME] and df[col].dropna().nunique() > 1
        ]

    @staticmethod
    def _get_email_column(search_keys: Dict[str, SearchKey]) -> Optional[str]:
        cols = [col for col, t in search_keys.items() if t == SearchKey.EMAIL]
        if len(cols) > 1:
            raise Exception("More than one email column found after unnest")
        if len(cols) == 1:
            return cols[0]

    @staticmethod
    def _get_hem_column(search_keys: Dict[str, SearchKey]) -> Optional[str]:
        cols = [col for col, t in search_keys.items() if t == SearchKey.HEM]
        if len(cols) > 1:
            raise Exception("More than one hem column found after unnest")
        if len(cols) == 1:
            return cols[0]

    @staticmethod
    def _get_phone_column(search_keys: Dict[str, SearchKey]) -> Optional[str]:
        for col, t in search_keys.items():
            if t == SearchKey.PHONE:
                return col

    def _explode_multiple_search_keys(
        self, df: pd.DataFrame, search_keys: Dict[str, SearchKey]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        # find groups of multiple search keys
        search_key_names_by_type: Dict[SearchKey, str] = {}
        for key_name, key_type in search_keys.items():
            search_key_names_by_type[key_type] = search_key_names_by_type.get(key_type, []) + [key_name]
        search_key_names_by_type = {
            key_type: key_names for key_type, key_names in search_key_names_by_type.items() if len(key_names) > 1
        }
        if len(search_key_names_by_type) == 0:
            return df, {}

        multiple_keys_columns = [col for cols in search_key_names_by_type.values() for col in cols]
        other_columns = [col for col in df.columns if col not in multiple_keys_columns]
        exploded_dfs = []
        unnest_search_keys = {}

        for key_type, key_names in search_key_names_by_type.items():
            new_search_key = f"upgini_{key_type.name.lower()}_unnest"
            exploded_df = pd.melt(
                df, id_vars=other_columns, value_vars=key_names, var_name=SEARCH_KEY_UNNEST, value_name=new_search_key
            )
            exploded_dfs.append(exploded_df)
            for old_key in key_names:
                del search_keys[old_key]
            search_keys[new_search_key] = key_type
            unnest_search_keys[new_search_key] = key_names

        df = pd.concat(exploded_dfs, ignore_index=True)
        return df, unnest_search_keys

    def __add_fit_system_record_id(
        self,
        df: pd.DataFrame,
        meaning_types: Dict[str, FileColumnMeaningType],
        search_keys: Dict[str, SearchKey],
        id_name: str,
    ) -> pd.DataFrame:
        # save original order or rows
        original_index_name = df.index.name
        index_name = df.index.name or DEFAULT_INDEX
        original_order_name = "original_order"
        df = df.reset_index().rename(columns={index_name: ORIGINAL_INDEX})
        df = df.reset_index().rename(columns={DEFAULT_INDEX: original_order_name})

        # order by date and idempotent order by other keys
        if self.cv not in [CVType.time_series, CVType.blocked_time_series]:
            sort_exclude_columns = [
                original_order_name,
                ORIGINAL_INDEX,
                EVAL_SET_INDEX,
                TARGET,
                "__target",
                ENTITY_SYSTEM_RECORD_ID,
            ]
            if DateTimeSearchKeyConverter.DATETIME_COL in df.columns:
                date_column = DateTimeSearchKeyConverter.DATETIME_COL
                sort_exclude_columns.append(self._get_date_column(search_keys))
            else:
                date_column = self._get_date_column(search_keys)
            sort_columns = [date_column] if date_column is not None else []

            other_columns = sorted(
                [
                    c
                    for c in df.columns
                    if c not in sort_columns and c not in sort_exclude_columns and df[c].nunique() > 1
                ]
                # [
                #     sk
                #     for sk, key_type in search_keys.items()
                #     if key_type not in [SearchKey.DATE, SearchKey.DATETIME]
                #     and sk in df.columns
                #     and df[sk].nunique() > 1  # don't use constant keys for hash
                # ]
            )

            search_keys_hash = "search_keys_hash"
            if len(other_columns) > 0:
                sort_columns.append(search_keys_hash)
                df[search_keys_hash] = pd.util.hash_pandas_object(df[other_columns], index=False)

            df = df.sort_values(by=sort_columns)

            if search_keys_hash in df.columns:
                df.drop(columns=search_keys_hash, inplace=True)

        if DateTimeSearchKeyConverter.DATETIME_COL in df.columns:
            df.drop(columns=DateTimeSearchKeyConverter.DATETIME_COL, inplace=True)

        df = df.reset_index(drop=True).reset_index()
        # system_record_id saves correct order for fit
        df = df.rename(columns={DEFAULT_INDEX: id_name})

        # return original order
        df = df.set_index(ORIGINAL_INDEX)
        df.index.name = original_index_name
        df = df.sort_values(by=original_order_name).drop(columns=original_order_name)

        meaning_types[id_name] = (
            FileColumnMeaningType.SYSTEM_RECORD_ID
            if id_name == SYSTEM_RECORD_ID
            else FileColumnMeaningType.ENTITY_SYSTEM_RECORD_ID
        )
        return df

    def __correct_target(self, df: pd.DataFrame) -> pd.DataFrame:
        target = df[self.TARGET_NAME]
        if is_string_dtype(target) or is_object_dtype(target):
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

        if self.country_code is not None and SearchKey.COUNTRY not in search_keys.values():
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
        X: Optional[pd.DataFrame] = None,
        is_transform=False,
        rows_to_drop: Optional[pd.DataFrame] = None,
        drop_system_record_id=True,
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        if result_features is None:
            self.logger.error(f"result features not found by search_task_id: {self.get_search_id()}")
            raise RuntimeError(self.bundle.get("features_wasnt_returned"))
        result_features = (
            result_features.drop(columns=EVAL_SET_INDEX)
            if EVAL_SET_INDEX in result_features.columns
            else result_features
        )

        comparing_columns = X.columns if is_transform else df_with_original_index.columns
        dup_features = [
            c
            for c in comparing_columns
            if c in result_features.columns and c not in [SYSTEM_RECORD_ID, ENTITY_SYSTEM_RECORD_ID]
        ]
        if len(dup_features) > 0:
            self.logger.warning(f"X contain columns with same name as returned from backend: {dup_features}")
            raise ValidationError(self.bundle.get("returned_features_same_as_passed").format(dup_features))

        # index overrites from result_features
        original_index_name = df_with_original_index.index.name
        df_with_original_index = df_with_original_index.reset_index()
        result_features = pd.merge(
            df_with_original_index,
            result_features,
            on=ENTITY_SYSTEM_RECORD_ID,
            how="left" if is_transform else "inner",
        )
        result_features = result_features.set_index(original_index_name or DEFAULT_INDEX)
        result_features.index.name = original_index_name

        if rows_to_drop is not None:
            self.logger.info(f"Before dropping target outliers size: {len(result_features)}")
            result_features = result_features[
                ~result_features[ENTITY_SYSTEM_RECORD_ID].isin(rows_to_drop[ENTITY_SYSTEM_RECORD_ID])
            ]
            self.logger.info(f"After dropping target outliers size: {len(result_features)}")

        result_eval_sets = {}
        if not is_transform and EVAL_SET_INDEX in result_features.columns:
            result_train_features = result_features.loc[result_features[EVAL_SET_INDEX] == 0].copy()
            eval_set_indices = list(result_features[EVAL_SET_INDEX].unique())
            if 0 in eval_set_indices:
                eval_set_indices.remove(0)
            for eval_set_index in eval_set_indices:
                result_eval_sets[eval_set_index] = result_features.loc[
                    result_features[EVAL_SET_INDEX] == eval_set_index
                ].copy()
            result_train_features = result_train_features.drop(columns=EVAL_SET_INDEX)
        else:
            result_train_features = result_features

        if is_transform:
            index_name = X.index.name
            renamed_column = None
            if index_name in X.columns:
                renamed_column = f"{index_name}_renamed"
                X = X.rename(columns={index_name: renamed_column})
            result_train = pd.concat([X.reset_index(), result_train_features.reset_index(drop=True)], axis=1).set_index(
                index_name or DEFAULT_INDEX
            )
            result_train.index.name = index_name
            if renamed_column is not None:
                result_train = result_train.rename(columns={renamed_column: index_name})
        else:
            result_train = result_train_features

        if drop_system_record_id:
            result_train = result_train.drop(columns=[SYSTEM_RECORD_ID, ENTITY_SYSTEM_RECORD_ID], errors="ignore")
            for eval_set_index in result_eval_sets.keys():
                result_eval_sets[eval_set_index] = result_eval_sets[eval_set_index].drop(
                    columns=[SYSTEM_RECORD_ID, ENTITY_SYSTEM_RECORD_ID], errors="ignore"
                )

        return result_train, result_eval_sets

    def __prepare_feature_importances(self, trace_id: str, x_columns: List[str], silent=False):
        llm_source = "LLM with external data augmentation"
        if self._search_task is None:
            raise NotFittedError(self.bundle.get("transform_unfitted_enricher"))
        features_meta = self._search_task.get_all_features_metadata_v2()
        if features_meta is None:
            raise Exception(self.bundle.get("missing_features_meta"))

        original_names_dict = {c.name: c.originalName for c in self._search_task.get_file_metadata(trace_id).columns}
        features_df = self._search_task.get_all_initial_raw_features(trace_id, metrics_calculation=True)

        self.feature_names_ = []
        self.feature_importances_ = []
        features_info = []
        features_info_without_links = []
        internal_features_info = []

        def round_shap_value(shap: float) -> float:
            if shap > 0.0 and shap < 0.0001:
                return 0.0001
            else:
                return round(shap, 4)

        def list_or_single(lst: List[str], single: str):
            return lst or ([single] if single else [])

        def to_anchor(link: str, value: str) -> str:
            if not value:
                return ""
            elif not link:
                return value
            elif value == llm_source:
                return value
            else:
                return f"<a href='{link}' target='_blank' rel='noopener noreferrer'>{value}</a>"

        def make_links(names: List[str], links: List[str]):
            all_links = [to_anchor(link, name) for name, link in itertools.zip_longest(names, links)]
            return ",".join(all_links)

        features_meta.sort(key=lambda m: (-m.shap_value, m.name))
        for feature_meta in features_meta:
            if feature_meta.name in original_names_dict.keys():
                feature_meta.name = original_names_dict[feature_meta.name]
            # Use only enriched features
            if (
                feature_meta.name in x_columns
                or feature_meta.name == COUNTRY
                or feature_meta.shap_value == 0.0
                or feature_meta.name in self.fit_generated_features
            ):
                continue

            feature_sample = []
            self.feature_names_.append(feature_meta.name)
            self.feature_importances_.append(round_shap_value(feature_meta.shap_value))
            if feature_meta.name in features_df.columns:
                feature_sample = np.random.choice(features_df[feature_meta.name].dropna().unique(), 3).tolist()
                if len(feature_sample) > 0 and isinstance(feature_sample[0], float):
                    feature_sample = [round(f, 4) for f in feature_sample]
                feature_sample = [str(f) for f in feature_sample]
                feature_sample = ", ".join(feature_sample)
                if len(feature_sample) > 30:
                    feature_sample = feature_sample[:30] + "..."

            internal_provider = feature_meta.data_provider or "Upgini"
            providers = list_or_single(feature_meta.data_providers, feature_meta.data_provider)
            provider_links = list_or_single(feature_meta.data_provider_links, feature_meta.data_provider_link)
            if providers:
                provider = make_links(providers, provider_links)
            else:
                provider = to_anchor("https://upgini.com", "Upgini")

            internal_source = feature_meta.data_source or (
                llm_source
                if not feature_meta.name.endswith("_country") and not feature_meta.name.endswith("_postal_code")
                else ""
            )
            sources = list_or_single(feature_meta.data_sources, feature_meta.data_source)
            source_links = list_or_single(feature_meta.data_source_links, feature_meta.data_source_link)
            if sources:
                source = make_links(sources, source_links)
            else:
                source = internal_source

            internal_feature_name = feature_meta.name
            if feature_meta.doc_link:
                feature_name = to_anchor(feature_meta.doc_link, feature_meta.name)
            else:
                feature_name = internal_feature_name

            features_info.append(
                {
                    self.bundle.get("features_info_name"): feature_name,
                    self.bundle.get("features_info_shap"): round_shap_value(feature_meta.shap_value),
                    self.bundle.get("features_info_hitrate"): feature_meta.hit_rate,
                    self.bundle.get("features_info_value_preview"): feature_sample,
                    self.bundle.get("features_info_provider"): provider,
                    self.bundle.get("features_info_source"): source,
                    self.bundle.get("features_info_update_frequency"): feature_meta.update_frequency,
                }
            )
            features_info_without_links.append(
                {
                    self.bundle.get("features_info_name"): internal_feature_name,
                    self.bundle.get("features_info_shap"): round_shap_value(feature_meta.shap_value),
                    self.bundle.get("features_info_hitrate"): feature_meta.hit_rate,
                    self.bundle.get("features_info_value_preview"): feature_sample,
                    self.bundle.get("features_info_provider"): internal_provider,
                    self.bundle.get("features_info_source"): internal_source,
                    self.bundle.get("features_info_update_frequency"): feature_meta.update_frequency,
                }
            )
            internal_features_info.append(
                {
                    self.bundle.get("features_info_name"): internal_feature_name,
                    "feature_link": feature_meta.doc_link,
                    self.bundle.get("features_info_shap"): round_shap_value(feature_meta.shap_value),
                    self.bundle.get("features_info_hitrate"): feature_meta.hit_rate,
                    self.bundle.get("features_info_value_preview"): feature_sample,
                    self.bundle.get("features_info_provider"): internal_provider,
                    "provider_link": feature_meta.data_provider_link,
                    self.bundle.get("features_info_source"): internal_source,
                    "source_link": feature_meta.data_source_link,
                    self.bundle.get("features_info_commercial_schema"): feature_meta.commercial_schema or "",
                    self.bundle.get("features_info_update_frequency"): feature_meta.update_frequency,
                }
            )

        if len(features_info) > 0:
            self.features_info = pd.DataFrame(features_info)
            self._features_info_without_links = pd.DataFrame(features_info_without_links)
            self._internal_features_info = pd.DataFrame(internal_features_info)
            if not silent:
                do_without_pandas_limits(lambda: self.logger.info(f"Features info:\n{self._internal_features_info}"))

            self.relevant_data_sources = self._group_relevant_data_sources(self.features_info, self.bundle)
            self._relevant_data_sources_wo_links = self._group_relevant_data_sources(
                self._features_info_without_links, self.bundle
            )
            if not silent:
                do_without_pandas_limits(
                    lambda: self.logger.info(f"Relevant data sources:\n{self._relevant_data_sources_wo_links}")
                )
        else:
            self.logger.warning("Empty features info")

    def get_autofe_features_description(self):
        try:
            autofe_meta = self._search_task.get_autofe_metadata()
            if autofe_meta is None:
                return None
            features_meta = self._search_task.get_all_features_metadata_v2()

            def get_feature_by_name(name: str):
                for m in features_meta:
                    if m.name == name:
                        return m

            descriptions = []
            for m in autofe_meta:
                orig_to_hashed = {base_column.original_name: base_column.hashed_name for base_column in m.base_columns}

                autofe_feature = (
                    Feature.from_formula(m.formula)
                    .set_display_index(m.display_index)
                    .set_alias(m.alias)
                    .set_op_params(m.operator_params or {})
                    .rename_columns(orig_to_hashed)
                )

                if autofe_feature.op.is_vector:
                    continue

                description = {}

                feature_meta = get_feature_by_name(autofe_feature.get_display_name(shorten=True))
                if feature_meta is None:
                    self.logger.warning(f"Feature meta for display index {m.display_index} not found")
                    continue
                description["shap"] = feature_meta.shap_value
                description["Sources"] = feature_meta.data_source.replace("AutoFE: features from ", "").replace(
                    "AutoFE: feature from ", ""
                )
                description["Feature name"] = feature_meta.name

                feature_idx = 1
                for bc in m.base_columns:
                    description[f"Feature {feature_idx}"] = bc.hashed_name
                    feature_idx += 1

                description["Function"] = ",".join(sorted(autofe_feature.get_all_operand_names()))

                descriptions.append(description)

            if len(descriptions) == 0:
                return None

            descriptions_df = pd.DataFrame(descriptions)
            descriptions_df.fillna("", inplace=True)
            descriptions_df.sort_values(by="shap", ascending=False, inplace=True)
            descriptions_df.drop(columns="shap", inplace=True)
            return descriptions_df

        except Exception:
            self.logger.exception("Failed to generate AutoFE features description")
            return None

    @staticmethod
    def _group_relevant_data_sources(df: pd.DataFrame, bundle: ResourceBundle) -> pd.DataFrame:
        return (
            df.query(f"{bundle.get('features_info_provider')} != ''")
            .groupby([bundle.get("features_info_provider"), bundle.get("features_info_source")])
            .agg(
                shap_sum=(bundle.get("features_info_shap"), "sum"),
                row_count=(bundle.get("features_info_shap"), "count"),
            )
            .sort_values(by="shap_sum", ascending=False)
            .reset_index()
            .rename(
                columns={
                    "shap_sum": bundle.get("relevant_data_sources_all_shap"),
                    "row_count": bundle.get("relevant_data_sources_number"),
                }
            )
        )

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

    def __prepare_search_keys(
        self,
        x: pd.DataFrame,
        search_keys: Dict[str, SearchKey],
        is_demo_dataset: bool,
        is_transform=False,
        silent_mode=False,
    ):
        for _, key_type in search_keys.items():
            if not isinstance(key_type, SearchKey):
                raise ValidationError(self.bundle.get("unsupported_type_of_search_key").format(key_type))
        valid_search_keys = {}
        unsupported_search_keys = {
            SearchKey.IP_RANGE_FROM,
            SearchKey.IP_RANGE_TO,
            SearchKey.MSISDN_RANGE_FROM,
            SearchKey.MSISDN_RANGE_TO,
            # SearchKey.EMAIL_ONE_DOMAIN,
        }
        passed_unsupported_search_keys = unsupported_search_keys.intersection(search_keys.values())
        if len(passed_unsupported_search_keys) > 0:
            raise ValidationError(self.bundle.get("unsupported_search_key").format(passed_unsupported_search_keys))

        for column_id, meaning_type in search_keys.items():
            column_name = None
            if isinstance(column_id, str):
                if column_id not in x.columns:
                    raise ValidationError(self.bundle.get("search_key_not_found").format(column_id, list(x.columns)))
                column_name = column_id
                valid_search_keys[column_name] = meaning_type
            elif isinstance(column_id, int):
                if column_id >= x.shape[1]:
                    raise ValidationError(self.bundle.get("numeric_search_key_not_found").format(column_id, x.shape[1]))
                column_name = x.columns[column_id]
                valid_search_keys[column_name] = meaning_type
            else:
                raise ValidationError(self.bundle.get("unsupported_search_key_type").format(type(column_id)))

            if meaning_type == SearchKey.COUNTRY and self.country_code is not None:
                msg = self.bundle.get("search_key_country_and_country_code")
                self.logger.warning(msg)
                print(msg)
                self.country_code = None

            if not self.__is_registered and not is_demo_dataset and meaning_type in SearchKey.personal_keys():
                msg = self.bundle.get("unregistered_with_personal_keys").format(meaning_type)
                self.logger.warning(msg)
                if not silent_mode:
                    self.warning_counter.increment()
                    print(msg)

                valid_search_keys[column_name] = SearchKey.CUSTOM_KEY
            else:
                if x[column_name].isnull().all() or (
                    (is_string_dtype(x[column_name]) or is_object_dtype(x[column_name]))
                    and (x[column_name].astype("string").str.strip() == "").all()
                ):
                    raise ValidationError(self.bundle.get("empty_search_key").format(column_name))

        if self.detect_missing_search_keys and (
            not is_transform or set(valid_search_keys.values()) != set(self.fit_search_keys.values())
        ):
            valid_search_keys = self.__detect_missing_search_keys(
                x, valid_search_keys, is_demo_dataset, silent_mode, is_transform
            )

        if all(k == SearchKey.CUSTOM_KEY for k in valid_search_keys.values()):
            msg = self.bundle.get("unregistered_only_personal_keys")
            self.logger.warning(msg + f" Provided search keys: {search_keys}")
            raise ValidationError(msg)

        if SearchKey.CUSTOM_KEY in valid_search_keys.values():
            custom_keys = [column for column, key in valid_search_keys.items() if key == SearchKey.CUSTOM_KEY]
            for key in custom_keys:
                del valid_search_keys[key]

        if (
            len(valid_search_keys.values()) == 1
            and self.country_code is None
            and next(iter(valid_search_keys.values())) == SearchKey.DATE
            and not silent_mode
        ):
            msg = self.bundle.get("date_only_search")
            print(msg)
            self.logger.warning(msg)
            self.warning_counter.increment()

        maybe_date = [k for k, v in valid_search_keys.items() if v in [SearchKey.DATE, SearchKey.DATETIME]]
        if (self.cv is None or self.cv == CVType.k_fold) and len(maybe_date) > 0 and not silent_mode:
            date_column = next(iter(maybe_date))
            if x[date_column].nunique() > 0.9 * _num_samples(x):
                msg = self.bundle.get("date_search_without_time_series")
                print(msg)
                self.logger.warning(msg)
                self.warning_counter.increment()

        if len(valid_search_keys) == 1:
            key, value = list(valid_search_keys.items())[0]
            # Show warning for country only if country is the only key
            if x[key].nunique() == 1:
                msg = self.bundle.get("single_constant_search_key").format(value, x[key].values[0])
                print(msg)
                self.logger.warning(msg)
                self.warning_counter.increment()

        self.logger.info(f"Prepared search keys: {valid_search_keys}")

        return valid_search_keys

    def __show_metrics(
        self,
        scoring: Union[Callable, str, None],
        estimator: Optional[Any],
        importance_threshold: Optional[float],
        max_features: Optional[int],
        remove_outliers_calc_metrics: Optional[bool],
        trace_id: str,
        progress_bar: Optional[ProgressBar] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
    ):
        self.metrics = self.calculate_metrics(
            scoring=scoring,
            estimator=estimator,
            importance_threshold=importance_threshold,
            max_features=max_features,
            remove_outliers_calc_metrics=remove_outliers_calc_metrics,
            trace_id=trace_id,
            silent=True,
            progress_bar=progress_bar,
            progress_callback=progress_callback,
        )
        if self.metrics is not None:
            msg = self.bundle.get("quality_metrics_header")
            display_html_dataframe(self.metrics, self.metrics, msg)

    def __show_selected_features(self, search_keys: Dict[str, SearchKey]):
        msg = self.bundle.get("features_info_header").format(len(self.feature_names_), list(search_keys.keys()))

        try:
            _ = get_ipython()  # type: ignore

            print(Format.GREEN + Format.BOLD + msg + Format.END)
            self.logger.info(msg)
            if len(self.feature_names_) > 0:
                display_html_dataframe(
                    self.features_info, self._features_info_without_links, self.bundle.get("relevant_features_header")
                )

                display_html_dataframe(
                    self.relevant_data_sources,
                    self._relevant_data_sources_wo_links,
                    self.bundle.get("relevant_data_sources_header"),
                )
            else:
                msg = self.bundle.get("features_info_zero_important_features")
                self.logger.warning(msg)
                self.__display_support_link(msg)
                self.warning_counter.increment()
        except (ImportError, NameError):
            print(msg)
            print(self._internal_features_info)

    def __show_report_button(self):
        try:
            prepare_and_show_report(
                relevant_features_df=self._features_info_without_links,
                relevant_datasources_df=self.relevant_data_sources,
                metrics_df=self.metrics,
                autofe_descriptions_df=self.get_autofe_features_description(),
                search_id=self._search_task.search_task_id,
                email=self.rest_client.get_current_email(),
                search_keys=[str(sk) for sk in self.search_keys.values()],
            )
        except Exception:
            pass

    def __validate_importance_threshold(self, importance_threshold: Optional[float]) -> float:
        try:
            return float(importance_threshold) if importance_threshold is not None else 0.0
        except ValueError:
            self.logger.exception(f"Invalid importance_threshold provided: {importance_threshold}")
            raise ValidationError(self.bundle.get("invalid_importance_threshold"))

    def __validate_max_features(self, max_features: Optional[int]) -> int:
        try:
            return int(max_features) if max_features is not None else 400
        except ValueError:
            self.logger.exception(f"Invalid max_features provided: {max_features}")
            raise ValidationError(self.bundle.get("invalid_max_features"))

    def __filtered_enriched_features(
        self,
        importance_threshold: Optional[float],
        max_features: Optional[int],
    ) -> List[str]:
        importance_threshold = self.__validate_importance_threshold(importance_threshold)
        max_features = self.__validate_max_features(max_features)

        return self.__filtered_importance_names(importance_threshold, max_features)

    def __detect_missing_search_keys(
        self,
        df: pd.DataFrame,
        search_keys: Dict[str, SearchKey],
        is_demo_dataset: bool,
        silent_mode=False,
        is_transform=False,
    ) -> Dict[str, SearchKey]:
        sample = df.head(100)

        def check_need_detect(search_key: SearchKey):
            return not is_transform or search_key in self.fit_search_keys.values()

        # if SearchKey.POSTAL_CODE not in search_keys.values() and check_need_detect(SearchKey.POSTAL_CODE):
        if check_need_detect(SearchKey.POSTAL_CODE):
            maybe_keys = PostalCodeSearchKeyDetector().get_search_key_columns(sample, search_keys)
            if maybe_keys:
                new_keys = {key: SearchKey.POSTAL_CODE for key in maybe_keys}
                search_keys.update(new_keys)
                self.autodetected_search_keys.update(new_keys)
                self.logger.info(f"Autodetected search key POSTAL_CODE in column {maybe_keys}")
                if not silent_mode:
                    print(self.bundle.get("postal_code_detected").format(maybe_keys))

        if (
            SearchKey.COUNTRY not in search_keys.values()
            and self.country_code is None
            and check_need_detect(SearchKey.COUNTRY)
        ):
            maybe_key = CountrySearchKeyDetector().get_search_key_columns(sample, search_keys)
            if maybe_key:
                search_keys[maybe_key[0]] = SearchKey.COUNTRY
                self.autodetected_search_keys[maybe_key[0]] = SearchKey.COUNTRY
                self.logger.info(f"Autodetected search key COUNTRY in column {maybe_key}")
                if not silent_mode:
                    print(self.bundle.get("country_detected").format(maybe_key))

        if (
            # SearchKey.EMAIL not in search_keys.values()
            SearchKey.HEM not in search_keys.values()
            and check_need_detect(SearchKey.HEM)
        ):
            maybe_keys = EmailSearchKeyDetector().get_search_key_columns(sample, search_keys)
            if maybe_keys:
                if self.__is_registered or is_demo_dataset:
                    new_keys = {key: SearchKey.EMAIL for key in maybe_keys}
                    search_keys.update(new_keys)
                    self.autodetected_search_keys.update(new_keys)
                    self.logger.info(f"Autodetected search key EMAIL in column {maybe_keys}")
                    if not silent_mode:
                        print(self.bundle.get("email_detected").format(maybe_keys))
                else:
                    self.logger.warning(
                        f"Autodetected search key EMAIL in column {maybe_keys}."
                        " But not used because not registered user"
                    )
                    if not silent_mode:
                        print(self.bundle.get("email_detected_not_registered").format(maybe_keys))
                    self.warning_counter.increment()

        # if SearchKey.PHONE not in search_keys.values() and check_need_detect(SearchKey.PHONE):
        if check_need_detect(SearchKey.PHONE):
            maybe_keys = PhoneSearchKeyDetector().get_search_key_columns(sample, search_keys)
            if maybe_keys:
                if self.__is_registered or is_demo_dataset:
                    new_keys = {key: SearchKey.PHONE for key in maybe_keys}
                    search_keys.update(new_keys)
                    self.autodetected_search_keys.update(new_keys)
                    self.logger.info(f"Autodetected search key PHONE in column {maybe_keys}")
                    if not silent_mode:
                        print(self.bundle.get("phone_detected").format(maybe_keys))
                else:
                    self.logger.warning(
                        f"Autodetected search key PHONE in column {maybe_keys}. "
                        "But not used because not registered user"
                    )
                    if not silent_mode:
                        print(self.bundle.get("phone_detected_not_registered"))
                    self.warning_counter.increment()

        return search_keys

    def _validate_binary_observations(self, y, task_type: ModelTaskType):
        if task_type == ModelTaskType.BINARY and (y.value_counts() < 1000).any():
            msg = self.bundle.get("binary_small_dataset")
            self.logger.warning(msg)
            print(msg)

    def _validate_PSI(self, df: pd.DataFrame):
        if EVAL_SET_INDEX in df.columns:
            train = df.query(f"{EVAL_SET_INDEX} == 0")
            eval1 = df.query(f"{EVAL_SET_INDEX} == 1")
        else:
            train = df
            eval1 = None

        # 1. Check train PSI
        half_train = round(len(train) / 2)
        part1 = train[:half_train]
        part2 = train[half_train:]
        train_psi = calculate_psi(part1[self.TARGET_NAME], part2[self.TARGET_NAME])
        if train_psi > 0.2:
            self.warning_counter.increment()
            msg = self.bundle.get("train_unstable_target").format(train_psi)
            print(msg)
            self.logger.warning(msg)

        # 2. Check train-test PSI
        if eval1 is not None:
            train_test_psi = calculate_psi(train[self.TARGET_NAME], eval1[self.TARGET_NAME])
            if train_test_psi > 0.2:
                self.warning_counter.increment()
                msg = self.bundle.get("eval_unstable_target").format(train_test_psi)
                print(msg)
                self.logger.warning(msg)

    def _dump_python_libs(self):
        try:
            from pip._internal.operations.freeze import freeze

            python_version = sys.version
            libs = list(freeze(local_only=True))
            self.logger.warning(f"User python {python_version} libs versions:\n{libs}")
        except Exception:
            self.logger.exception("Failed to dump python libs")

    def __display_support_link(self, link_text: Optional[str] = None):
        support_link = self.bundle.get("support_link")
        link_text = link_text or self.bundle.get("support_text")
        try:
            from IPython.display import HTML, display

            _ = get_ipython()  # type: ignore
            self.logger.warning(f"Showing support link: {link_text}")
            display(
                HTML(
                    f"""<br/>{link_text} <a href='{support_link}' target='_blank' rel='noopener noreferrer'>
                    here</a>"""
                )
            )
        except (ImportError, NameError):
            print(f"{link_text} at {support_link}")

    def _show_error(self, msg):
        try:
            _ = get_ipython()  # type: ignore
            print(Format.RED + Format.BOLD + msg + Format.END)
        except (ImportError, NameError):
            print(msg)

    def dump_input(
        self,
        trace_id: str,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series, None] = None,
        eval_set: Union[Tuple, None] = None,
    ):
        def dump_task():
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
                    if isinstance(inp, (pd.DataFrame, pd.Series)):
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
                        if eval_set and _num_samples(eval_set[0][0]) > 0:
                            eval_xy_sample_index = rnd.randint(0, _num_samples(eval_set[0][0]), size=1000)
                            with open(f"{tmp_dir}/eval_x.pickle", "wb") as eval_x_file:
                                pickle.dump(sample(eval_set[0][0], eval_xy_sample_index), eval_x_file)
                            with open(f"{tmp_dir}/eval_y.pickle", "wb") as eval_y_file:
                                pickle.dump(sample(eval_set[0][1], eval_xy_sample_index), eval_y_file)
                            self.rest_client.dump_input_files(
                                trace_id,
                                f"{tmp_dir}/x.pickle",
                                f"{tmp_dir}/y.pickle",
                                f"{tmp_dir}/eval_x.pickle",
                                f"{tmp_dir}/eval_y.pickle",
                            )
                        else:
                            self.rest_client.dump_input_files(
                                trace_id,
                                f"{tmp_dir}/x.pickle",
                                f"{tmp_dir}/y.pickle",
                            )
                    else:
                        self.rest_client.dump_input_files(
                            trace_id,
                            f"{tmp_dir}/x.pickle",
                        )
            except Exception:
                self.logger.warning("Failed to dump input files", exc_info=True)

        try:
            Thread(target=dump_task, daemon=True).start()
        except Exception:
            self.logger.warning("Failed to dump input files", exc_info=True)


def _num_samples(x):
    """Return number of samples in array-like x."""
    if x is None:
        return 0
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


def is_frames_equal(first, second, bundle: ResourceBundle) -> bool:
    if (isinstance(first, pd.DataFrame) and isinstance(second, pd.DataFrame)) or (
        isinstance(first, pd.Series) and isinstance(second, pd.Series)
    ):
        return first.equals(second)
    elif isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return np.array_equal(first, second)
    elif type(first) is type(second):
        return first == second
    else:
        raise ValidationError(bundle.get("x_and_eval_x_diff_types").format(type(first), type(second)))


def drop_duplicates(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.drop_duplicates()
    elif isinstance(df, np.ndarray):
        return pd.DataFrame(df).drop_duplicates()
    else:
        return df


def hash_input(X: pd.DataFrame, y: Optional[pd.Series] = None, eval_set: Optional[List[Tuple]] = None) -> str:
    hashed_objects = []
    try:
        hashed_objects.append(pd.util.hash_pandas_object(X, index=False).values)
        if y is not None:
            hashed_objects.append(pd.util.hash_pandas_object(y, index=False).values)
        if eval_set is not None:
            for eval_X, eval_y in eval_set:
                hashed_objects.append(pd.util.hash_pandas_object(eval_X, index=False).values)
                hashed_objects.append(pd.util.hash_pandas_object(eval_y, index=False).values)
        common_hash = hashlib.sha256(np.concatenate(hashed_objects)).hexdigest()
        return common_hash
    except Exception:
        return ""
