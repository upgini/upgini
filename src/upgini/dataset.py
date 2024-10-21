import csv
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from upgini.errors import ValidationError
from upgini.http import ProgressStage, SearchProgress, _RestClient
from upgini.metadata import (
    ENTITY_SYSTEM_RECORD_ID,
    EVAL_SET_INDEX,
    SYSTEM_RECORD_ID,
    TARGET,
    DataType,
    FeaturesFilter,
    FileColumnMeaningType,
    FileColumnMetadata,
    FileMetadata,
    FileMetrics,
    ModelTaskType,
    NumericInterval,
    RuntimeParameters,
    SearchCustomization,
)
from upgini.resource_bundle import ResourceBundle, get_custom_bundle
from upgini.search_task import SearchTask
from upgini.utils.email_utils import EmailSearchKeyConverter
from upgini.utils.target_utils import balance_undersample

try:
    from upgini.utils.progress_bar import CustomProgressBar as ProgressBar
except Exception:
    from upgini.utils.fallback_progress_bar import CustomFallbackProgressBar as ProgressBar

from upgini.utils.warning_counter import WarningCounter


class Dataset:  # (pd.DataFrame):
    MIN_ROWS_COUNT = 100
    MAX_ROWS = 200_000
    FIT_SAMPLE_ROWS = 200_000
    FIT_SAMPLE_THRESHOLD = 200_000
    FIT_SAMPLE_WITH_EVAL_SET_ROWS = 200_000
    FIT_SAMPLE_WITH_EVAL_SET_THRESHOLD = 200_000
    BINARY_MIN_SAMPLE_THRESHOLD = 5_000
    MULTICLASS_MIN_SAMPLE_THRESHOLD = 25_000
    IMBALANCE_THESHOLD = 0.6
    BINARY_BOOTSTRAP_LOOPS = 5
    MULTICLASS_BOOTSTRAP_LOOPS = 2
    MIN_TARGET_CLASS_ROWS = 100
    MAX_MULTICLASS_CLASS_COUNT = 100
    MIN_SUPPORTED_DATE_TS = 946684800000  # 2000-01-01
    MAX_FEATURES_COUNT = 3500
    MAX_UPLOADING_FILE_SIZE = 268435456  # 256 Mb
    MAX_STRING_FEATURE_LENGTH = 24573

    def __init__(
        self,
        dataset_name: str,
        description: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        path: Optional[str] = None,
        meaning_types: Optional[Dict[str, FileColumnMeaningType]] = None,
        search_keys: Optional[List[Tuple[str, ...]]] = None,
        unnest_search_keys: Optional[Dict[str, str]] = None,
        model_task_type: Optional[ModelTaskType] = None,
        random_state: Optional[int] = None,
        rest_client: Optional[_RestClient] = None,
        logger: Optional[logging.Logger] = None,
        warning_counter: Optional[WarningCounter] = None,
        bundle: Optional[ResourceBundle] = None,
        **kwargs,
    ):
        self.bundle = bundle or get_custom_bundle()
        if df is not None:
            data = df.copy()
        elif path is not None:
            if "sep" in kwargs:
                data = pd.read_csv(path, **kwargs)
            else:
                # try different separators: , ; \t ...
                with open(path) as csvfile:
                    sep = csv.Sniffer().sniff(csvfile.read(2048)).delimiter
                kwargs["sep"] = sep
                data = pd.read_csv(path, **kwargs)
        else:
            raise ValueError(self.bundle.get("dataset_dataframe_or_path_empty"))
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, pd.io.parsers.TextFileReader):  # type: ignore
            raise ValueError(self.bundle.get("dataset_dataframe_iterator"))
        else:
            raise ValueError(self.bundle.get("dataset_dataframe_not_pandas"))

        self.dataset_name = dataset_name
        self.task_type = model_task_type
        self.description = description
        self.meaning_types = meaning_types
        self.search_keys = search_keys
        self.unnest_search_keys = unnest_search_keys
        self.hierarchical_group_keys = []
        self.hierarchical_subgroup_keys = []
        self.file_upload_id: Optional[str] = None
        self.etalon_def: Optional[Dict[str, str]] = None
        self.rest_client = rest_client
        self.random_state = random_state
        self.columns_renaming: Dict[str, str] = {}
        self.imbalanced: bool = False
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")
        self.warning_counter = warning_counter or WarningCounter()

    def __len__(self):
        return len(self.data) if self.data is not None else None

    @property
    def columns(self):
        return self.data.columns if self.data is not None else None

    @property
    def meaning_types_checked(self) -> Dict[str, FileColumnMeaningType]:
        if self.meaning_types is None:
            raise ValueError(self.bundle.get("dataset_empty_meaning_types"))
        else:
            return self.meaning_types

    @property
    def search_keys_checked(self) -> List[Tuple[str, ...]]:
        if self.search_keys is None:
            raise ValueError(self.bundle.get("dataset_empty_search_keys"))
        else:
            return self.search_keys

    @property
    def etalon_def_checked(self) -> Dict[str, str]:
        if self.etalon_def is None:
            self.etalon_def = {
                v.value: k for k, v in self.meaning_types_checked.items() if v != FileColumnMeaningType.FEATURE
            }

        return self.etalon_def

    def __validate_min_rows_count(self):
        if len(self.data) < self.MIN_ROWS_COUNT:
            raise ValidationError(self.bundle.get("dataset_too_few_rows").format(self.MIN_ROWS_COUNT))

    def __validate_max_row_count(self):
        if ENTITY_SYSTEM_RECORD_ID in self.data.columns:
            rows_count = self.data[ENTITY_SYSTEM_RECORD_ID].nunique()
        else:
            rows_count = len(self.data)
        if rows_count > self.MAX_ROWS:
            raise ValidationError(self.bundle.get("dataset_too_many_rows_registered").format(self.MAX_ROWS))

    def __target_value(self) -> pd.Series:
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, "")
        target: pd.Series = self.data[target_column]
        # clean target from nulls
        target.dropna(inplace=True)
        if is_numeric_dtype(target):
            target = target.loc[np.isfinite(target)]  # type: ignore
        else:
            target = target.loc[target != ""]

        return target

    def __validate_target(self):
        # self.logger.info("Validating target")
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, "")
        target = self.data[target_column]

        if self.task_type == ModelTaskType.BINARY:
            if not is_integer_dtype(target):
                try:
                    target = target.astype("category").cat.codes
                except ValueError:
                    self.logger.exception("Failed to cast target to category codes for binary task type")
                    raise ValidationError(self.bundle.get("dataset_invalid_target_type").format(target.dtype))
            target_classes_count = target.nunique()
            if target_classes_count != 2:
                msg = self.bundle.get("dataset_invalid_binary_target").format(target_classes_count)
                self.logger.warning(msg)
                raise ValidationError(msg)
        elif self.task_type == ModelTaskType.MULTICLASS:
            if not is_integer_dtype(target):
                try:
                    target = self.data[target_column].astype("category").cat.codes
                except Exception:
                    self.logger.exception("Failed to cast target to category codes for multiclass task type")
                    raise ValidationError(self.bundle.get("dataset_invalid_multiclass_target").format(target.dtype))
        elif self.task_type == ModelTaskType.REGRESSION:
            if not is_float_dtype(target):
                try:
                    self.data[target_column] = self.data[target_column].astype("float64")
                except ValueError:
                    self.logger.exception("Failed to cast target to float for regression task type")
                    raise ValidationError(self.bundle.get("dataset_invalid_regression_target").format(target.dtype))
        elif self.task_type == ModelTaskType.TIMESERIES:
            if not is_float_dtype(target):
                try:
                    self.data[target_column] = self.data[target_column].astype("float64")
                except ValueError:
                    self.logger.exception("Failed to cast target to float for timeseries task type")
                    raise ValidationError(self.bundle.get("dataset_invalid_timeseries_target").format(target.dtype))

    def __resample(self):
        # self.logger.info("Resampling etalon")
        # Resample imbalanced target. Only train segment (without eval_set)
        if EVAL_SET_INDEX in self.data.columns:
            train_segment = self.data[self.data[EVAL_SET_INDEX] == 0]
        else:
            train_segment = self.data

        if self.task_type == ModelTaskType.MULTICLASS or (
            self.task_type == ModelTaskType.BINARY and len(train_segment) > self.BINARY_MIN_SAMPLE_THRESHOLD
        ):
            count = len(train_segment)
            target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, TARGET)
            target = train_segment[target_column]
            target_classes_count = target.nunique()

            if target_classes_count > self.MAX_MULTICLASS_CLASS_COUNT:
                msg = self.bundle.get("dataset_to_many_multiclass_targets").format(
                    target_classes_count, self.MAX_MULTICLASS_CLASS_COUNT
                )
                self.logger.warning(msg)
                raise ValidationError(msg)

            vc = target.value_counts()
            min_class_value = vc.index[len(vc) - 1]
            min_class_count = vc[min_class_value]

            if min_class_count < self.MIN_TARGET_CLASS_ROWS:
                msg = self.bundle.get("dataset_rarest_class_less_min").format(
                    min_class_value, min_class_count, self.MIN_TARGET_CLASS_ROWS
                )
                self.logger.warning(msg)
                raise ValidationError(msg)

            min_class_percent = self.IMBALANCE_THESHOLD / target_classes_count
            min_class_threshold = min_class_percent * count

            # If min class count less than 30% for binary or (60 / classes_count)% for multiclass
            if min_class_count < min_class_threshold:
                self.imbalanced = True
                self.data = balance_undersample(
                    df=train_segment,
                    target_column=target_column,
                    task_type=self.task_type,
                    random_state=self.random_state,
                    binary_min_sample_threshold=self.BINARY_MIN_SAMPLE_THRESHOLD,
                    multiclass_min_sample_threshold=self.MULTICLASS_MIN_SAMPLE_THRESHOLD,
                    binary_bootstrap_loops=self.BINARY_BOOTSTRAP_LOOPS,
                    multiclass_bootstrap_loops=self.MULTICLASS_BOOTSTRAP_LOOPS,
                    logger=self.logger,
                    bundle=self.bundle,
                    warning_counter=self.warning_counter,
                )

        # Resample over fit threshold
        if not self.imbalanced and EVAL_SET_INDEX in self.data.columns:
            sample_threshold = self.FIT_SAMPLE_WITH_EVAL_SET_THRESHOLD
            sample_rows = self.FIT_SAMPLE_WITH_EVAL_SET_ROWS
        else:
            sample_threshold = self.FIT_SAMPLE_THRESHOLD
            sample_rows = self.FIT_SAMPLE_ROWS

        if len(self.data) > sample_threshold:
            self.logger.info(
                f"Etalon has size {len(self.data)} more than threshold {sample_threshold} "
                f"and will be downsampled to {sample_rows}"
            )
            resampled_data = self.data.sample(n=sample_rows, random_state=self.random_state)
            self.data = resampled_data
            self.logger.info(f"Shape after threshold resampling: {self.data.shape}")

    def __validate_dataset(self, validate_target: bool, silent_mode: bool):
        """Validate DataSet"""
        # self.logger.info("validating etalon")
        target = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        if validate_target:
            if target is None:
                raise ValidationError(self.bundle.get("dataset_missing_target"))

            target_value = self.__target_value()
            target_items = target_value.nunique()
            if target_items == 1:
                raise ValidationError(self.bundle.get("dataset_constant_target"))
            elif target_items == 0:
                raise ValidationError(self.bundle.get("dataset_empty_target"))

            # if self.task_type != ModelTaskType.MULTICLASS:
            #     self.data[target] = self.data[target].apply(pd.to_numeric, errors="coerce")

        keys_to_validate = {
            key
            for search_group in self.search_keys_checked
            for key in search_group
            if not self.columns_renaming.get(key).endswith(EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX)
        }
        ipv4_column = self.etalon_def_checked.get(FileColumnMeaningType.IP_ADDRESS.value)
        if (
            FileColumnMeaningType.IPV6_ADDRESS.value in self.etalon_def_checked
            and ipv4_column is not None
            and ipv4_column in keys_to_validate
        ):
            keys_to_validate.remove(ipv4_column)

        mandatory_columns = [target]
        columns_to_validate = mandatory_columns.copy()
        columns_to_validate.extend(keys_to_validate)
        columns_to_validate = set([i for i in columns_to_validate if i is not None])

        nrows = len(self.data)
        validation_stats = {}
        self.data["valid_keys"] = 0
        self.data["valid_mandatory"] = True

        all_valid_status = self.bundle.get("validation_all_valid_status")
        some_invalid_status = self.bundle.get("validation_some_invalid_status")
        all_invalid_status = self.bundle.get("validation_all_invalid_status")
        all_valid_message = self.bundle.get("validation_all_valid_message")
        invalid_message = self.bundle.get("validation_invalid_message")

        for col in columns_to_validate:
            self.data[f"{col}_is_valid"] = ~self.data[col].isnull()
            if validate_target and target is not None and col == target:
                self.data.loc[self.data[target] == np.Inf, f"{col}_is_valid"] = False

            if col in mandatory_columns:
                self.data["valid_mandatory"] = self.data["valid_mandatory"] & self.data[f"{col}_is_valid"]

            invalid_values = list(self.data.loc[self.data[f"{col}_is_valid"] == 0, col].head().values)  # type: ignore
            valid_share = self.data[f"{col}_is_valid"].sum() / nrows
            original_col_name = self.columns_renaming[col]
            validation_stats[original_col_name] = {}
            if valid_share == 1:
                valid_status = all_valid_status
                valid_message = all_valid_message
            elif 0 < valid_share < 1:
                valid_status = some_invalid_status
                valid_message = invalid_message.format(100 * (1 - valid_share), invalid_values)
            else:
                valid_status = all_invalid_status
                valid_message = invalid_message.format(100 * (1 - valid_share), invalid_values)
            validation_stats[original_col_name]["valid_status"] = valid_status
            validation_stats[original_col_name]["valid_message"] = valid_message

            if col in keys_to_validate:
                self.data["valid_keys"] = self.data["valid_keys"] + self.data[f"{col}_is_valid"]
            self.data.drop(columns=f"{col}_is_valid", inplace=True)

        self.data["is_valid"] = self.data["valid_keys"] > 0
        self.data["is_valid"] = self.data["is_valid"] & self.data["valid_mandatory"]
        self.data.drop(columns=["valid_keys", "valid_mandatory"], inplace=True)

        drop_idx = self.data[self.data["is_valid"] != 1].index  # type: ignore
        self.data.drop(index=drop_idx, inplace=True)  # type: ignore
        self.data.drop(columns=["is_valid"], inplace=True)

        if not silent_mode:
            df_stats = pd.DataFrame.from_dict(validation_stats, orient="index")
            df_stats.reset_index(inplace=True)
            name_header = self.bundle.get("validation_column_name_header")
            status_header = self.bundle.get("validation_status_header")
            description_header = self.bundle.get("validation_descr_header")
            df_stats.columns = [name_header, status_header, description_header]
            try:
                import html

                from IPython.display import HTML, display  # type: ignore

                _ = get_ipython()  # type: ignore

                text_color = self.bundle.get("validation_text_color")
                colormap = {
                    all_valid_status: self.bundle.get("validation_all_valid_color"),
                    some_invalid_status: self.bundle.get("validation_some_invalid_color"),
                    all_invalid_status: self.bundle.get("validation_all_invalid_color"),
                }

                def map_color(text) -> str:
                    return (
                        f"<td style='background-color:{colormap[text]};color:{text_color}'>{text}</td>"
                        if text in colormap
                        else f"<td>{text}</td>"
                    )

                df_stats[description_header] = df_stats[description_header].apply(lambda x: html.escape(x))
                html_stats = (
                    "<table>"
                    + "<tr>"
                    + "".join(f"<th style='font-weight:bold'>{column}</th>" for column in df_stats.columns)
                    + "</tr>"
                    + "".join("<tr>" + "".join(map(map_color, row[1:])) + "</tr>" for row in df_stats.itertuples())
                    + "</table>"
                )
                print()
                display(HTML(html_stats))
            except (ImportError, NameError):
                print()
                print(df_stats)

        if len(self.data) == 0:
            raise ValidationError(self.bundle.get("all_search_keys_invalid"))

    def validate(self, validate_target: bool = True, silent_mode: bool = False):
        self.__validate_dataset(validate_target, silent_mode)

        if validate_target:
            self.__validate_target()

            self.__resample()

            self.__validate_min_rows_count()

        self.__validate_max_row_count()

        self.data = self.data.sort_values(by=SYSTEM_RECORD_ID).reset_index(drop=True)

    def __construct_metadata(self, exclude_features_sources: Optional[List[str]] = None) -> FileMetadata:
        # self.logger.info("Constructing dataset metadata")
        columns = []
        for index, (column_name, column_type) in enumerate(zip(self.data.columns, self.data.dtypes)):
            if column_name in self.meaning_types_checked:
                meaning_type = self.meaning_types_checked[column_name]
                # Temporary workaround while backend doesn't support datetime
                if meaning_type == FileColumnMeaningType.DATETIME:
                    meaning_type = FileColumnMeaningType.DATE
            else:
                meaning_type = FileColumnMeaningType.FEATURE
            if meaning_type in {
                FileColumnMeaningType.DATE,
                FileColumnMeaningType.DATETIME,
                # FileColumnMeaningType.IP_ADDRESS,
            }:
                min_value = self.data[column_name].astype("Int64").min()
                max_value = self.data[column_name].astype("Int64").max()
                min_max_values = NumericInterval(
                    minValue=min_value,
                    maxValue=max_value,
                )
            else:
                min_max_values = None
            column_meta = FileColumnMetadata(
                index=index,
                name=column_name,
                originalName=self.columns_renaming.get(column_name) or column_name,
                dataType=self.__get_data_type(column_type, column_name),
                meaningType=meaning_type,
                minMaxValues=min_max_values,
            )
            if self.unnest_search_keys and column_meta.originalName in self.unnest_search_keys:
                column_meta.isUnnest = True
                column_meta.unnestKeyNames = self.unnest_search_keys[column_meta.originalName]

            columns.append(column_meta)

        return FileMetadata(
            name=self.dataset_name,
            description=self.description,
            columns=columns,
            searchKeys=self.search_keys,
            excludeFeaturesSources=exclude_features_sources,
            hierarchicalGroupKeys=self.hierarchical_group_keys,
            hierarchicalSubgroupKeys=self.hierarchical_subgroup_keys,
            taskType=self.task_type,
        )

    def __get_data_type(self, pandas_data_type, column_name: str) -> DataType:
        if is_integer_dtype(pandas_data_type):
            return DataType.INT
        elif is_float_dtype(pandas_data_type):
            return DataType.DECIMAL
        elif is_string_dtype(pandas_data_type) or is_object_dtype(pandas_data_type):
            return DataType.STRING
        else:
            msg = self.bundle.get("dataset_invalid_column_type").format(column_name, pandas_data_type)
            self.logger.warning(msg)
            raise ValidationError(msg)

    def __construct_search_customization(
        self,
        return_scores: bool,
        extract_features: bool,
        accurate_model: Optional[bool] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        filter_features: Optional[dict] = None,
        runtime_parameters: Optional[RuntimeParameters] = None,
        metrics_calculation: Optional[bool] = False,
    ) -> SearchCustomization:
        # self.logger.info("Constructing search customization")
        search_customization = SearchCustomization(
            extractFeatures=extract_features,
            accurateModel=accurate_model,
            importanceThreshold=importance_threshold,
            maxFeatures=max_features,
            returnScores=return_scores,
            runtimeParameters=runtime_parameters,
            metricsCalculation=metrics_calculation,
        )
        if filter_features:
            if [
                key
                for key in filter_features
                if key not in {"min_importance", "max_psi", "max_count", "selected_features"}
            ]:
                raise ValidationError(self.bundle.get("dataset_invalid_filter"))
            feature_filter = FeaturesFilter(
                minImportance=filter_features.get("min_importance"),
                maxPSI=filter_features.get("max_psi"),
                maxCount=filter_features.get("max_count"),
                selectedFeatures=filter_features.get("selected_features"),
            )
            search_customization.featuresFilter = feature_filter

        search_customization.runtimeParameters.properties["etalon_imbalanced"] = self.imbalanced

        return search_customization

    def _rename_generate_features(self, runtime_parameters: Optional[RuntimeParameters]) -> Optional[RuntimeParameters]:
        if (
            runtime_parameters is not None
            and runtime_parameters.properties is not None
            and "generate_features" in runtime_parameters.properties
        ):
            generate_features = runtime_parameters.properties["generate_features"].split(",")
            renamed_generate_features = []
            for f in generate_features:
                for new_column, orig_column in self.columns_renaming.items():
                    if f == orig_column:
                        renamed_generate_features.append(new_column)
            runtime_parameters.properties["generate_features"] = ",".join(renamed_generate_features)

        return runtime_parameters

    def _clean_generate_features(self, runtime_parameters: Optional[RuntimeParameters]) -> Optional[RuntimeParameters]:
        if (
            runtime_parameters is not None
            and runtime_parameters.properties is not None
            and "generate_features" in runtime_parameters.properties
        ):
            del runtime_parameters.properties["generate_features"]

        return runtime_parameters

    def search(
        self,
        trace_id: str,
        progress_bar: Optional[ProgressBar],
        start_time: float,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
        return_scores: bool = False,
        extract_features: bool = False,
        accurate_model: bool = False,
        exclude_features_sources: Optional[List[str]] = None,
        importance_threshold: Optional[float] = None,  # deprecated
        max_features: Optional[int] = None,  # deprecated
        filter_features: Optional[dict] = None,  # deprecated
        runtime_parameters: Optional[RuntimeParameters] = None,
    ) -> SearchTask:
        if self.etalon_def is None:
            self.validate()
        file_metrics = FileMetrics()

        runtime_parameters = self._rename_generate_features(runtime_parameters)

        file_metadata = self.__construct_metadata(exclude_features_sources)
        search_customization = self.__construct_search_customization(
            return_scores=return_scores,
            extract_features=extract_features,
            accurate_model=accurate_model,
            importance_threshold=importance_threshold,
            max_features=max_features,
            filter_features=filter_features,
            runtime_parameters=runtime_parameters,
        )

        if self.file_upload_id is not None and self.rest_client.check_uploaded_file_v2(
            trace_id, self.file_upload_id, file_metadata
        ):
            search_task_response = self.rest_client.initial_search_without_upload_v2(
                trace_id, self.file_upload_id, file_metadata, file_metrics, search_customization
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                parquet_file_path = self.prepare_uploading_file(tmp_dir)
                time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
                time_left = time.time() - start_time
                search_progress = SearchProgress(2.0, ProgressStage.CREATING_FIT, time_left)
                if progress_bar is not None:
                    progress_bar.progress = search_progress.to_progress_bar()
                if progress_callback is not None:
                    progress_callback(search_progress)
                search_task_response = self.rest_client.initial_search_v2(
                    trace_id, parquet_file_path, file_metadata, file_metrics, search_customization
                )
                # if progress_bar is not None:
                #     progress_bar.progress = (6.0, self.bundle.get(ProgressStage.MATCHING.value))
                # if progress_callback is not None:
                #     progress_callback(SearchProgress(6.0, ProgressStage.MATCHING))
                self.file_upload_id = search_task_response.file_upload_id

        return SearchTask(
            search_task_response.search_task_id,
            self,
            return_scores,
            extract_features,
            accurate_model,
            task_type=self.task_type,
            rest_client=self.rest_client,
            logger=self.logger,
        )

    def validation(
        self,
        trace_id: str,
        initial_search_task_id: str,
        start_time: int,
        return_scores: bool = True,
        extract_features: bool = False,
        runtime_parameters: Optional[RuntimeParameters] = None,
        exclude_features_sources: Optional[List[str]] = None,
        metrics_calculation: bool = False,
        silent_mode: bool = False,
        progress_bar: Optional[ProgressBar] = None,
        progress_callback: Optional[Callable[[SearchProgress], Any]] = None,
    ) -> SearchTask:
        if self.etalon_def is None:
            self.validate(validate_target=False, silent_mode=silent_mode)
        file_metrics = FileMetrics()

        runtime_parameters = self._clean_generate_features(runtime_parameters)

        file_metadata = self.__construct_metadata(exclude_features_sources=exclude_features_sources)
        search_customization = self.__construct_search_customization(
            return_scores,
            extract_features,
            runtime_parameters=runtime_parameters,
            metrics_calculation=metrics_calculation,
        )
        seconds_left = time.time() - start_time
        search_progress = SearchProgress(1.0, ProgressStage.CREATING_TRANSFORM, seconds_left)
        if progress_bar is not None:
            progress_bar.progress = search_progress.to_progress_bar()
        if progress_callback is not None:
            progress_callback(search_progress)
        if self.file_upload_id is not None and self.rest_client.check_uploaded_file_v2(
            trace_id, self.file_upload_id, file_metadata
        ):
            search_task_response = self.rest_client.validation_search_without_upload_v2(
                trace_id, self.file_upload_id, initial_search_task_id, file_metadata, file_metrics, search_customization
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                parquet_file_path = self.prepare_uploading_file(tmp_dir)
                # To avoid rate limit
                time.sleep(1)

                search_task_response = self.rest_client.validation_search_v2(
                    trace_id,
                    parquet_file_path,
                    initial_search_task_id,
                    file_metadata,
                    file_metrics,
                    search_customization,
                )
                self.file_upload_id = search_task_response.file_upload_id
        # if progress_bar is not None:
        #     progress_bar.progress = (6.0, self.bundle.get(ProgressStage.ENRICHING.value))
        # if progress_callback is not None:
        #     progress_callback(SearchProgress(6.0, ProgressStage.ENRICHING))

        return SearchTask(
            search_task_response.search_task_id,
            self,
            return_scores,
            extract_features,
            initial_search_task_id=initial_search_task_id,
            rest_client=self.rest_client,
            logger=self.logger,
        )

    def prepare_uploading_file(self, base_path: str) -> str:
        parquet_file_path = f"{base_path}/{self.dataset_name}.parquet"
        self.data.to_parquet(path=parquet_file_path, index=False, compression="gzip", engine="fastparquet")
        uploading_file_size = Path(parquet_file_path).stat().st_size
        self.logger.info(f"Size of prepared uploading file: {uploading_file_size}. {len(self.data)} rows")
        if uploading_file_size > self.MAX_UPLOADING_FILE_SIZE:
            raise ValidationError(self.bundle.get("dataset_too_big_file"))
        return parquet_file_path
