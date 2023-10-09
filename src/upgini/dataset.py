import csv
import hashlib
import logging
import tempfile
import time
from ipaddress import IPv4Address, ip_address
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.common import is_period_dtype

from upgini.errors import ValidationError
from upgini.http import ProgressStage, SearchProgress, get_rest_client
from upgini.metadata import (
    EVAL_SET_INDEX,
    SYSTEM_COLUMNS,
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
from upgini.normalizer.phone_normalizer import PhoneNormalizer
from upgini.resource_bundle import bundle
from upgini.sampler.random_under_sampler import RandomUnderSampler
from upgini.search_task import SearchTask
from upgini.utils.email_utils import EmailSearchKeyConverter

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
    MIN_SAMPLE_THRESHOLD = 20_000
    IMBALANCE_THESHOLD = 0.4
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
        model_task_type: Optional[ModelTaskType] = None,
        random_state: Optional[int] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        client_ip: Optional[str] = None,
        warning_counter: Optional[WarningCounter] = None,
        **kwargs,
    ):
        if df is not None:
            data = df.copy()
        elif path is not None:
            if "sep" in kwargs:
                data = pd.read_csv(path, **kwargs)
            else:
                # try different separators: , ; \t ...
                with open(path, mode="r") as csvfile:
                    sep = csv.Sniffer().sniff(csvfile.read(2048)).delimiter
                kwargs["sep"] = sep
                data = pd.read_csv(path, **kwargs)
        else:
            raise ValueError(bundle.get("dataset_dataframe_or_path_empty"))
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, pd.io.parsers.TextFileReader):  # type: ignore
            raise ValueError(bundle.get("dataset_dataframe_iterator"))
        else:
            raise ValueError(bundle.get("dataset_dataframe_not_pandas"))

        self.dataset_name = dataset_name
        self.task_type = model_task_type
        self.description = description
        self.meaning_types = meaning_types
        self.search_keys = search_keys
        self.ignore_columns = []
        self.hierarchical_group_keys = []
        self.hierarchical_subgroup_keys = []
        self.file_upload_id: Optional[str] = None
        self.etalon_def: Optional[Dict[str, str]] = None
        self.endpoint = endpoint
        self.api_key = api_key
        self.random_state = random_state
        self.columns_renaming: Dict[str, str] = {}
        self.imbalanced: bool = False
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")
        self.client_ip = client_ip
        self.warning_counter = warning_counter or WarningCounter()

    def __len__(self):
        return len(self.data) if self.data is not None else None

    @property
    def columns(self):
        return self.data.columns if self.data is not None else None

    @property
    def meaning_types_checked(self) -> Dict[str, FileColumnMeaningType]:
        if self.meaning_types is None:
            raise ValueError(bundle.get("dataset_empty_meaning_types"))
        else:
            return self.meaning_types

    @property
    def search_keys_checked(self) -> List[Tuple[str, ...]]:
        if self.search_keys is None:
            raise ValueError(bundle.get("dataset_empty_search_keys"))
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
            raise ValidationError(bundle.get("dataset_too_few_rows").format(self.MIN_ROWS_COUNT))

    def __validate_max_row_count(self):
        if len(self.data) > self.MAX_ROWS:
            raise ValidationError(bundle.get("dataset_too_many_rows_registered").format(self.MAX_ROWS))

    def __rename_columns(self):
        # self.logger.info("Replace restricted symbols in column names")
        new_columns = []
        dup_counter = 0
        for column in self.data.columns:
            if column in [TARGET, EVAL_SET_INDEX, SYSTEM_RECORD_ID]:
                self.columns_renaming[column] = column
                new_columns.append(column)
                continue

            new_column = str(column)
            suffix = hashlib.sha256(new_column.encode()).hexdigest()[:6]
            if len(new_column) == 0:
                raise ValidationError(bundle.get("dataset_empty_column_names"))
            # db limit for column length
            if len(new_column) > 250:
                new_column = new_column[:250]

            # make column name unique relative to server features
            new_column = f"{new_column}_{suffix}"

            new_column = new_column.lower()

            # if column starts with non alphabetic symbol then add "a" to the beginning of string
            if ord(new_column[0]) not in range(ord("a"), ord("z") + 1):
                new_column = "a" + new_column

            # replace unsupported characters to "_"
            for idx, c in enumerate(new_column):
                if ord(c) not in range(ord("a"), ord("z") + 1) and ord(c) not in range(ord("0"), ord("9") + 1):
                    new_column = new_column[:idx] + "_" + new_column[idx + 1 :]

            if new_column in new_columns:
                new_column = f"{new_column}_{dup_counter}"
                dup_counter += 1
            new_columns.append(new_column)

            # self.data.columns.values[col_idx] = new_column
            # self.rename(columns={column: new_column}, inplace=True)
            self.meaning_types = {
                (new_column if key == str(column) else key): value for key, value in self.meaning_types_checked.items()
            }
            self.search_keys = [
                tuple(new_column if key == str(column) else key for key in keys) for keys in self.search_keys_checked
            ]
            self.columns_renaming[new_column] = str(column)
        self.data.columns = new_columns
        self.etalon_def = None

    def __validate_too_long_string_values(self):
        """Check that string values less than maximum characters for LLM"""
        # self.logger.info("Validate too long string values")
        for col in self.data.columns:
            if is_string_dtype(self.data[col]):
                max_length: int = self.data[col].astype("str").str.len().max()
                if max_length > self.MAX_STRING_FEATURE_LENGTH:
                    self.data[col] = self.data[col].astype("str").str.slice(stop=self.MAX_STRING_FEATURE_LENGTH)

    def __clean_duplicates(self, silent_mode: bool = False):
        """Clean DataSet from full duplicates."""
        # self.logger.info("Clean full duplicates")
        nrows = len(self.data)
        if nrows == 0:
            return
        # Remove absolute duplicates (exclude system_record_id)
        unique_columns = self.data.columns.tolist()
        unique_columns.remove(SYSTEM_RECORD_ID)
        self.logger.info(f"Dataset shape before clean duplicates: {self.data.shape}")
        self.data.drop_duplicates(subset=unique_columns, inplace=True)
        self.logger.info(f"Dataset shape after clean duplicates: {self.data.shape}")
        nrows_after_full_dedup = len(self.data)
        share_full_dedup = 100 * (1 - nrows_after_full_dedup / nrows)
        if share_full_dedup > 0:
            msg = bundle.get("dataset_full_duplicates").format(share_full_dedup)
            self.logger.warning(msg)
            # if not silent_mode:
            #     print(msg)
            # self.warning_counter.increment()
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        if target_column is not None:
            unique_columns.remove(target_column)
            marked_duplicates = self.data.duplicated(subset=unique_columns, keep=False)
            if marked_duplicates.sum() > 0:
                dups_indices = self.data[marked_duplicates].index.to_list()
                nrows_after_tgt_dedup = len(self.data.drop_duplicates(subset=unique_columns))
                num_dup_rows = nrows_after_full_dedup - nrows_after_tgt_dedup
                share_tgt_dedup = 100 * num_dup_rows / nrows_after_full_dedup

                msg = bundle.get("dataset_diff_target_duplicates").format(share_tgt_dedup, num_dup_rows, dups_indices)
                self.logger.warning(msg)
                if not silent_mode:
                    print(msg)
                self.data.drop_duplicates(subset=unique_columns, keep=False, inplace=True)
                self.logger.info(f"Dataset shape after clean invalid target duplicates: {self.data.shape}")

    def __convert_bools(self):
        """Convert bool columns True -> 1, False -> 0"""
        # self.logger.info("Converting bool to int")
        for col in self.data.columns:
            if is_bool(self.data[col]):
                self.data[col] = self.data[col].astype("Int64")

    def __convert_float16(self):
        """Convert float16 to float"""
        # self.logger.info("Converting float16 to float")
        for col in self.data.columns:
            if is_float_dtype(self.data[col]):
                self.data[col] = self.data[col].astype("float64")

    def __correct_decimal_comma(self):
        """Check DataSet for decimal commas and fix them"""
        # self.logger.info("Correct decimal commas")
        tmp = self.data.head(10)
        # all columns with sep="," will have dtype == 'object', i.e string
        # sep="." will be casted to numeric automatically
        cls_to_check = [i for i in tmp.columns if is_string_dtype(tmp[i])]
        for col in cls_to_check:
            if tmp[col].astype("string").str.match("^[0-9]+,[0-9]*$").any():
                self.data[col] = self.data[col].astype("string").str.replace(",", ".").astype(np.float64)

    @staticmethod
    def __ip_to_int(ip: Union[str, int, IPv4Address]) -> Optional[int]:
        try:
            ip_addr = ip_address(ip)
            if isinstance(ip_addr, IPv4Address):
                return int(ip_addr)
            else:
                return None
        except Exception:
            return None

    def __convert_ip(self):
        """Convert ip address to int"""
        ip = self.etalon_def_checked.get(FileColumnMeaningType.IP_ADDRESS.value)
        if ip is not None and ip in self.data.columns:
            # self.logger.info("Convert ip address to int")
            self.data[ip] = self.data[ip].apply(self.__ip_to_int).astype("Int64")
            if self.data[ip].isnull().all():
                raise ValidationError(bundle.get("invalid_ip").format(ip))

    def __normalize_iso_code(self):
        iso_code = self.etalon_def_checked.get(FileColumnMeaningType.COUNTRY.value)
        if iso_code is not None and iso_code in self.data.columns:
            # self.logger.info("Normalize iso code column")
            self.data[iso_code] = (
                self.data[iso_code]
                .astype("string")
                .str.upper()
                .str.replace(r"[^A-Z]", "", regex=True)
                .str.replace("UK", "GB", regex=False)
            )
            if (self.data[iso_code] == "").all():
                raise ValidationError(bundle.get("invalid_country").format(iso_code))

    def __normalize_postal_code(self):
        postal_code = self.etalon_def_checked.get(FileColumnMeaningType.POSTAL_CODE.value)
        if postal_code is not None and postal_code in self.data.columns:
            # self.logger.info("Normalize postal code")

            if is_string_dtype(self.data[postal_code]):
                try:
                    self.data[postal_code] = self.data[postal_code].astype("Float64").astype("Int64").astype("string")
                except Exception:
                    pass
            elif is_float_dtype(self.data[postal_code]):
                self.data[postal_code] = self.data[postal_code].astype("Int64").astype("string")

            self.data[postal_code] = (
                self.data[postal_code]
                .astype("string")
                .str.upper()
                .str.replace(r"[^0-9A-Z]", "", regex=True)  # remove non alphanumeric characters
                .str.replace(r"^0+\B", "", regex=True)  # remove leading zeros
            )
            if (self.data[postal_code] == "").all():
                raise ValidationError(bundle.get("invalid_postal_code").format(postal_code))

    def __normalize_hem(self):
        hem = self.etalon_def_checked.get(FileColumnMeaningType.HEM.value)
        if hem is not None and hem in self.data.columns:
            self.data[hem] = self.data[hem].str.lower()

    def __remove_old_dates(self, silent_mode: bool = False):
        date_column = self.etalon_def_checked.get(FileColumnMeaningType.DATE.value) or self.etalon_def_checked.get(
            FileColumnMeaningType.DATETIME.value
        )
        if date_column is not None and is_numeric_dtype(self.data[date_column]):
            old_subset = self.data[self.data[date_column] < self.MIN_SUPPORTED_DATE_TS]
            if len(old_subset) > 0:
                self.logger.info(f"df before dropping old rows: {self.data.shape}")
                self.data.drop(index=old_subset.index, inplace=True)  # type: ignore
                self.logger.info(f"df after dropping old rows: {self.data.shape}")
                if len(self.data) == 0:
                    raise ValidationError(bundle.get("dataset_all_dates_old"))
                else:
                    msg = bundle.get("dataset_drop_old_dates")
                    self.logger.warning(msg)
                    if not silent_mode:
                        print(msg)
                    self.warning_counter.increment()

    def __drop_ignore_columns(self):
        """Drop ignore columns"""
        columns_to_drop = list(set(self.data.columns) & set(self.ignore_columns))
        if len(columns_to_drop) > 0:
            # self.logger.info(f"Dropping ignore columns: {self.ignore_columns}")
            self.data.drop(columns_to_drop, axis=1, inplace=True)

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
                    raise ValidationError(bundle.get("dataset_invalid_target_type").format(target.dtype))
            target_classes_count = target.nunique()
            if target_classes_count != 2:
                msg = bundle.get("dataset_invalid_binary_target").format(target_classes_count)
                self.logger.warning(msg)
                raise ValidationError(msg)
        elif self.task_type == ModelTaskType.MULTICLASS:
            if not is_integer_dtype(target):
                try:
                    target = self.data[target_column].astype("category").cat.codes
                except Exception:
                    self.logger.exception("Failed to cast target to category codes for multiclass task type")
                    raise ValidationError(bundle.get("dataset_invalid_multiclass_target").format(target.dtype))
        elif self.task_type == ModelTaskType.REGRESSION:
            if not is_float_dtype(target):
                try:
                    self.data[target_column] = self.data[target_column].astype("float")
                except ValueError:
                    self.logger.exception("Failed to cast target to float for regression task type")
                    raise ValidationError(bundle.get("dataset_invalid_regression_target").format(target.dtype))
        elif self.task_type == ModelTaskType.TIMESERIES:
            if not is_float_dtype(target):
                try:
                    self.data[target_column] = self.data[target_column].astype("float")
                except ValueError:
                    self.logger.exception("Failed to cast target to float for timeseries task type")
                    raise ValidationError(bundle.get("dataset_invalid_timeseries_target").format(target.dtype))

    def __resample(self):
        # self.logger.info("Resampling etalon")
        # Resample imbalanced target. Only train segment (without eval_set)
        if EVAL_SET_INDEX in self.data.columns:
            train_segment = self.data[self.data[EVAL_SET_INDEX] == 0]
        else:
            train_segment = self.data

        if self.task_type == ModelTaskType.MULTICLASS or (
            self.task_type == ModelTaskType.BINARY and len(train_segment) > self.MIN_SAMPLE_THRESHOLD
        ):
            count = len(train_segment)
            min_class_count = count
            min_class_value = None
            target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, "")
            target = train_segment[target_column].copy()
            target_classes_count = target.nunique()

            if target_classes_count > self.MAX_MULTICLASS_CLASS_COUNT:
                msg = bundle.get("dataset_to_many_multiclass_targets").format(
                    target_classes_count, self.MAX_MULTICLASS_CLASS_COUNT
                )
                self.logger.warning(msg)
                raise ValidationError(msg)

            unique_target = target.unique()
            for v in list(unique_target):  # type: ignore
                current_class_count = len(train_segment.loc[target == v])
                if current_class_count < min_class_count:
                    min_class_count = current_class_count
                    min_class_value = v

            if min_class_count < self.MIN_TARGET_CLASS_ROWS:
                msg = bundle.get("dataset_rarest_class_less_min").format(
                    min_class_value, min_class_count, self.MIN_TARGET_CLASS_ROWS
                )
                self.logger.warning(msg)
                raise ValidationError(msg)

            min_class_percent = self.IMBALANCE_THESHOLD / target_classes_count
            min_class_threshold = min_class_percent * count

            if min_class_count < min_class_threshold:
                msg = bundle.get("dataset_rarest_class_less_threshold").format(
                    min_class_value, min_class_count, min_class_threshold, min_class_percent * 100
                )
                self.logger.warning(msg)
                print(msg)
                self.warning_counter.increment()

                train_segment = train_segment.copy().sort_values(by=SYSTEM_RECORD_ID)
                if self.task_type == ModelTaskType.MULTICLASS:
                    # Sort classes by rows count and find 25% quantile class
                    classes = target.value_counts().index
                    quantile25_idx = int(0.75 * len(classes))
                    quantile25_class = classes[quantile25_idx]
                    count_of_quantile25_class = len(target[target == quantile25_class])
                    msg = bundle.get("imbalance_multiclass").format(quantile25_class, count_of_quantile25_class)
                    self.logger.warning(msg)
                    print(msg)
                    # 25% and lower classes will stay as is. Higher classes will be downsampled
                    parts = []
                    for class_idx in range(quantile25_idx):
                        sampled = train_segment[train_segment[target_column] == classes[class_idx]].sample(
                            n=count_of_quantile25_class, random_state=self.random_state
                        )
                        parts.append(sampled)
                    for class_idx in range(quantile25_idx, len(classes)):
                        parts.append(train_segment[train_segment[target_column] == classes[class_idx]])
                    resampled_data = pd.concat(parts)
                elif self.task_type == ModelTaskType.BINARY and min_class_count < self.MIN_SAMPLE_THRESHOLD / 2:
                    minority_class = train_segment[train_segment[target_column] == min_class_value]
                    majority_class = train_segment[train_segment[target_column] != min_class_value]
                    sampled_majority_class = majority_class.sample(
                        n=self.MIN_SAMPLE_THRESHOLD - min_class_count, random_state=self.random_state
                    )
                    resampled_data = train_segment[
                        (train_segment[SYSTEM_RECORD_ID].isin(minority_class[SYSTEM_RECORD_ID]))
                        | (train_segment[SYSTEM_RECORD_ID].isin(sampled_majority_class[SYSTEM_RECORD_ID]))
                    ]
                else:
                    sampler = RandomUnderSampler(random_state=self.random_state)
                    X = train_segment[SYSTEM_RECORD_ID]
                    X = X.to_frame(SYSTEM_RECORD_ID)
                    new_x, _ = sampler.fit_resample(X, target)  # type: ignore
                    resampled_data = train_segment[train_segment[SYSTEM_RECORD_ID].isin(new_x[SYSTEM_RECORD_ID])]

                self.data = resampled_data
                self.logger.info(f"Shape after rebalance resampling: {self.data.shape}")
                self.imbalanced = True

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

    def __convert_phone(self):
        """Convert phone/msisdn to int"""
        # self.logger.info("Convert phone to int")
        msisdn_column = self.etalon_def_checked.get(FileColumnMeaningType.MSISDN.value)
        country_column = self.etalon_def_checked.get(FileColumnMeaningType.COUNTRY.value)
        if msisdn_column is not None and msisdn_column in self.data.columns:
            normalizer = PhoneNormalizer(self.data, msisdn_column, country_column)
            self.data[msisdn_column] = normalizer.normalize()
            if self.data[msisdn_column].isnull().all():
                raise ValidationError(f"All values of PHONE column `{msisdn_column}` are invalid")

    def __features(self):
        return [
            f for f, meaning_type in self.meaning_types_checked.items() if meaning_type == FileColumnMeaningType.FEATURE
        ]

    def __remove_dates_from_features(self, silent_mode: bool = False):
        # self.logger.info("Remove date columns from features")

        removed_features = []
        for f in self.__features():
            if is_datetime(self.data[f]) or is_period_dtype(self.data[f]):
                removed_features.append(f)
                self.data.drop(columns=f, inplace=True)
                del self.meaning_types_checked[f]

        if removed_features:
            msg = bundle.get("dataset_date_features").format(removed_features)
            self.logger.warning(msg)
            if not silent_mode:
                print(msg)
            self.warning_counter.increment()

    def __validate_features_count(self):
        if len(self.__features()) > self.MAX_FEATURES_COUNT:
            msg = bundle.get("dataset_too_many_features").format(self.MAX_FEATURES_COUNT)
            self.logger.warning(msg)
            raise ValidationError(msg)

    def __convert_features_types(self):
        # self.logger.info("Convert features to supported data types")

        for f in self.__features():
            if not is_numeric_dtype(self.data[f]):
                self.data[f] = self.data[f].astype("string")

    def __validate_dataset(self, validate_target: bool, silent_mode: bool):
        """Validate DataSet"""
        # self.logger.info("validating etalon")
        target = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        if validate_target:
            if target is None:
                raise ValidationError(bundle.get("dataset_missing_target"))

            target_value = self.__target_value()
            target_items = target_value.nunique()
            if target_items == 1:
                raise ValidationError(bundle.get("dataset_constant_target"))
            elif target_items == 0:
                raise ValidationError(bundle.get("dataset_empty_target"))

            # if self.task_type != ModelTaskType.MULTICLASS:
            #     self.data[target] = self.data[target].apply(pd.to_numeric, errors="coerce")

        keys_to_validate = [
            key
            for search_group in self.search_keys_checked
            for key in search_group
            if self.columns_renaming.get(key) != EmailSearchKeyConverter.EMAIL_ONE_DOMAIN_COLUMN_NAME
        ]
        mandatory_columns = [target]
        columns_to_validate = mandatory_columns.copy()
        columns_to_validate.extend(keys_to_validate)
        columns_to_validate = set([i for i in columns_to_validate if i is not None])

        nrows = len(self.data)
        validation_stats = {}
        self.data["valid_keys"] = 0
        self.data["valid_mandatory"] = True

        all_valid_status = bundle.get("validation_all_valid_status")
        some_invalid_status = bundle.get("validation_some_invalid_status")
        all_invalid_status = bundle.get("validation_all_invalid_status")
        all_valid_message = bundle.get("validation_all_valid_message")
        invalid_message = bundle.get("validation_invalid_message")

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
            name_header = bundle.get("validation_column_name_header")
            status_header = bundle.get("validation_status_header")
            description_header = bundle.get("validation_descr_header")
            df_stats.columns = [name_header, status_header, description_header]
            try:
                import html

                from IPython.display import HTML, display  # type: ignore

                _ = get_ipython()  # type: ignore

                text_color = bundle.get("validation_text_color")
                colormap = {
                    all_valid_status: bundle.get("validation_all_valid_color"),
                    some_invalid_status: bundle.get("validation_some_invalid_color"),
                    all_invalid_status: bundle.get("validation_all_invalid_color"),
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
                display(HTML(html_stats))
            except (ImportError, NameError):
                print(df_stats)

        if len(self.data) == 0:
            raise ValidationError(bundle.get("all_search_keys_invalid"))

    def __validate_meaning_types(self, validate_target: bool):
        # self.logger.info("Validating meaning types")
        if self.meaning_types is None or len(self.meaning_types) == 0:
            raise ValueError(bundle.get("dataset_missing_meaning_types"))

        if SYSTEM_RECORD_ID not in self.data.columns:
            raise ValueError("Internal error")

        for column in self.meaning_types:
            if column not in self.data.columns:
                raise ValueError(bundle.get("dataset_missing_meaning_column").format(column, self.data.columns))
        if validate_target and FileColumnMeaningType.TARGET not in self.meaning_types.values():
            raise ValueError(bundle.get("dataset_missing_target"))

    def __validate_search_keys(self):
        # self.logger.info("Validating search keys")
        if self.search_keys is None or len(self.search_keys) == 0:
            raise ValueError(bundle.get("dataset_missing_search_keys"))
        for keys_group in self.search_keys:
            for key in keys_group:
                if key not in self.data.columns:
                    showing_columns = set(self.data.columns) - SYSTEM_COLUMNS
                    raise ValidationError(bundle.get("dataset_missing_search_key_column").format(key, showing_columns))

    def validate(self, validate_target: bool = True, silent_mode: bool = False):
        # self.logger.info("Validating dataset")

        self.__validate_search_keys()

        self.__validate_meaning_types(validate_target=validate_target)

        self.__drop_ignore_columns()

        self.__rename_columns()

        self.__remove_dates_from_features(silent_mode)

        self.__validate_features_count()

        self.__validate_too_long_string_values()

        self.__convert_bools()

        self.__convert_float16()

        self.__correct_decimal_comma()

        self.__remove_old_dates(silent_mode)

        self.__convert_ip()

        self.__convert_phone()

        self.__normalize_iso_code()

        self.__normalize_postal_code()

        self.__normalize_hem()

        self.__convert_features_types()

        self.__clean_duplicates(silent_mode)

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
            if column_name not in self.ignore_columns:
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
                    FileColumnMeaningType.IP_ADDRESS,
                }:
                    min_max_values = NumericInterval(
                        minValue=self.data[column_name].astype("Int64").min(),
                        maxValue=self.data[column_name].astype("Int64").max(),
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

    def __get_data_type(self, pandas_data_type, column_name) -> DataType:
        if is_integer_dtype(pandas_data_type):
            return DataType.INT
        elif is_float_dtype(pandas_data_type):
            return DataType.DECIMAL
        elif is_string_dtype(pandas_data_type):
            return DataType.STRING
        else:
            msg = bundle.get("dataset_invalid_column_type").format(column_name, pandas_data_type)
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
                raise ValidationError(bundle.get("dataset_invalid_filter"))
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

        if self.file_upload_id is not None and get_rest_client(self.endpoint, self.api_key).check_uploaded_file_v2(
            trace_id, self.file_upload_id, file_metadata
        ):
            search_task_response = get_rest_client(self.endpoint, self.api_key).initial_search_without_upload_v2(
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
                search_task_response = get_rest_client(self.endpoint, self.api_key).initial_search_v2(
                    trace_id, parquet_file_path, file_metadata, file_metrics, search_customization
                )
                # if progress_bar is not None:
                #     progress_bar.progress = (6.0, bundle.get(ProgressStage.MATCHING.value))
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
            endpoint=self.endpoint,
            api_key=self.api_key,
            client_ip=self.client_ip,
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
        if self.file_upload_id is not None and get_rest_client(self.endpoint, self.api_key).check_uploaded_file_v2(
            trace_id, self.file_upload_id, file_metadata
        ):
            search_task_response = get_rest_client(self.endpoint, self.api_key).validation_search_without_upload_v2(
                trace_id, self.file_upload_id, initial_search_task_id, file_metadata, file_metrics, search_customization
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                parquet_file_path = self.prepare_uploading_file(tmp_dir)
                # To avoid rate limit
                time.sleep(1)

                search_task_response = get_rest_client(self.endpoint, self.api_key).validation_search_v2(
                    trace_id,
                    parquet_file_path,
                    initial_search_task_id,
                    file_metadata,
                    file_metrics,
                    search_customization,
                )
                self.file_upload_id = search_task_response.file_upload_id
        # if progress_bar is not None:
        #     progress_bar.progress = (6.0, bundle.get(ProgressStage.ENRICHING.value))
        # if progress_callback is not None:
        #     progress_callback(SearchProgress(6.0, ProgressStage.ENRICHING))

        return SearchTask(
            search_task_response.search_task_id,
            self,
            return_scores,
            extract_features,
            initial_search_task_id=initial_search_task_id,
            endpoint=self.endpoint,
            api_key=self.api_key,
            client_ip=self.client_ip,
        )

    def prepare_uploading_file(self, base_path: str) -> str:
        parquet_file_path = f"{base_path}/{self.dataset_name}.parquet"
        self.data.to_parquet(path=parquet_file_path, index=False, compression="gzip", engine="fastparquet")
        uploading_file_size = Path(parquet_file_path).stat().st_size
        self.logger.info(f"Size of prepared uploading file: {uploading_file_size}")
        if uploading_file_size > self.MAX_UPLOADING_FILE_SIZE:
            raise ValidationError(bundle.get("dataset_too_big_file"))
        return parquet_file_path
