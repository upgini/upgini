import csv
import logging
import os
import re
import tempfile
import time
from hashlib import sha256
from ipaddress import IPv4Address, ip_address
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from upgini.http import UPGINI_API_KEY, get_rest_client
from upgini.metadata import (
    COUNTRY,
    EVAL_SET_INDEX,
    SYSTEM_COLUMNS,
    SYSTEM_RECORD_ID,
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
from upgini.normalizer.phone_normalizer import phone_to_int
from upgini.sampler.random_under_sampler import RandomUnderSampler
from upgini.search_task import SearchTask


class Dataset(pd.DataFrame):
    MIN_ROWS_COUNT = 100
    MAX_ROWS_REGISTERED = 299_999
    MAX_ROWS_UNREGISTERED = 149_999
    FIT_SAMPLE_ROWS = 100_000
    FIT_SAMPLE_THRESHOLD = FIT_SAMPLE_ROWS * 3
    IMBALANCE_THESHOLD = 0.4
    MIN_TARGET_CLASS_ROWS = 100
    MAX_MULTICLASS_CLASS_COUNT = 100
    MIN_SUPPORTED_DATE_TS = 946684800000  # 2000-01-01
    MAX_FEATURES_COUNT = 1100
    MAX_UPLOADING_FILE_SIZE = 268435456  # 256 Mb
    EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")

    _metadata = [
        "dataset_name",
        "description",
        "meaning_types",
        "search_keys",
        "ignore_columns",
        "hierarchical_group_keys",
        "hierarchical_subgroup_keys",
        "date_format",
        "random_state",
        "task_type",
        "initial_data",
        "file_upload_id",
        "etalon_def",
        "endpoint",
        "api_key",
        "columns_renaming",
        "sampled",
    ]

    def __init__(
        self,
        dataset_name: str,
        description: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        path: Optional[str] = None,
        meaning_types: Optional[Dict[str, FileColumnMeaningType]] = None,
        search_keys: Optional[List[Tuple[str, ...]]] = None,
        model_task_type: Optional[ModelTaskType] = None,
        date_format: Optional[str] = None,
        random_state: Optional[int] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
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
            raise ValueError("DataFrame or path to file should be passed")
        if isinstance(data, pd.DataFrame):
            super(Dataset, self).__init__(data)  # type: ignore
        else:
            raise ValueError("Iteration is not supported. Remove `iterator` and `chunksize` arguments and try again")

        self.dataset_name = dataset_name
        self.task_type = model_task_type
        self.description = description
        self.meaning_types = meaning_types
        self.search_keys = search_keys
        self.ignore_columns = []
        self.hierarchical_group_keys = []
        self.hierarchical_subgroup_keys = []
        self.date_format = date_format
        self.initial_data = data.copy()
        self.file_upload_id: Optional[str] = None
        self.etalon_def: Optional[Dict[str, str]] = None
        self.endpoint = endpoint
        self.api_key = api_key
        self.random_state = random_state
        self.columns_renaming: Dict[str, str] = {}
        self.sampled: bool = False
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")

    @property
    def meaning_types_checked(self) -> Dict[str, FileColumnMeaningType]:
        if self.meaning_types is None:
            raise ValueError("meaning_types is empty")
        else:
            return self.meaning_types

    @property
    def search_keys_checked(self) -> List[Tuple[str, ...]]:
        if self.search_keys is None:
            raise ValueError("search_keys is empty")
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
        if self.shape[0] < self.MIN_ROWS_COUNT:
            raise ValueError(f"X should contain at least {self.MIN_ROWS_COUNT} valid distinct rows")

    def __validate_max_row_count(self):
        api_key = self.api_key or os.environ.get(UPGINI_API_KEY)
        is_registered = api_key is not None and api_key != ""
        if is_registered:
            if len(self) > self.MAX_ROWS_REGISTERED:
                raise ValueError(
                    f"Total X + eval_set rows count limit is {self.MAX_ROWS_REGISTERED}. "
                    "Please sample X and eval_set"
                )
        else:
            if len(self) > self.MAX_ROWS_UNREGISTERED:
                raise ValueError(
                    f"For unregistered users total rows count limit for X + eval_set is {self.MAX_ROWS_UNREGISTERED}. "
                    "Please register to increase the limit"
                )

    def __rename_columns(self):
        # self.logger.info("Replace restricted symbols in column names")
        for column in self.columns:
            if len(column) == 0:
                raise ValueError("Some of column names are empty. Add names and try again, please")
            new_column = str(column).lower()
            if ord(new_column[0]) not in range(ord("a"), ord("z") + 1):
                new_column = "a" + new_column
            for idx, c in enumerate(new_column):
                if ord(c) not in range(ord("a"), ord("z") + 1) and ord(c) not in range(ord("0"), ord("9") + 1):
                    new_column = new_column[:idx] + "_" + new_column[idx + 1 :]
            self.rename(columns={column: new_column}, inplace=True)
            self.meaning_types = {
                (new_column if key == str(column) else key): value for key, value in self.meaning_types_checked.items()
            }
            self.search_keys = [
                tuple(new_column if key == str(column) else key for key in keys) for keys in self.search_keys_checked
            ]
            self.columns_renaming[new_column] = str(column)

    def __validate_too_long_string_values(self):
        """Check that string values less than 400 characters"""
        # self.logger.info("Validate too long string values")
        for col in self.columns:
            if is_string_dtype(self[col]):
                max_length: int = self[col].astype("str").str.len().max()
                if max_length > 400:
                    raise ValueError(
                        f"Columns {col} are too long: {max_length} characters. "
                        "Remove this columns or trim length to 50 characters"
                    )

    def __clean_duplicates(self):
        """Clean DataSet from full duplicates."""
        # self.logger.info("Clean full duplicates")
        nrows = len(self)
        if nrows == 0:
            return
        # Remove absolute duplicates (exclude system_record_id)
        unique_columns = self.columns.tolist()
        unique_columns.remove(SYSTEM_RECORD_ID)
        self.logger.info(f"Dataset shape before clean duplicates: {self.shape}")
        self.drop_duplicates(subset=unique_columns, inplace=True)
        self.logger.info(f"Dataset shape after clean duplicates: {self.shape}")
        nrows_after_full_dedup = len(self)
        share_full_dedup = 100 * (1 - nrows_after_full_dedup / nrows)
        if share_full_dedup > 0:
            print(f"{share_full_dedup:.5f}% of the rows are fully duplicated")
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        if target_column is not None:
            unique_columns.remove(target_column)
            # unique_columns.remove(SYSTEM_RECORD_ID)
            self.drop_duplicates(subset=unique_columns, inplace=True)
            nrows_after_tgt_dedup = len(self)
            num_dup_rows = nrows_after_full_dedup - nrows_after_tgt_dedup
            share_tgt_dedup = 100 * num_dup_rows / nrows_after_full_dedup
            if nrows_after_tgt_dedup < nrows_after_full_dedup:
                msg = (
                    f"{share_tgt_dedup:.4f}% of rows ({num_dup_rows}) in X are duplicates with different y values. "
                    "Please check X dataframe"
                )
                self.logger.error(msg)
                raise ValueError(msg)

    def __convert_bools(self):
        """Convert bool columns True -> 1, False -> 0"""
        # self.logger.info("Converting bool to int")
        for col in self.columns:
            if is_bool(self[col]):
                self[col] = self[col].astype("Int64")

    def __convert_float16(self):
        """Convert float16 to float"""
        # self.logger.info("Converting float16 to float")
        for col in self.columns:
            if is_float_dtype(self[col]):
                self[col] = self[col].astype("float64")

    def __correct_decimal_comma(self):
        """Check DataSet for decimal commas and fix them"""
        # self.logger.info("Correct decimal commas")
        tmp = self.head(10)
        # all columns with sep="," will have dtype == 'object', i.e string
        # sep="." will be casted to numeric automatically
        cls_to_check = [i for i in tmp.columns if is_string_dtype(tmp[i])]
        for col in cls_to_check:
            if tmp[col].astype(str).str.match("^[0-9]+,[0-9]*$").any():
                self[col] = self[col].astype(str).str.replace(",", ".").astype(np.float64)

    def __to_millis(self):
        """Parse date column and transform it to millis"""
        date = self.etalon_def_checked.get(FileColumnMeaningType.DATE.value) or self.etalon_def_checked.get(
            FileColumnMeaningType.DATETIME.value
        )

        def intToOpt(i: int) -> Optional[int]:
            if i == -9223372036855:
                return None
            else:
                return i

        if date is not None and date in self.columns:
            # self.logger.info("Transform date column to millis")
            if is_string_dtype(self[date]):
                self[date] = (
                    pd.to_datetime(self[date], format=self.date_format).dt.floor("D").view(np.int64) // 1_000_000
                )
            elif is_datetime(self[date]):
                self[date] = self[date].dt.floor("D").view(np.int64) // 1_000_000
            elif is_period_dtype(self[date]):
                self[date] = pd.to_datetime(self[date].astype("string")).dt.floor("D").view(np.int64) // 1_000_000
            elif is_numeric_dtype(self[date]):
                msg = f"Unsupported type of date column {date}. Convert to datetime please."
                self.logger.error(msg)
                raise Exception(msg)

            self[date] = self[date].apply(lambda x: intToOpt(x)).astype("Int64")

    @staticmethod
    def __email_to_hem(email: str) -> Optional[str]:
        if email is None or not isinstance(email, str) or email == "":
            return None

        if not Dataset.EMAIL_REGEX.match(email):
            return None

        return sha256(email.lower().encode("utf-8")).hexdigest()

    def __hash_email(self):
        """Add column with HEM if email presented in search keys"""
        email = self.etalon_def_checked.get(FileColumnMeaningType.EMAIL.value)
        hem = self.etalon_def_checked.get(FileColumnMeaningType.HEM.value)
        if email is not None and email in self.columns:
            # self.logger.info("Hashing email")
            if hem is None:
                generated_hem_name = "generated_hem"
                self[generated_hem_name] = self[email].apply(self.__email_to_hem)
                self.meaning_types_checked[generated_hem_name] = FileColumnMeaningType.HEM
                self.etalon_def_checked[FileColumnMeaningType.HEM.value] = generated_hem_name

                self.search_keys = [
                    tuple(key if key != email else generated_hem_name for key in search_group)
                    for search_group in self.search_keys_checked
                ]

            self.meaning_types_checked.pop(email)
            del self.etalon_def_checked[FileColumnMeaningType.EMAIL.value]

            self["email_domain"] = self[email].str.split("@").str[1]
            self.drop(columns=email, inplace=True)

    @staticmethod
    def __ip_to_int(ip: Union[str, int, IPv4Address]) -> Optional[int]:
        try:
            return int(ip_address(ip))
        except Exception:
            return None

    def __convert_ip(self):
        """Convert ip address to int"""
        ip = self.etalon_def_checked.get(FileColumnMeaningType.IP_ADDRESS.value)
        if ip is not None and ip in self.columns:
            # self.logger.info("Convert ip address to int")
            self[ip] = self[ip].apply(self.__ip_to_int).astype("Int64")

    def __normalize_iso_code(self):
        iso_code = self.etalon_def_checked.get(FileColumnMeaningType.COUNTRY.value)
        if iso_code is not None and iso_code in self.columns:
            # self.logger.info("Normalize iso code column")
            self[iso_code] = (
                self[iso_code]
                .astype(str)
                .str.upper()
                .str.replace(r"[^A-Z]", "", regex=True)
                .str.replace("UK", "GB", regex=False)
            )

    def __normalize_postal_code(self):
        postal_code = self.etalon_def_checked.get(FileColumnMeaningType.POSTAL_CODE.value)
        if postal_code is not None and postal_code in self.columns:
            # self.logger.info("Normalize postal code")

            if is_float_dtype(self[postal_code]):
                self[postal_code] = self[postal_code].astype("Int64").astype(str)

            self[postal_code] = (
                self[postal_code]
                .astype(str)
                .str.upper()
                .replace(r"[^0-9A-Z]", "", regex=True)  # remove non alphanumeric characters
                .replace(r"^0+\B", "", regex=True)  # remove leading zeros
                .replace("NA", "")
            )

    def __remove_old_dates(self):
        date_column = self.etalon_def_checked.get(FileColumnMeaningType.DATE.value) or self.etalon_def_checked.get(
            FileColumnMeaningType.DATETIME.value
        )
        if date_column is not None:
            old_subset = self[self[date_column] < self.MIN_SUPPORTED_DATE_TS]
            if len(old_subset) > 0:
                self.logger.info(f"df before dropping old rows: {self.shape}")
                self.drop(index=old_subset.index, inplace=True)  # type: ignore
                self.logger.info(f"df after dropping old rows: {self.shape}")
                msg = "We don't have data before '2000-01-01' and removed all earlier records from the search dataset"
                self.logger.warning(msg)
                print("WARN: ", msg)
                if len(self) == 0:
                    raise Exception("There is empty train dataset after dropping old rows")

    def __drop_ignore_columns(self):
        """Drop ignore columns"""
        columns_to_drop = list(set(self.columns) & set(self.ignore_columns))
        if len(columns_to_drop) > 0:
            # self.logger.info(f"Dropping ignore columns: {self.ignore_columns}")
            self.drop(columns_to_drop, axis=1, inplace=True)

    def __target_value(self) -> pd.Series:
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, "")
        target: pd.Series = self[target_column]
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
        target = self[target_column]

        if self.task_type == ModelTaskType.BINARY:
            if not is_integer_dtype(target):
                try:
                    self[target_column] = self[target_column].astype("int")
                except ValueError:
                    self.logger.exception("Failed to cast target to integer for binary task type")
                    raise ValidationError(
                        f"Unexpected dtype of target for binary task type: {target.dtype}." " Expected int or bool"
                    )
            target_classes_count = target.nunique()
            if target_classes_count != 2:
                msg = f"Binary task type should contain only 2 target values, but {target_classes_count} presented"
                self.logger.error(msg)
                raise ValidationError(msg)
        elif self.task_type == ModelTaskType.MULTICLASS:
            if not is_integer_dtype(target) and not is_string_dtype(target):
                if is_numeric_dtype(target):
                    try:
                        self[target_column] = self[target_column].astype("int")
                    except ValueError:
                        self.logger.exception("Failed to cast target to integer for multiclass task type")
                        raise ValidationError(
                            f"Unexpected dtype of target for multiclass task type: {target.dtype}."
                            "Expected int or str"
                        )
                else:
                    msg = f"Unexpected dtype of target for multiclass task type: {target.dtype}. Expected int or str"
                    self.logger.exception(msg)
                    raise ValidationError(msg)
        elif self.task_type == ModelTaskType.REGRESSION:
            if not is_float_dtype(target):
                try:
                    self[target_column] = self[target_column].astype("float")
                except ValueError:
                    self.logger.exception("Failed to cast target to float for regression task type")
                    raise ValidationError(
                        f"Unexpected dtype of target for regression task type: {target.dtype}. Expected float"
                    )
        elif self.task_type == ModelTaskType.TIMESERIES:
            if not is_float_dtype(target):
                try:
                    self[target_column] = self[target_column].astype("float")
                except ValueError:
                    self.logger.exception("Failed to cast target to float for timeseries task type")
                    raise ValidationError(
                        f"Unexpected dtype of target for timeseries task type: {target.dtype}. Expected float"
                    )

    def __resample(self):
        # self.logger.info("Resampling etalon")
        # Resample imbalanced target. Only train segment (without eval_set)
        if self.task_type in [ModelTaskType.BINARY, ModelTaskType.MULTICLASS]:
            if EVAL_SET_INDEX in self.columns:
                train_segment = self[self[EVAL_SET_INDEX] == 0]
                validation_segment = self[self[EVAL_SET_INDEX] != 0]
            else:
                train_segment = self
                validation_segment = None

            count = len(train_segment)
            min_class_count = count
            min_class_value = None
            target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, "")
            target = train_segment[target_column]
            target_classes_count = target.nunique()

            if target_classes_count > self.MAX_MULTICLASS_CLASS_COUNT:
                msg = (
                    f"The number of target classes {target_classes_count} exceeds the allowed threshold: "
                    f"{self.MAX_MULTICLASS_CLASS_COUNT}. Please, correct your data and try again"
                )
                self.logger.error(msg)
                raise ValidationError(msg)

            unique_target = target.unique()
            for v in list(unique_target):  # type: ignore
                current_class_count = len(train_segment.loc[target == v])
                if current_class_count < min_class_count:
                    min_class_count = current_class_count
                    min_class_value = v

            if min_class_count < self.MIN_TARGET_CLASS_ROWS:
                msg = (
                    f"The rarest class `{min_class_value}` occurs {min_class_count}. "
                    "The minimum number of observations for each class in a train dataset must be "
                    f"grater than {self.MIN_TARGET_CLASS_ROWS}. Please, correct your data and try again"
                )
                self.logger.error(msg)
                raise ValidationError(msg)

            min_class_percent = self.IMBALANCE_THESHOLD / target_classes_count
            min_class_threshold = min_class_percent * count

            if min_class_count < min_class_threshold:
                self.logger.info(
                    f"Target is imbalanced. The rarest class `{min_class_value}` occurs {min_class_count} times. "
                    "The minimum number of observations for each class in a train dataset must be "
                    f"grater than or equal to {min_class_threshold} ({min_class_percent * 100} %). "
                    "It will be undersampled"
                )

                if is_string_dtype(target):
                    target_replacement = {v: i for i, v in enumerate(unique_target)}  # type: ignore
                    prepared_target = target.replace(target_replacement)
                else:
                    prepared_target = target

                sampler = RandomUnderSampler(random_state=self.random_state)
                X = train_segment[SYSTEM_RECORD_ID]
                X = X.to_frame(SYSTEM_RECORD_ID)
                new_x, _ = sampler.fit_resample(X, prepared_target)  # type: ignore
                resampled_data = train_segment[train_segment[SYSTEM_RECORD_ID].isin(new_x[SYSTEM_RECORD_ID])]
                if validation_segment is not None:
                    resampled_data = pd.concat([resampled_data, validation_segment], ignore_index=True)
                self._update_inplace(resampled_data)
                self.logger.info(f"Shape after rebalance resampling: {self.shape}")
                self.sampled = True

        # Resample over fit threshold
        if EVAL_SET_INDEX in self.columns:
            train_segment = self[self[EVAL_SET_INDEX] == 0]
            validation_segment = self[self[EVAL_SET_INDEX] != 0]
        else:
            train_segment = self
            validation_segment = None
        if len(train_segment) > self.FIT_SAMPLE_THRESHOLD:
            self.logger.info(
                f"Etalon has size {len(train_segment)} more than threshold {self.FIT_SAMPLE_THRESHOLD} "
                f"and will be downsampled to {self.FIT_SAMPLE_ROWS}"
            )
            resampled_data = train_segment.sample(n=self.FIT_SAMPLE_ROWS, random_state=self.random_state)
            if validation_segment is not None:
                resampled_data = pd.concat([resampled_data, validation_segment], ignore_index=True)  # type: ignore
            self._update_inplace(resampled_data)
            self.logger.info(f"Shape after threshold resampling: {self.shape}")
            self.sampled = True

    def __convert_phone(self):
        """Convert phone/msisdn to int"""
        # self.logger.info("Convert phone to int")
        msisdn_column = self.etalon_def_checked.get(FileColumnMeaningType.MSISDN.value)
        if msisdn_column is not None and msisdn_column in self.columns:
            # self.logger.info(f"going to apply phone_to_int for column {msisdn_column}")
            phone_to_int(self, msisdn_column)
            self[msisdn_column] = self[msisdn_column].astype("Int64")

    def __features(self):
        return [
            f for f, meaning_type in self.meaning_types_checked.items() if meaning_type == FileColumnMeaningType.FEATURE
        ]

    def __remove_dates_from_features(self):
        # self.logger.info("Remove date columns from features")

        removed_features = []
        for f in self.__features():
            if is_datetime(self[f]) or is_period_dtype(self[f]):
                removed_features.append(f)
                self.drop(columns=f, inplace=True)
                del self.meaning_types_checked[f]

        if removed_features:
            msg = (
                f"Columns {removed_features} is a datetime or period type "
                "but not used as a search key and has been droped from X"
            )
            print(msg)
            self.logger.warning(msg)

    def __remove_empty_and_constant_features(self):
        # self.logger.info("Remove almost constant and almost empty columns")
        removed_features = []
        for f in self.__features():
            value_counts = self[f].value_counts(dropna=False, normalize=True)
            # most_frequent_value = value_counts.index[0]
            most_frequent_percent = value_counts.iloc[0]
            if most_frequent_percent >= 0.99:
                removed_features.append(f)
                self.drop(columns=f, inplace=True)
                del self.meaning_types_checked[f]

        if removed_features:
            msg = f"Columns {removed_features} has value with frequency more than 99% and has been droped from X"
            print(msg)
            self.logger.warning(msg)

    def __remove_high_cardinality_features(self):
        # self.logger.info("Remove columns with high cardinality")

        count = len(self)
        removed_features = []
        for f in self.__features():
            if (is_string_dtype(self[f]) or is_integer_dtype(self[f])) and self[f].nunique() / count >= 0.9:
                removed_features.append(f)
                self.drop(columns=f, inplace=True)
                del self.meaning_types_checked[f]
        if removed_features:
            msg = f"Columns {removed_features} has high cardinality (>90% unique values) " "and has been droped from X"
            print(msg)
            self.logger.warning(msg)

    def __validate_features_count(self):
        if len(self.__features()) > self.MAX_FEATURES_COUNT:
            msg = f"Maximum count of features is {self.MAX_FEATURES_COUNT}"
            self.logger.error(msg)
            raise Exception(msg)

    def __convert_features_types(self):
        # self.logger.info("Convert features to supported data types")

        for f in self.__features():
            if self[f].dtype == object:
                self[f] = self[f].astype(str)
            elif not is_numeric_dtype(self[f].dtype):
                self[f] = self[f].astype(str)

    def __validate_dataset(self, validate_target: bool, silent_mode: bool):
        """Validate DataSet"""
        # self.logger.info("validating etalon")
        date_millis = self.etalon_def_checked.get(FileColumnMeaningType.DATE.value) or self.etalon_def_checked.get(
            FileColumnMeaningType.DATETIME.value
        )
        target = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        score = self.etalon_def_checked.get(FileColumnMeaningType.SCORE.value)
        if validate_target:
            if target is None:
                raise ValidationError("Target column is absent in meaning_types")

            target_value = self.__target_value()
            target_items = target_value.nunique()
            if target_items == 1:
                raise ValidationError("Target contains only one distinct value")
            elif target_items == 0:
                raise ValidationError("Target contains only NaN or incorrect values.")

            if self.task_type != ModelTaskType.MULTICLASS:
                self[target] = self[target].apply(pd.to_numeric, errors="coerce")

        keys_to_validate = [key for search_group in self.search_keys_checked for key in search_group if key != COUNTRY]
        mandatory_columns = [date_millis, target, score]
        columns_to_validate = mandatory_columns.copy()
        columns_to_validate.extend(keys_to_validate)
        columns_to_validate = set([i for i in columns_to_validate if i is not None])

        nrows = len(self)
        validation_stats = {}
        self["valid_keys"] = 0
        self["valid_mandatory"] = True
        for col in columns_to_validate:
            self[f"{col}_is_valid"] = ~self[col].isnull()
            if validate_target and target is not None and col == target:
                self.loc[self[target] == np.Inf, f"{col}_is_valid"] = False

            if col in mandatory_columns:
                self["valid_mandatory"] = self["valid_mandatory"] & self[f"{col}_is_valid"]

            invalid_values = list(self.loc[self[f"{col}_is_valid"] == 0, col].head().values)  # type: ignore
            valid_share = self[f"{col}_is_valid"].sum() / nrows
            validation_stats[col] = {}
            optional_drop_message = "Invalid rows will be dropped. " if col in mandatory_columns else ""
            if valid_share == 1:
                valid_status = "All valid"
                valid_message = "All values in this column are good to go"
            elif 0 < valid_share < 1:
                valid_status = "Some invalid"
                valid_message = (
                    f"{100 * (1 - valid_share):.5f}% of the values of this column failed validation "
                    f"{optional_drop_message}"
                    f"Some examples of invalid values: {invalid_values}"
                )
            else:
                valid_status = "All invalid"
                valid_message = (
                    f"{100 * (1 - valid_share):.5f}% of the values of this column failed validation "
                    f"{optional_drop_message}"
                    f"Some examples of invalid values: {invalid_values}"
                )
            validation_stats[col]["valid_status"] = valid_status
            validation_stats[col]["valid_message"] = valid_message

            if col in keys_to_validate:
                self["valid_keys"] = self["valid_keys"] + self[f"{col}_is_valid"]
            self.drop(columns=f"{col}_is_valid", inplace=True)

        self["is_valid"] = self["valid_keys"] > 0
        self["is_valid"] = self["is_valid"] & self["valid_mandatory"]
        self.drop(columns=["valid_keys", "valid_mandatory"], inplace=True)

        drop_idx = self[self["is_valid"] != 1].index  # type: ignore
        self.drop(index=drop_idx, inplace=True)  # type: ignore
        self.drop(columns=["is_valid"], inplace=True)

        if not silent_mode:
            df_stats = pd.DataFrame.from_dict(validation_stats, orient="index")
            df_stats.reset_index(inplace=True)
            df_stats.columns = ["Column name", "Status", "Description"]  # type: ignore
            try:
                import html

                from IPython.display import HTML, display  # type: ignore

                def map_color(text):
                    colormap = {"All valid": "#DAF7A6", "Some invalid": "#FFC300", "All invalid": "#FF5733"}
                    return (
                        f"<td style='background-color:{colormap[text]};color:black'>{text}</td>"
                        if text in colormap
                        else f"<td>{text}</td>"
                    )

                df_stats["Description"] = df_stats["Description"].apply(lambda x: html.escape(x))
                html_stats = (
                    "<table>"
                    + "<tr>"
                    + "".join(f"<th style='font-weight:bold'>{col}</th>" for col in df_stats.columns)
                    + "</tr>"
                    + "".join("<tr>" + "".join(map(map_color, row[1:])) + "</tr>" for row in df_stats.itertuples())
                    + "</table>"
                )
                display(HTML(html_stats))
            except ImportError:
                print(df_stats)

    def __validate_meaning_types(self, validate_target: bool):
        # self.logger.info("Validating meaning types")
        if self.meaning_types is None or len(self.meaning_types) == 0:
            raise ValueError("Please pass the `meaning_types` argument before validation")

        if SYSTEM_RECORD_ID not in self.columns:
            self[SYSTEM_RECORD_ID] = self.apply(lambda row: hash(tuple(row)), axis=1)
            self.meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID

        for column in self.meaning_types:
            if column not in self.columns:
                raise ValueError(f"Meaning column `{column}` doesn't exist in dataframe columns: {self.columns}")
        if validate_target and FileColumnMeaningType.TARGET not in self.meaning_types.values():
            raise ValueError("Target column is not presented in meaning types. Specify it, please")

    def __validate_search_keys(self):
        # self.logger.info("Validating search keys")
        if self.search_keys is None or len(self.search_keys) == 0:
            raise ValueError("Please pass `search_keys` argument before validation")
        for keys_group in self.search_keys:
            for key in keys_group:
                if key not in self.columns:
                    showing_columns = set(self.columns) - SYSTEM_COLUMNS
                    raise ValueError(f"Search key `{key}` doesn't exist in dataframe columns: {showing_columns}")

    def validate(self, validate_target: bool = True, silent_mode: bool = False):
        # self.logger.info("Validating dataset")

        self.__validate_search_keys()

        self.__validate_meaning_types(validate_target=validate_target)

        self.__rename_columns()

        self.__drop_ignore_columns()

        self.__remove_dates_from_features()

        self.__remove_empty_and_constant_features()

        self.__remove_high_cardinality_features()

        self.__validate_features_count()

        self.__validate_too_long_string_values()

        self.__clean_duplicates()

        self.__convert_bools()

        self.__convert_float16()

        self.__correct_decimal_comma()

        self.__to_millis()

        self.__remove_old_dates()

        self.__hash_email()

        self.__convert_ip()

        self.__convert_phone()

        self.__normalize_iso_code()

        self.__normalize_postal_code()

        self.__convert_features_types()

        self.__validate_dataset(validate_target, silent_mode)

        if validate_target:
            self.__validate_target()

            self.__resample()

            self.__validate_min_rows_count()

        self.__validate_max_row_count()

    def __construct_metadata(self) -> FileMetadata:
        # self.logger.info("Constructing dataset metadata")
        columns = []
        for index, (column_name, column_type) in enumerate(zip(self.columns, self.dtypes)):
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
                        minValue=self[column_name].astype("Int64").min(),
                        maxValue=self[column_name].astype("Int64").max(),
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
            msg = f"Unsupported data type of column {column_name}: {pandas_data_type}"
            self.logger.error(msg)
            raise Exception(msg)

    def __construct_search_customization(
        self,
        return_scores: bool,
        extract_features: bool,
        accurate_model: Optional[bool] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        filter_features: Optional[dict] = None,
        runtime_parameters: Optional[RuntimeParameters] = None,
    ) -> SearchCustomization:
        # self.logger.info("Constructing search customization")
        search_customization = SearchCustomization(
            extractFeatures=extract_features,
            accurateModel=accurate_model,
            importanceThreshold=importance_threshold,
            maxFeatures=max_features,
            returnScores=return_scores,
            runtimeParameters=runtime_parameters,
        )
        if filter_features:
            if [
                key
                for key in filter_features
                if key not in {"min_importance", "max_psi", "max_count", "selected_features"}
            ]:
                raise ValueError(
                    "Unknown field in filter_features. "
                    "Should be {'min_importance', 'max_psi', 'max_count', 'selected_features'}."
                )
            feature_filter = FeaturesFilter(
                minImportance=filter_features.get("min_importance"),
                maxPSI=filter_features.get("max_psi"),
                maxCount=filter_features.get("max_count"),
                selectedFeatures=filter_features.get("selected_features"),
            )
            search_customization.featuresFilter = feature_filter

        return search_customization

    def search(
        self,
        trace_id: str,
        return_scores: bool = False,
        extract_features: bool = False,
        accurate_model: bool = False,
        importance_threshold: Optional[float] = None,  # deprecated
        max_features: Optional[int] = None,  # deprecated
        filter_features: Optional[dict] = None,  # deprecated
        runtime_parameters: Optional[RuntimeParameters] = None,
    ) -> SearchTask:
        if self.etalon_def is None:
            self.validate()
        file_metrics = FileMetrics()

        file_metadata = self.__construct_metadata()
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
                search_task_response = get_rest_client(self.endpoint, self.api_key).initial_search_v2(
                    trace_id, parquet_file_path, file_metadata, file_metrics, search_customization
                )
                self.file_upload_id = search_task_response.file_upload_id

        search_task = SearchTask(
            search_task_response.search_task_id,
            self,
            return_scores,
            extract_features,
            accurate_model,
            task_type=self.task_type,
            endpoint=self.endpoint,
            api_key=self.api_key,
        )
        return search_task.poll_result(trace_id)

    def validation(
        self,
        trace_id: str,
        initial_search_task_id: str,
        return_scores: bool = True,
        extract_features: bool = False,
        runtime_parameters: Optional[RuntimeParameters] = None,
        silent_mode: bool = False,
    ) -> SearchTask:
        if self.etalon_def is None:
            self.validate(validate_target=False, silent_mode=silent_mode)
        file_metrics = FileMetrics()

        file_metadata = self.__construct_metadata()
        search_customization = self.__construct_search_customization(
            return_scores, extract_features, runtime_parameters=runtime_parameters
        )

        if self.file_upload_id is not None and get_rest_client(self.endpoint, self.api_key).check_uploaded_file_v2(
            trace_id, self.file_upload_id, file_metadata
        ):
            search_task_response = get_rest_client(self.endpoint, self.api_key).validation_search_without_upload_v2(
                trace_id, self.file_upload_id, initial_search_task_id, file_metadata, file_metrics, search_customization
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                parquet_file_path = self.prepare_uploading_file(tmp_dir)
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

        search_task = SearchTask(
            search_task_response.search_task_id,
            self,
            return_scores,
            extract_features,
            initial_search_task_id=initial_search_task_id,
            endpoint=self.endpoint,
            api_key=self.api_key,
        )

        return search_task.poll_result(trace_id, quiet=silent_mode)

    def prepare_uploading_file(self, base_path: str) -> str:
        parquet_file_path = f"{base_path}/{self.dataset_name}.parquet"
        self.to_parquet(path=parquet_file_path, index=False, compression="gzip", engine="fastparquet")
        uploading_file_size = Path(parquet_file_path).stat().st_size
        self.logger.info(f"Size of prepared uploading file: {uploading_file_size}")
        if uploading_file_size > self.MAX_UPLOADING_FILE_SIZE:
            raise Exception("Dataset size is too big. Please try to reduce rows or columns count")
        return parquet_file_path
