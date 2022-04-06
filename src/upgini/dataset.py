import csv
import html
import logging
import tempfile
from hashlib import sha256
from ipaddress import IPv4Address, ip_address
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype
from pandas.api.types import is_string_dtype as is_string
from pandas.core.dtypes.common import is_period_dtype

from upgini.http import get_rest_client
from upgini.metadata import (
    SYSTEM_FAKE_DATE,
    SYSTEM_RECORD_ID,
    BinaryTask,
    DataType,
    FeaturesFilter,
    FileColumnMeaningType,
    FileColumnMetadata,
    FileMetadata,
    FileMetrics,
    ModelLabelType,
    ModelTaskType,
    MulticlassTask,
    NumericInterval,
    RegressionTask,
    RuntimeParameters,
    SearchCustomization,
)
from upgini.normalizer.phone_normalizer import phone_to_int
from upgini.search_task import SearchTask


class Dataset(pd.DataFrame):
    MIN_ROWS_COUNT: int = 100

    name: str
    description: Optional[str]
    meaning_types: Optional[Dict[str, FileColumnMeaningType]]
    search_keys: Optional[List[Tuple[str, ...]]]
    ignore_columns: List[str]
    hierarchical_group_keys: List[Tuple[str, ...]]
    hierarchical_subgroup_keys: List[Tuple[str, ...]]
    date_format: Optional[str]
    random_state: Optional[int]
    task_type: Optional[ModelTaskType]
    initial_data: pd.DataFrame
    file_upload_id: Optional[str]
    etalon_def: Optional[Dict[str, str]]
    columns_renaming: Dict[str, str] = {}

    _metadata = [
        "name",
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
    ]

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        path: Optional[str] = None,
        meaning_types: Optional[Dict[str, FileColumnMeaningType]] = None,
        search_keys: Optional[List[Tuple[str, ...]]] = None,
        date_format: Optional[str] = None,
        random_state: Optional[int] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
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
            raise ValueError("DataFrame or path to file should be passed.")
        if isinstance(data, pd.DataFrame):
            super(Dataset, self).__init__(data)
        else:
            raise ValueError("Iteration is not supported. Remove `iterator` and `chunksize` arguments and try again.")

        self.name = name
        self.description = description
        self.meaning_types = meaning_types
        self.search_keys = search_keys
        self.ignore_columns = []
        self.hierarchical_group_keys = []
        self.hierarchical_subgroup_keys = []
        self.date_format = date_format
        self.task_type = None
        self.initial_data = data.copy()
        self.file_upload_id = None
        self.etalon_def = None
        self.endpoint = endpoint
        self.api_key = api_key
        self.random_state = random_state

    @property
    def meaning_types_checked(self) -> Dict[str, FileColumnMeaningType]:
        if self.meaning_types is None:
            raise ValueError("meaning_types is empty.")
        else:
            return self.meaning_types

    @property
    def search_keys_checked(self) -> List[Tuple[str, ...]]:
        if self.search_keys is None:
            raise ValueError("search_keys is empty.")
        else:
            return self.search_keys

    @property
    def etalon_def_checked(self) -> Dict[str, str]:
        if self.etalon_def is None:
            self.etalon_def = {
                v.value: k for k, v in self.meaning_types_checked.items() if v != FileColumnMeaningType.FEATURE
            }

        return self.etalon_def

    def __validate_rows_count(self):
        logging.debug("Validate rows count")
        if self.shape[0] < self.MIN_ROWS_COUNT:
            raise ValueError(f"X should contain at least {self.MIN_ROWS_COUNT} valid distinct rows.")

    def __rename_columns(self):
        logging.debug("Replace restricted symbols in column names")
        for column in self.columns:
            if len(column) == 0:
                raise ValueError("Some of column names are empty. Fill them and try again, please.")
            new_column = str(column).lower()
            if ord(new_column[0]) not in range(ord("a"), ord("z")):
                new_column = "a" + new_column
            for idx, c in enumerate(new_column):
                if ord(c) not in range(ord("a"), ord("z")) and ord(c) not in range(ord("0"), ord("9")):
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
        logging.debug("Validate too long string values")
        for col in self.columns:
            if is_string(self[col]):
                max_length: int = self[col].astype("str").str.len().max()
                if max_length > 400:
                    raise ValueError(
                        f"Some of column {col} values are too long: {max_length} characters. "
                        "Remove this column or trim values to 50 characters."
                    )

    def __clean_duplicates(self):
        """Clean DataSet from full duplicates."""
        nrows = len(self)
        logging.debug("cleaning duplicates")
        unique_columns = self.columns.tolist()
        logging.info(f"Input data shape: {self.shape}")
        self.drop_duplicates(subset=unique_columns, inplace=True)
        logging.info(f"Data without duplicates: {self.shape}")
        nrows_after_full_dedup = len(self)
        share_full_dedup = 100 * (1 - nrows_after_full_dedup / nrows)
        if share_full_dedup > 0:
            print(f"{share_full_dedup:.5f}% of the rows are fully duplicated")
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        if target_column is not None:
            unique_columns.remove(target_column)
            unique_columns.remove(SYSTEM_RECORD_ID)
            self.drop_duplicates(subset=unique_columns, inplace=True)
            nrows_after_tgt_dedup = len(self)
            share_tgt_dedup = 100 * (1 - nrows_after_tgt_dedup / nrows_after_full_dedup)
            if nrows_after_tgt_dedup < nrows_after_full_dedup:
                raise ValueError(
                    f"{share_tgt_dedup:.5f}% of rows in X are duplicates with different y values. "
                    "Please check the dataframe and restart fit"
                )

    def __convert_bools(self):
        """Convert bool columns True -> 1, False -> 0"""
        logging.debug("Converting bool to int")
        for col in self.columns:
            if is_bool(self[col]):
                logging.debug(f"Converting {col} to int")
                self[col] = self[col].astype("Int64")

    def __convert_float16(self):
        """Convert float16 to float"""
        logging.debug("Converting float16 to float")
        for col in self.columns:
            if is_float_dtype(self[col]):
                self[col] = self[col].astype("float64")

    def __correct_decimal_comma(self):
        """Check DataSet for decimal commas and fix them"""
        logging.debug("Correct decimal commas")
        tmp = self.head(10)
        # all columns with sep="," will have dtype == 'object', i.e string
        # sep="." will be casted to numeric automatically
        cls_to_check = [i for i in tmp.columns if is_string(tmp[i])]
        for col in cls_to_check:
            if tmp[col].astype(str).str.match("^[0-9]+,[0-9]*$").any():
                logging.info(f"Correcting comma sep in {col}")
                self[col] = self[col].astype(str).str.replace(",", ".").astype(np.float64)

    def __to_millis(self):
        """Parse date column and transform it to millis"""
        logging.debug("Transform date column to millis")
        date = self.etalon_def_checked.get(FileColumnMeaningType.DATE.value) or self.etalon_def_checked.get(
            FileColumnMeaningType.DATETIME.value
        )

        def intToOpt(i: int) -> Optional[int]:
            if i == -9223372036855:
                return None
            else:
                return i

        if date is not None and date in self.columns:
            if is_string(self[date]):
                self[date] = pd.to_datetime(self[date], format=self.date_format).values.astype(np.int64) // 1_000_000
            elif is_datetime(self[date]):
                self[date] = self[date].view(np.int64) // 1_000_000
            elif is_period_dtype(self[date]):
                self[date] = (
                    pd.to_datetime(self[date].astype("string"), format=self.date_format).values.astype(np.int64)
                    // 1_000_000
                )
            self[date] = self[date].apply(lambda x: intToOpt(x)).astype("Int64")

    @staticmethod
    def __email_to_hem(email: str) -> Optional[str]:
        if email is None or not isinstance(email, str) or email == "":
            return None
        else:
            return sha256(email.lower().encode("utf-8")).hexdigest()

    def __hash_email(self):
        """Add column with HEM if email presented in search keys"""
        logging.debug("Hashing email")
        email = self.etalon_def_checked.get(FileColumnMeaningType.EMAIL.value)
        if email is not None and email in self.columns:
            generated_hem_name = "generated_hem"
            self[generated_hem_name] = self[email].apply(self.__email_to_hem)
            self.meaning_types_checked[generated_hem_name] = FileColumnMeaningType.HEM
            self.meaning_types_checked.pop(email)
            self.etalon_def_checked[FileColumnMeaningType.HEM.value] = generated_hem_name
            del self.etalon_def_checked[FileColumnMeaningType.EMAIL.value]
            self.search_keys = [
                tuple(key if key != email else generated_hem_name for key in search_group)
                for search_group in self.search_keys_checked
            ]
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
        logging.debug("Convert ip address to int")
        ip = self.etalon_def_checked.get(FileColumnMeaningType.IP_ADDRESS.value)
        if ip is not None and ip in self.columns:
            self[ip] = self[ip].apply(self.__ip_to_int).astype("Int64")

    def __remove_empty_date_rows(self):
        """Clean DataSet from empty date rows"""
        logging.debug("cleaning empty rows")
        date_column = self.etalon_def_checked.get(FileColumnMeaningType.DATE.value) or self.etalon_def_checked.get(
            FileColumnMeaningType.DATETIME.value
        )
        if date_column is not None:
            drop_idx = self[(self[date_column] == "") | self[date_column].isna()].index
            self.drop(drop_idx, inplace=True)
            logging.info(f"df with valid date column: {self.shape}")

    def __drop_ignore_columns(self):
        """Drop ignore columns"""
        logging.debug("Dropping ignore columns")
        columns_to_drop = list(set(self.columns) & set(self.ignore_columns))
        self.drop(columns_to_drop, axis=1, inplace=True)

    def __target_value(self) -> pd.Series:
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, "")
        target: pd.Series = self[target_column]
        # clean target from nulls
        target.dropna(inplace=True)
        if is_numeric_dtype(target):
            target = target.loc[np.isfinite(target)]
        else:
            target = target.loc[target != ""]

        return target

    def __define_task(self) -> ModelTaskType:
        logging.debug("defining task")
        target = self.__target_value()
        target_items = target.nunique()
        target_ratio = target_items / len(target)
        if (target_items > 50 or (target_items > 2 and target_ratio > 0.2)) and is_numeric_dtype(target):
            task = ModelTaskType.REGRESSION
        elif target_items == 2:
            if is_numeric_dtype(target):
                task = ModelTaskType.BINARY
            else:
                raise ValueError("Binary target should be numerical.")
        else:
            task = ModelTaskType.MULTICLASS
        print(f"Detected task type: {task}")
        return task

    def __validate_target(self):
        target = self.__target_value()
        target_column = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value, "")

        def imbalance_check(target_classes_count: int):
            count = len(self)
            min_class_count = count
            min_class_value = None
            for v in target.unique():
                current_class_count = len(self.loc[target == v])
                if current_class_count < min_class_count:
                    min_class_count = current_class_count
                    min_class_value = v
            if min_class_count < (2 / (5 * target_classes_count) * count):
                if is_string(target):
                    logging.warning(
                        f"Imbalanced target min class `{min_class_value}` count: {min_class_count}. "
                        "But target is string and not supported by RandomUnderSampler"
                    )
                else:
                    rus = RandomUnderSampler(random_state=self.random_state)
                    new_x, new_y = rus.fit_resample(self.drop(columns=target_column), self[target_column])
                    self._update_inplace(pd.merge(new_x, new_y, left_index=True, right_index=True))

        if self.task_type == ModelTaskType.BINARY:
            if not is_integer_dtype(target):
                try:
                    self[target_column] = self[target_column].astype("int")
                except ValueError:
                    logging.exception("Failed to cast target to integer for binary task type")
                    raise Exception(
                        f"Unexpected dtype of target for binary task type: {target.dtype}." " Expected int or bool"
                    )
            target_classes_count = target.nunique()
            if target_classes_count != 2:
                msg = f"Binary task type should contain only 2 target values, but {target_classes_count} presented"
                logging.error(msg)
                raise Exception(msg)
            imbalance_check(target_classes_count)
        elif self.task_type == ModelTaskType.MULTICLASS:
            if not is_integer_dtype(target) and not is_string(target):
                if is_numeric_dtype(target):
                    try:
                        self[target_column] = self[target_column].astype("int")
                    except ValueError:
                        logging.exception("Failed to cast target to integer for multiclass task type")
                        raise Exception(
                            f"Unexpected dtype of target for multiclass task type: {target.dtype}."
                            "Expected int or str"
                        )
                else:
                    msg = f"Unexpected dtype of target for multiclass task type: {target.dtype}. Expected int or str"
                    logging.exception(msg)
                    raise Exception(msg)
            imbalance_check(target.nunique())
        elif self.task_type == ModelTaskType.REGRESSION:
            if not is_float_dtype(target):
                try:
                    self[target_column] = self[target_column].astype("float")
                except ValueError:
                    logging.exception("Failed to cast target to float for regression task type")
                    raise Exception(
                        f"Unexpected dtype of target for regression task type: {target.dtype}. Expected float"
                    )
        elif self.task_type == ModelTaskType.TIMESERIES:
            if not is_float_dtype(target):
                try:
                    self[target_column] = self[target_column].astype("float")
                except ValueError:
                    logging.exception("Failed to cast target to float for timeseries task type")
                    raise Exception(
                        f"Unexpected dtype of target for timeseries task type: {target.dtype}. Expected float"
                    )

    def __convert_phone(self):
        """Convert phone/msisdn to int"""
        logging.debug("Convert phone to int")
        msisdn_column = self.etalon_def_checked.get(FileColumnMeaningType.MSISDN.value)
        if msisdn_column is not None and msisdn_column in self.columns:
            logging.debug(f"going to apply phone_to_int for column {msisdn_column}")
            phone_to_int(self, msisdn_column)
            self[msisdn_column] = self[msisdn_column].astype("Int64")

    def __features(self):
        return [
            f for f, meaning_type in self.meaning_types_checked.items() if meaning_type == FileColumnMeaningType.FEATURE
        ]

    def __remove_dates_from_features(self):
        logging.debug("Remove date columns from features")

        for f in self.__features():
            if is_datetime(self[f]) or is_period_dtype(self[f]):
                logging.warning(f"Column {f} has datetime or period type but is feature and will be dropped from tds")
                self.drop(columns=f, inplace=True)
                del self.meaning_types_checked[f]

    def __remove_empty_and_constant_features(self):
        logging.debug("Remove constant and empty columns")

        for f in self.__features():
            if self[f].nunique() <= 1:
                logging.warning(f"Column {f} is constant or empty and will be droped from tds")
                self.drop(columns=f, inplace=True)
                del self.meaning_types_checked[f]

    def __remove_high_cardinality_features(self):
        logging.debug("Remove columns with high cardinality")

        count = len(self)
        for f in self.__features():
            if (is_string(self[f]) or is_integer_dtype(self[f])) and self[f].nunique() / count >= 0.9:
                logging.warning(
                    f"Column {f} has high cardinality (more than 90% uniques and string or integer type) "
                    "and will be droped from tds"
                )
                self.drop(columns=f, inplace=True)
                del self.meaning_types_checked[f]

    def __convert_features_types(self):
        logging.debug("Convert features to supported data types")

        for f in self.__features():
            if self[f].dtype == np.object:
                self[f] = self[f].astype(str)
            elif not is_numeric_dtype(self[f].dtype):
                self[f] = self[f].astype(str)

    def __validate_dataset(self, validate_target: bool):
        """Validate DataSet"""
        logging.debug("validating etalon")
        date_millis = self.etalon_def_checked.get(FileColumnMeaningType.DATE.value) or self.etalon_def_checked.get(
            FileColumnMeaningType.DATETIME.value
        )
        target = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        score = self.etalon_def_checked.get(FileColumnMeaningType.SCORE.value)
        if self.task_type != ModelTaskType.MULTICLASS and validate_target:
            if target is None:
                raise ValueError("Target column is absent in meaning_types.")

            target_value = self.__target_value()
            target_items = target_value.nunique()
            if target_items == 1:
                raise ValueError("Target contains only one distinct value.")
            elif target_items == 0:
                raise ValueError("Target contains only NaN or incorrect values.")

            self[target] = self[target].apply(pd.to_numeric, errors="coerce")
        keys_to_validate = [key for search_group in self.search_keys_checked for key in search_group]
        mandatory_columns = [date_millis, target, score]
        columns_to_validate = mandatory_columns.copy()
        columns_to_validate.extend(keys_to_validate)
        columns_to_validate = set([i for i in columns_to_validate if i is not None])
        columns_to_validate.discard(SYSTEM_FAKE_DATE)

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

            invalid_values = list(self.loc[self[f"{col}_is_valid"] == 0, col].head().values)
            valid_share = self[f"{col}_is_valid"].sum() / nrows
            validation_stats[col] = {}
            optional_drop_message = "Invalid rows will be dropped. " if col in mandatory_columns else ""
            if valid_share == 1:
                valid_status = "All valid"
                valid_message = "All values in this column are good to go"
            elif 0 < valid_share < 1:
                valid_status = "Some invalid"
                valid_message = (
                    f"{100 * (1 - valid_share):.5f}% of the values of this column failed validation. "
                    f"{optional_drop_message}"
                    f"Some examples of invalid values: {invalid_values}"
                )
            else:
                valid_status = "All invalid"
                valid_message = (
                    f"{100 * (1 - valid_share):.5f}% of the values of this column failed validation. "
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

        df_stats = pd.DataFrame.from_dict(validation_stats, orient="index")
        df_stats.reset_index(inplace=True)
        df_stats.columns = ["Column name", "Status", "Description"]
        styled_df_stats = df_stats.copy()
        styled_df_stats["Description"] = styled_df_stats["Description"].apply(lambda x: html.escape(x))  # type: ignore
        colormap = {"All valid": "#DAF7A6", "Some invalid": "#FFC300", "All invalid": "#FF5733"}
        styled_df_stats = styled_df_stats.style
        styled_df_stats.applymap(lambda x: f"background-color: {colormap[x]}", subset="Status")
        try:
            from IPython.display import display  # type: ignore

            display(styled_df_stats)
        except ImportError:
            print(df_stats)

    def __validate_meaning_types(self, validate_target: bool):
        logging.debug("Validating meaning types")
        if self.meaning_types is None or len(self.meaning_types) == 0:
            raise ValueError("Please pass the `meaning_types` argument before validation.")

        if SYSTEM_RECORD_ID not in self.columns:
            self[SYSTEM_RECORD_ID] = self.apply(lambda row: hash(tuple(row)), axis=1)
            self.meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID

        for column in self.meaning_types:
            if column not in self.columns:
                raise ValueError(f"Meaning column {column} doesn't exist in dataframe columns: {self.columns}.")
        if validate_target and FileColumnMeaningType.TARGET not in self.meaning_types.values():
            raise ValueError("Target column is not presented in meaning types. Specify it, please.")

    def __validate_search_keys(self):
        logging.debug("Validating search keys")
        if self.search_keys is None or len(self.search_keys) == 0:
            raise ValueError("Please pass `search_keys` argument before validation.")
        for keys_group in self.search_keys:
            for key in keys_group:
                if key not in self.columns:
                    raise ValueError(f"Search key {key} doesn't exist in dataframe columns: {self.columns}.")

    def validate(self, validate_target: bool = True, silent_mode: bool = False):
        logging.info("Validating dataset")

        self.__rename_columns()

        self.__validate_meaning_types(validate_target=validate_target)

        self.__validate_search_keys()

        self.__drop_ignore_columns()

        self.__validate_too_long_string_values()

        self.__clean_duplicates()

        self.__convert_bools()

        self.__convert_float16()

        self.__correct_decimal_comma()

        if validate_target and self.task_type is None:
            self.task_type = self.__define_task()

        self.__to_millis()

        self.__hash_email()

        self.__convert_ip()

        self.__convert_phone()

        self.__remove_dates_from_features()

        self.__remove_empty_and_constant_features()

        self.__remove_high_cardinality_features()

        self.__convert_features_types()

        if not silent_mode:
            self.__validate_dataset(validate_target)

    def calculate_metrics(self) -> FileMetrics:
        """Calculate initial metadata for DataSet

        Returns:
            InitialMetadata: initial metadata
        """
        logging.info("Calculating metrics")
        if self.etalon_def is None:
            self.validate()

        self.__remove_empty_date_rows()
        date_millis = (
            self.etalon_def_checked.get(FileColumnMeaningType.DATE.value)
            or self.etalon_def_checked.get(FileColumnMeaningType.DATETIME.value)
            or ""
        )
        target = self.etalon_def_checked.get(FileColumnMeaningType.TARGET.value)
        score = self.etalon_def_checked.get(FileColumnMeaningType.SCORE.value)
        cls_metadata = [date_millis, target, score, "is_valid"]
        cls_metadata = [i for i in cls_metadata if i is not None]
        # df with columns for metadata calculation
        df: pd.DataFrame = self[cls_metadata].copy()  # type: ignore
        count = len(df)
        valid_count = int(df["is_valid"].sum())
        valid_rate = 100 * valid_count / count
        avg_target = None
        metrics_binary_etalon = None
        metrics_regression_etalon = None
        metrics_multiclass_etalon = None

        if target is None:
            raise ValueError("Target column is absent in meaning_types.")
        target_values: pd.Series = df.loc[df.is_valid == 1, target]  # type: ignore
        tgt = target_values.values
        if self.task_type != ModelTaskType.MULTICLASS:
            avg_target = target_values.mean()

        if self.task_type == ModelTaskType.BINARY:
            label = ModelLabelType.AUC
        elif self.task_type == ModelTaskType.REGRESSION:
            label = ModelLabelType.RMSE
        else:
            label = ModelLabelType.ACCURACY

        if score is not None:
            try:
                from sklearn.metrics import (  # type: ignore
                    accuracy_score,
                    mean_squared_error,
                    mean_squared_log_error,
                    roc_auc_score,
                )
            except ModuleNotFoundError:
                raise ModuleNotFoundError("To calculate score performance please install scikit-learn.")
            sc = df.loc[df.is_valid == 1, score].values  # type: ignore
            try:
                if self.task_type == ModelTaskType.REGRESSION:
                    sc = sc.astype(float)
                    tgt = tgt.astype(float)
                    metrics_regression_etalon = RegressionTask(
                        mse=mean_squared_error(tgt, sc),
                        rmse=np.sqrt(mean_squared_error(tgt, sc)),
                        msle=mean_squared_log_error(tgt, sc),
                        rmsle=np.sqrt(mean_squared_log_error(tgt, sc)),
                    )
                elif self.task_type == ModelTaskType.BINARY:
                    auc = roc_auc_score(tgt, sc)
                    auc = max(auc, 1 - auc)
                    gini = 100 * (2 * auc - 1)
                    metrics_binary_etalon = BinaryTask(auc=auc, gini=gini)
                else:
                    accuracy100 = 100 * max(0.0, accuracy_score(tgt, sc))
                    metrics_multiclass_etalon = MulticlassTask(accuracy=accuracy100)
            except Exception:
                logging.error("Can't calculate etalon's score performance")
        else:
            sc = []

        df["date_cut"], cuts = pd.cut(df[date_millis], bins=6, include_lowest=True, retbins=True)
        # save bins for future
        cuts = cuts.tolist()
        df["date_cut"] = df.date_cut.apply(lambda x: x.mid).astype(int)
        df["count_tgt"] = df.groupby("date_cut")[target].transform(lambda x: len(set(x)))

        if self.task_type == ModelTaskType.MULTICLASS:
            df_stat = df.groupby("date_cut").apply(
                lambda x: pd.Series(
                    {
                        "count": len(x),
                        "valid_count": x["is_valid"].sum(),
                        "valid_rate": 100 * x["is_valid"].mean(),
                    }
                )
            )
        else:
            df_stat = df.groupby("date_cut").apply(
                lambda x: pd.Series(
                    {
                        "avg_target": x[target].mean(),
                        "count": len(x),
                        "valid_count": x["is_valid"].sum(),
                        "valid_rate": 100 * x["is_valid"].mean(),
                    }
                )
            )

        if score is not None and len(sc) > 0:
            df_stat["avg_score_etalon"] = df[df.is_valid == 1].groupby("date_cut")[score].mean()

        df_stat.dropna(inplace=True)
        interval = df_stat.reset_index().to_dict(orient="records")
        return FileMetrics(
            task_type=self.task_type,
            label=label,
            count=count,
            valid_count=valid_count,
            valid_rate=valid_rate,
            avg_target=avg_target,
            metrics_binary_etalon=metrics_binary_etalon,
            metrics_regression_etalon=metrics_regression_etalon,
            metrics_multiclass_etalon=metrics_multiclass_etalon,
            cuts=cuts,
            interval=interval,
        )

    def __construct_metadata(self) -> FileMetadata:
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
            name=self.name,
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
        elif is_string(pandas_data_type):
            return DataType.STRING
        else:
            msg = f"Unsupported data type of column {column_name}: {pandas_data_type}"
            logging.error(msg)
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
        return_scores: bool = False,
        extract_features: bool = False,
        accurate_model: bool = False,
        model_task_type: Optional[ModelTaskType] = None,
        importance_threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        filter_features: Optional[dict] = None,
        runtime_parameters: Optional[RuntimeParameters] = None,
    ) -> SearchTask:
        self.task_type = model_task_type
        if self.etalon_def is None:
            self.validate()
        file_metrics = self.calculate_metrics()
        # keep only valid rows
        if "is_valid" in self.columns:
            drop_idx = self[self["is_valid"] != 1].index  # type: ignore
            self.drop(drop_idx, inplace=True)
            self.drop(columns=["is_valid"], inplace=True)
        self.__validate_target()
        self.__validate_rows_count()

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
            self.file_upload_id, file_metadata
        ):
            search_task_response = get_rest_client(self.endpoint, self.api_key).initial_search_without_upload_v2(
                self.file_upload_id, file_metadata, file_metrics, search_customization
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                parquet_file_path = f"{tmp_dir}/{self.name}.parquet"
                self.to_parquet(path=parquet_file_path, index=False, compression="gzip")
                logging.debug(f"Size of prepared uploading file: {Path(parquet_file_path).stat().st_size}")
                search_task_response = get_rest_client(self.endpoint, self.api_key).initial_search_v2(
                    parquet_file_path, file_metadata, file_metrics, search_customization
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
        return search_task.poll_result()

    def validation(
        self,
        initial_search_task_id: str,
        return_scores: bool = True,
        extract_features: bool = False,
        runtime_parameters: Optional[RuntimeParameters] = None,
        silent_mode: bool = False,
    ) -> SearchTask:
        if self.etalon_def is None:
            self.validate(validate_target=False, silent_mode=silent_mode)
        if extract_features:
            file_metrics = FileMetrics()
        else:
            file_metrics = self.calculate_metrics()
        # keep only valid rows
        if "is_valid" in self.columns:
            self.query("is_valid == 1", inplace=True)
            self.drop(["is_valid"], axis=1, inplace=True)

        file_metadata = self.__construct_metadata()
        search_customization = self.__construct_search_customization(
            return_scores, extract_features, runtime_parameters=runtime_parameters
        )

        if self.file_upload_id is not None and get_rest_client(self.endpoint, self.api_key).check_uploaded_file_v2(
            self.file_upload_id, file_metadata
        ):
            search_task_response = get_rest_client(self.endpoint, self.api_key).validation_search_without_upload_v2(
                self.file_upload_id, initial_search_task_id, file_metadata, file_metrics, search_customization
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                parquet_file_path = f"{tmp_dir}/{self.name}.parquet"
                self.to_parquet(path=parquet_file_path, index=False, compression="gzip")
                logging.debug(f"Size of uploading file: {Path(parquet_file_path).stat().st_size}")
                search_task_response = get_rest_client(self.endpoint, self.api_key).validation_search_v2(
                    parquet_file_path, initial_search_task_id, file_metadata, file_metrics, search_customization
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

        return search_task.poll_result(quiet=silent_mode)
