import hashlib
from logging import Logger, getLogger
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import (
    is_float_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from upgini.errors import ValidationError
from upgini.metadata import (
    ENTITY_SYSTEM_RECORD_ID,
    EVAL_SET_INDEX,
    SEARCH_KEY_UNNEST,
    SYSTEM_RECORD_ID,
    TARGET,
    SearchKey,
)
from upgini.resource_bundle import ResourceBundle, get_custom_bundle
from upgini.utils import find_numbers_with_decimal_comma
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.phone_utils import PhoneSearchKeyConverter
from upgini.utils.warning_counter import WarningCounter


class Normalizer:

    MAX_STRING_FEATURE_LENGTH = 24573

    def __init__(
        self,
        search_keys: Dict[str, SearchKey],
        generated_features: List[str],
        bundle: ResourceBundle = None,
        logger: Logger = None,
        warnings_counter: WarningCounter = None,
        silent_mode=False,
    ):
        self.search_keys = search_keys
        self.generated_features = generated_features
        self.bundle = bundle or get_custom_bundle()
        self.logger = logger or getLogger()
        self.warnings_counter = warnings_counter or WarningCounter()
        self.silent_mode = silent_mode
        self.columns_renaming = {}

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._rename_columns(df)

        df = self._remove_dates_from_features(df)

        df = self._cut_too_long_string_values(df)

        df = self._convert_bools(df)

        df = self._convert_float16(df)

        df = self._correct_decimal_comma(df)

        df = self._convert_phone_numbers(df)

        df = self.__convert_features_types(df)

        return df

    def _rename_columns(self, df: pd.DataFrame):
        # logger.info("Replace restricted symbols in column names")
        new_columns = []
        dup_counter = 0
        for column in df.columns:
            if column in [
                TARGET,
                EVAL_SET_INDEX,
                SYSTEM_RECORD_ID,
                ENTITY_SYSTEM_RECORD_ID,
                SEARCH_KEY_UNNEST,
                DateTimeSearchKeyConverter.DATETIME_COL,
            ] + self.generated_features:
                self.columns_renaming[column] = column
                new_columns.append(column)
                continue

            new_column = str(column)
            suffix = hashlib.sha256(new_column.encode()).hexdigest()[:6]
            if len(new_column) == 0:
                raise ValidationError(self.bundle.get("dataset_empty_column_names"))
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

            # df.columns.values[col_idx] = new_column
            # rename(columns={column: new_column}, inplace=True)

            if new_column != column and column in self.search_keys:
                self.search_keys[new_column] = self.search_keys[column]
                del self.search_keys[column]
            self.columns_renaming[new_column] = str(column)
        df.columns = new_columns
        return df

    def _get_features(self, df: pd.DataFrame) -> List[str]:
        system_columns = [ENTITY_SYSTEM_RECORD_ID, EVAL_SET_INDEX, SEARCH_KEY_UNNEST, SYSTEM_RECORD_ID, TARGET]
        features = set(df.columns) - set(self.search_keys.keys()) - set(system_columns)
        return sorted(list(features))

    def _remove_dates_from_features(self, df: pd.DataFrame):
        features = self._get_features(df)

        removed_features = []
        for f in features:
            if is_datetime(df[f]) or isinstance(df[f].dtype, pd.PeriodDtype):
                removed_features.append(f)
                df.drop(columns=f, inplace=True)

        if removed_features:
            msg = self.bundle.get("dataset_date_features").format(removed_features)
            self.logger.warning(msg)
            if not self.silent_mode:
                print(msg)
            self.warnings_counter.increment()

        return df

    def _cut_too_long_string_values(self, df: pd.DataFrame):
        """Check that string values less than maximum characters for LLM"""
        # logger.info("Validate too long string values")
        for col in df.columns:
            if is_string_dtype(df[col]) or is_object_dtype(df[col]):
                max_length: int = df[col].astype("str").str.len().max()
                if max_length > self.MAX_STRING_FEATURE_LENGTH:
                    df[col] = df[col].astype("str").str.slice(stop=self.MAX_STRING_FEATURE_LENGTH)

        return df

    @staticmethod
    def _convert_bools(df: pd.DataFrame):
        """Convert bool columns to string"""
        # logger.info("Converting bool to int")
        for col in df.columns:
            if is_bool(df[col]):
                df[col] = df[col].astype("str")
        return df

    @staticmethod
    def _convert_float16(df: pd.DataFrame):
        """Convert float16 to float"""
        # logger.info("Converting float16 to float")
        for col in df.columns:
            if is_float_dtype(df[col]):
                df[col] = df[col].astype("float64")
        return df

    def _correct_decimal_comma(self, df: pd.DataFrame):
        """Check DataSet for decimal commas and fix them"""
        # logger.info("Correct decimal commas")
        columns_to_fix = find_numbers_with_decimal_comma(df)
        if len(columns_to_fix) > 0:
            self.logger.warning(f"Convert strings with decimal comma to float: {columns_to_fix}")
            for col in columns_to_fix:
                df[col] = df[col].astype("string").str.replace(",", ".", regex=False).astype(np.float64)
        return df

    def _convert_phone_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        maybe_country_col = SearchKey.find_key(self.search_keys, SearchKey.COUNTRY)
        for phone_col in SearchKey.find_all_keys(self.search_keys, SearchKey.PHONE):
            converter = PhoneSearchKeyConverter(phone_col, maybe_country_col)
            df = converter.convert(df)
        return df

    def __convert_features_types(self, df: pd.DataFrame):
        # self.logger.info("Convert features to supported data types")

        for f in self._get_features(df):
            if not is_numeric_dtype(df[f]):
                df[f] = df[f].astype("string")
        return df
