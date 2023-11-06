import datetime
import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.api.types import is_numeric_dtype, is_period_dtype, is_string_dtype

from upgini.errors import ValidationError

DATE_FORMATS = ["%Y-%m-%d", "%d.%m.%y", "%d.%m.%Y", "%m.%d.%y", "%m.%d.%Y", "%Y-%m-%dT%H:%M:%S.%f"]

DATETIME_PATTERN = r"^[\d\s\.\-:T]+$"


class DateTimeSearchKeyConverter:
    DATETIME_COL = "_date_time"

    def __init__(self, date_column: str, date_format: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.date_column = date_column
        self.date_format = date_format
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")
        self.generated_features: List[str] = []

    @staticmethod
    def _int_to_opt(i: int) -> Optional[int]:
        if i == -9223372036855:
            return None
        else:
            return i

    @staticmethod
    def clean_date(s: Optional[str]):
        try:
            if s is None or len(str(s).strip()) == 0:
                return None
            if not re.match(DATETIME_PATTERN, str(s)):
                return None
            return s
        except Exception:
            return None

    def convert(self, df: pd.DataFrame, keep_time=False) -> pd.DataFrame:
        if len(df) == 0:
            return df

        df = df.copy()
        if df[self.date_column].apply(lambda x: isinstance(x, datetime.datetime)).all():
            df[self.date_column] = df[self.date_column].apply(lambda x: x.replace(tzinfo=None))
        elif isinstance(df[self.date_column].values[0], datetime.date):
            df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        elif is_string_dtype(df[self.date_column]):
            df[self.date_column] = df[self.date_column].apply(self.clean_date)
            df[self.date_column] = self.parse_date(df)
        elif is_period_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column].astype("string"))
        elif is_numeric_dtype(df[self.date_column]):
            msg = f"Unsupported type of date column {self.date_column}. Convert to datetime please."
            self.logger.warning(msg)
            raise ValidationError(msg)

        # If column with date is datetime then extract seconds of the day and minute of the hour
        # as additional features
        seconds = "datetime_seconds"
        df[self.date_column] = df[self.date_column].dt.tz_localize(None)
        df[seconds] = (df[self.date_column] - df[self.date_column].dt.floor("D")).dt.seconds

        seconds_without_na = df[seconds].dropna()
        if (seconds_without_na != 0).any() and seconds_without_na.nunique() > 1:
            self.logger.info("Time found in date search key. Add extra features based on time")
            seconds_in_day = 60 * 60 * 24
            orders = [1, 2, 24, 48]
            for order in orders:
                sin_feature = f"datetime_time_sin_{order}"
                cos_feature = f"datetime_time_cos_{order}"
                df[sin_feature] = np.round(np.sin(2 * np.pi * order * df[seconds] / seconds_in_day), 10)
                df[cos_feature] = np.round(np.cos(2 * np.pi * order * df[seconds] / seconds_in_day), 10)
                self.generated_features.append(sin_feature)
                self.generated_features.append(cos_feature)

        df.drop(columns=seconds, inplace=True)

        if keep_time:
            df[self.DATETIME_COL] = df[self.date_column].view(np.int64) // 1_000_000
            df[self.DATETIME_COL] = df[self.DATETIME_COL].apply(self._int_to_opt).astype("Int64")
        df[self.date_column] = df[self.date_column].dt.floor("D").view(np.int64) // 1_000_000
        df[self.date_column] = df[self.date_column].apply(self._int_to_opt).astype("Int64")

        self.logger.info(f"Date after convertion to timestamp: {df[self.date_column]}")

        return df

    def parse_date(self, df: pd.DataFrame):
        if self.date_format is not None:
            try:
                return pd.to_datetime(df[self.date_column], format=self.date_format)
            except ValueError as e:
                raise ValidationError(e)
        else:
            for date_format in DATE_FORMATS:
                try:
                    return pd.to_datetime(df[self.date_column], format=date_format)
                except ValueError:
                    pass
            raise ValidationError(
                f"Failed to parse date in column `{self.date_column}`. "
                "Try to pass explicit date format in date_format argument of FeaturesEnricher constructor"
            )


def is_time_series(df: pd.DataFrame, date_col: str) -> bool:
    try:
        if df[date_col].isnull().any():
            return False

        df = pd.to_datetime(df[date_col]).to_frame()

        def rel(row):
            if not pd.isnull(row[date_col]) and not pd.isnull(row["shifted_date"]):
                return relativedelta(row[date_col], row["shifted_date"])

        value_counts = df[date_col].value_counts()
        # count with each date is constant
        if value_counts.nunique() == 1:
            # Univariate timeseries
            if value_counts.unique()[0] == 1:
                df["shifted_date"] = df[date_col].shift(1)
                # if dates cover full interval without gaps
                return df.apply(rel, axis=1).nunique() == 1

            # Multivariate timeseries
            df_with_unique_dates = df.drop_duplicates().copy()

            df_with_unique_dates["shifted_date"] = df_with_unique_dates[date_col].shift(1)
            # if unique dates cover full interval without gaps
            return df_with_unique_dates.apply(rel, axis=1).nunique() == 1

        return False
    except Exception:
        return False


def is_blocked_time_series(df: pd.DataFrame, date_col: str, search_keys: List[str]) -> bool:
    df = df.copy()
    seconds = "datetime_seconds"
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].dt.tz_localize(None)
    df[seconds] = (df[date_col] - df[date_col].dt.floor("D")).dt.seconds

    seconds_without_na = df[seconds].dropna()
    columns_to_drop = [c for c in search_keys if c != date_col] + [seconds]
    df.drop(columns=columns_to_drop, inplace=True)
    # Date, not datetime
    if (seconds_without_na != 0).any() and seconds_without_na.nunique() > 1:
        return False

    nunique_dates = df[date_col].nunique()
    # Unique dates count more than 270
    if nunique_dates < 270:
        return False

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    days_delta = (max_date - min_date).days + 1
    # Missing dates less than 30% (unique dates count and days delta between earliest and latest dates)
    if nunique_dates / days_delta < 0.3:
        return False

    def check_differences(group):
        data = group.drop(date_col, axis=1)
        diffs = data.values[:, None] != data.values
        diff_counts = diffs.sum(axis=2)
        max_diff = np.max(diff_counts)
        return max_diff <= 2

    def is_multiple_rows(group):
        return group.shape[0] > 1

    grouped = df.groupby(date_col)
    dates_with_multiple_rows = len(grouped.apply(is_multiple_rows))

    # share of dates with more than one record is more than 99%
    if dates_with_multiple_rows / nunique_dates < 0.99:
        return False

    if df.shape[1] <= 3:
        return True

    is_diff_less_than_two_columns = grouped.apply(check_differences)
    return is_diff_less_than_two_columns.all()
