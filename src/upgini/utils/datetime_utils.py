import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_period_dtype, is_string_dtype
from dateutil.relativedelta import relativedelta

from upgini.errors import ValidationError


class DateTimeSearchKeyConverter:
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

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if is_string_dtype(df[self.date_column]):
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format)
            except ValueError as e:
                raise ValidationError(e)
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

        df[self.date_column] = df[self.date_column].dt.floor("D").view(np.int64) // 1_000_000
        df[self.date_column] = df[self.date_column].apply(self._int_to_opt).astype("Int64")

        return df


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
            df_with_unique_dates = df.drop_duplicates()

            df_with_unique_dates["shifted_date"] = df_with_unique_dates[date_col].shift(1)
            # if unique dates cover full interval without gaps
            return df_with_unique_dates.apply(rel, axis=1).nunique() == 1

        return False
    except Exception:
        return False
