import logging
from typing import List, Optional
import pandas as pd


def get_most_frequent_time_unit(df: pd.DataFrame, id_columns: List[str], date_column: str) -> Optional[pd.DateOffset]:

    def closest_unit(diff):
        return pd.tseries.frequencies.to_offset(pd.Timedelta(diff, unit="s"))

    all_diffs = []
    groups = df.groupby(id_columns) if id_columns else [(None, df)]
    for _, group in groups:
        group_dates = group[date_column].sort_values().unique()
        if len(group_dates) > 1:
            diff_series = pd.Series(group_dates[1:] - group_dates[:-1])
            diff_ns = diff_series.dt.total_seconds()
            all_diffs.extend(diff_ns)

    all_diffs = pd.Series(all_diffs)

    most_frequent_unit = all_diffs.apply(closest_unit).mode().min()

    return most_frequent_unit if isinstance(most_frequent_unit, pd.DateOffset) else None


def trunc_datetime(
    df: pd.DataFrame,
    id_columns: List[str],
    date_column: str,
    length: pd.DateOffset,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if logger is not None:
        logger.info(f"Truncating time series dataset to {length}")

    if id_columns:
        min_datetime = df.groupby(id_columns)[date_column].transform(lambda group: group.max() - length)
    else:
        min_datetime = df[date_column].max() - length
    return df[df[date_column] > min_datetime]
