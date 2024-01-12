from logging import Logger
from typing import Dict, List, Optional, Union

import pandas as pd

from upgini.metadata import TARGET, ModelTaskType, SearchKey
from upgini.resource_bundle import bundle
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.target_utils import define_task


def remove_fintech_duplicates(df: pd.DataFrame, 
                              search_keys: Dict[str, SearchKey], 
                              logger: Optional[Logger] = None) -> pd.DataFrame:
    if define_task(df.target, silent=True) != ModelTaskType.BINARY:
        return df
    
    date_col = _get_column_by_key(search_keys, [SearchKey.DATE, SearchKey.DATETIME])
    if date_col is None:
        return df
    
    personal_cols = []
    phone_col = _get_column_by_key(search_keys, SearchKey.PHONE)
    if phone_col:
        personal_cols.append(phone_col)
    email_col = _get_column_by_key(search_keys, SearchKey.EMAIL)
    if email_col:
        personal_cols.append(email_col)
    hem_col = _get_column_by_key(search_keys, SearchKey.HEM)
    if hem_col:
        personal_cols.append(hem_col)
    if len(personal_cols) == 0:
        return df
    
    duplicates = df.duplicated(personal_cols, keep=False)
    duplicate_rows = df[duplicates]
    if len(duplicate_rows) == 0:
        return df
    
    grouped_by_personal_cols = df.groupby(personal_cols, group_keys=False)
    
    uniques = grouped_by_personal_cols[date_col].nunique()
    total = len(uniques)
    diff_dates = len(uniques[uniques > 1])
    if diff_dates / total >= 0.6:
        return df
    
    if grouped_by_personal_cols[TARGET].apply(lambda x: len(x.unique()) == 1).all():
        return df

    def has_diff_target_within_60_days(rows):
        rows = rows.sort_values(by=date_col)
        return len(rows[rows[TARGET].ne(rows[TARGET].shift()) & (rows[date_col].diff() < 60 * 24 * 60 * 60 * 1000)]) > 0
    
    df = DateTimeSearchKeyConverter(date_col).convert(df)
    grouped_by_personal_cols = df.groupby(personal_cols, group_keys=False)
    rows_with_diff_target = grouped_by_personal_cols.filter(has_diff_target_within_60_days)
    if len(rows_with_diff_target) > 0:
        perc = len(rows_with_diff_target) * 100 / len(df)
        msg = bundle.get("dataset_diff_target_duplicates_fintech").format(perc, len(rows_with_diff_target), rows_with_diff_target.index.to_list())
        print(msg)
        if logger:
            logger.warning(msg)
        df = df[~df.index.isin(rows_with_diff_target.index)]
    
    return df


def _get_column_by_key(search_keys: Dict[str, SearchKey], keys: Union[SearchKey, List[SearchKey]]) -> Optional[str]:
    for col, key_type in search_keys.items():
        if (isinstance(keys, list) and key_type in keys) or key_type == keys:
            return col