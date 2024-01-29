from logging import Logger
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from upgini.metadata import SYSTEM_RECORD_ID, TARGET, ModelTaskType, SearchKey
from upgini.resource_bundle import bundle
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.target_utils import define_task


def remove_fintech_duplicates(
    df: pd.DataFrame, search_keys: Dict[str, SearchKey], logger: Optional[Logger] = None, silent=False
) -> Tuple[bool, pd.DataFrame]:
    # Base checks
    need_full_deduplication = True

    date_col = _get_column_by_key(search_keys, [SearchKey.DATE, SearchKey.DATETIME])
    if define_task(df[TARGET], date_col is not None, silent=True) != ModelTaskType.BINARY:
        return need_full_deduplication, df

    date_col = _get_column_by_key(search_keys, [SearchKey.DATE, SearchKey.DATETIME])
    if date_col is None:
        return need_full_deduplication, df

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
        return need_full_deduplication, df

    sub_df = df[personal_cols + [date_col, TARGET]]

    # Fast check for duplicates by personal keys
    if not sub_df[personal_cols].duplicated().any():
        return need_full_deduplication, df

    grouped_by_personal_cols = sub_df.groupby(personal_cols, group_keys=False)

    # counts of diff dates by set of personal keys
    uniques = grouped_by_personal_cols[date_col].nunique()
    total = len(uniques)
    diff_dates = len(uniques[uniques > 1])
    if diff_dates / total >= 0.6:
        return need_full_deduplication, df

    # Additional checks

    need_full_deduplication = False

    duplicates = sub_df.duplicated(personal_cols, keep=False)
    duplicate_rows = sub_df[duplicates]
    if len(duplicate_rows) == 0:
        return need_full_deduplication, df

    # if there is no different target values in personal keys duplicate rows
    nonunique_target_groups = grouped_by_personal_cols[TARGET].nunique() > 1
    if nonunique_target_groups.sum() == 0:
        return need_full_deduplication, df

    def has_diff_target_within_60_days(rows):
        rows = rows.sort_values(by=date_col)
        return len(rows[rows[TARGET].ne(rows[TARGET].shift()) & (rows[date_col].diff() < 60 * 24 * 60 * 60 * 1000)]) > 0

    nonunique_target_rows = nonunique_target_groups[nonunique_target_groups].reset_index().drop(columns=TARGET)
    sub_df = pd.merge(sub_df, nonunique_target_rows, on=personal_cols)

    sub_df = DateTimeSearchKeyConverter(date_col).convert(sub_df)
    grouped_by_personal_cols = sub_df.groupby(personal_cols, group_keys=False)
    rows_with_diff_target = grouped_by_personal_cols.filter(has_diff_target_within_60_days)
    if len(rows_with_diff_target) > 0:
        unique_keys_to_delete = rows_with_diff_target[personal_cols].drop_duplicates()
        rows_to_remove = pd.merge(df.reset_index(), unique_keys_to_delete, on=personal_cols)
        rows_to_remove = rows_to_remove.set_index(df.index.name or "index")
        perc = len(rows_to_remove) * 100 / len(df)
        msg = bundle.get("dataset_diff_target_duplicates_fintech").format(
            perc, len(rows_to_remove), rows_to_remove.index.to_list()
        )
        if not silent:
            print(msg)
        if logger:
            logger.warning(msg)
        logger.info(f"Dataset shape before clean fintech duplicates: {df.shape}")
        df = df[~df.index.isin(rows_to_remove.index)]
        logger.info(f"Dataset shape after clean fintech duplicates: {df.shape}")

    return need_full_deduplication, df


def clean_full_duplicates(
    df: pd.DataFrame, logger: Optional[Logger] = None, silent=False
) -> pd.DataFrame:
    nrows = len(df)
    if nrows == 0:
        return df
    # Remove absolute duplicates (exclude system_record_id)
    unique_columns = df.columns.tolist()
    if SYSTEM_RECORD_ID in unique_columns:
        unique_columns.remove(SYSTEM_RECORD_ID)
    if "sort_id" in unique_columns:
        unique_columns.remove("sort_id")
    logger.info(f"Dataset shape before clean duplicates: {df.shape}")
    df = df.drop_duplicates(subset=unique_columns)
    logger.info(f"Dataset shape after clean duplicates: {df.shape}")
    nrows_after_full_dedup = len(df)
    share_full_dedup = 100 * (1 - nrows_after_full_dedup / nrows)
    if share_full_dedup > 0:
        msg = bundle.get("dataset_full_duplicates").format(share_full_dedup)
        logger.warning(msg)
        # if not silent_mode:
        #     print(msg)
        # self.warning_counter.increment()
    if TARGET in df.columns:
        unique_columns.remove(TARGET)
        marked_duplicates = df.duplicated(subset=unique_columns, keep=False)
        if marked_duplicates.sum() > 0:
            dups_indices = df[marked_duplicates].index.to_list()
            nrows_after_tgt_dedup = len(df.drop_duplicates(subset=unique_columns))
            num_dup_rows = nrows_after_full_dedup - nrows_after_tgt_dedup
            share_tgt_dedup = 100 * num_dup_rows / nrows_after_full_dedup

            msg = bundle.get("dataset_diff_target_duplicates").format(share_tgt_dedup, num_dup_rows, dups_indices)
            logger.warning(msg)
            if not silent:
                print(msg)
            df = df.drop_duplicates(subset=unique_columns, keep=False)
            logger.info(f"Dataset shape after clean invalid target duplicates: {df.shape}")
    return df


def _get_column_by_key(search_keys: Dict[str, SearchKey], keys: Union[SearchKey, List[SearchKey]]) -> Optional[str]:
    for col, key_type in search_keys.items():
        if (isinstance(keys, list) and key_type in keys) or key_type == keys:
            return col
