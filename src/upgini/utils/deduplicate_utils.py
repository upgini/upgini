import logging
from logging import Logger
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from upgini.metadata import (
    ENTITY_SYSTEM_RECORD_ID,
    EVAL_SET_INDEX,
    SORT_ID,
    SYSTEM_RECORD_ID,
    TARGET,
    ModelTaskType,
    SearchKey,
)
from upgini.resource_bundle import ResourceBundle, get_custom_bundle
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.target_utils import define_task


def remove_fintech_duplicates(
    df: pd.DataFrame,
    search_keys: Dict[str, SearchKey],
    date_format: Optional[str] = None,
    logger: Optional[Logger] = None,
    bundle: ResourceBundle = None,
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    # Initial checks for target type and date column
    bundle = bundle or get_custom_bundle()
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.FATAL)
    date_col = _get_column_by_key(search_keys, [SearchKey.DATE, SearchKey.DATETIME])
    if define_task(df[TARGET], date_col is not None, silent=True) != ModelTaskType.BINARY:
        return df, []

    if date_col is None:
        return df, []

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
        return df, []

    # Splitting into train and eval_set parts
    if EVAL_SET_INDEX in df.columns:
        train_df = df[df[EVAL_SET_INDEX] == 0]
        eval_dfs = [df[df[EVAL_SET_INDEX] == idx] for idx in df[EVAL_SET_INDEX].unique() if idx != 0]
    else:
        train_df = df
        eval_dfs = []

    warning_messages = []

    def process_df(segment_df: pd.DataFrame, eval_index=0) -> Tuple[pd.DataFrame, Optional[str]]:
        """Process a subset of the dataset to remove duplicates based on personal keys."""
        # Fast check for duplicates based on personal keys
        if not segment_df[personal_cols].duplicated().any():
            return segment_df, None

        sub_df = segment_df[personal_cols + [date_col, TARGET]].copy()

        # Group by personal columns to check for unique dates
        grouped_by_personal_cols = sub_df.groupby(personal_cols, group_keys=False)

        # Checking for different dates by the same personal keys
        uniques = grouped_by_personal_cols[date_col].nunique()
        total = len(uniques)
        if total == 0:
            return segment_df, None
        diff_dates = len(uniques[uniques > 1])
        if diff_dates / total >= 0.6:
            return segment_df, None

        # Check for duplicate rows
        duplicates = sub_df.duplicated(personal_cols, keep=False)
        duplicate_rows = sub_df[duplicates]
        if len(duplicate_rows) == 0:
            return segment_df, None

        # Check if there are different target values for the same personal keys
        nonunique_target_groups = grouped_by_personal_cols[TARGET].nunique() > 1
        if nonunique_target_groups.sum() == 0:
            return segment_df, None

        # Helper function to check if there are different target values within 60 days
        def has_diff_target_within_60_days(rows: pd.DataFrame):
            rows = rows.sort_values(by=date_col)
            return (
                len(rows[rows[TARGET].ne(rows[TARGET].shift()) & (rows[date_col].diff() < 60 * 24 * 60 * 60 * 1000)])
                > 0
            )

        # Filter rows with different target values within 60 days
        nonunique_target_rows = nonunique_target_groups[nonunique_target_groups].reset_index().drop(columns=TARGET)
        sub_df = pd.merge(sub_df, nonunique_target_rows, on=personal_cols)

        # Convert date columns for further checks
        sub_df = DateTimeSearchKeyConverter(
            date_col, date_format=date_format, logger=logger, bundle=bundle, generate_cyclical_features=False
        ).convert(sub_df)
        grouped_by_personal_cols = sub_df.groupby(personal_cols, group_keys=False)
        rows_with_diff_target = grouped_by_personal_cols.filter(has_diff_target_within_60_days)

        if len(rows_with_diff_target) > 0:
            unique_keys_to_delete = rows_with_diff_target[personal_cols].drop_duplicates()
            rows_to_remove = pd.merge(segment_df.reset_index(), unique_keys_to_delete, on=personal_cols)
            rows_to_remove = rows_to_remove.set_index(segment_df.index.name or "index")
            perc = len(rows_to_remove) * 100 / len(segment_df)
            if eval_index == 0:
                msg = bundle.get("dataset_train_diff_target_duplicates_fintech").format(
                    perc, len(rows_to_remove), rows_to_remove.index.to_list()
                )
            else:
                msg = bundle.get("dataset_eval_diff_target_duplicates_fintech").format(
                    perc, len(rows_to_remove), eval_index, rows_to_remove.index.to_list()
                )
            return segment_df[~segment_df.index.isin(rows_to_remove.index)], msg
        return segment_df, None

    # Process the train part separately
    logger.info(f"Train dataset shape before clean fintech duplicates: {train_df.shape}")
    train_df, train_warning = process_df(train_df)
    if train_warning:
        warning_messages.append(train_warning)
    logger.info(f"Train dataset shape after clean fintech duplicates: {train_df.shape}")

    # Process each eval_set part separately
    oot_eval_dfs = []
    new_eval_dfs = []
    for i, eval_df in enumerate(eval_dfs, 1):
        # Skip OOT
        if eval_df[TARGET].isna().all():
            oot_eval_dfs.append(eval_df)
            continue
        logger.info(f"Eval {i} dataset shape before clean fintech duplicates: {eval_df.shape}")
        cleaned_eval_df, eval_warning = process_df(eval_df, i)
        if eval_warning:
            warning_messages.append(eval_warning)
        logger.info(f"Eval {i} dataset shape after clean fintech duplicates: {cleaned_eval_df.shape}")
        new_eval_dfs.append(cleaned_eval_df)

    # Combine the processed train and eval parts back into one dataset
    logger.info(f"Dataset shape before clean fintech duplicates: {df.shape}")
    if new_eval_dfs or oot_eval_dfs:
        df = pd.concat([train_df] + new_eval_dfs + oot_eval_dfs, ignore_index=False)
    else:
        df = train_df
    logger.info(f"Dataset shape after clean fintech duplicates: {df.shape}")

    return df, warning_messages


def clean_full_duplicates(
    df: pd.DataFrame, logger: Optional[Logger] = None, bundle: Optional[ResourceBundle] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.FATAL)
    if bundle is None:
        bundle = get_custom_bundle()

    nrows = len(df)
    if nrows == 0:
        return df, None
    # Remove full duplicates (exclude system_record_id, sort_id and eval_set_index)
    unique_columns = df.columns.tolist()
    if SYSTEM_RECORD_ID in unique_columns:
        unique_columns.remove(SYSTEM_RECORD_ID)
    if ENTITY_SYSTEM_RECORD_ID in unique_columns:
        unique_columns.remove(ENTITY_SYSTEM_RECORD_ID)
    if SORT_ID in unique_columns:
        unique_columns.remove(SORT_ID)
    if EVAL_SET_INDEX in unique_columns:
        unique_columns.remove(EVAL_SET_INDEX)

    logger.info(f"Dataset shape before clean duplicates: {df.shape}")
    # Train segment goes first so if duplicates are found in train and eval set
    # then we keep unique rows in train segment
    df = df.drop_duplicates(subset=unique_columns, keep="first")
    logger.info(f"Dataset shape after clean duplicates: {df.shape}")
    nrows_after_full_dedup = len(df)
    share_full_dedup = 100 * (1 - nrows_after_full_dedup / nrows)
    if share_full_dedup > 0:
        logger.warning(bundle.get("dataset_full_duplicates").format(share_full_dedup))

    msg = None
    if TARGET in df.columns:
        unique_columns.remove(TARGET)

        # Separate rows to exclude from deduplication:
        # for each eval_set_index != 0 check separately, all TARGET values are NaN
        df_for_dedup = df
        oot_df = None

        if EVAL_SET_INDEX in df.columns:
            oot_eval_dfs = []
            other_dfs = []
            for eval_idx in df[EVAL_SET_INDEX].unique():
                eval_subset = df[df[EVAL_SET_INDEX] == eval_idx]
                # Check that all TARGET values for this specific eval_set_index are NaN
                if eval_idx != 0 and eval_subset[TARGET].isna().all():
                    oot_eval_dfs.append(eval_subset)
                    logger.info(
                        f"Excluded {len(eval_subset)} rows from deduplication "
                        f"(eval_set_index={eval_idx} and all TARGET values are NaN)"
                    )
                else:
                    other_dfs.append(eval_subset)

            if oot_eval_dfs:
                oot_df = pd.concat(oot_eval_dfs, ignore_index=False)
                df_for_dedup = pd.concat(other_dfs, ignore_index=False)
            else:
                df_for_dedup = df

        marked_duplicates = df_for_dedup.duplicated(subset=unique_columns, keep=False)
        if marked_duplicates.sum() > 0:
            dups_indices = df_for_dedup[marked_duplicates].index.to_list()[:100]
            nrows_after_tgt_dedup = len(df_for_dedup.drop_duplicates(subset=unique_columns, keep=False))
            num_dup_rows = len(df_for_dedup) - nrows_after_tgt_dedup
            share_tgt_dedup = 100 * num_dup_rows / len(df_for_dedup)

            msg = bundle.get("dataset_diff_target_duplicates").format(share_tgt_dedup, num_dup_rows, dups_indices)
            df_for_dedup = df_for_dedup.drop_duplicates(subset=unique_columns, keep=False)
            logger.info(f"Dataset shape after clean invalid target duplicates: {df_for_dedup.shape}")
        # Combine back excluded rows
        if oot_df is not None:
            df = pd.concat([df_for_dedup, oot_df], ignore_index=False)
            marked_duplicates = df.duplicated(subset=unique_columns, keep=False)
            if marked_duplicates.sum() > 0:
                dups_indices = df[marked_duplicates].index.to_list()[:100]
                nrows_after_tgt_dedup = len(df.drop_duplicates(subset=unique_columns, keep=False))
                num_dup_rows = len(df) - nrows_after_tgt_dedup
                share_tgt_dedup = 100 * num_dup_rows / len(df)
                msg = bundle.get("dataset_diff_target_duplicates_oot").format(
                    share_tgt_dedup, num_dup_rows, dups_indices
                )
                df = df.drop_duplicates(subset=unique_columns, keep="first")
            logger.info(f"Final dataset shape after adding back excluded rows: {df.shape}")
        else:
            df = df_for_dedup

    return df, msg


def _get_column_by_key(search_keys: Dict[str, SearchKey], keys: Union[SearchKey, List[SearchKey]]) -> Optional[str]:
    for col, key_type in search_keys.items():
        if (isinstance(keys, list) and key_type in keys) or key_type == keys:
            return col
