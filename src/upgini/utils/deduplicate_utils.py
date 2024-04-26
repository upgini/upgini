from logging import Logger
from typing import Dict, List, Optional, Union

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
from upgini.resource_bundle import ResourceBundle
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.target_utils import define_task


def remove_fintech_duplicates(
    df: pd.DataFrame,
    search_keys: Dict[str, SearchKey],
    date_format: Optional[str] = None,
    logger: Optional[Logger] = None,
    silent=False,
    bundle: ResourceBundle = None,
) -> pd.DataFrame:
    # Base checks
    date_col = _get_column_by_key(search_keys, [SearchKey.DATE, SearchKey.DATETIME])
    if define_task(df[TARGET], date_col is not None, silent=True) != ModelTaskType.BINARY:
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

    sub_df = df[personal_cols + [date_col, TARGET]]

    # Fast check for duplicates by personal keys
    if not sub_df[personal_cols].duplicated().any():
        return df

    grouped_by_personal_cols = sub_df.groupby(personal_cols, group_keys=False)

    # counts of diff dates by set of personal keys
    uniques = grouped_by_personal_cols[date_col].nunique()
    total = len(uniques)
    diff_dates = len(uniques[uniques > 1])
    if diff_dates / total >= 0.6:
        return df

    # Additional checks

    duplicates = sub_df.duplicated(personal_cols, keep=False)
    duplicate_rows = sub_df[duplicates]
    if len(duplicate_rows) == 0:
        return df

    # if there is no different target values in personal keys duplicate rows
    nonunique_target_groups = grouped_by_personal_cols[TARGET].nunique() > 1
    if nonunique_target_groups.sum() == 0:
        return df

    def has_diff_target_within_60_days(rows):
        rows = rows.sort_values(by=date_col)
        return len(rows[rows[TARGET].ne(rows[TARGET].shift()) & (rows[date_col].diff() < 60 * 24 * 60 * 60 * 1000)]) > 0

    nonunique_target_rows = nonunique_target_groups[nonunique_target_groups].reset_index().drop(columns=TARGET)
    sub_df = pd.merge(sub_df, nonunique_target_rows, on=personal_cols)

    sub_df = DateTimeSearchKeyConverter(date_col, date_format=date_format, logger=logger, bundle=bundle).convert(sub_df)
    grouped_by_personal_cols = sub_df.groupby(personal_cols, group_keys=False)
    rows_with_diff_target = grouped_by_personal_cols.filter(has_diff_target_within_60_days)
    if len(rows_with_diff_target) > 0:
        unique_keys_to_delete = rows_with_diff_target[personal_cols].drop_duplicates()
        if EVAL_SET_INDEX not in df.columns:
            rows_to_remove = pd.merge(df.reset_index(), unique_keys_to_delete, on=personal_cols)
            rows_to_remove = rows_to_remove.set_index(df.index.name or "index")
            perc = len(rows_to_remove) * 100 / len(df)
            msg = bundle.get("dataset_train_diff_target_duplicates_fintech").format(
                perc, len(rows_to_remove), rows_to_remove.index.to_list()
            )
            if not silent:
                print(msg)
            if logger:
                logger.warning(msg)
            logger.info(f"Dataset shape before clean fintech duplicates: {df.shape}")
            df = df[~df.index.isin(rows_to_remove.index)]
            logger.info(f"Dataset shape after clean fintech duplicates: {df.shape}")
        else:
            # Indices in train and eval_set can be the same so we remove rows from them separately
            train = df.query(f"{EVAL_SET_INDEX} == 0")
            train_rows_to_remove = pd.merge(train.reset_index(), unique_keys_to_delete, on=personal_cols)
            train_rows_to_remove = train_rows_to_remove.set_index(train.index.name or "index")
            train_perc = len(train_rows_to_remove) * 100 / len(train)
            msg = bundle.get("dataset_train_diff_target_duplicates_fintech").format(
                train_perc, len(train_rows_to_remove), train_rows_to_remove.index.to_list()
            )
            if not silent:
                print(msg)
            if logger:
                logger.warning(msg)
            logger.info(f"Train dataset shape before clean fintech duplicates: {train.shape}")
            train = train[~train.index.isin(train_rows_to_remove.index)]
            logger.info(f"Train dataset shape after clean fintech duplicates: {train.shape}")

            evals = [df.query(f"{EVAL_SET_INDEX} == {i}") for i in df[EVAL_SET_INDEX].unique() if i != 0]
            new_evals = []
            for i, eval in enumerate(evals):
                eval_rows_to_remove = pd.merge(eval.reset_index(), unique_keys_to_delete, on=personal_cols)
                eval_rows_to_remove = eval_rows_to_remove.set_index(eval.index.name or "index")
                eval_perc = len(eval_rows_to_remove) * 100 / len(eval)
                msg = bundle.get("dataset_eval_diff_target_duplicates_fintech").format(
                    eval_perc, len(eval_rows_to_remove), i + 1, eval_rows_to_remove.index.to_list()
                )
                if not silent:
                    print(msg)
                if logger:
                    logger.warning(msg)
                logger.info(f"Eval {i + 1} dataset shape before clean fintech duplicates: {eval.shape}")
                eval = eval[~eval.index.isin(eval_rows_to_remove.index)]
                logger.info(f"Eval {i + 1} dataset shape after clean fintech duplicates: {eval.shape}")
                new_evals.append(eval)

            logger.info(f"Dataset shape before clean fintech duplicates: {df.shape}")
            df = pd.concat([train] + new_evals)
            logger.info(f"Dataset shape after clean fintech duplicates: {df.shape}")
    return df


def clean_full_duplicates(
    df: pd.DataFrame, logger: Optional[Logger] = None, silent=False, bundle: ResourceBundle = None
) -> pd.DataFrame:
    nrows = len(df)
    if nrows == 0:
        return df
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
            nrows_after_tgt_dedup = len(df.drop_duplicates(subset=unique_columns, keep=False))
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
