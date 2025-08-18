import logging
import numbers
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from upgini.metadata import (
    EVAL_SET_INDEX,
    SYSTEM_RECORD_ID,
    TARGET,
    CVType,
    ModelTaskType,
)
from upgini.resource_bundle import ResourceBundle, get_custom_bundle
from upgini.utils.config import (
    TS_DEFAULT_HIGH_FREQ_TRUNC_LENGTHS,
    TS_DEFAULT_LOW_FREQ_TRUNC_LENGTHS,
    TS_DEFAULT_TIME_UNIT_THRESHOLD,
    TS_MIN_DIFFERENT_IDS_RATIO,
    SampleConfig,
)
from upgini.utils.target_utils import balance_undersample
from upgini.utils.ts_utils import get_most_frequent_time_unit, trunc_datetime


@dataclass
class SampleColumns:
    date: str
    target: str
    ids: Optional[List[str]] = None
    eval_set_index: Optional[str] = None


def sample(
    df: pd.DataFrame,
    task_type: Optional[ModelTaskType],
    cv_type: Optional[CVType],
    sample_config: SampleConfig,
    sample_columns: SampleColumns,
    random_state: int = 42,
    balance: bool = True,
    force_downsampling: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> pd.DataFrame:
    if force_downsampling:
        return balance_undersample_forced(
            df,
            sample_columns.target,
            sample_columns.ids,
            sample_columns.date,
            task_type,
            cv_type,
            random_state,
            sample_config.force_sample_size,
            logger=logger,
            **kwargs,
        )

    if sample_columns.eval_set_index in df.columns:
        fit_sample_threshold = sample_config.fit_sample_threshold_with_eval_set
        fit_sample_rows = sample_config.fit_sample_rows_with_eval_set
    else:
        fit_sample_threshold = sample_config.fit_sample_threshold
        fit_sample_rows = sample_config.fit_sample_rows

    if cv_type is not None and cv_type.is_time_series():
        return sample_time_series_train_eval(
            df,
            sample_columns,
            sample_config.fit_sample_rows_ts,
            trim_threshold=fit_sample_threshold,
            max_rows=fit_sample_rows,
            random_state=random_state,
            logger=logger,
            **kwargs,
        )

    if task_type is not None and task_type.is_classification() and balance:
        df = balance_undersample(
            df=df,
            target_column=sample_columns.target,
            task_type=task_type,
            random_state=random_state,
            binary_min_sample_threshold=sample_config.binary_min_sample_threshold,
            multiclass_min_sample_threshold=sample_config.multiclass_min_sample_threshold,
            binary_bootstrap_loops=sample_config.binary_bootstrap_loops,
            multiclass_bootstrap_loops=sample_config.multiclass_bootstrap_loops,
            logger=logger,
            **kwargs,
        )

    # separate OOT
    oot_dfs = []
    other_dfs = []
    if EVAL_SET_INDEX in df.columns:
        for eval_set_index in df[EVAL_SET_INDEX].unique():
            eval_df = df[df[EVAL_SET_INDEX] == eval_set_index]
            if TARGET in eval_df.columns and eval_df[TARGET].isna().all():
                oot_dfs.append(eval_df)
            else:
                other_dfs.append(eval_df)
    if len(oot_dfs) > 0:
        oot_df = pd.concat(oot_dfs, ignore_index=False)
        df = pd.concat(other_dfs, ignore_index=False)
    else:
        oot_df = None

    num_samples = _num_samples(df)
    if num_samples > fit_sample_threshold:
        logger.info(
            f"Etalon has size {num_samples} more than threshold {fit_sample_threshold} "
            f"and will be downsampled to {fit_sample_rows}"
        )
        df = df.sample(n=fit_sample_rows, random_state=random_state)
        logger.info(f"Shape after threshold resampling: {df.shape}")

    if oot_df is not None:
        num_samples_oot = _num_samples(oot_df)
        if num_samples_oot > fit_sample_threshold:
            logger.info(
                f"OOT has size {num_samples_oot} more than threshold {fit_sample_threshold} "
                f"and will be downsampled to {fit_sample_rows}"
            )
            oot_df = oot_df.sample(n=fit_sample_rows, random_state=random_state)
        df = pd.concat([df, oot_df], ignore_index=False)

    logger.info(f"Dataset size after downsampling: {len(df)}")

    return df


def sample_time_series_train_eval(
    df: pd.DataFrame,
    sample_columns: SampleColumns,
    sample_size: int,
    trim_threshold: int,
    max_rows: int,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
    bundle: Optional[ResourceBundle] = None,
    **kwargs,
):
    if sample_columns.eval_set_index in df.columns:
        train_df = df[df[sample_columns.eval_set_index] == 0]
        eval_df = df[df[sample_columns.eval_set_index] > 0]
    else:
        train_df = df
        eval_df = None

    train_df = sample_time_series_trunc(
        train_df, sample_columns.ids, sample_columns.date, sample_size, random_state, logger=logger, **kwargs
    )
    if sample_columns.ids and eval_df is not None:
        missing_ids = (
            eval_df[~eval_df[sample_columns.ids].isin(np.unique(train_df[sample_columns.ids]))][sample_columns.ids]
            .dropna()
            .drop_duplicates()
            .values.tolist()
        )
        if missing_ids:
            bundle = bundle or get_custom_bundle()
            print(bundle.get("missing_ids_in_eval_set").format(missing_ids))
            eval_df = eval_df.merge(train_df[sample_columns.ids].drop_duplicates())

    if eval_df is not None:
        if len(eval_df) > trim_threshold - len(train_df):
            eval_df = sample_time_series_trunc(
                eval_df,
                sample_columns.ids,
                sample_columns.date,
                max_rows - len(train_df),
                random_state,
                logger=logger,
                **kwargs,
            )
        if logger is not None:
            logger.info(f"Eval set size: {len(eval_df)}")
        df = pd.concat([train_df, eval_df], ignore_index=False)

    elif len(train_df) > max_rows:
        df = sample_time_series_trunc(
            train_df,
            sample_columns.ids,
            sample_columns.date,
            max_rows,
            random_state,
            logger=logger,
            **kwargs,
        )
    else:
        df = train_df

    if logger is not None:
        logger.info(f"Train set size: {len(df)}")

    return df


def sample_time_series_trunc(
    df: pd.DataFrame,
    id_columns: Optional[List[str]],
    date_column: str,
    sample_size: int,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
    highfreq_trunc_lengths: List[pd.DateOffset] = TS_DEFAULT_HIGH_FREQ_TRUNC_LENGTHS,
    lowfreq_trunc_lengths: List[pd.DateOffset] = TS_DEFAULT_LOW_FREQ_TRUNC_LENGTHS,
    time_unit_threshold: pd.Timedelta = TS_DEFAULT_TIME_UNIT_THRESHOLD,
    **kwargs,
):
    if id_columns is None:
        id_columns = []
    # Convert date column to datetime
    dates_df = df[id_columns + [date_column]].copy().reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(dates_df[date_column]):
        dates_df[date_column] = pd.to_datetime(dates_df[date_column], unit="ms")
    else:
        dates_df[date_column] = pd.to_datetime(dates_df[date_column])

    time_unit = get_most_frequent_time_unit(dates_df, id_columns, date_column)
    if logger is not None:
        logger.info(f"Time unit: {time_unit}")

    if time_unit is None:
        if logger is not None:
            logger.info("Cannot detect time unit, returning original dataset")
        return df

    if time_unit < time_unit_threshold:
        for trunc_length in highfreq_trunc_lengths:
            sampled_df = trunc_datetime(dates_df, id_columns, date_column, trunc_length, logger=logger)
            if len(sampled_df) <= sample_size:
                break
        if len(sampled_df) > sample_size:
            sampled_df = sample_time_series(
                sampled_df, id_columns, date_column, sample_size, random_state, logger=logger, **kwargs
            )
    else:
        for trunc_length in lowfreq_trunc_lengths:
            sampled_df = trunc_datetime(dates_df, id_columns, date_column, trunc_length, logger=logger)
            if len(sampled_df) <= sample_size:
                break
        if len(sampled_df) > sample_size:
            sampled_df = sample_time_series(
                sampled_df, id_columns, date_column, sample_size, random_state, logger=logger, **kwargs
            )

    return df.iloc[sampled_df.index]


def sample_time_series(
    df: pd.DataFrame,
    id_columns: List[str],
    date_column: str,
    sample_size: int,
    random_state: int = 42,
    min_different_ids_ratio: float = TS_MIN_DIFFERENT_IDS_RATIO,
    prefer_recent_dates: bool = True,
    logger: Optional[logging.Logger] = None,
    **kwargs,
):
    def ensure_tuple(x):
        return tuple([x]) if not isinstance(x, tuple) else x

    random_state = np.random.RandomState(random_state)

    if not id_columns:
        id_columns = [date_column]
    ids_sort = df.groupby(id_columns)[date_column].aggregate(["max", "count"]).T.to_dict()
    ids_sort = {
        ensure_tuple(k): (
            (v["max"], v["count"], random_state.rand()) if prefer_recent_dates else (v["count"], random_state.rand())
        )
        for k, v in ids_sort.items()
    }
    id_counts = df[id_columns].value_counts()
    id_counts.index = [ensure_tuple(i) for i in id_counts.index]
    id_counts = id_counts.sort_index(key=lambda x: [ids_sort[y] for y in x], ascending=False).cumsum()
    id_counts = id_counts[id_counts <= sample_size]
    min_different_ids = max(int(len(df[id_columns].drop_duplicates()) * min_different_ids_ratio), 1)

    def id_mask(sample_index: pd.Index) -> pd.Index:
        if isinstance(sample_index, pd.MultiIndex):
            return pd.MultiIndex.from_frame(df[id_columns]).isin(sample_index)
        else:
            return df[id_columns[0]].isin(sample_index)

    if len(id_counts) < min_different_ids:
        if logger is not None:
            logger.info(
                f"Different ids count {len(id_counts)} for sample size {sample_size}"
                f" is less than min different ids {min_different_ids}, sampling time window"
            )
        date_counts = df.groupby(id_columns)[date_column].nunique().sort_values(ascending=False)
        ids_to_sample = date_counts.index[:min_different_ids] if len(id_counts) > 0 else date_counts.index
        mask = id_mask(ids_to_sample)
        df = df[mask]
        sample_date_counts = df[date_column].value_counts().sort_index(ascending=False).cumsum()
        sample_date_counts = sample_date_counts[sample_date_counts <= sample_size]
        df = df[df[date_column].isin(sample_date_counts.index)]
    else:
        if len(id_columns) > 1:
            id_counts.index = pd.MultiIndex.from_tuples(id_counts.index)
        else:
            id_counts.index = [i[0] for i in id_counts.index]
        mask = id_mask(id_counts.index)
        df = df[mask]

    return df


def balance_undersample_forced(
    df: pd.DataFrame,
    sample_columns: SampleColumns,
    task_type: ModelTaskType,
    cv_type: Optional[CVType],
    random_state: int,
    sample_size: int = 7000,
    logger: Optional[logging.Logger] = None,
    bundle: Optional[ResourceBundle] = None,
    warning_callback: Optional[Callable] = None,
):
    if len(df) <= sample_size:
        return df

    if logger is None:
        logger = logging.getLogger("muted_logger")
        logger.setLevel("FATAL")
    bundle = bundle or get_custom_bundle()
    if SYSTEM_RECORD_ID not in df.columns:
        raise Exception("System record id must be presented for undersampling")

    msg = bundle.get("forced_balance_undersample")
    logger.info(msg)
    if warning_callback is not None:
        warning_callback(msg)

    target = df[sample_columns.target].copy()

    vc = target.value_counts()
    max_class_value = vc.index[0]
    min_class_value = vc.index[len(vc) - 1]
    max_class_count = vc[max_class_value]
    min_class_count = vc[min_class_value]

    resampled_data = df
    df = df.copy().sort_values(by=SYSTEM_RECORD_ID)
    if cv_type is not None and cv_type.is_time_series():
        logger.warning(f"Sampling time series dataset from {len(df)} to {sample_size}")
        resampled_data = sample_time_series_train_eval(
            df,
            sample_columns=sample_columns,
            sample_size=sample_size,
            trim_threshold=sample_size,
            max_rows=sample_size,
            random_state=random_state,
            logger=logger,
        )
    elif task_type in [ModelTaskType.MULTICLASS, ModelTaskType.REGRESSION]:
        logger.warning(f"Sampling dataset from {len(df)} to {sample_size}")
        resampled_data = df.sample(n=sample_size, random_state=random_state)
    else:
        msg = bundle.get("imbalanced_target").format(min_class_value, min_class_count)
        logger.warning(msg)

        # fill up to min_sample_threshold by majority class
        minority_class = df[df[sample_columns.target] == min_class_value]
        majority_class = df[df[sample_columns.target] != min_class_value]
        logger.info(
            f"Min class count: {min_class_count}. Max class count: {max_class_count}."
            f" Rebalance sample size: {sample_size}"
        )
        if len(minority_class) > (sample_size / 2):
            sampled_minority_class = minority_class.sample(n=int(sample_size / 2), random_state=random_state)
        else:
            sampled_minority_class = minority_class

        if len(majority_class) > (sample_size) / 2:
            sampled_majority_class = majority_class.sample(n=int(sample_size / 2), random_state=random_state)

        resampled_data = df[
            (df[SYSTEM_RECORD_ID].isin(sampled_minority_class[SYSTEM_RECORD_ID]))
            | (df[SYSTEM_RECORD_ID].isin(sampled_majority_class[SYSTEM_RECORD_ID]))
        ]

    logger.info(f"Shape after forced rebalance resampling: {resampled_data}")
    return resampled_data


def _num_samples(x):
    """Return number of samples in array-like x."""
    if x is None:
        return 0
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error
