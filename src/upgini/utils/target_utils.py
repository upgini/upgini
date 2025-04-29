import logging
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from upgini.errors import ValidationError
from upgini.metadata import SYSTEM_RECORD_ID, CVType, ModelTaskType
from upgini.resource_bundle import ResourceBundle, bundle, get_custom_bundle
from upgini.sampler.random_under_sampler import RandomUnderSampler
from upgini.utils.ts_utils import get_most_frequent_time_unit, trunc_datetime

TS_MIN_DIFFERENT_IDS_RATIO = 0.2


def prepare_target(y: Union[pd.Series, np.ndarray], target_type: ModelTaskType) -> Union[pd.Series, np.ndarray]:
    if target_type != ModelTaskType.REGRESSION or (not is_numeric_dtype(y) and not is_datetime64_any_dtype(y)):
        if isinstance(y, pd.Series):
            y = y.astype(str).astype("category").cat.codes
        elif isinstance(y, np.ndarray):
            y = pd.Series(y).astype(str).astype("category").cat.codes.values

    return y


def define_task(
    y: pd.Series, has_date: bool = False, logger: Optional[logging.Logger] = None, silent: bool = False
) -> ModelTaskType:
    if logger is None:
        logger = logging.getLogger()

    # Replace inf and -inf with NaN to handle extreme values correctly
    y = y.replace([np.inf, -np.inf], np.nan, inplace=False)

    # Drop NaN values from the target
    target = y.dropna()

    # Check if target is numeric and finite
    if is_numeric_dtype(target):
        target = target.loc[np.isfinite(target)]
    else:
        # If not numeric, drop empty strings as well
        target = target.loc[target != ""]

    # Raise error if there are no valid values left in the target
    if len(target) == 0:
        raise ValidationError(bundle.get("empty_target"))

    # Count unique values in the target
    target_items = target.nunique()

    # Raise error if all target values are the same
    if target_items == 1:
        raise ValidationError(bundle.get("dataset_constant_target"))

    reason = ""  # Will store the reason for selecting the task type

    # Binary classification case: exactly two unique values
    if target_items == 2:
        task = ModelTaskType.BINARY
        reason = bundle.get("binary_target_reason")
    else:
        # Attempt to convert target to numeric
        try:
            target = pd.to_numeric(target)
            is_numeric = True
        except Exception:
            is_numeric = False

        # If target cannot be converted to numeric, assume multiclass classification
        if not is_numeric:
            task = ModelTaskType.MULTICLASS
            reason = bundle.get("non_numeric_multiclass_reason")
        else:
            # Multiclass classification: few unique values and integer encoding
            if target.nunique() <= 50 and is_int_encoding(target.unique()):
                task = ModelTaskType.MULTICLASS
                reason = bundle.get("few_unique_label_multiclass_reason")
            # Regression case: if there is date, assume regression
            elif has_date:
                task = ModelTaskType.REGRESSION
                reason = bundle.get("date_search_key_regression_reason")
            else:
                # Remove zero values and recalculate unique ratio
                non_zero_target = target[target != 0]
                target_items = non_zero_target.nunique()
                target_ratio = target_items / len(non_zero_target)

                # Use unique_ratio to determine whether to classify as regression or multiclass
                if (
                    (target.dtype.kind == "f" and np.any(target != target.astype(int)))  # Non-integer float values
                    or target_items > 50
                    or target_ratio > 0.2  # If non-zero values have high ratio of uniqueness
                ):
                    task = ModelTaskType.REGRESSION
                    reason = bundle.get("many_unique_label_regression_reason")
                else:
                    task = ModelTaskType.MULTICLASS
                    reason = bundle.get("limited_int_multiclass_reason")

    # Log or print the reason for the selected task type
    logger.info(f"Detected task type: {task} (Reason: {reason})")

    # Print task type and reason if silent mode is off
    if not silent:
        print(bundle.get("target_type_detected").format(task, reason))

    return task


def is_int_encoding(unique_values):
    return set(unique_values) == set(range(len(unique_values))) or set(unique_values) == set(
        range(1, len(unique_values) + 1)
    )


def balance_undersample(
    df: pd.DataFrame,
    target_column: str,
    task_type: ModelTaskType,
    random_state: int,
    binary_min_sample_threshold: int = 5000,
    multiclass_min_sample_threshold: int = 25000,
    binary_bootstrap_loops: int = 5,
    multiclass_bootstrap_loops: int = 2,
    logger: Optional[logging.Logger] = None,
    bundle: Optional[ResourceBundle] = None,
    warning_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger("muted_logger")
        logger.setLevel("FATAL")
    bundle = bundle or get_custom_bundle()
    if SYSTEM_RECORD_ID not in df.columns:
        raise Exception("System record id must be presented for undersampling")

    target = df[target_column].copy()

    vc = target.value_counts()
    max_class_value = vc.index[0]
    min_class_value = vc.index[len(vc) - 1]
    max_class_count = vc[max_class_value]
    min_class_count = vc[min_class_value]
    num_classes = len(vc)

    resampled_data = df
    df = df.copy().sort_values(by=SYSTEM_RECORD_ID)
    if task_type == ModelTaskType.MULTICLASS:
        if len(df) > multiclass_min_sample_threshold and max_class_count > (
            min_class_count * multiclass_bootstrap_loops
        ):

            msg = bundle.get("imbalanced_target").format(min_class_value, min_class_count)
            logger.warning(msg)
            if warning_callback is not None:
                warning_callback(msg)

            sample_strategy = dict()
            for class_value in vc.index:
                if class_value == min_class_value:
                    continue
                class_count = vc[class_value]
                sample_size = min(
                    class_count,
                    multiclass_bootstrap_loops
                    * (
                        min_class_count
                        + max((multiclass_min_sample_threshold - num_classes * min_class_count) / (num_classes - 1), 0)
                    ),
                )
                sample_strategy[class_value] = int(sample_size)
            logger.info(f"Rebalance sample strategy: {sample_strategy}. Min class count: {min_class_count}")
            sampler = RandomUnderSampler(sampling_strategy=sample_strategy, random_state=random_state)
            X = df[SYSTEM_RECORD_ID]
            X = X.to_frame(SYSTEM_RECORD_ID)
            new_x, _ = sampler.fit_resample(X, target)  # type: ignore

            resampled_data = df[df[SYSTEM_RECORD_ID].isin(new_x[SYSTEM_RECORD_ID])]
    elif len(df) > binary_min_sample_threshold:
        msg = bundle.get("imbalanced_target").format(min_class_value, min_class_count)
        logger.warning(msg)
        if warning_callback is not None:
            warning_callback(msg)

        # fill up to min_sample_threshold by majority class
        minority_class = df[df[target_column] == min_class_value]
        majority_class = df[df[target_column] != min_class_value]
        sample_size = min(
            max_class_count,
            binary_bootstrap_loops * (min_class_count + max(binary_min_sample_threshold - 2 * min_class_count, 0)),
        )
        logger.info(
            f"Min class count: {min_class_count}. Max class count: {max_class_count}."
            f" Rebalance sample size: {sample_size}"
        )
        sampled_majority_class = majority_class.sample(n=sample_size, random_state=random_state)
        resampled_data = df[
            (df[SYSTEM_RECORD_ID].isin(minority_class[SYSTEM_RECORD_ID]))
            | (df[SYSTEM_RECORD_ID].isin(sampled_majority_class[SYSTEM_RECORD_ID]))
        ]

    logger.info(f"Shape after rebalance resampling: {resampled_data}")
    return resampled_data


def balance_undersample_forced(
    df: pd.DataFrame,
    target_column: str,
    id_columns: Optional[List[str]],
    date_column: str,
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

    target = df[target_column].copy()

    vc = target.value_counts()
    max_class_value = vc.index[0]
    min_class_value = vc.index[len(vc) - 1]
    max_class_count = vc[max_class_value]
    min_class_count = vc[min_class_value]

    resampled_data = df
    df = df.copy().sort_values(by=SYSTEM_RECORD_ID)
    if cv_type is not None and cv_type.is_time_series():
        logger.warning(f"Sampling time series dataset from {len(df)} to {sample_size}")
        resampled_data = balance_undersample_time_series_trunc(
            df,
            id_columns=id_columns,
            date_column=date_column,
            sample_size=sample_size,
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
        minority_class = df[df[target_column] == min_class_value]
        majority_class = df[df[target_column] != min_class_value]
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


DEFAULT_HIGH_FREQ_TRUNC_LENGTHS = [pd.DateOffset(years=2, months=6), pd.DateOffset(years=2, days=7)]
DEFAULT_LOW_FREQ_TRUNC_LENGTHS = [pd.DateOffset(years=7), pd.DateOffset(years=5)]
DEFAULT_TIME_UNIT_THRESHOLD = pd.Timedelta(weeks=4)


def balance_undersample_time_series_trunc(
    df: pd.DataFrame,
    id_columns: Optional[List[str]],
    date_column: str,
    sample_size: int,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
    highfreq_trunc_lengths: List[pd.DateOffset] = DEFAULT_HIGH_FREQ_TRUNC_LENGTHS,
    lowfreq_trunc_lengths: List[pd.DateOffset] = DEFAULT_LOW_FREQ_TRUNC_LENGTHS,
    time_unit_threshold: pd.Timedelta = DEFAULT_TIME_UNIT_THRESHOLD,
    **kwargs,
):
    if id_columns is None:
        id_columns = []
    # Convert date column to datetime
    dates_df = df[id_columns + [date_column]].copy()
    dates_df[date_column] = pd.to_datetime(dates_df[date_column], unit="ms")

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
            sampled_df = balance_undersample_time_series(
                sampled_df, id_columns, date_column, sample_size, random_state, logger=logger, **kwargs
            )
    else:
        for trunc_length in lowfreq_trunc_lengths:
            sampled_df = trunc_datetime(dates_df, id_columns, date_column, trunc_length, logger=logger)
            if len(sampled_df) <= sample_size:
                break
        if len(sampled_df) > sample_size:
            sampled_df = balance_undersample_time_series(
                sampled_df, id_columns, date_column, sample_size, random_state, logger=logger, **kwargs
            )

    return df.loc[sampled_df.index]


def balance_undersample_time_series(
    df: pd.DataFrame,
    id_columns: List[str],
    date_column: str,
    sample_size: int,
    random_state: int = 42,
    min_different_ids_ratio: float = TS_MIN_DIFFERENT_IDS_RATIO,
    prefer_recent_dates: bool = True,
    logger: Optional[logging.Logger] = None,
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


def calculate_psi(expected: pd.Series, actual: pd.Series) -> Union[float, Exception]:
    try:
        df = pd.concat([expected, actual])

        if is_bool_dtype(df):
            df = np.where(df, 1, 0)

        # Define the bins for the target variable
        df_min = df.min()
        df_max = df.max()
        bins = [df_min, (df_min + df_max) / 2, df_max]

        # Calculate the base distribution
        train_distribution = expected.value_counts(bins=bins, normalize=True).sort_index().values

        # Calculate the target distribution
        test_distribution = actual.value_counts(bins=bins, normalize=True).sort_index().values

        # Calculate the PSI
        return np.sum((train_distribution - test_distribution) * np.log(train_distribution / test_distribution))
    except Exception as e:
        return e
