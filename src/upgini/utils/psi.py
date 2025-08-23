import itertools
import logging
import operator
from functools import reduce
from typing import Callable, Dict, Optional

import more_itertools
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel

from upgini.metadata import TARGET, ModelTaskType


class StabilityParams(BaseModel):
    threshold: float = 999
    n_intervals: int = 12
    min_intervals: int = 10
    max_intervals: Optional[int] = None
    min_values_in_interval: Optional[int] = None
    n_bins: int = 10
    min_values_in_bin: Optional[int] = None
    cat_top_pct: float = 0.7
    agg: str = "max"


DEFAULT_TARGET_PARAMS = StabilityParams(
    n_intervals=12,
    min_intervals=10,
    max_intervals=None,
    min_values_in_interval=None,
    n_bins=5,
)

DEFAULT_FEATURES_PARAMS = StabilityParams(
    n_intervals=12,
    min_intervals=10,
    max_intervals=None,
    min_values_in_interval=None,
    n_bins=10,
)


def calculate_sparsity_psi(
    df: pd.DataFrame,
    cat_features: list[str],
    date_column: str,
    logger: logging.Logger,
    model_task_type: ModelTaskType,
    stability_agg_func: str | None = None,
    psi_features_params: StabilityParams = DEFAULT_FEATURES_PARAMS,
    psi_target_params: StabilityParams = DEFAULT_TARGET_PARAMS,
) -> Dict[str, float]:
    sparse_features = df.columns[df.isna().sum() > 0].to_list()
    if len(sparse_features) > 0:
        logger.info(f"Calculating sparsity stability for {len(sparse_features)} sparse features")
        sparse_df = df[sparse_features].notna()
        sparse_df[date_column] = df[date_column]
        return calculate_features_psi(
            sparse_df,
            cat_features,
            date_column,
            logger,
            model_task_type,
            stability_agg_func,
            psi_target_params,
            psi_features_params,
        )
    return {}


def calculate_features_psi(
    df: pd.DataFrame,
    cat_features: list[str],
    date_column: str,
    logger: logging.Logger,
    model_task_type: ModelTaskType,
    stability_agg_func: str | None = None,
    psi_features_params: StabilityParams = DEFAULT_FEATURES_PARAMS,
    psi_target_params: StabilityParams = DEFAULT_TARGET_PARAMS,
) -> dict[str, float]:
    empty_res = {col: 0.0 for col in df.columns if col not in [TARGET, date_column]}

    # Filter out rows with missing dates
    df = df[df[date_column].notna()].copy()

    n_months = pd.to_datetime(df[date_column], unit="ms").dt.month.nunique()

    if TARGET in df.columns:
        psi_target_params.n_intervals = min(
            psi_target_params.max_intervals or np.inf, max(psi_target_params.min_intervals, n_months)
        )
        logger.info(f"Setting {psi_target_params.n_intervals} intervals for target PSI check")

        logger.info(f"Calculating target PSI for {psi_target_params.n_intervals} intervals")
        reference_mask, current_masks = _split_intervals(df, date_column, psi_target_params.n_intervals, logger)

        if psi_target_params.min_values_in_interval is not None and any(
            len(mask) < psi_target_params.min_values_in_interval
            for mask in itertools.chain(current_masks, [reference_mask])
        ):
            logger.info(
                f"Some intervals have less than {psi_target_params.min_values_in_interval} values. Skip PSI check"
            )
            return empty_res

        target_agg_func = _get_agg_func(stability_agg_func or psi_target_params.agg)
        logger.info(f"Calculating target PSI with agg function {target_agg_func}")
        target_psi = _stability_agg(
            [df[TARGET][cur] for cur in current_masks],
            reference_data=df[TARGET][reference_mask],
            is_numerical=model_task_type == ModelTaskType.REGRESSION,
            min_values_in_bin=psi_target_params.min_values_in_bin,
            n_bins=psi_target_params.n_bins,
            cat_top_pct=psi_target_params.cat_top_pct,
            agg_func=target_agg_func,
        )
        if target_psi is None or np.isnan(target_psi):
            logger.info("Cannot determine target PSI. Skip feature PSI check")
            return empty_res

        if target_psi > psi_target_params.threshold:
            logger.info(
                f"Target PSI {target_psi} is more than threshold {psi_target_params.threshold}. Skip feature PSI check"
            )
            return empty_res

    psi_features_params.n_intervals = min(
        psi_features_params.max_intervals or np.inf, max(psi_features_params.min_intervals, n_months)
    )
    logger.info(f"Setting {psi_features_params.n_intervals} intervals for features PSI check")

    logger.info(f"Calculating PSI for {len(df.columns)} features")
    reference_mask, current_masks = _split_intervals(df, date_column, psi_features_params.n_intervals, logger)
    features_agg_func = _get_agg_func(stability_agg_func or psi_features_params.agg)
    logger.info(f"Calculating features PSI with agg function {features_agg_func}")
    psi_values = [
        _stability_agg(
            [df[feature][cur] for cur in current_masks],
            reference_data=df[feature][reference_mask],
            is_numerical=feature not in cat_features,
            min_values_in_bin=psi_features_params.min_values_in_bin,
            n_bins=psi_features_params.n_bins,
            cat_top_pct=psi_features_params.cat_top_pct,
            agg_func=features_agg_func,
        )
        for feature in df.columns
        if feature not in [TARGET, date_column]
    ]
    return {feature: psi for feature, psi in zip(df.columns, psi_values)}


def _split_intervals(
    df: pd.DataFrame, date_column: str, n_intervals: int, logger: logging.Logger
) -> tuple[pd.Series, list[pd.Series]]:
    date_series = df[date_column]

    # Check if we have enough unique values for the requested number of intervals
    unique_values = date_series.nunique()

    # If we have fewer unique values than requested intervals, adjust n_intervals
    if unique_values < n_intervals:
        logger.warning(f"Date column '{date_column}' has only {unique_values} unique values")

    time_intervals = pd.qcut(date_series, q=n_intervals, duplicates="drop")
    interval_labels = time_intervals.unique()
    reference_mask = time_intervals == interval_labels[0]
    current_masks = [time_intervals == label for label in interval_labels[1:]]
    return reference_mask, current_masks


def _get_agg_func(agg: str):
    np_agg = getattr(np, agg, None)
    if np_agg is None and agg.startswith("q"):
        q = int(agg[1:])
        return lambda x: np.quantile(list(x), q / 100, method="higher")
    return np_agg


def _psi(reference_percent: np.ndarray, current_percent: np.ndarray) -> float:
    return np.sum((reference_percent - current_percent) * np.log(reference_percent / current_percent))


def _stability_agg(
    current_data: list[pd.Series],
    reference_data: pd.Series,
    is_numerical: bool = True,
    min_values_in_bin: int | None = None,
    n_bins: int = 10,
    cat_top_pct: float = 0.7,
    agg_func: Callable = max,
) -> float | None:
    """Calculate the PSI
    Args:
        current_data: current data
        reference_data: reference data
        is_numerical: whether the feature is numerical
        reference_ratio: ratio of current data to use as reference if reference_data is not provided
        min_values_in_bin: minimum number of values in a bin to calculate PSI
        n_bins: number of bins to use for numerical features
    Returns:
        psi_value: calculated PSI
    """
    reference, current = _get_binned_data(reference_data, current_data, is_numerical, n_bins, cat_top_pct)

    if len(reference) == 0 or len(current) == 0:
        return None

    nonempty_current = [i for i, c in enumerate(current) if len(c) > 0]
    current = [current[i] for i in nonempty_current]
    current_data = [current_data[i] for i in nonempty_current]

    if len(current) == 0:
        return None

    if min_values_in_bin is not None and (
        np.array(reference).min() < min_values_in_bin or any(np.array(c).min() < min_values_in_bin for c in current)
    ):
        return None

    reference = _fill_zeroes(reference / len(reference_data))
    current = [_fill_zeroes(c / len(d)) for c, d in zip(current, current_data)]

    psi_value = agg_func([_psi(reference, c) for c in current])

    return float(psi_value)


def _get_binned_data(
    reference_data: pd.Series,
    current_data: list[pd.Series],
    is_numerical: bool,
    n_bins: int,
    cat_top_pct: float,
):
    """Split variable into n buckets based on reference quantiles
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        n: number of quantiles
    Returns:
        reference_counts: number of records in each bucket for reference
        current_counts: number of records in each bucket for current
    """
    n_vals = reference_data.nunique()

    if is_numerical and n_vals > 20:
        bins = _get_bin_edges(reference_data, n_bins)
        reference_counts = np.histogram(reference_data, bins)[0]
        current_counts = [np.histogram(d, bins)[0] for d in current_data]

    else:
        keys = _get_unique_not_nan_values_list_from_series([reference_data] + current_data)
        ref_feature_dict = {**dict.fromkeys(keys, 0), **dict(reference_data.value_counts())}
        current_feature_dict = [{**dict.fromkeys(keys, 0), **dict(d.value_counts())} for d in current_data]
        key_dict = more_itertools.map_reduce(
            itertools.chain(ref_feature_dict.items(), *(d.items() for d in current_feature_dict)),
            keyfunc=operator.itemgetter(0),
            valuefunc=operator.itemgetter(1),
            reducefunc=sum,
        )
        key_dict = pd.Series(key_dict)
        keys = key_dict.index[key_dict.rank(pct=True) >= cat_top_pct]
        reference_counts = np.array([ref_feature_dict[key] for key in keys])
        current_counts = [np.array([current_feature_dict[i][key] for key in keys]) for i in range(len(current_data))]

    reference_counts = np.append(reference_counts, reference_data.isna().sum())
    current_counts = [np.append(d, current_data[i].isna().sum()) for i, d in enumerate(current_counts)]

    return reference_counts, current_counts


def _fill_zeroes(percents: np.ndarray) -> np.ndarray:
    eps = 0.0001
    if (percents == 0).all():
        np.place(percents, percents == 0, eps)
    else:
        min_value = min(percents[percents != 0])
        if min_value <= eps:
            np.place(percents, percents == 0, eps)
        else:
            np.place(percents, percents == 0, min_value / 10**6)
    return percents


def _get_bin_edges(data: pd.Series, n_bins: int) -> np.ndarray:
    bins = np.nanquantile(data, np.linspace(0, 1, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def _get_unique_not_nan_values_list_from_series(series: list[pd.Series]) -> list:
    """Get unique values from current and reference series, drop NaNs"""
    return list(reduce(set.union, (set(s.dropna().unique()) for s in series)))
