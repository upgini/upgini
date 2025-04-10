import hashlib
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from psutil import cpu_count
from scipy.stats import skew, spearmanr

from upgini.metadata import ModelTaskType, SearchKey
from upgini.utils import mstats


def sort_columns(
    df: pd.DataFrame,
    target_column: Union[str, pd.Series],
    search_keys: Dict[str, SearchKey],
    model_task_type: ModelTaskType,
    exclude_columns: Optional[List[str]] = None,
    sort_all_columns: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    if exclude_columns is None:
        exclude_columns = []
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.FATAL)
    df = df.copy()  # avoid side effects
    search_keys = {k: v for k, v in search_keys.items() if v != SearchKey.CUSTOM_KEY}

    # Check multiple search keys
    search_key_values = list(search_keys.values())
    has_duplicate_search_keys = len(search_key_values) != len(set(search_key_values))
    if has_duplicate_search_keys:
        logger.warning(f"WARNING: Found duplicate SearchKey values in search_keys: {search_keys}")

    sorted_keys = sorted(search_keys.keys(), key=lambda x: str(search_keys.get(x)))
    sorted_keys = [k for k in sorted_keys if k in df.columns and k not in exclude_columns]

    duplicate_names = df.columns[df.columns.duplicated()].unique()
    if len(duplicate_names) > 0:
        logger.warning(f"WARNING: Found columns with duplicate names: {list(duplicate_names)}")
        df = df[list(set(df.columns))]

    other_columns = sorted(
        [
            c
            for c in df.columns
            if c not in sorted_keys and c not in exclude_columns and (df[c].nunique() > 1 or sort_all_columns)
        ]
    )
    target = target_column if isinstance(target_column, pd.Series) else df[target_column]
    target = prepare_target(target, model_task_type)
    sort_dict = get_sort_columns_dict(
        df[sorted_keys + other_columns], target, sorted_keys, sort_all_columns=sort_all_columns
    )
    other_columns = [c for c in other_columns if c in sort_dict]
    columns_for_sort = sorted_keys + sorted(other_columns, key=lambda e: sort_dict[e], reverse=True)
    return columns_for_sort


def get_sort_columns_dict(
    df: pd.DataFrame,
    target: pd.Series,
    sorted_keys: List[str],
    n_jobs: Optional[int] = None,
    sort_all_columns: bool = False,
) -> Dict[str, Any]:
    string_features = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in sorted_keys]
    columns_for_sort = [c for c in df.columns if c not in sorted_keys + string_features]
    if len(string_features) > 0:
        if len(df) > len(df.drop(columns=string_features).drop_duplicates()) or sort_all_columns:
            # factorize string features
            df = df.copy()
            for c in string_features:
                df = df.assign(**{c: pd.factorize(df[c], sort=True)[0].astype(int)})
            columns_for_sort.extend(string_features)

    if len(columns_for_sort) == 0:
        return {}

    df = df[columns_for_sort]
    df_with_target = pd.concat([df, target], axis=1)
    # Drop rows where target is NaN
    df_with_target = df_with_target.loc[~target.isna()]
    df = df_with_target.iloc[:, :-1]
    target = df_with_target.iloc[:, -1]
    df = df.fillna(df.apply(lambda x: int(x.mean()) if pd.api.types.is_integer_dtype(x) else x.mean()))
    omit_nan = False
    hashes = [hash_series(df[col]) for col in columns_for_sort]
    df = np.asarray(df, dtype=np.float32)
    correlations = get_sort_columns_correlations(df, target, omit_nan, n_jobs)

    sort_dict = {col: (corr, h) for col, corr, h in zip(columns_for_sort, correlations, hashes)}
    return sort_dict


def get_sort_columns_correlations(df: np.ndarray, target: pd.Series, omit_nan: bool, n_jobs: Optional[int] = None):
    target_correlations = get_target_correlations(df, target, omit_nan, n_jobs, precision=7)

    return np.max(target_correlations, axis=0)


def get_target_correlations(
    df: np.ndarray, target: pd.Series, omit_nan: bool, n_jobs: Optional[int] = None, precision: int = 15
):
    df = np.asarray(df, dtype=np.float32)
    target_correlations = np.zeros((2, df.shape[1]))
    target_correlations[0, :] = np.nan_to_num(
        calculate_spearman_corr_with_target(df, target, omit_nan, n_jobs), copy=False
    )
    target_correlations[1, :] = np.nan_to_num(np.abs(np.corrcoef(df.T, target.T, rowvar=True)[-1, :-1]))

    target_correlations = np.trunc(target_correlations * 10**precision) / (10**precision)

    return target_correlations


def calculate_spearman_corr_with_target(
    X: Union[pd.DataFrame, np.ndarray], y: pd.Series, omit_nan: bool = False, n_jobs: Optional[int] = None
) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = np.asarray(X, dtype=np.float32)

    if X.size == 0:
        return np.ndarray(shape=(0,))

    all_correlations = np.zeros(X.shape[1])
    all_correlations.fill(np.nan)
    cols2calc = np.where([c.size > 0 and not (c == c[0]).all() for c in X.T])[0]

    if omit_nan:
        results = Parallel(n_jobs=n_jobs or cpu_count(logical=False))(
            delayed(mstats.spearmanr)(
                X[:, i],
                y,
                nan_policy="omit",
                axis=0,
            )
            for i in cols2calc
        )
        target_correlations = np.array([abs(res.correlation) for res in results])
    else:
        cols2calc = cols2calc[np.where(~np.isnan(X[:, cols2calc]).any(axis=0))[0]]
        target_correlations = calculate_spearman(X[:, cols2calc], y, nan_policy="raise")
        if isinstance(target_correlations, float):
            target_correlations = np.abs([target_correlations])
        else:
            target_correlations = np.abs(target_correlations)[-1, :-1]

    all_correlations[cols2calc] = target_correlations

    return all_correlations


def calculate_spearman(X: np.ndarray, y: Optional[pd.Series], nan_policy: str):
    features_num = X.shape[1]
    if y is not None:
        features_num += 1

    if features_num < 2:
        return 1.0
    else:
        return spearmanr(X, y, nan_policy=nan_policy).correlation


def hash_series(series: pd.Series) -> int:
    return int(hashlib.sha256(pd.util.hash_pandas_object(series, index=True).values).hexdigest(), 16)


def prepare_target(target: pd.Series, model_task_type: ModelTaskType) -> pd.Series:
    target_name = target.name
    if model_task_type != ModelTaskType.REGRESSION or (
        not is_numeric_dtype(target) and not is_datetime64_any_dtype(target)
    ):
        target = target.astype(str).astype("category").cat.codes

    elif model_task_type == ModelTaskType.REGRESSION:
        skewness = round(abs(skew(target)), 2)
        if (target.min() >= 0) and (skewness >= 0.9):
            target = np.log1p(target)

    return pd.Series(target, name=target_name)
