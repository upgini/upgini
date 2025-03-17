import functools
import hashlib
from typing import Any, Callable
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from psutil import cpu_count
from upgini.utils import mstats


# def ...


def _sort_list(lst: list, dct: dict) -> list:
    return sorted(lst, key=lambda e: dct[e], reverse=True)


def sort_by_dict_desc(dct: dict) -> Callable | None:
    if dct is None:
        return None
    return functools.partial(_sort_list, dct=dct)


def get_sort_columns_dict(
    df: pd.DataFrame,
    target: pd.Series,
    omit_nan: bool,
    n_jobs: int | None = None,
) -> dict[str, Any]:

    non_number_corrs = {col: (0.0, hash_series(df[col])) for col in df.select_dtypes(exclude=[np.number]).columns}
    df = df.select_dtypes(include=[np.number])

    columns = df.columns
    hashes = [hash_series(df[col]) for col in columns]
    df = np.asarray(df, dtype=np.float32)
    correlations = get_sort_columns_correlations(df, target, omit_nan, n_jobs)

    sort_dict = {col: (corr, h) for col, corr, h in zip(columns, correlations, hashes)}
    sort_dict.update(non_number_corrs)
    return sort_dict


def get_sort_columns_correlations(df: np.ndarray, target: pd.Series, omit_nan: bool, n_jobs: int | None = None):
    target_correlations = get_target_correlations(df, target, omit_nan, n_jobs, precision=7)

    return np.max(target_correlations, axis=0)


def get_target_correlations(
    df: np.ndarray, target: pd.Series, omit_nan: bool, n_jobs: int | None = None, precision: int = 15
):
    df = np.asarray(df, dtype=np.float32)
    target_correlations = np.zeros((2, df.shape[1]))
    target_correlations[0, :] = np.nan_to_num(
        calculate_spearman_corr_with_target(df, target, omit_nan, n_jobs), copy=False
    )
    target_correlations[1, :] = np.nan_to_num(np.abs(np.corrcoef(df.T, target.T, rowvar=True)[-1, :-1]))

    target_correlations = np.trunc(target_correlations * 10**precision) / (10**precision)

    return target_correlations


def corr_dict_from_sort_dict(sort_dict: dict[str, tuple[float, int]]) -> dict[str, float]:
    return {k: v[0] for k, v in sort_dict.items()}


def calculate_spearman_corr_with_target(
    X: pd.DataFrame | np.ndarray, y: pd.Series, omit_nan: bool = False, n_jobs: int | None = None
) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = np.asarray(X, dtype=np.float32)

    if X.size == 0:
        return np.array()

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


def calculate_spearman(X: np.ndarray, y: pd.Series | None, nan_policy: str):
    features_num = X.shape[1]
    if y is not None:
        features_num += 1

    if features_num < 2:
        return 1.0
    else:
        return spearmanr(X, y, nan_policy=nan_policy).correlation


def hash_series(series: pd.Series) -> int:
    return int(hashlib.sha256(pd.util.hash_pandas_object(series, index=True).values).hexdigest(), 16)
