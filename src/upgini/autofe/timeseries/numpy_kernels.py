"""Numpy kernels for time-series AutoFE operators.

Window semantics match pandas time-based rolling with a left-open interval
``(t - window, t]`` (see ``Lag`` / ``Roll`` operators and their tests).
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd

# Aggregation applied to values in a time window (may be length 1+).
WindowAgg = Callable[[np.ndarray], float]


def _timedelta_ns(size: int, unit: str) -> int:
    return int(pd.Timedelta(size, unit).to_timedelta64().astype("timedelta64[ns]").astype(np.int64))


def window_left_indices(times_ns: np.ndarray, window_ns: int) -> np.ndarray:
    """For each t_i, first index j with times[j] > t_i - window (left-open)."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    return np.searchsorted(times_ns, times_ns - window_ns, side="right")


def lag_values(times_ns: np.ndarray, values: np.ndarray, lag_size: int, lag_unit: str) -> np.ndarray:
    """Lag: oldest value in ``(t - (lag+1)·unit, t]`` if span covers ``lag``, else NaN."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(times_ns)
    if n == 0:
        return np.array([], dtype=np.float64)

    lag_ns = _timedelta_ns(lag_size, lag_unit)
    window_ns = _timedelta_ns(lag_size + 1, lag_unit)
    lefts = window_left_indices(times_ns, window_ns)

    out = np.full(n, np.nan, dtype=np.float64)
    # Gate: oldest in window is at or before t - lag  <=>  times[left] <= t - lag
    gate = times_ns[lefts] <= times_ns - lag_ns
    out[gate] = values[lefts[gate]]
    return out


def _quantile_linear(sorted_vals: np.ndarray, q: float) -> float:
    """pandas/numpy linear quantile on an already-sorted 1d array."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = lo + 1 if lo + 1 < n else lo
    w = pos - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _rolling_mean(window: np.ndarray) -> float:
    return float(np.mean(window))


def _rolling_min(window: np.ndarray) -> float:
    return float(np.min(window))


def _rolling_max(window: np.ndarray) -> float:
    return float(np.max(window))


def _rolling_median(window: np.ndarray) -> float:
    return _quantile_linear(np.sort(window), 0.5)


def _rolling_std(window: np.ndarray) -> float:
    # pandas rolling.std default ddof=1
    if len(window) < 2:
        return np.nan
    return float(np.std(window, ddof=1))


def _rolling_norm_mean(window: np.ndarray) -> float:
    mean = np.mean(window)
    return float(window[-1] / mean)


def _rolling_q25(window: np.ndarray) -> float:
    return _quantile_linear(np.sort(window), 0.25)


def _rolling_q75(window: np.ndarray) -> float:
    return _quantile_linear(np.sort(window), 0.75)


def _rolling_iqr(window: np.ndarray) -> float:
    sorted_vals = np.sort(window)
    return _quantile_linear(sorted_vals, 0.75) - _quantile_linear(sorted_vals, 0.25)


ROLL_AGGS: dict[str, WindowAgg] = {
    "mean": _rolling_mean,
    "min": _rolling_min,
    "max": _rolling_max,
    "median": _rolling_median,
    "std": _rolling_std,
    "norm_mean": _rolling_norm_mean,
    "q25": _rolling_q25,
    "q75": _rolling_q75,
    "iqr": _rolling_iqr,
}


def roll_values(
    times_ns: np.ndarray,
    values: np.ndarray,
    window_size: int,
    window_unit: str,
    aggregation: str,
) -> np.ndarray:
    """Time-based rolling aggregation with ``min_periods=1`` semantics."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(times_ns)
    if n == 0:
        return np.array([], dtype=np.float64)

    if aggregation not in ROLL_AGGS:
        raise ValueError(f"Unsupported roll aggregation for numpy path: {aggregation}")

    window_ns = _timedelta_ns(window_size, window_unit)
    lefts = window_left_indices(times_ns, window_ns)

    # pandas rolling skipna=True: ignore NaNs inside the window
    finite = np.isfinite(values)

    if aggregation in {"mean", "norm_mean"}:
        safe = np.where(finite, values, 0.0)
        csum = np.empty(n + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(safe, out=csum[1:])
        ccount = np.empty(n + 1, dtype=np.float64)
        ccount[0] = 0.0
        np.cumsum(finite.astype(np.float64), out=ccount[1:])
        window_sum = csum[1:] - csum[lefts]
        counts = ccount[1:] - ccount[lefts]
        out = np.full(n, np.nan, dtype=np.float64)
        valid = counts > 0
        means = np.empty(n, dtype=np.float64)
        means[valid] = window_sum[valid] / counts[valid]
        if aggregation == "mean":
            out[valid] = means[valid]
            return out
        # last finite observation in window / mean; if current value NaN → NaN
        out[valid & finite] = values[valid & finite] / means[valid & finite]
        return out

    if aggregation == "std":
        return _roll_std_skipna(values, lefts, finite)

    if aggregation in {"q25", "q75", "iqr", "median"}:
        return _roll_order_stats_skipna(values, lefts, aggregation)

    if aggregation in {"min", "max"}:
        # pandas rolling min/max skip NaN and ±inf
        reducer = np.min if aggregation == "min" else np.max
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            sl = slice(int(lefts[i]), i + 1)
            finite_window = values[sl][finite[sl]]
            if finite_window.size:
                out[i] = reducer(finite_window)
        return out

    agg = ROLL_AGGS[aggregation]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        window = values[lefts[i] : i + 1]
        finite_window = window[np.isfinite(window)]
        out[i] = agg(finite_window) if len(finite_window) else np.nan
    return out


def _roll_std_skipna(values: np.ndarray, lefts: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Sliding Welford std (ddof=1), skipna=True; same add/remove as pandas rolling.std."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    mean = 0.0
    m2 = 0.0
    count = 0
    left = 0

    for i in range(n):
        new_left = int(lefts[i])
        for j in range(left, new_left):
            if not finite[j]:
                continue
            if count <= 1:
                mean = 0.0
                m2 = 0.0
                count = 0
                continue
            x = values[j]
            if count == 2:
                mean = 2.0 * mean - x
                m2 = 0.0
                count = 1
                continue
            count_new = count - 1
            delta = x - mean
            mean = mean - delta / count_new
            m2 -= delta * (x - mean)
            count = count_new
        left = new_left

        if finite[i]:
            count += 1
            x = values[i]
            delta = x - mean
            mean += delta / count
            m2 += delta * (x - mean)

        if count >= 2:
            out[i] = np.sqrt(m2 / (count - 1)) if m2 > 0.0 else 0.0

    return out


def _roll_order_stats_skipna(values: np.ndarray, lefts: np.ndarray, aggregation: str) -> np.ndarray:
    """Fast q25/q75/iqr/median with NaN skipping (pandas rolling skipna=True)."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    q = {"q25": 0.25, "q75": 0.75, "median": 0.5, "iqr": None}[aggregation]

    for i in range(n):
        window = values[lefts[i] : i + 1]
        finite_vals = window[np.isfinite(window)]
        w = len(finite_vals)
        if w == 0:
            continue
        if aggregation == "iqr":
            if w == 1:
                out[i] = 0.0
            elif w == 2:
                a, b = finite_vals[0], finite_vals[1]
                out[i] = 0.5 * abs(b - a)
            else:
                sorted_vals = np.sort(finite_vals)
                out[i] = _quantile_linear(sorted_vals, 0.75) - _quantile_linear(sorted_vals, 0.25)
            continue

        if w == 1:
            out[i] = finite_vals[0]
        elif w == 2:
            a, b = finite_vals[0], finite_vals[1]
            lo, hi = (a, b) if a <= b else (b, a)
            out[i] = 0.5 * (lo + hi) if q == 0.5 else lo * (1.0 - q) + hi * q
        else:
            out[i] = _quantile_linear(np.sort(finite_vals), q)
    return out


def freq_pct_change(times_ns: np.ndarray, values: np.ndarray, step_size: int, step_unit: str) -> np.ndarray:
    """Match ``Series.pct_change(freq=..., fill_method='pad').fillna(0)`` on sorted unique dates."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(times_ns)
    if n == 0:
        return np.array([], dtype=np.float64)

    # pandas pct_change fill_method='pad': forward-fill NaN only, then fillna(0) (NaN only)
    filled = values.copy()
    is_na = np.isnan(filled)
    if is_na.any():
        not_na = ~is_na
        idx = np.where(not_na, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        first_valid = int(np.argmax(not_na)) if not_na.any() else n
        filled = filled[idx]
        filled[:first_valid] = np.nan

    step_ns = _timedelta_ns(step_size, step_unit)
    targets = times_ns - step_ns
    idx = np.searchsorted(times_ns, targets, side="left")
    out = np.zeros(n, dtype=np.float64)
    in_range = idx < n
    matches = np.zeros(n, dtype=bool)
    matches[in_range] = times_ns[idx[in_range]] == targets[in_range]
    prev = filled[idx[matches]]
    cur = filled[matches]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = cur / prev - 1.0
    out[matches] = pct
    out[np.isnan(out)] = 0.0
    return out


def delta_values(times_ns: np.ndarray, values: np.ndarray, delta_size: int, delta_unit: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - lag_values(times_ns, values, delta_size, delta_unit)


def delta2_values(times_ns: np.ndarray, values: np.ndarray, delta_size: int, delta_unit: str) -> np.ndarray:
    first = delta_values(times_ns, values, delta_size, delta_unit)
    return delta_values(times_ns, first, delta_size, delta_unit)


def rolling_volatility_values(
    times_ns: np.ndarray,
    values: np.ndarray,
    step_size: int,
    step_unit: str,
    window_size: int,
    window_unit: str,
    abs_returns: bool = False,
) -> np.ndarray:
    returns = freq_pct_change(times_ns, values, step_size, step_unit)
    if abs_returns:
        returns = np.abs(returns)
    return roll_values(times_ns, returns, window_size, window_unit, "std")


def offset_values(
    times_ns: np.ndarray, values: np.ndarray, offset_size: int, offset_unit: str
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``Series.shift(freq=offset)`` aligned onto the same timestamps.

    Returns
    -------
    out, matched
        ``out`` is the shifted values (NaN where no exact timestamp match).
        ``matched`` is True where an exact ``t - offset`` source timestamp existed
        (same rows kept by pandas ``merge(..., how='inner')`` with ``shift(freq)``).
    """
    times_ns = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(times_ns)
    if n == 0 or offset_size == 0:
        return values.copy(), np.ones(n, dtype=bool)

    offset_ns = _timedelta_ns(offset_size, offset_unit)
    targets = times_ns - offset_ns
    idx = np.searchsorted(times_ns, targets, side="left")
    out = np.full(n, np.nan, dtype=np.float64)
    matched = np.zeros(n, dtype=bool)
    in_range = idx < n
    matched[in_range] = times_ns[idx[in_range]] == targets[in_range]
    out[matched] = values[idx[matched]]
    return out, matched


def _kernel_on_arrays(
    times_ns: np.ndarray,
    values: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    return kernel(times_ns, values)


def _apply_kernel_by_group_ids(
    times_ns: np.ndarray,
    values: np.ndarray,
    group_ids: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply kernel on contiguous runs of the same group_id (caller must sort by group, then time)."""
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    # Boundaries where group_id changes
    changes = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))
    for start, end in zip(starts, ends):
        out[start:end] = kernel(times_ns[start:end], values[start:end])
    return out


def apply_grouped_kernel(
    ts: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Apply ``kernel(times_ns, values) -> values`` on a DatetimeIndex frame or GroupBy."""
    if isinstance(ts, pd.DataFrame):
        if ts.empty:
            return ts.iloc[:, [-1]].astype(np.float64) if len(ts.columns) else ts
        value_col = ts.columns[-1]
        series = pd.to_numeric(ts[value_col], errors="coerce").astype(np.float64)
        index = ts.index
        times_ns = index.get_level_values(-1).asi8 if isinstance(index, pd.MultiIndex) else index.asi8
        out = kernel(times_ns, series.to_numpy(dtype=np.float64, copy=False))
        return pd.DataFrame({value_col: out}, index=ts.index, dtype=np.float64)

    # DataFrameGroupBy — avoid GroupBy.apply; use positional indices
    obj = ts.obj
    keys = ts.keys
    if isinstance(keys, list):
        grouper_keys = keys
    elif isinstance(keys, tuple):
        grouper_keys = list(keys)
    else:
        grouper_keys = [keys]

    value_col = obj.columns[-1]
    values = pd.to_numeric(obj[value_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    index = obj.index
    times_ns = index.get_level_values(-1).asi8 if isinstance(index, pd.MultiIndex) else index.asi8
    # GroupBy.indices skips null keys (dropna=True); NaN preserves those rows.
    out = np.full(len(obj), np.nan, dtype=np.float64)

    for indexer in ts.indices.values():
        idx = np.asarray(indexer, dtype=np.intp)
        order = np.argsort(times_ns[idx], kind="mergesort")
        sorted_idx = idx[order]
        out[sorted_idx] = kernel(times_ns[sorted_idx], values[sorted_idx])

    # MultiIndex (group keys..., date) so TimeSeriesBase.reindex works with duplicate dates
    date_level = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
    levels = [obj[c].to_numpy() for c in grouper_keys] + [date_level]
    mi = pd.MultiIndex.from_arrays(levels, names=list(grouper_keys) + [date_level.name])
    return pd.DataFrame({value_col: out}, index=mi, dtype=np.float64)


def apply_offset_grouped(
    ts: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    offset_size: int,
    offset_unit: str,
) -> pd.DataFrame:
    """Apply frequency offset within groups without ``groupby.apply``.

    Matches ``DataFrame.merge(series.shift(freq=...), how='inner')`` used by the
    legacy ``_shift`` helper: rows without an exact offset match are dropped
    before aggregation (then restored as NaN by the final ``reindex``).
    """
    if offset_size <= 0:
        return ts

    date_name = ts.index.name or "index"
    if not group_cols:
        times_ns = np.asarray(ts.index.asi8, dtype=np.int64)
        values = pd.to_numeric(ts[value_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        out_vals, matched = offset_values(times_ns, values, offset_size, offset_unit)
        result = ts.copy()
        result[value_col] = out_vals
        return result.loc[matched]

    flat = ts.reset_index()
    flat = flat.sort_values(list(group_cols) + [date_name], kind="mergesort")
    times_ns = np.asarray(pd.DatetimeIndex(pd.to_datetime(flat[date_name])).asi8, dtype=np.int64)
    values = pd.to_numeric(flat[value_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    group_ids = pd.factorize(pd.MultiIndex.from_arrays([flat[c].to_numpy() for c in group_cols]), sort=False)[0]

    out_vals = np.empty(len(flat), dtype=np.float64)
    matched = np.zeros(len(flat), dtype=bool)
    changes = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(flat)]))
    for start, end in zip(starts, ends):
        group_out, group_matched = offset_values(
            times_ns[start:end], values[start:end], offset_size, offset_unit
        )
        out_vals[start:end] = group_out
        matched[start:end] = group_matched

    flat[value_col] = out_vals
    flat = flat.loc[matched]
    return flat.set_index(date_name)[list(group_cols) + [value_col]]
