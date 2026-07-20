"""Numpy kernels for time-series AutoFE operators.

Window semantics match pandas time-based rolling with a left-open interval
``(t - window, t]`` (see ``Lag`` / ``Roll`` operators and their tests).
"""

from __future__ import annotations

from typing import Callable, Union

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


def _rolling_mean(window: np.ndarray) -> float:
    return float(np.mean(window))


def _rolling_min(window: np.ndarray) -> float:
    return float(np.min(window))


def _rolling_max(window: np.ndarray) -> float:
    return float(np.max(window))


def _rolling_median(window: np.ndarray) -> float:
    return float(np.median(window))


def _rolling_std(window: np.ndarray) -> float:
    # pandas rolling.std default ddof=1
    if len(window) < 2:
        return np.nan
    return float(np.std(window, ddof=1))


def _rolling_norm_mean(window: np.ndarray) -> float:
    mean = np.mean(window)
    return float(window[-1] / mean)


def _rolling_q25(window: np.ndarray) -> float:
    return float(np.quantile(window, 0.25, method="linear"))


def _rolling_q75(window: np.ndarray) -> float:
    return float(np.quantile(window, 0.75, method="linear"))


def _rolling_iqr(window: np.ndarray) -> float:
    q75, q25 = np.quantile(window, [0.75, 0.25], method="linear")
    return float(q75 - q25)


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
    counts = np.arange(n, dtype=np.int64) - lefts + 1

    if aggregation in {"mean", "norm_mean", "std"}:
        csum = np.empty(n + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(values, out=csum[1:])
        window_sum = csum[1:] - csum[lefts]
        means = window_sum / counts
        if aggregation == "mean":
            return means
        if aggregation == "norm_mean":
            return values / means
        # std, ddof=1
        csum2 = np.empty(n + 1, dtype=np.float64)
        csum2[0] = 0.0
        np.cumsum(values * values, out=csum2[1:])
        window_sum2 = csum2[1:] - csum2[lefts]
        out = np.full(n, np.nan, dtype=np.float64)
        multi = counts >= 2
        # sample variance: (sumsq - sum^2/n) / (n-1)
        var = (window_sum2[multi] - window_sum[multi] * window_sum[multi] / counts[multi]) / (counts[multi] - 1)
        # numerical noise can be slightly negative
        out[multi] = np.sqrt(np.maximum(var, 0.0))
        return out

    if aggregation == "min":
        # Falling window minima via brute force is fine for correctness; still O(n·w) worst case.
        # Use a simple loop — windows are typically small calendar spans.
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = np.min(values[lefts[i] : i + 1])
        return out

    if aggregation == "max":
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = np.max(values[lefts[i] : i + 1])
        return out

    agg = ROLL_AGGS[aggregation]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = agg(values[lefts[i] : i + 1])
    return out


def freq_pct_change(times_ns: np.ndarray, values: np.ndarray, step_size: int, step_unit: str) -> np.ndarray:
    """Match ``Series.pct_change(freq=..., fill_method='pad').fillna(0)`` on sorted unique dates."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(times_ns)
    if n == 0:
        return np.array([], dtype=np.float64)

    # pandas pct_change default fill_method='pad' forward-fills before differencing
    filled = values.copy()
    is_valid = np.isfinite(filled)
    if not is_valid.all():
        idx = np.where(is_valid, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        # keep leading NaNs as NaN
        first_valid = int(np.argmax(is_valid)) if is_valid.any() else n
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
    out[~np.isfinite(out)] = 0.0
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


def _kernel_on_frame(
    frame: pd.DataFrame,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[:, [-1]].astype(np.float64) if len(frame.columns) else frame

    value_col = frame.columns[-1]
    series = pd.to_numeric(frame[value_col], errors="coerce").astype(np.float64)
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        times_ns = index.get_level_values(-1).asi8
    else:
        times_ns = index.asi8
    out = kernel(times_ns, series.to_numpy(dtype=np.float64, copy=False))
    return pd.DataFrame({value_col: out}, index=frame.index, dtype=np.float64)


def apply_grouped_kernel(
    ts: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Apply ``kernel(times_ns, values) -> values`` on a DatetimeIndex frame or GroupBy."""
    if isinstance(ts, pd.DataFrame):
        return _kernel_on_frame(ts, kernel)

    try:
        applied = ts.apply(lambda g: _kernel_on_frame(g, kernel), include_groups=False)
    except TypeError:
        # pandas without include_groups
        applied = ts.apply(lambda g: _kernel_on_frame(g, kernel))
    if isinstance(applied, pd.Series):
        return applied.to_frame()
    return applied.iloc[:, [-1]] if getattr(applied, "shape", (0, 0))[1] > 1 else applied
