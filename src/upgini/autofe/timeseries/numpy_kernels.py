"""Numpy kernels for time-series AutoFE operators.

Window semantics match pandas time-based rolling with a left-open interval
``(t - window, t]`` (see ``Lag`` / ``Roll`` operators and their tests).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


@lru_cache(maxsize=64)
def _tick_ns(size: int, unit: str) -> Optional[int]:
    """Nanoseconds for a fixed duration, or None for calendar/business offsets (M, B, Q, …).

    Cached: grouped kernels call this once per group, not per row, but re-parsing
    ``Timedelta`` on every group dominated lag/roll time.
    """
    try:
        return int(pd.Timedelta(size, unit).to_timedelta64().astype("timedelta64[ns]").astype(np.int64))
    except ValueError:
        return None


@lru_cache(maxsize=64)
def _as_offset(size: int, unit: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return to_offset(f"{size}{unit}")


def _times_minus_offset(
    times_ns: np.ndarray,
    size: int,
    unit: str,
    dates: Optional[pd.DatetimeIndex] = None,
) -> np.ndarray:
    """``times - offset`` as int64 ns. Tick units use integer arithmetic; others use DateOffset."""
    if dates is None:
        tick_ns = _tick_ns(size, unit)
        if tick_ns is not None:
            return times_ns - tick_ns
        dates = pd.DatetimeIndex(times_ns.astype("datetime64[ns]", copy=False))
    return _subtract_calendar_offset(dates, size, unit)


def _subtract_calendar_offset(dates: pd.DatetimeIndex, size: int, unit: str) -> np.ndarray:
    """``dates - offset`` as int64 ns.

    ``to_offset('M')`` is MonthEnd, which misses a series sampled on day X of each
    month (Apr 15 → Mar 31). Non-month-end timestamps use ``DateOffset(months=)``
    so the same day-of-month matches; month-end timestamps keep MonthEnd so
    Jan 31 / Feb 29 / Apr 30 still line up.
    """
    if unit.upper() == "M":
        same_day = dates - pd.DateOffset(months=size)
        month_end = dates.is_month_end
        if not bool(np.any(month_end)):
            return np.asarray(same_day.asi8, dtype=np.int64)
        anchored = dates - _as_offset(size, unit)
        return np.asarray(np.where(month_end, anchored.asi8, same_day.asi8), dtype=np.int64)
    return np.asarray((dates - _as_offset(size, unit)).asi8, dtype=np.int64)


def window_left_indices(
    times_ns: np.ndarray,
    window_size: int,
    window_unit: str,
    dates: Optional[pd.DatetimeIndex] = None,
) -> np.ndarray:
    """For each t_i, first index j with times[j] > t_i - window (left-open)."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    starts = _times_minus_offset(times_ns, window_size, window_unit, dates)
    return np.searchsorted(times_ns, starts, side="right")


def lag_values(times_ns: np.ndarray, values: np.ndarray, lag_size: int, lag_unit: str) -> np.ndarray:
    """Lag: oldest value in ``(t - (lag+1)·unit, t]`` if span covers ``lag``, else NaN."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(times_ns)
    if n == 0:
        return np.array([], dtype=np.float64)

    dates = (
        None
        if _tick_ns(lag_size, lag_unit) is not None
        else pd.DatetimeIndex(times_ns.astype("datetime64[ns]", copy=False))
    )
    lag_targets = _times_minus_offset(times_ns, lag_size, lag_unit, dates)
    lefts = window_left_indices(times_ns, lag_size + 1, lag_unit, dates)

    out = np.full(n, np.nan, dtype=np.float64)
    # Gate: oldest in window is at or before t - lag  <=>  times[left] <= t - lag
    gate = times_ns[lefts] <= lag_targets
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


ROLL_AGGS = frozenset({"mean", "min", "max", "median", "std", "norm_mean", "q25", "q75", "iqr"})

# Per-window Python scans are quadratic in width; pandas' Cython kernels are used
# for these on fixed Tick windows. Calendar offsets (M, B, …) stay on numpy.
PANDAS_ROLL_AGGS = frozenset({"min", "max", "median", "q25", "q75", "iqr"})


def uses_pandas_rolling(window_size: int, window_unit: str, aggregation: str) -> bool:
    return aggregation in PANDAS_ROLL_AGGS and _tick_ns(window_size, window_unit) is not None


def apply_pandas_rolling(obj, window_size: int, window_unit: str, aggregation: str):
    """Time-based rolling via pandas (DataFrame, Series, or GroupBy)."""
    roller = obj.rolling(f"{window_size}{window_unit}", min_periods=1)
    if aggregation == "q25":
        return roller.quantile(0.25)
    if aggregation == "q75":
        return roller.quantile(0.75)
    if aggregation == "iqr":
        return roller.quantile(0.75) - roller.quantile(0.25)
    return getattr(roller, aggregation)()


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

    if uses_pandas_rolling(window_size, window_unit, aggregation):
        series = pd.Series(
            values, index=pd.DatetimeIndex(times_ns.astype("datetime64[ns]", copy=False))
        )
        return np.asarray(apply_pandas_rolling(series, window_size, window_unit, aggregation), dtype=np.float64)

    lefts = window_left_indices(times_ns, window_size, window_unit)

    # pandas rolling skipna=True: ignore NaNs inside the window
    finite = np.isfinite(values)

    if aggregation in {"mean", "norm_mean", "std"}:
        return _roll_compensated(values, lefts, finite, aggregation)

    if aggregation in {"q25", "q75", "iqr", "median"}:
        return _roll_order_stats_skipna(values, lefts, aggregation)

    # min / max: skip NaN and ±inf (pandas rolling skipna)
    reducer = np.min if aggregation == "min" else np.max
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        sl = slice(int(lefts[i]), i + 1)
        finite_window = values[sl][finite[sl]]
        if finite_window.size:
            out[i] = reducer(finite_window)
    return out


def _roll_compensated(values: np.ndarray, lefts: np.ndarray, finite: np.ndarray, aggregation: str) -> np.ndarray:
    """Rolling mean / norm_mean / std via prefix sums of ``x - baseline``.

    Subtracting the first finite value makes the sums track the signal, not the
    level: ``mean(x) = mean(x - c) + c`` and ``std(x) = std(x - c)``. Avoids
    cancellation on long high-baseline series without a Python Welford loop.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    baseline = 0.0
    if finite.any():
        baseline = float(values[int(np.argmax(finite))])
    shifted = np.where(finite, values - baseline, 0.0)

    csum = np.empty(n + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(shifted, out=csum[1:])
    ccount = np.empty(n + 1, dtype=np.float64)
    ccount[0] = 0.0
    np.cumsum(finite.astype(np.float64), out=ccount[1:])
    window_sum = csum[1:] - csum[lefts]
    counts = ccount[1:] - ccount[lefts]
    valid = counts > 0

    if aggregation == "std":
        csum2 = np.empty(n + 1, dtype=np.float64)
        csum2[0] = 0.0
        np.cumsum(shifted * shifted, out=csum2[1:])
        window_sum2 = csum2[1:] - csum2[lefts]
        ok = counts >= 2
        var = (window_sum2[ok] - window_sum[ok] * window_sum[ok] / counts[ok]) / (counts[ok] - 1.0)
        out[ok] = np.sqrt(np.maximum(var, 0.0))
        return out

    means = np.empty(n, dtype=np.float64)
    means[valid] = window_sum[valid] / counts[valid] + baseline
    if aggregation == "mean":
        out[valid] = means[valid]
        return out
    out[valid & finite] = values[valid & finite] / means[valid & finite]
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

    targets = _times_minus_offset(times_ns, step_size, step_unit)
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

    targets = _times_minus_offset(times_ns, offset_size, offset_unit)
    idx = np.searchsorted(times_ns, targets, side="left")
    out = np.full(n, np.nan, dtype=np.float64)
    matched = np.zeros(n, dtype=bool)
    in_range = idx < n
    matched[in_range] = times_ns[idx[in_range]] == targets[in_range]
    out[matched] = values[idx[matched]]
    return out, matched


def _contiguous_group_bounds(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start/end indices of contiguous equal-id runs (caller must sort by group)."""
    n = len(group_ids)
    if n == 0:
        empty = np.array([], dtype=np.intp)
        return empty, empty
    changes = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))
    return starts, ends


def pack_group_ids(columns: Sequence[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer group codes and a mask of rows whose group key has a null."""
    if not columns:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    codes: list[np.ndarray] = []
    sizes: list[int] = []
    null = np.zeros(n, dtype=bool)
    for col in columns:
        c = pd.factorize(col, sort=False)[0].astype(np.int64, copy=False)
        null |= c < 0
        codes.append(np.where(c < 0, 0, c))
        mx = int(c.max()) if n else -1
        sizes.append(mx + 1 if mx >= 0 else 1)
    prod = 1
    for sz in sizes:
        prod *= max(sz, 1)
        if prod > 2**62:
            mi = pd.MultiIndex.from_arrays(list(columns))
            packed = pd.factorize(mi, sort=False)[0].astype(np.int64, copy=False)
            return packed, packed < 0
    packed = codes[0]
    for c, sz in zip(codes[1:], sizes[1:]):
        packed = packed * sz + c
    return packed, null


def unique_sorted_by_group_time(
    times_ns: np.ndarray,
    values: np.ndarray,
    group_ids: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First occurrence of each (group, time) among ``keep``, sorted by group then time.

    Returns
    -------
    times_s, values_s, group_ids_s
        Unique rows in (group, time) order.
    scatter_idx, scatter_slot
        ``out[scatter_idx] = values_s[scatter_slot]`` maps results onto every
        kept original row (duplicates included). Invalid after rows are dropped
        from the unique arrays (e.g. offset inner-join).
    """
    n = len(times_ns)
    empty_i8 = np.array([], dtype=np.int64)
    empty_p = np.array([], dtype=np.intp)
    if n == 0 or not keep.any():
        return empty_i8, np.array([], dtype=np.float64), empty_i8, empty_p, empty_p

    keep_all = bool(keep.all())
    if keep_all:
        if _group_time_strictly_increasing(group_ids, times_ns):
            scatter = np.arange(n, dtype=np.intp)
            return times_ns, values, group_ids, scatter, scatter
        idx = np.arange(n, dtype=np.intp)
        g_k, t_k = group_ids, times_ns
    else:
        idx = np.flatnonzero(keep)
        g_k, t_k = group_ids[idx], times_ns[idx]
        if _group_time_strictly_increasing(g_k, t_k):
            slot = np.arange(idx.size, dtype=np.intp)
            return t_k, values[idx], g_k, idx, slot

    # Primary: group, then time, then original row so keep='first' is the run head.
    order = np.lexsort((idx, t_k, g_k))
    scatter_idx = idx[order]
    g = g_k[order]
    t = t_k[order]
    n_keep = scatter_idx.size
    first = np.empty(n_keep, dtype=bool)
    first[0] = True
    if n_keep > 1:
        first[1:] = (g[1:] != g[:-1]) | (t[1:] != t[:-1])
    if bool(first.all()):
        scatter_slot = np.arange(n_keep, dtype=np.intp)
        uniq_idx = scatter_idx
    else:
        uniq_idx = scatter_idx[first]
        scatter_slot = np.cumsum(first, dtype=np.intp) - 1
    return times_ns[uniq_idx], values[uniq_idx], group_ids[uniq_idx], scatter_idx, scatter_slot


def _group_time_strictly_increasing(group_ids: np.ndarray, times_ns: np.ndarray) -> bool:
    """True when rows are unique and already in (group, time) order."""
    n = len(times_ns)
    if n <= 1:
        return True
    dg = np.diff(group_ids)
    return bool(np.all((dg > 0) | ((dg == 0) & (np.diff(times_ns) > 0))))


def scatter_grouped_times(
    times_s: np.ndarray,
    group_ids_s: np.ndarray,
    values_s: np.ndarray,
    times_o: np.ndarray,
    group_ids_o: np.ndarray,
    fill: np.ndarray,
    scatter_idx: Optional[np.ndarray] = None,
    scatter_slot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map unique sorted (group, time) results onto original rows."""
    out = np.full(len(times_o), np.nan, dtype=np.float64)
    k = len(values_s)
    if k == 0 or not fill.any():
        return out
    if scatter_idx is not None and scatter_slot is not None:
        out[scatter_idx] = values_s[scatter_slot]
        return out

    orig = np.flatnonzero(fill)
    order = np.lexsort((times_o[orig], group_ids_o[orig]))
    oi = orig[order]
    g_orig = group_ids_o[oi]
    t_orig = times_o[oi]

    s_starts, s_ends = _contiguous_group_bounds(group_ids_s)
    o_starts, o_ends = _contiguous_group_bounds(g_orig)

    si = oi_i = 0
    n_s, n_o = s_starts.size, o_starts.size
    while si < n_s and oi_i < n_o:
        gs = group_ids_s[s_starts[si]]
        go = g_orig[o_starts[oi_i]]
        if gs < go:
            si += 1
            continue
        if gs > go:
            oi_i += 1
            continue
        sl = slice(int(s_starts[si]), int(s_ends[si]))
        ol = slice(int(o_starts[oi_i]), int(o_ends[oi_i]))
        times_g = times_s[sl]
        pos = np.searchsorted(times_g, t_orig[ol])
        sl_len = times_g.size
        in_range = pos < sl_len
        pos_c = np.where(in_range, pos, 0)
        match = in_range & (times_g[pos_c] == t_orig[ol])
        dest = oi[ol]
        out[dest[match]] = values_s[sl][pos_c[match]]
        si += 1
        oi_i += 1
    return out


def apply_offset_arrays(
    times_ns: np.ndarray,
    values: np.ndarray,
    group_ids: np.ndarray,
    offset_size: int,
    offset_unit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offset values within contiguous groups; drop rows with no exact match."""
    if offset_size <= 0 or len(values) == 0:
        return times_ns, values, group_ids
    out_vals = np.empty(len(values), dtype=np.float64)
    matched = np.zeros(len(values), dtype=bool)
    starts, ends = _contiguous_group_bounds(group_ids)
    for start, end in zip(starts, ends):
        group_out, group_matched = offset_values(
            times_ns[start:end], values[start:end], offset_size, offset_unit
        )
        out_vals[start:end] = group_out
        matched[start:end] = group_matched
    return times_ns[matched], out_vals[matched], group_ids[matched]


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

    starts, ends = _contiguous_group_bounds(group_ids)
    for start, end in zip(starts, ends):
        out[start:end] = kernel(times_ns[start:end], values[start:end])
    return out


def apply_offset_grouped(
    ts: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    offset_size: int,
    offset_unit: str,
) -> pd.DataFrame:
    """Apply frequency offset within groups without ``groupby.apply``.

    Rows without an exact offset match are dropped before aggregation (then
    restored as NaN by the final ``reindex``).
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
    starts, ends = _contiguous_group_bounds(group_ids)
    for start, end in zip(starts, ends):
        group_out, group_matched = offset_values(
            times_ns[start:end], values[start:end], offset_size, offset_unit
        )
        out_vals[start:end] = group_out
        matched[start:end] = group_matched

    flat[value_col] = out_vals
    flat = flat.loc[matched]
    return flat.set_index(date_name)[list(group_cols) + [value_col]]
