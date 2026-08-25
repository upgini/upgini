"""Compare kernel operators to pandas ``rolling`` / ``pct_change`` on the same frames.

Tick windows only (D/h). Calendar M/B/Q and 1e12 baselines are covered elsewhere:
month offsets use DateOffset, and high-baseline mean/std are *meant* to beat pandas.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries import Delta, Delta2, Lag, Roll, RollingVolatility, RollingVolatility2, VolatilityRatio

ROLL_AGGS = ("mean", "std", "norm_mean", "min", "max", "median", "q25", "q75", "iqr")
_ROLL_CALLABLES = {
    "norm_mean": lambda x: x.iloc[-1] / x.mean(),
    "q25": lambda x: x.quantile(0.25),
    "q75": lambda x: x.quantile(0.75),
    "iqr": lambda x: x.quantile(0.75) - x.quantile(0.25),
}


def pandas_roll(s: pd.Series, window_size: int, window_unit: str, aggregation: str) -> pd.Series:
    roller = s.rolling(f"{window_size}{window_unit}", min_periods=1)
    return roller.agg(_ROLL_CALLABLES.get(aggregation, aggregation))


def pandas_lag(s: pd.Series, lag_size: int, lag_unit: str) -> pd.Series:
    def _lag(x):
        if x.index.min() > (x.index.max() - pd.Timedelta(lag_size, lag_unit)):
            return np.nan
        return x.iloc[0]

    return s.rolling(f"{lag_size + 1}{lag_unit}", min_periods=1).agg(_lag)


def pandas_delta(s: pd.Series, delta_size: int, delta_unit: str) -> pd.Series:
    return s - pandas_lag(s, delta_size, delta_unit)


def pandas_delta2(s: pd.Series, delta_size: int, delta_unit: str) -> pd.Series:
    first = pandas_delta(s, delta_size, delta_unit)
    return pandas_delta(first, delta_size, delta_unit)


def pandas_vol(
    s: pd.Series,
    window_size: int,
    window_unit: str,
    step_size: int = 1,
    step_unit: str = "D",
    abs_returns: bool = False,
) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        returns = s.pct_change(freq=f"{step_size}{step_unit}").fillna(0)
    if abs_returns:
        returns = returns.abs()
    return returns.rolling(f"{window_size}{window_unit}", min_periods=1).std()


def pandas_vol2(s: pd.Series, window_size: int, window_unit: str, step_size: int = 1, step_unit: str = "D") -> pd.Series:
    vol1 = pandas_vol(s, window_size, window_unit, step_size, step_unit, abs_returns=True)
    return pandas_vol(vol1, window_size, window_unit, step_size, step_unit, abs_returns=False)


def pandas_vol_ratio(
    s: pd.Series,
    short_window_size: int,
    window_size: int,
    short_window_unit: str = "D",
    window_unit: str = "D",
    step_size: int = 1,
    step_unit: str = "D",
) -> pd.Series:
    short = pandas_vol(s, short_window_size, short_window_unit, step_size, step_unit)
    long = pandas_vol(s, window_size, window_unit, step_size, step_unit)
    return (short / long).replace([np.inf, -np.inf], np.nan).fillna(1)


def _shift_frame(ts: pd.DataFrame, offset_size: int, offset_unit: str) -> pd.DataFrame:
    if offset_size <= 0:
        return ts
    return ts.iloc[:, :-1].merge(
        ts.iloc[:, -1].shift(freq=f"{offset_size}{offset_unit}"),
        left_index=True,
        right_index=True,
    )


def pandas_on_frame(df, col_names, series_fn, date_unit=None, offset_size=0, offset_unit="D") -> pd.Series:
    """Unique (date, groups), optional ``shift(freq)`` inner join, ``series_fn`` per group, reindex."""
    date_name, value_name = col_names[0], col_names[-1]
    group_cols = col_names[1:-1]
    date = pd.to_datetime(df[date_name], unit=date_unit, errors="coerce")
    ts = pd.concat([date.rename(date_name)] + [df[c] for c in group_cols + [value_name]], axis=1)
    ts.drop_duplicates(subset=ts.columns[:-1], keep="first", inplace=True)
    ts.set_index(date_name, inplace=True)
    ts = ts[ts.index.notna()].sort_index()

    if group_cols:
        parts = [_shift_frame(g, offset_size, offset_unit) for _, g in ts.groupby(group_cols, sort=False)]
        ts = pd.concat(parts) if parts else ts.iloc[0:0]
        out_parts = []
        for key, g in ts.groupby(group_cols, sort=False):
            s = pd.to_numeric(g[value_name], errors="coerce").astype("float64")
            r = series_fn(s)
            keys = key if isinstance(key, tuple) else (key,)
            r.index = pd.MultiIndex.from_tuples(
                [keys + (t,) for t in r.index],
                names=group_cols + [date_name],
            )
            out_parts.append(r)
        result = pd.concat(out_parts) if out_parts else pd.Series(dtype=float)
        result = result.reindex(pd.MultiIndex.from_arrays([df[c] for c in group_cols] + [date]))
    else:
        ts = _shift_frame(ts, offset_size, offset_unit)
        s = pd.to_numeric(ts[value_name], errors="coerce").astype("float64")
        result = series_fn(s).reindex(date)

    result.index = date.index
    result.name = value_name
    return result


def _assert_matches_pandas(df, op, children, series_fn):
    got = Feature(op=op, children=children).calculate(df)
    expected = pandas_on_frame(
        df,
        [c.name for c in children],
        series_fn,
        date_unit=op.date_unit,
        offset_size=op.offset_size,
        offset_unit=op.offset_unit,
    )
    assert_series_equal(got, expected, check_dtype=False)


def _daily_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-05-01",
                "2024-05-02",
                "2024-05-03",
                "---",
                "2024-05-05",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-08",
                "2024-05-10",
            ],
            "value": [100.0, 110.0, np.nan, 999.0, 121.0, 115.0, 105.0, 112.0, 999.0, 108.0],
        }
    )


def _grouped_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-05-01",
                "2024-05-01",
                "2024-05-02",
                "2024-05-02",
                "2024-05-03",
                "2024-05-04",
                "2024-05-04",
            ],
            "group": ["A", "B", "A", "B", "A", "A", "B"],
            "value": [100.0, 200.0, 110.0, 220.0, 99.0, 121.0, 198.0],
        }
    )


def _hourly_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-05-05 22:00", periods=8, freq="h"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


def _offset_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-05-01", periods=6, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _ungrouped_children():
    return [Column("date"), Column("value")]


def _grouped_children():
    return [Column("date"), Column("group"), Column("value")]


def test_roll_matches_pandas():
    daily, hourly, grouped, offset = _daily_df(), _hourly_df(), _grouped_df(), _offset_df()
    ungrouped, groups = _ungrouped_children(), _grouped_children()
    for agg in ROLL_AGGS:
        _assert_matches_pandas(
            daily, Roll(window_size=3, aggregation=agg), ungrouped, lambda s, a=agg: pandas_roll(s, 3, "D", a)
        )
        _assert_matches_pandas(
            grouped, Roll(window_size=2, aggregation=agg), groups, lambda s, a=agg: pandas_roll(s, 2, "D", a)
        )
        _assert_matches_pandas(
            hourly,
            Roll(window_size=3, window_unit="h", aggregation=agg),
            ungrouped,
            lambda s, a=agg: pandas_roll(s, 3, "h", a),
        )
        _assert_matches_pandas(
            offset,
            Roll(window_size=2, aggregation=agg, offset_size=1),
            ungrouped,
            lambda s, a=agg: pandas_roll(s, 2, "D", a),
        )


def test_lag_matches_pandas():
    daily, hourly, grouped, offset = _daily_df(), _hourly_df(), _grouped_df(), _offset_df()
    ungrouped, groups = _ungrouped_children(), _grouped_children()
    for lag_size in (1, 2):
        _assert_matches_pandas(daily, Lag(lag_size=lag_size), ungrouped, lambda s, n=lag_size: pandas_lag(s, n, "D"))
        _assert_matches_pandas(grouped, Lag(lag_size=lag_size), groups, lambda s, n=lag_size: pandas_lag(s, n, "D"))
    _assert_matches_pandas(hourly, Lag(lag_size=2, lag_unit="h"), ungrouped, lambda s: pandas_lag(s, 2, "h"))
    _assert_matches_pandas(offset, Lag(lag_size=1, offset_size=1), ungrouped, lambda s: pandas_lag(s, 1, "D"))


def test_delta_matches_pandas():
    daily, hourly, grouped, offset = _daily_df(), _hourly_df(), _grouped_df(), _offset_df()
    ungrouped, groups = _ungrouped_children(), _grouped_children()
    for size in (1, 2):
        _assert_matches_pandas(daily, Delta(delta_size=size), ungrouped, lambda s, n=size: pandas_delta(s, n, "D"))
        _assert_matches_pandas(grouped, Delta(delta_size=size), groups, lambda s, n=size: pandas_delta(s, n, "D"))
        _assert_matches_pandas(daily, Delta2(delta_size=size), ungrouped, lambda s, n=size: pandas_delta2(s, n, "D"))
    _assert_matches_pandas(grouped, Delta2(delta_size=1), groups, lambda s: pandas_delta2(s, 1, "D"))
    _assert_matches_pandas(
        hourly, Delta(delta_size=2, delta_unit="h"), ungrouped, lambda s: pandas_delta(s, 2, "h")
    )
    _assert_matches_pandas(offset, Delta(delta_size=1, offset_size=1), ungrouped, lambda s: pandas_delta(s, 1, "D"))
    _assert_matches_pandas(offset, Delta2(delta_size=1, offset_size=1), ungrouped, lambda s: pandas_delta2(s, 1, "D"))


def test_rolling_volatility_matches_pandas():
    daily, hourly, grouped, offset = _daily_df(), _hourly_df(), _grouped_df(), _offset_df()
    ungrouped, groups = _ungrouped_children(), _grouped_children()
    _assert_matches_pandas(daily, RollingVolatility(window_size=3), ungrouped, lambda s: pandas_vol(s, 3, "D"))
    _assert_matches_pandas(grouped, RollingVolatility(window_size=2), groups, lambda s: pandas_vol(s, 2, "D"))
    _assert_matches_pandas(
        hourly,
        RollingVolatility(window_size=3, window_unit="h", step_unit="h"),
        ungrouped,
        lambda s: pandas_vol(s, 3, "h", step_unit="h"),
    )
    _assert_matches_pandas(
        offset, RollingVolatility(window_size=3, offset_size=1), ungrouped, lambda s: pandas_vol(s, 3, "D")
    )


def test_rolling_volatility2_matches_pandas():
    daily, grouped, offset = _daily_df(), _grouped_df(), _offset_df()
    ungrouped, groups = _ungrouped_children(), _grouped_children()
    _assert_matches_pandas(daily, RollingVolatility2(window_size=3), ungrouped, lambda s: pandas_vol2(s, 3, "D"))
    _assert_matches_pandas(grouped, RollingVolatility2(window_size=2), groups, lambda s: pandas_vol2(s, 2, "D"))
    _assert_matches_pandas(
        offset, RollingVolatility2(window_size=3, offset_size=1), ungrouped, lambda s: pandas_vol2(s, 3, "D")
    )


def test_volatility_ratio_matches_pandas():
    daily, grouped, offset = _daily_df(), _grouped_df(), _offset_df()
    ungrouped, groups = _ungrouped_children(), _grouped_children()
    _assert_matches_pandas(
        daily,
        VolatilityRatio(short_window_size=2, window_size=4),
        ungrouped,
        lambda s: pandas_vol_ratio(s, 2, 4),
    )
    _assert_matches_pandas(
        grouped,
        VolatilityRatio(short_window_size=2, window_size=3),
        groups,
        lambda s: pandas_vol_ratio(s, 2, 3),
    )
    _assert_matches_pandas(
        offset,
        VolatilityRatio(short_window_size=2, window_size=4, offset_size=1),
        ungrouped,
        lambda s: pandas_vol_ratio(s, 2, 4),
    )
