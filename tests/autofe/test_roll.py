from typing import List

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries import Roll
from upgini.autofe.utils import pydantic_parse_method


def test_roll_date():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-09", "---", "2024-05-07", "2024-05-08", "2024-05-08", "2024-05-08"],
            "value": [1, 2, 3, 4, 5, 5, 6],
        }
    )

    def check_agg(agg: str, expected_values: List[float]):
        feature = Feature(op=Roll(window_size=2, aggregation=agg), children=[Column("date"), Column("value")])
        assert feature.op.to_formula() == f"roll_2D_{agg}"
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_agg("mean", [1.0, 3.5, np.nan, 2.5, 4.5, 4.5, 4.5])
    check_agg("min", [1.0, 2.0, np.nan, 1.0, 4.0, 4.0, 4.0])
    check_agg("max", [1.0, 5.0, np.nan, 4.0, 5.0, 5.0, 5.0])
    check_agg(
        "std",
        [
            np.nan,
            2.1213203435596424,
            np.nan,
            2.1213203435596424,
            0.7071067811865476,
            0.7071067811865476,
            0.7071067811865476,
        ],
    )
    check_agg("median", [1.0, 3.5, np.nan, 2.5, 4.5, 4.5, 4.5])
    check_agg(
        "norm_mean",
        [1.0, 0.5714285714285714, np.nan, 1.6, 1.1111111111111112, 1.1111111111111112, 1.1111111111111112],
    )
    check_agg("q25", [1.0, 2.75, np.nan, 1.75, 4.25, 4.25, 4.25])
    check_agg("q75", [1.0, 4.25, np.nan, 3.25, 4.75, 4.75, 4.75])
    check_agg("iqr", [0.0, 1.5, np.nan, 1.5, 0.5, 0.5, 0.5])


def test_roll_date_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-06", "---", "2024-05-07", "2024-05-07", "2024-05-07"],
            "f1": ["a", "b", "a", "a", "a", "c"],
            "f2": [1, 2, 1, 1, 1, 2],
            "value": [1, 2, 3, 4, 4, 5],
        },
        index=[9, 8, 7, 6, 5, 4],
    )

    def check_period(period: int, agg: str, expected_values: List[float]):
        feature = Feature(
            op=Roll(window_size=period, aggregation=agg),
            children=[Column("date"), Column("f1"), Column("f2"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value", index=df.index)
        assert_series_equal(feature.calculate(df), expected_res)

    check_period(1, "mean", [1.0, 2.0, np.nan, 4.0, 4.0, 5.0])
    check_period(2, "mean", [1.0, 2.0, np.nan, 2.5, 2.5, 5.0])
    check_period(2, "norm_mean", [1.0, 1.0, np.nan, 1.6, 1.6, 1.0])


def _pandas_rolling(dates, values, window_size: int, aggregation: str, window_unit: str = "D") -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    roller = pd.Series(values, index=idx).rolling(f"{window_size}{window_unit}", min_periods=1)
    rolled = getattr(roller, aggregation)()
    return pd.Series(rolled.to_numpy(), name="value")


def test_roll_std_large_magnitude():
    dates = pd.date_range("2024-05-01", periods=4, freq="D")
    for baseline in (1e8, 1e12):
        values = baseline + np.arange(4, dtype=np.float64)
        df = pd.DataFrame({"date": dates, "value": values})
        result = Feature(
            op=Roll(window_size=10, aggregation="std"),
            children=[Column("date"), Column("value")],
        ).calculate(df)
        assert_series_equal(result, _pandas_rolling(dates, values, 10, "std"))


def test_roll_std_skipna_and_window_bounds():
    dates = pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-06"])
    values = np.array([1.0, np.nan, 3.0, 4.0, 6.0])
    df = pd.DataFrame({"date": dates, "value": values})
    result = Feature(
        op=Roll(window_size=3, aggregation="std"),
        children=[Column("date"), Column("value")],
    ).calculate(df)
    assert_series_equal(result, _pandas_rolling(dates, values, 3, "std"))


def test_roll_minmax_nonfinite():
    dates = pd.date_range("2024-05-01", periods=4, freq="D")
    values_inf = np.array([1.0, np.inf, 3.0, 4.0])
    df_inf = pd.DataFrame({"date": dates, "value": values_inf})
    assert_series_equal(
        Feature(op=Roll(window_size=10, aggregation="max"), children=[Column("date"), Column("value")]).calculate(
            df_inf
        ),
        _pandas_rolling(dates, values_inf, 10, "max"),
    )

    values_ninf = np.array([1.0, -np.inf, 3.0, 4.0])
    df_ninf = pd.DataFrame({"date": dates, "value": values_ninf})
    assert_series_equal(
        Feature(op=Roll(window_size=10, aggregation="min"), children=[Column("date"), Column("value")]).calculate(
            df_ninf
        ),
        _pandas_rolling(dates, values_ninf, 10, "min"),
    )


def test_roll_null_group_keys():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-07", "2024-05-08"],
            "group": ["a", None, "a"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    feature = Feature(
        op=Roll(window_size=2, aggregation="mean"),
        children=[Column("date"), Column("group"), Column("value")],
    )
    assert_series_equal(feature.calculate(df), pd.Series([1.0, np.nan, 3.0], name="value"))

    df_composite = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-07", "2024-05-08", "2024-05-08"],
            "f1": ["a", "a", None, "b"],
            "f2": [1, None, 1, 2],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    feature_composite = Feature(
        op=Roll(window_size=2, aggregation="mean"),
        children=[Column("date"), Column("f1"), Column("f2"), Column("value")],
    )
    assert_series_equal(
        feature_composite.calculate(df_composite),
        pd.Series([1.0, np.nan, np.nan, 4.0], name="value"),
    )


def test_roll_from_formula():
    roll = Roll.from_formula("roll_3d_mean")
    assert roll.window_size == 3
    assert roll.window_unit == "d"
    assert roll.aggregation == "mean"
    assert roll.to_formula() == "roll_3d_mean"

    roll = Roll.from_formula("roll_10D_max_offset_1D")
    assert roll.window_size == 10
    assert roll.window_unit == "D"
    assert roll.offset_size == 1
    assert roll.offset_unit == "D"
    assert roll.aggregation == "max"
    assert roll.to_formula() == "roll_10D_max_offset_1D"

    # Test invalid formulas
    roll = Roll.from_formula("not_a_roll_formula")
    assert roll is None

    roll = Roll.from_formula("roll_abc_mean")
    assert roll is None

    roll = Roll.from_formula("roll_3d")
    assert roll is None

    # Test that constructed name matches formula pattern
    roll = Roll(window_size=5, window_unit="D", aggregation="median")
    assert roll.to_formula() == "roll_5D_median"


def test_roll_with_offset():
    df = pd.DataFrame(
        {
            "date": [
                "2024-05-05",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-08",
            ],
            "value": [1, 2, 3, 4, 5],
        },
    )

    def check_roll(
        window_size: int, window_unit: str, offset_size: int, aggregation: str, expected_values: List[float]
    ):
        feature = Feature(
            op=Roll(window_size=window_size, window_unit=window_unit, offset_size=offset_size, aggregation=aggregation),
            children=[Column("date"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_roll(2, "d", 0, "mean", [1.0, 1.5, 2.5, 3.5, 3.5])
    check_roll(2, "d", 1, "mean", [np.nan, 1.0, 1.5, 2.5, 2.5])
    check_roll(3, "d", 1, "median", [np.nan, 1.0, 1.5, 2.0, 2.0])


def test_roll_month_offset():
    df = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )
    feature = Feature(
        op=Roll(window_size=1, window_unit="D", aggregation="mean", offset_size=1, offset_unit="M"),
        children=[Column("date"), Column("value")],
    )
    assert_series_equal(feature.calculate(df), pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0], name="value"))

    mid = pd.DataFrame(
        {
            "date": ["2024-03-15", "2024-03-31", "2024-04-15", "2024-04-30"],
            "value": [1.0, 2.0, 3.0, 4.0],
        },
    )
    mid_feature = Feature(
        op=Roll(window_size=1, window_unit="D", aggregation="mean", offset_size=1, offset_unit="M"),
        children=[Column("date"), Column("value")],
    )
    # Apr 15 looks up Mar 15; Apr 30 is month-end and looks up Mar 31.
    assert_series_equal(mid_feature.calculate(mid), pd.Series([np.nan, np.nan, 1.0, 2.0], name="value"))


def test_roll_month_same_day_of_month():
    df = pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"],
            "value": [1.0, 2.0, 3.0, 4.0],
        },
    )
    offset_feature = Feature(
        op=Roll(window_size=1, window_unit="D", aggregation="mean", offset_size=1, offset_unit="M"),
        children=[Column("date"), Column("value")],
    )
    assert_series_equal(offset_feature.calculate(df), pd.Series([np.nan, 1.0, 2.0, 3.0], name="value"))

    window_feature = Feature(
        op=Roll(window_size=2, window_unit="M", aggregation="mean"),
        children=[Column("date"), Column("value")],
    )
    assert_series_equal(window_feature.calculate(df), pd.Series([1.0, 1.5, 2.5, 3.5], name="value"))


def test_roll_month_window():
    df = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"],
            "value": [1.0, 3.0, 5.0, 7.0],
        },
    )
    feature = Feature(
        op=Roll(window_size=2, window_unit="M", aggregation="mean"),
        children=[Column("date"), Column("value")],
    )
    # Left-open (t - 2M, t]: previous month-end is excluded, current + one prior month-end remain.
    assert_series_equal(feature.calculate(df), pd.Series([1.0, 2.0, 4.0, 6.0], name="value"))


def test_roll_with_offset_and_groups():
    df = pd.DataFrame(
        {
            "date": [
                "2024-05-05",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-08",
                "2024-05-05",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-08",
            ],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "value": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        },
    )

    def check_roll_groups(
        window_size: int, window_unit: str, offset_size: int, aggregation: str, expected_values: List[float]
    ):
        feature = Feature(
            op=Roll(window_size=window_size, window_unit=window_unit, offset_size=offset_size, aggregation=aggregation),
            children=[Column("date"), Column("group"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_roll_groups(2, "d", 0, "mean", [1.0, 1.5, 2.5, 3.5, 3.5, 10.0, 15.0, 25.0, 35.0, 35.0])
    check_roll_groups(2, "d", 1, "mean", [np.nan, 1.0, 1.5, 2.5, 2.5, np.nan, 10.0, 15.0, 25.0, 25.0])
    check_roll_groups(3, "d", 1, "median", [np.nan, 1.0, 1.5, 2.0, 2.0, np.nan, 10.0, 15.0, 20.0, 20.0])


def test_roll_hours():
    df = pd.DataFrame(
        {
            "date": [
                "2024-05-05 22:00",
                "2024-05-06 23:00",
                "2024-05-07 00:00",
                "2024-05-08 01:00",
                "2024-05-08 02:00",
            ],
            "value": [1, 2, 3, 4, 5],
        },
    )

    def check_roll(window_size: int, window_unit: str, aggregation: str, expected_values: List[float]):
        feature = Feature(
            op=Roll(window_size=window_size, window_unit=window_unit, aggregation=aggregation),
            children=[Column("date"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_roll(1, "d", "mean", [1.0, 2.0, 2.5, 4.0, 4.5])
    check_roll(2, "d", "median", [1.0, 1.5, 2.0, 3.0, 3.5])
    check_roll(2, "H", "norm_mean", [1.0, 1.0, 1.2, 1.0, 1.111111])


def test_roll_operator_parse_obj():
    roll = Roll(window_size=5, window_unit="d", aggregation="mean", offset_size=1, offset_unit="D")

    roll_dict = roll.get_params()
    parsed_roll = pydantic_parse_method(Roll)(roll_dict)

    assert parsed_roll.window_size == 5
    assert parsed_roll.window_unit == "d"
    assert parsed_roll.aggregation == "mean"
    assert parsed_roll.offset_size == 1
    assert parsed_roll.offset_unit == "D"
    assert parsed_roll.to_formula() == "roll_5d_mean_offset_1D"
