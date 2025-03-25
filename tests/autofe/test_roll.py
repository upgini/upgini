from typing import List

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries import Roll


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


def test_roll_from_formula():
    roll = Roll.from_formula("roll_3d_mean")
    assert roll.window_size == 3
    assert roll.window_unit == "d"
    assert roll.aggregation == "mean"
    assert roll.to_formula() == "roll_3d_mean"

    roll = Roll.from_formula("roll_10D_offset_1D_max")
    assert roll.window_size == 10
    assert roll.window_unit == "D"
    assert roll.offset_size == 1
    assert roll.offset_unit == "D"
    assert roll.aggregation == "max"
    assert roll.to_formula() == "roll_10D_offset_1D_max"

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
    check_roll(2, "d", 0, "last", [1.0, 2.0, 3.0, 4.0, 4.0])
    check_roll(2, "d", 1, "last", [np.nan, 1.0, 2.0, 3.0, 3.0])
    check_roll(3, "d", 1, "last", [np.nan, 1.0, 2.0, 3.0, 3.0])


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
