from typing import List

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries import Lag
from upgini.autofe.utils import pydantic_parse_method


def test_lag_date():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-07", "2024-05-08", "2024-05-09"],
            "value": [1, 2, 3, 4],
        },
    )

    def check_lag(lag_size: int, expected_values: List[float]):
        feature = Feature(
            op=Lag(lag_size=lag_size),
            children=[Column("date"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_lag(1, [np.nan, 1.0, 2.0, 3.0])
    check_lag(2, [np.nan, np.nan, 1.0, 2.0])


def test_lag_date_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-06", "---", "2024-05-07", "2024-05-07", "2024-05-07"],
            "f1": ["a", "b", "a", "a", "a", "c"],
            "f2": [1, 2, 1, 1, 1, 2],
            "value": [1, 2, 3, 4, 4, 5],
        },
        index=[9, 8, 7, 6, 5, 4],
    )

    def check_lag(lag_size: int, expected_values: List[float]):
        feature = Feature(
            op=Lag(lag_size=lag_size),
            children=[Column("date"), Column("f1"), Column("f2"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value", index=df.index)
        assert_series_equal(feature.calculate(df), expected_res)

    check_lag(1, [np.nan, np.nan, np.nan, 1.0, 1.0, np.nan])
    check_lag(2, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def test_lag_hours():
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

    def check_lag(lag_size: int, lag_unit: str, expected_values: List[float]):
        feature = Feature(
            op=Lag(lag_size=lag_size, lag_unit=lag_unit),
            children=[Column("date"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_lag(1, "d", [np.nan, 1.0, 1.0, 2.0, 2.0])
    check_lag(2, "d", [np.nan, np.nan, np.nan, 1.0, 1.0])
    check_lag(1, "H", [np.nan, np.nan, 2.0, np.nan, 4.0])


def test_lag_with_offset():
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

    def check_lag(lag_size: int, lag_unit: str, offset_size: int, expected_values: List[float]):
        feature = Feature(
            op=Lag(lag_size=lag_size, lag_unit=lag_unit, offset_size=offset_size),
            children=[Column("date"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_lag(1, "d", 0, [np.nan, 1.0, 2.0, 3.0, 3.0])
    check_lag(1, "d", 1, [np.nan, np.nan, 1.0, 2.0, 2.0])
    check_lag(2, "d", 1, [np.nan, np.nan, np.nan, 1.0, 1.0])


def test_lag_from_formula():
    lag = Lag.from_formula("lag_3d")
    assert lag.lag_size == 3
    assert lag.lag_unit == "d"
    assert lag.to_formula() == "lag_3d"

    lag = Lag.from_formula("lag_10D_offset_1D")
    assert lag.lag_size == 10
    assert lag.lag_unit == "D"
    assert lag.offset_size == 1
    assert lag.offset_unit == "D"
    assert lag.to_formula() == "lag_10D_offset_1D"

    # Test invalid formulas
    lag = Lag.from_formula("not_a_lag_formula")
    assert lag is None

    lag = Lag.from_formula("lag_abc")
    assert lag is None

    # Test that constructed name matches formula pattern
    lag = Lag(lag_size=5, lag_unit="D")
    assert lag.to_formula() == "lag_5D"


def test_lag_parse_obj():
    lag = Lag(lag_size=3, lag_unit="d", offset_size=2, offset_unit="D")

    lag_dict = lag.get_params()
    parsed_lag = pydantic_parse_method(Lag)(lag_dict)

    assert parsed_lag.lag_size == 3
    assert parsed_lag.lag_unit == "d"
    assert parsed_lag.offset_size == 2
    assert parsed_lag.offset_unit == "D"
    assert parsed_lag.to_formula() == "lag_3d_offset_2D"
