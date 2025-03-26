import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries import TrendCoefficient


def test_trend_coef():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-07", "2024-05-08", "2024-05-09"],
            "value": [1, 2, 3, 4],
        },
    )

    feature = Feature(
        op=TrendCoefficient(),
        children=[Column("date"), Column("value")],
    )
    expected_res = pd.Series([1.0] * 4, name="value")
    assert_series_equal(feature.calculate(df), expected_res)


def test_trend_coef_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-06", "2024-05-07", "2024-05-07", "2024-05-08"],
            "group": ["A", "B", "A", "B", "A"],
            "value": [1, 10, 2, 20, 3],
        },
    )

    feature = Feature(
        op=TrendCoefficient(),
        children=[Column("date"), Column("group"), Column("value")],
    )
    expected_res = pd.Series([1.0, 10.0, 1.0, 10.0, 1.0], name="value")
    assert_series_equal(feature.calculate(df), expected_res)


def test_trend_coef_missing():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-06", "2024-05-07", "2024-05-07", "2024-05-08"],
            "group": ["A", "B", "A", "B", "A"],
            "value": [1, 10, np.nan, 20, 3],
        },
    )

    feature = Feature(
        op=TrendCoefficient(),
        children=[Column("date"), Column("group"), Column("value")],
    )
    expected_res = pd.Series([1.0, 10.0, 1.0, 10.0, 1.0], name="value")
    assert_series_equal(feature.calculate(df), expected_res)
