import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries.volatility import RollingVolatility


def test_rolling_volatility_calculate():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
            "value": [100, 110, 99, 121, 115],
        }
    )

    def check_volatility(window_size, expected_values):
        feature = Feature(
            op=RollingVolatility(window_size=window_size, window_unit="D"),
            children=[Column("date"), Column("value")],
        )
        result = feature.calculate(df)
        expected_result = pd.Series(expected_values, name="value")
        assert_series_equal(result, expected_result)

    check_volatility(
        window_size=3,
        expected_values=[np.nan, 0.07071067811865482, 0.10000000000000003, 0.16266808337776115, 0.17332473642609442],
    )

    # Calculate the expected values for 2-day rolling window
    check_volatility(
        window_size=2,
        expected_values=[np.nan, 0.07071067811865482, 0.14142135623730956, 0.22784551838233202, 0.1921979864382168],
    )

    # Calculate the expected values for 5-day rolling window
    check_volatility(
        window_size=5,
        expected_values=[np.nan, 0.07071067811865482, 0.10000000000000003, 0.1378852627332318, 0.12833643782026619],
    )


def test_rolling_volatility_with_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-01", "2024-05-02", "2024-05-02", "2024-05-03"],
            "group": ["A", "B", "A", "B", "A"],
            "value": [100, 200, 110, 220, 99],
        }
    )

    feature = Feature(
        op=RollingVolatility(window_size=2, window_unit="D"),
        children=[Column("date"), Column("group"), Column("value")],
    )

    result = feature.calculate(df)

    # Calculate expected values for each group
    group_a_returns = pd.Series([0.0, 0.1, -0.1])
    group_b_returns = pd.Series([0.0, 0.1])

    # Calculate rolling std for each group
    rolling_vol_a = group_a_returns.rolling(window=2, min_periods=1).std()
    rolling_vol_b = group_b_returns.rolling(window=2, min_periods=1).std()

    # Combine results
    expected_series = pd.Series(
        [
            rolling_vol_a.iloc[0],
            rolling_vol_b.iloc[0],
            rolling_vol_a.iloc[1],
            rolling_vol_b.iloc[1],
            rolling_vol_a.iloc[2],
        ],
        name="value",
    )

    assert_series_equal(result, expected_series)


def test_rolling_volatility_with_missing_values():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-06"],
            "value": [100, np.nan, 99, 121, 115],
        }
    )

    feature = Feature(
        op=RollingVolatility(window_size=3, window_unit="D"),
        children=[Column("date"), Column("value")],
    )

    result = feature.calculate(df)

    expected_result = pd.Series(
        [np.nan, 0.0, 0.005773502691896263, 0.13128206055614883, 0.1571348402636773], name="value"
    )

    assert_series_equal(result, expected_result)


def test_rolling_volatility_formula():
    # Test to_formula
    op = RollingVolatility(window_size=5, window_unit="D")
    assert op.to_formula() == "roll_vol_5D"

    # Test from_formula
    op_from_formula = RollingVolatility.from_formula("roll_vol_10D")
    assert op_from_formula is not None
    assert op_from_formula.window_size == 10
    assert op_from_formula.window_unit == "D"

    # Test different time unit
    op_week = RollingVolatility(window_size=2, window_unit="W")
    assert op_week.to_formula() == "roll_vol_2W"

    op_from_week = RollingVolatility.from_formula("roll_vol_2W")
    assert op_from_week is not None
    assert op_from_week.window_size == 2
    assert op_from_week.window_unit == "W"

    # Test lowercase frequency
    op_hour = RollingVolatility(window_size=3, window_unit="h")
    assert op_hour.to_formula() == "roll_vol_3h"

    op_from_hour = RollingVolatility.from_formula("roll_vol_3h")
    assert op_from_hour is not None
    assert op_from_hour.window_size == 3
    assert op_from_hour.window_unit == "h"

    # Test with offset
    op_with_offset = RollingVolatility(window_size=5, window_unit="D", offset_size=1, offset_unit="D")
    assert op_with_offset.to_formula() == "roll_vol_5D_offset_1D"

    op_from_offset_formula = RollingVolatility.from_formula("roll_vol_7D_offset_2D")
    assert op_from_offset_formula is not None
    assert op_from_offset_formula.window_size == 7
    assert op_from_offset_formula.window_unit == "D"
    assert op_from_offset_formula.offset_size == 2
    assert op_from_offset_formula.offset_unit == "D"

    # Test offset with lowercase frequency
    op_offset_hour = RollingVolatility(window_size=4, window_unit="h", offset_size=3, offset_unit="h")
    assert op_offset_hour.to_formula() == "roll_vol_4h_offset_3h"

    op_from_offset_hour = RollingVolatility.from_formula("roll_vol_4h_offset_3h")
    assert op_from_offset_hour is not None
    assert op_from_offset_hour.window_size == 4
    assert op_from_offset_hour.window_unit == "h"
    assert op_from_offset_hour.offset_size == 3
    assert op_from_offset_hour.offset_unit == "h"

    # Test invalid formula
    invalid_op = RollingVolatility.from_formula("roll_volatility_5D")
    assert invalid_op is None


def test_rolling_volatility_with_offset():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
            "value": [100, 110, 99, 121, 115],
        }
    )

    # No offset
    feature_no_offset = Feature(
        op=RollingVolatility(window_size=2, window_unit="D"),
        children=[Column("date"), Column("value")],
    )
    result_no_offset = feature_no_offset.calculate(df)

    # With offset
    feature_with_offset = Feature(
        op=RollingVolatility(window_size=2, window_unit="D", offset_size=1, offset_unit="D"),
        children=[Column("date"), Column("value")],
    )
    result_with_offset = feature_with_offset.calculate(df)

    # The offset should shift the result by one day
    assert_series_equal(
        result_with_offset.iloc[2:].reset_index(drop=True),
        result_no_offset.iloc[1:-1].reset_index(drop=True),
        check_names=False,
    )
