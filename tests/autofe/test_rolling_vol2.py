import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries.volatility import RollingVolatility, RollingVolatility2


def first_order_vol(window_size: int, abs_returns: bool = False):
    return Feature(
        op=RollingVolatility(window_size=window_size, window_unit="D", abs_returns=abs_returns),
        children=[Column("date"), Column("value")],
    )


def calc_second_order_vol(df: pd.DataFrame, window_size: int):
    df2 = df.copy()
    df2["value"] = first_order_vol(window_size, abs_returns=True).calculate(df)
    return first_order_vol(window_size, abs_returns=False).calculate(df2)


def test_rolling_volatility2_calculate():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
            "value": [100, 110, 99, 121, 115],
        }
    )

    def check_volatility(window_size, expected_values):
        feature = Feature(
            op=RollingVolatility2(window_size=window_size, window_unit="D"),
            children=[Column("date"), Column("value")],
        )
        result = feature.calculate(df)
        expected_result = pd.Series(expected_values, name="value")
        assert_series_equal(result, expected_result)

    expected_values = calc_second_order_vol(df, window_size=3)

    check_volatility(
        window_size=3,
        expected_values=expected_values,
    )


def test_rolling_volatility2_with_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-01", "2024-05-02", "2024-05-02", "2024-05-03"],
            "group": ["A", "B", "A", "B", "A"],
            "value": [100, 200, 110, 220, 99],
        }
    )

    window_size = 2

    feature = Feature(
        op=RollingVolatility2(window_size=window_size, window_unit="D"),
        children=[Column("date"), Column("group"), Column("value")],
    )

    result = feature.calculate(df)

    # Calculate expected values for each group
    expected_results = []
    for _, group_df in df.groupby("group"):
        vol_result = calc_second_order_vol(group_df, window_size)

        expected_results.append(vol_result)

    # Sort by original index and extract values
    expected_series = pd.concat(expected_results).sort_index()

    assert_series_equal(result, expected_series)


def test_rolling_volatility2_with_missing_values():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-06"],
            "value": [100, np.nan, 99, 121, 115],
        }
    )

    window_size = 3

    feature = Feature(
        op=RollingVolatility2(window_size=window_size, window_unit="D"),
        children=[Column("date"), Column("value")],
    )

    result = feature.calculate(df)
    expected_result = calc_second_order_vol(df, window_size=3)

    assert_series_equal(result, expected_result)


def test_rolling_volatility2_formula():
    # Test to_formula
    op = RollingVolatility2(window_size=5, window_unit="D")
    assert op.to_formula() == "roll_vol2_5D"

    # Test from_formula
    op_from_formula = RollingVolatility2.from_formula("roll_vol2_10D")
    assert op_from_formula is not None
    assert op_from_formula.window_size == 10
    assert op_from_formula.window_unit == "D"

    # Test different time unit
    op_week = RollingVolatility2(window_size=2, window_unit="W")
    assert op_week.to_formula() == "roll_vol2_2W"

    op_from_week = RollingVolatility2.from_formula("roll_vol2_2W")
    assert op_from_week is not None
    assert op_from_week.window_size == 2
    assert op_from_week.window_unit == "W"

    # Test lowercase frequency
    op_hour = RollingVolatility2(window_size=3, window_unit="h")
    assert op_hour.to_formula() == "roll_vol2_3h"

    op_from_hour = RollingVolatility2.from_formula("roll_vol2_3h")
    assert op_from_hour is not None
    assert op_from_hour.window_size == 3
    assert op_from_hour.window_unit == "h"

    # Test with offset
    op_with_offset = RollingVolatility2(window_size=5, window_unit="D", offset_size=1, offset_unit="D")
    assert op_with_offset.to_formula() == "roll_vol2_5D_offset_1D"

    op_from_offset_formula = RollingVolatility2.from_formula("roll_vol2_7D_offset_2D")
    assert op_from_offset_formula is not None
    assert op_from_offset_formula.window_size == 7
    assert op_from_offset_formula.window_unit == "D"
    assert op_from_offset_formula.offset_size == 2
    assert op_from_offset_formula.offset_unit == "D"

    # Test offset with lowercase frequency
    op_offset_hour = RollingVolatility2(window_size=4, window_unit="h", offset_size=3, offset_unit="h")
    assert op_offset_hour.to_formula() == "roll_vol2_4h_offset_3h"

    op_from_offset_hour = RollingVolatility2.from_formula("roll_vol2_4h_offset_3h")
    assert op_from_offset_hour is not None
    assert op_from_offset_hour.window_size == 4
    assert op_from_offset_hour.window_unit == "h"
    assert op_from_offset_hour.offset_size == 3
    assert op_from_offset_hour.offset_unit == "h"

    # Test invalid formula
    invalid_op = RollingVolatility2.from_formula("roll_vol_5D")
    assert invalid_op is None


def test_rolling_volatility2_with_offset():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
            "value": [100, 110, 99, 121, 115],
        }
    )

    # No offset
    feature_no_offset = Feature(
        op=RollingVolatility2(window_size=2, window_unit="D"),
        children=[Column("date"), Column("value")],
    )
    result_no_offset = feature_no_offset.calculate(df)

    # With offset
    feature_with_offset = Feature(
        op=RollingVolatility2(window_size=2, window_unit="D", offset_size=1, offset_unit="D"),
        children=[Column("date"), Column("value")],
    )
    result_with_offset = feature_with_offset.calculate(df)

    # The offset should shift the result by one day
    assert_series_equal(
        result_with_offset.iloc[2:].reset_index(drop=True),
        result_no_offset.iloc[1:-1].reset_index(drop=True),
        check_names=False,
    )
