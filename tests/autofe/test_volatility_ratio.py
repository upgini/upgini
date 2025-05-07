import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries.volatility import VolatilityRatio, RollingVolatility
from upgini.autofe.utils import pydantic_parse_method


def create_volatility_ratio(
    short_window_size,
    window_size,
    offset_size=0,
    offset_unit="D",
):
    return Feature(
        op=VolatilityRatio(
            short_window_size=short_window_size,
            window_size=window_size,
            offset_size=offset_size,
            offset_unit=offset_unit,
        ),
        children=[Column("date"), Column("value")],
    )


def compute_expected_ratio(df, short_size, long_size=None):
    # Calculate short-term volatility
    short_vol = RollingVolatility(window_size=short_size)
    short_feature = Feature(op=short_vol, children=[Column("date"), Column("value")])
    short_result = short_feature.calculate(df)

    # Calculate long-term volatility
    long_vol = RollingVolatility(window_size=long_size)
    long_feature = Feature(op=long_vol, children=[Column("date"), Column("value")])
    long_result = long_feature.calculate(df)

    # Calculate ratio with handling of division by zero
    ratio = VolatilityRatio._handle_div_errors(short_result / long_result)
    return ratio


def test_volatility_ratio_calculate():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06", "2024-05-07"],
            "value": [100, 110, 99, 121, 115, 105, 112],
        }
    )

    def check_volatility_ratio(short_window_size, window_size, expected_values):
        feature = create_volatility_ratio(short_window_size=short_window_size, window_size=window_size)
        result = feature.calculate(df)
        expected_result = pd.Series(expected_values, name="value")
        assert_series_equal(result, expected_result)

    # Test with short=2, long=4
    expected_values = compute_expected_ratio(df, short_size=2, long_size=4)
    check_volatility_ratio(
        short_window_size=2,
        window_size=4,
        expected_values=expected_values,
    )

    # Test with short=3, long=6
    expected_values = compute_expected_ratio(df, short_size=3, long_size=6)
    check_volatility_ratio(
        short_window_size=3,
        window_size=6,
        expected_values=expected_values,
    )


def test_volatility_ratio_with_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-01", "2024-05-02", "2024-05-02", "2024-05-03", "2024-05-03"],
            "group": ["A", "B", "A", "B", "A", "B"],
            "value": [100, 200, 110, 220, 99, 198],
        }
    )

    short_window = 2
    long_window = 3

    feature = Feature(
        op=VolatilityRatio(short_window_size=short_window, window_size=long_window),
        children=[Column("date"), Column("group"), Column("value")],
    )

    result = feature.calculate(df)

    # Calculate expected values for each group
    expected_results = []
    for _, group_df in df.groupby("group"):
        ratio_result = compute_expected_ratio(group_df, short_size=short_window, long_size=long_window)
        expected_results.append(ratio_result)

    # Sort by original index and extract values
    expected_series = pd.concat(expected_results).sort_index()

    assert_series_equal(result, expected_series)


def test_volatility_ratio_with_missing_values():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-06"],
            "value": [100, np.nan, 99, 121, 115],
        }
    )

    short_window = 2
    long_window = 4

    feature = create_volatility_ratio(short_window_size=short_window, window_size=long_window)
    result = feature.calculate(df)

    expected_result = compute_expected_ratio(df, short_size=short_window, long_size=long_window)
    assert_series_equal(result, expected_result)


def test_volatility_ratio_formula():
    # Test to_formula
    op = VolatilityRatio(short_window_size=5, short_window_unit="D", window_size=20, window_unit="D")
    assert op.to_formula() == "vol_ratio_5D_to_20D"

    # Test from_formula
    op_from_formula = VolatilityRatio.from_formula("vol_ratio_10D_to_30D")
    assert op_from_formula is not None
    assert op_from_formula.short_window_size == 10
    assert op_from_formula.short_window_unit == "D"
    assert op_from_formula.window_size == 30
    assert op_from_formula.window_unit == "D"

    # Test different time units
    op_week = VolatilityRatio(short_window_size=1, short_window_unit="W", window_size=4, window_unit="W")
    assert op_week.to_formula() == "vol_ratio_1W_to_4W"

    op_from_week = VolatilityRatio.from_formula("vol_ratio_1W_to_4W")
    assert op_from_week is not None
    assert op_from_week.short_window_size == 1
    assert op_from_week.short_window_unit == "W"
    assert op_from_week.window_size == 4
    assert op_from_week.window_unit == "W"

    # Test lowercase frequency
    op_hour = VolatilityRatio(short_window_size=3, short_window_unit="h", window_size=12, window_unit="h")
    assert op_hour.to_formula() == "vol_ratio_3h_to_12h"

    op_from_hour = VolatilityRatio.from_formula("vol_ratio_3h_to_12h")
    assert op_from_hour is not None
    assert op_from_hour.short_window_size == 3
    assert op_from_hour.short_window_unit == "h"
    assert op_from_hour.window_size == 12
    assert op_from_hour.window_unit == "h"

    # Test with offset
    op_with_offset = VolatilityRatio(
        short_window_size=3,
        short_window_unit="D",
        window_size=10,
        window_unit="D",
        offset_size=1,
        offset_unit="D",
    )
    assert op_with_offset.to_formula() == "vol_ratio_3D_to_10D_offset_1D"

    op_from_offset_formula = VolatilityRatio.from_formula("vol_ratio_5D_to_15D_offset_2D")
    assert op_from_offset_formula is not None
    assert op_from_offset_formula.short_window_size == 5
    assert op_from_offset_formula.short_window_unit == "D"
    assert op_from_offset_formula.window_size == 15
    assert op_from_offset_formula.window_unit == "D"
    assert op_from_offset_formula.offset_size == 2
    assert op_from_offset_formula.offset_unit == "D"

    # Test invalid formula
    invalid_op = VolatilityRatio.from_formula("vol_ratio_5D_15D")  # Missing "_to_"
    assert invalid_op is None


def test_volatility_ratio_with_offset():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06", "2024-05-07"],
            "value": [100, 110, 99, 121, 115, 105, 112],
        }
    )

    # No offset
    feature_no_offset = create_volatility_ratio(short_window_size=2, window_size=4)
    result_no_offset = feature_no_offset.calculate(df)

    # With offset
    feature_with_offset = create_volatility_ratio(short_window_size=2, window_size=4, offset_size=1, offset_unit="D")
    result_with_offset = feature_with_offset.calculate(df)

    # The offset should shift the result by one day
    assert_series_equal(
        result_with_offset.iloc[2:].reset_index(drop=True),
        result_no_offset.iloc[1:-1].reset_index(drop=True),
        check_names=False,
    )


def test_volatility_ratio_parse_obj():
    volatility_ratio = VolatilityRatio(
        short_window_size=3, short_window_unit="d", window_size=10, window_unit="D", offset_size=1, offset_unit="D"
    )

    vol_ratio_dict = volatility_ratio.get_params()
    parsed_vol_ratio = pydantic_parse_method(VolatilityRatio)(vol_ratio_dict)

    assert parsed_vol_ratio.short_window_size == 3
    assert parsed_vol_ratio.short_window_unit == "d"
    assert parsed_vol_ratio.window_size == 10
    assert parsed_vol_ratio.window_unit == "D"
    assert parsed_vol_ratio.offset_size == 1
    assert parsed_vol_ratio.offset_unit == "D"
    assert parsed_vol_ratio.to_formula() == "vol_ratio_3d_to_10D_offset_1D"
