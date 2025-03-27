import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries.volatility import VolatilityBase, EWMAVolatility


def test_volatility_base_get_returns():
    df = pd.Series(
        [100, 110, 99, 121, np.nan],
        index=pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-05", "2024-05-06"]),
        name="value",
    )

    returns = VolatilityBase._get_returns(df, "1D")

    expected_returns = pd.Series([0.0, 0.1, -0.1, 0.0, 0.0], index=df.index, name="value")

    assert_series_equal(returns, expected_returns)


def test_ewma_volatility_calculate():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
            "value": [100, 110, 99, 121, 115],
        }
    )

    def check_volatility(window_size, expected_values):
        feature = Feature(
            op=EWMAVolatility(window_size=window_size),
            children=[Column("date"), Column("value")],
        )
        result = feature.calculate(df)
        expected_result = pd.Series(expected_values, name="value")
        assert_series_equal(result, expected_result)

    check_volatility(
        window_size=3,
        expected_values=[np.nan, 0.07071067811865477, 0.1164964745021435, 0.17599803590374327, 0.15403830565097604],
    )

    check_volatility(
        window_size=2,
        expected_values=[np.nan, 0.07071067811865481, 0.12403473458920847, 0.19273115769200153, 0.16516062582879912],
    )

    check_volatility(
        window_size=5,
        expected_values=[np.nan, 0.07071067811865482, 0.1100239208440362, 0.16118516299625862, 0.14428760780515082],
    )


def test_ewma_volatility_with_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-01", "2024-05-02", "2024-05-02", "2024-05-03"],
            "group": ["A", "B", "A", "B", "A"],
            "value": [100, 200, 110, 220, 99],
        }
    )

    feature = Feature(
        op=EWMAVolatility(window_size=2),
        children=[Column("date"), Column("group"), Column("value")],
    )

    result = feature.calculate(df)

    # Calculate expected values for each group
    group_a = pd.Series([0.0, 0.1, -0.1])
    group_b = pd.Series([0.0, 0.1])

    # Calculate EWMA std for each group
    ewma_vol_a = group_a.ewm(span=2).std()
    ewma_vol_b = group_b.ewm(span=2).std()

    # Combine results
    expected_series = pd.Series(
        [ewma_vol_a.iloc[0], ewma_vol_b.iloc[0], ewma_vol_a.iloc[1], ewma_vol_b.iloc[1], ewma_vol_a.iloc[2]],
        name="value",
    )

    assert_series_equal(result, expected_series)


def test_ewma_volatility_with_missing_values():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-06"],
            "value": [100, np.nan, 99, 121, 115],
        }
    )

    feature = Feature(
        op=EWMAVolatility(window_size=3),
        children=[Column("date"), Column("value")],
    )

    result = feature.calculate(df)
    expected_result = pd.Series(
        [np.nan, 0.0, 0.006546536707079771, 0.14422339618471988, 0.12207508249690784], name="value"
    )

    assert_series_equal(result, expected_result)


def test_ewma_volatility_formula():
    # Test to_formula
    op = EWMAVolatility(window_size=5)
    assert op.to_formula() == "ewma_vol_5"

    # Test from_formula
    op_from_formula = EWMAVolatility.from_formula("ewma_vol_10")
    assert op_from_formula is not None
    assert op_from_formula.window_size == 10

    # Test with offset
    op_with_offset = EWMAVolatility(window_size=5, offset_size=1, offset_unit="D")
    assert op_with_offset.to_formula() == "ewma_vol_5_offset_1D"

    op_from_offset_formula = EWMAVolatility.from_formula("ewma_vol_7_offset_2D")
    assert op_from_offset_formula is not None
    assert op_from_offset_formula.window_size == 7
    assert op_from_offset_formula.offset_size == 2
    assert op_from_offset_formula.offset_unit == "D"

    # Test offset with lowercase frequency
    op_offset_hour = EWMAVolatility(window_size=4, offset_size=3, offset_unit="h")
    assert op_offset_hour.to_formula() == "ewma_vol_4_offset_3h"

    op_from_offset_hour = EWMAVolatility.from_formula("ewma_vol_4_offset_3h")
    assert op_from_offset_hour is not None
    assert op_from_offset_hour.window_size == 4
    assert op_from_offset_hour.offset_size == 3
    assert op_from_offset_hour.offset_unit == "h"

    # Test invalid formula
    invalid_op = EWMAVolatility.from_formula("ewma_volatility_5")
    assert invalid_op is None


def test_ewma_volatility_with_offset():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
            "value": [100, 110, 99, 121, 115],
        }
    )

    # No offset
    feature_no_offset = Feature(
        op=EWMAVolatility(window_size=2),
        children=[Column("date"), Column("value")],
    )
    result_no_offset = feature_no_offset.calculate(df)

    # With offset
    feature_with_offset = Feature(
        op=EWMAVolatility(window_size=2, offset_size=1, offset_unit="D"),
        children=[Column("date"), Column("value")],
    )
    result_with_offset = feature_with_offset.calculate(df)

    # The offset should shift the result by one day
    # So the result at position i with offset 1 should be similar to the result at position i-1 without offset
    # We skip the first position as it would be NaN in both cases
    for i in range(2, len(df)):
        # Allow for small floating-point differences
        assert abs(result_with_offset.iloc[i] - result_no_offset.iloc[i - 1]) < 1e-10 or (
            pd.isna(result_with_offset.iloc[i]) and pd.isna(result_no_offset.iloc[i - 1])
        )
