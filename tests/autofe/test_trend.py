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


def test_trend_coef_formula():
    # Test basic formula
    op = TrendCoefficient()
    assert op.to_formula() == "trend_coef"

    # Test with offset
    op_with_offset = TrendCoefficient(offset_size=1, offset_unit="D")
    assert op_with_offset.to_formula() == "trend_coef_offset_1D"

    # Test from_formula
    op_from_formula = TrendCoefficient.from_formula("trend_coef")
    assert op_from_formula is not None
    assert op_from_formula.offset_size == 0

    op_from_offset_formula = TrendCoefficient.from_formula("trend_coef_offset_2D")
    assert op_from_offset_formula is not None
    assert op_from_offset_formula.offset_size == 2
    assert op_from_offset_formula.offset_unit == "D"

    # Test invalid formula
    invalid_op = TrendCoefficient.from_formula("trend_coefficient")
    assert invalid_op is None


def test_trend_coef_with_offset():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"] * 2,
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "value": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        }
    )

    # No offset
    feature_no_offset = Feature(
        op=TrendCoefficient(),
        children=[Column("date"), Column("group"), Column("value")],
    )
    result_no_offset = feature_no_offset.calculate(df)

    # With offset
    feature_with_offset = Feature(
        op=TrendCoefficient(offset_size=1, offset_unit="D"),
        children=[Column("date"), Column("group"), Column("value")],
    )
    result_with_offset = feature_with_offset.calculate(df)

    # The offset should shift the results - for trend coefficient, values should be the same but shifted
    # Check group A
    assert_series_equal(
        result_with_offset.iloc[2:5].reset_index(drop=True),
        result_no_offset.iloc[1:4].reset_index(drop=True),
        check_names=False,
    )

    # Check group B
    assert_series_equal(
        result_with_offset.iloc[7:10].reset_index(drop=True),
        result_no_offset.iloc[6:9].reset_index(drop=True),
        check_names=False,
    )


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
            "date": ["2024-05-06", "2024-05-06", "2024-05-07", "2024-05-07", "2024-05-09"],
            "group": ["A", "B", "A", "B", "A"],
            "value": [1, 10, np.nan, 20, 3],
        },
    )

    feature = Feature(
        op=TrendCoefficient(),
        children=[Column("date"), Column("group"), Column("value")],
    )
    expected_res = pd.Series([0.6, 10.0, 0.6, 10.0, 0.6], name="value")
    assert_series_equal(feature.calculate(df), expected_res)


def test_trend_coef_polyfit_fails():
    # Create a dataframe with only one value which will cause np.polyfit to fail
    # as it can't fit a line to a single point
    df = pd.DataFrame(
        {
            "date": ["2024-05-06"],
            "value": [5.0],
        },
    )

    feature = Feature(
        op=TrendCoefficient(),
        children=[Column("date"), Column("value")],
    )

    # When polyfit fails, the method should return 0
    expected_res = pd.Series([0.0], name="value")
    assert_series_equal(feature.calculate(df), expected_res)

    # Alternative test case with constant values
    df_constant = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-07", "2024-05-08"],
            "value": [5, 5, 5],  # Constant values can also cause LinAlgError
        },
    )

    feature_constant = Feature(
        op=TrendCoefficient(),
        children=[Column("date"), Column("value")],
    )

    expected_res_constant = pd.Series([0.0, 0.0, 0.0], name="value")
    assert_series_equal(feature_constant.calculate(df_constant), expected_res_constant)


def test_trend_coefficient_parse_obj():
    trend_coef = TrendCoefficient(offset_size=2, offset_unit="D")

    trend_dict = trend_coef.get_params()
    parsed_trend = TrendCoefficient.parse_obj(trend_dict)

    assert parsed_trend.offset_size == 2
    assert parsed_trend.offset_unit == "D"
    assert parsed_trend.to_formula() == "trend_coef_offset_2D"
