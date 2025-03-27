import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries import CrossSeriesInteraction
from upgini.autofe.binary import Add, Subtract, Multiply, Divide


def test_cross_series_basic():
    df = pd.DataFrame(
        {
            "date": [
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
            ],
            "category": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "value": [10, 20, 30, 40, 5, 15, 25, 35],
        },
    )

    def check_op(op, expected_values):
        feature = Feature(
            op=CrossSeriesInteraction(op=op, descriptor_indices=[1], left_descriptor=["A"], right_descriptor=["B"]),
            children=[Column("date"), Column("category"), Column("value")],
        )
        result = feature.calculate(df)
        expected_res = pd.Series(expected_values, name="value", dtype=float)
        assert_series_equal(result, expected_res)

    # Test addition
    check_op(Add(), [10 + 5, 20 + 15, 30 + 25, 40 + 35, 10 + 5, 20 + 15, 30 + 25, 40 + 35])

    # Test subtraction
    check_op(Subtract(), [10 - 5, 20 - 15, 30 - 25, 40 - 35, 10 - 5, 20 - 15, 30 - 25, 40 - 35])

    # Test multiplication
    check_op(Multiply(), [10 * 5, 20 * 15, 30 * 25, 40 * 35, 10 * 5, 20 * 15, 30 * 25, 40 * 35])

    # Test division
    check_op(Divide(), [10 / 5, 20 / 15, 30 / 25, 40 / 35, 10 / 5, 20 / 15, 30 / 25, 40 / 35])


def test_cross_series_formula():
    # Test formula generation
    cross_add = CrossSeriesInteraction(op=Add())
    assert cross_add.to_formula() == "cross_add"

    cross_sub = CrossSeriesInteraction(op=Subtract())
    assert cross_sub.to_formula() == "cross_sub"

    cross_mul = CrossSeriesInteraction(op=Multiply())
    assert cross_mul.to_formula() == "cross_mul"

    cross_div = CrossSeriesInteraction(op=Divide())
    assert cross_div.to_formula() == "cross_div"

    # Test with offset
    cross_add_offset = CrossSeriesInteraction(op=Add(), offset_size=2, offset_unit="D")
    assert cross_add_offset.to_formula() == "cross_add_offset_2D"

    # Test formula parsing
    parsed_add = CrossSeriesInteraction.from_formula("cross_add")
    assert isinstance(parsed_add, CrossSeriesInteraction)
    assert parsed_add.interaction_op.alias == "add"

    parsed_sub = CrossSeriesInteraction.from_formula("cross_sub")
    assert isinstance(parsed_sub, CrossSeriesInteraction)
    assert parsed_sub.interaction_op.alias == "sub"

    parsed_mul = CrossSeriesInteraction.from_formula("cross_mul")
    assert isinstance(parsed_mul, CrossSeriesInteraction)
    assert parsed_mul.interaction_op.alias == "mul"

    parsed_div = CrossSeriesInteraction.from_formula("cross_div")
    assert isinstance(parsed_div, CrossSeriesInteraction)
    assert parsed_div.interaction_op.alias == "div"

    # Test offset parsing
    parsed_with_offset = CrossSeriesInteraction.from_formula("cross_add_offset_2D")
    assert isinstance(parsed_with_offset, CrossSeriesInteraction)
    assert parsed_with_offset.interaction_op.alias == "add"
    assert parsed_with_offset.offset_size == 2
    assert parsed_with_offset.offset_unit == "D"


def test_cross_series_with_missing_data():
    df = pd.DataFrame(
        {
            "date": [
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-10",
            ],
            "category": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "value": [10, np.nan, 30, 40, 5, 15, np.nan, 35],
        },
    )

    feature = Feature(
        op=CrossSeriesInteraction(op=Add(), descriptor_indices=[1], left_descriptor=["A"], right_descriptor=["B"]),
        children=[Column("date"), Column("category"), Column("value")],
    )

    result = feature.calculate(df)
    expected_values = [10 + 5, np.nan, np.nan, np.nan, 10 + 5, np.nan, np.nan, np.nan]
    expected_res = pd.Series(expected_values, name="value")
    assert_series_equal(result, expected_res)


def test_cross_series_with_offset():
    df = pd.DataFrame(
        {
            "date": [
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
            ],
            "category": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "value": [10, 20, 30, 40, 5, 15, 25, 35],
        },
    )

    feature = Feature(
        op=CrossSeriesInteraction(
            op=Add(),
            descriptor_indices=[1],
            left_descriptor=["A"],
            right_descriptor=["B"],
            offset_size=1,
            offset_unit="D",
        ),
        children=[Column("date"), Column("category"), Column("value")],
    )

    result = feature.calculate(df)

    # Expected: NaN for first day of each series, then the offset values are added
    expected_values = [np.nan, 10 + 5, 20 + 15, 30 + 25, np.nan, 10 + 5, 20 + 15, 30 + 25]
    expected_res = pd.Series(expected_values, name="value")
    assert_series_equal(result, expected_res)


def test_cross_series_complex_descriptor():
    df = pd.DataFrame(
        {
            "date": [
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
                "2024-05-06",
                "2024-05-07",
                "2024-05-08",
                "2024-05-09",
            ],
            "region": [
                "East",
                "East",
                "East",
                "East",
                "West",
                "West",
                "West",
                "West",
                "East",
                "East",
                "East",
                "East",
                "West",
                "West",
                "West",
                "West",
            ],
            "product": ["X", "X", "X", "X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 15, 25, 35, 45, 55, 65, 75, 85],
        },
    )

    # Interaction between East/X and West/Y product lines
    feature = Feature(
        op=CrossSeriesInteraction(
            op=Divide(),
            descriptor_indices=[1, 2],
            left_descriptor=["East", "X"],  # East region, product X
            right_descriptor=["West", "Y"],  # West region, product Y
        ),
        children=[Column("date"), Column("region"), Column("product"), Column("value")],
    )

    result = feature.calculate(df)

    # Expected: Division of East/X values by West/Y values
    expected_values = [
        # East/X divided by West/Y in place of East/X
        10 / 55,
        20 / 65,
        30 / 75,
        40 / 85,
        # The remaining values aren't part of either descriptor group
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        # East/X divided by West/Y in place of West/Y
        10 / 55,
        20 / 65,
        30 / 75,
        40 / 85,
    ]
    expected_res = pd.Series(expected_values, name="value")
    assert_series_equal(result, expected_res)
