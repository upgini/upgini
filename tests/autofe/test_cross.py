import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Column, Feature
from upgini.autofe.timeseries import CrossSeriesInteraction
from upgini.autofe.binary import Add, Subtract, Multiply, Divide
from upgini.autofe.all_operators import find_op


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
            op=CrossSeriesInteraction(
                interaction_op=op, descriptor_indices=[1], left_descriptor=["A"], right_descriptor=["B"]
            ),
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
    cross_add = CrossSeriesInteraction(
        interaction_op=Add(), descriptor_indices=[0], left_descriptor=["A"], right_descriptor=["B"]
    )
    assert cross_add.to_formula() == "cross_add"

    cross_sub = CrossSeriesInteraction(
        interaction_op=Subtract(), descriptor_indices=[0], left_descriptor=["A"], right_descriptor=["B"]
    )
    assert cross_sub.to_formula() == "cross_sub"

    cross_mul = CrossSeriesInteraction(
        interaction_op=Multiply(), descriptor_indices=[0], left_descriptor=["A"], right_descriptor=["B"]
    )
    assert cross_mul.to_formula() == "cross_mul"

    cross_div = CrossSeriesInteraction(
        interaction_op=Divide(), descriptor_indices=[0], left_descriptor=["A"], right_descriptor=["B"]
    )
    assert cross_div.to_formula() == "cross_div"

    # Test with offset
    cross_add_offset = CrossSeriesInteraction(
        interaction_op=Add(),
        descriptor_indices=[0],
        left_descriptor=["A"],
        right_descriptor=["B"],
        offset_size=2,
        offset_unit="D",
    )
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
        op=CrossSeriesInteraction(
            interaction_op=Add(), descriptor_indices=[1], left_descriptor=["A"], right_descriptor=["B"]
        ),
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
            "category": [1, 1, 1, 1, 2, 2, 2, 2],
            "value": [10, 20, 30, 40, 5, 15, 25, 35],
        },
    )

    feature = Feature(
        op=CrossSeriesInteraction(
            interaction_op=Add(),
            descriptor_indices=[1],
            left_descriptor=["1"],
            right_descriptor=["2"],
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
            interaction_op=Divide(),
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


def test_cross_series_validator_empty_indices():

    # Test that empty descriptor_indices raises a ValueError
    with pytest.raises(ValueError) as excinfo:
        CrossSeriesInteraction(
            interaction_op=Add(),
            descriptor_indices=[],  # Empty - should raise error
            left_descriptor=["A"],
            right_descriptor=["B"],
        )

    assert "descriptor_indices cannot be empty" in str(excinfo.value)


def test_cross_series_validator_mismatched_lengths():

    # Test that left_descriptor with incorrect length raises a ValueError
    with pytest.raises(ValueError) as excinfo:
        CrossSeriesInteraction(
            interaction_op=Add(),
            descriptor_indices=[1, 2],  # Two indices
            left_descriptor=["A"],  # Only one value - should raise error
            right_descriptor=["B", "C"],  # Correct length
        )

    assert "left_descriptor length" in str(excinfo.value)
    assert "must match descriptor_indices length" in str(excinfo.value)

    # Test that right_descriptor with incorrect length raises a ValueError
    with pytest.raises(ValueError) as excinfo:
        CrossSeriesInteraction(
            interaction_op=Add(),
            descriptor_indices=[1, 2, 3],  # Three indices
            left_descriptor=["A", "B", "C"],  # Correct length
            right_descriptor=["X", "Y"],  # Only two values - should raise error
        )

    assert "right_descriptor length" in str(excinfo.value)
    assert "must match descriptor_indices length" in str(excinfo.value)


def test_cross_series_hash_component():
    op1 = CrossSeriesInteraction(
        interaction_op=Add(),
        descriptor_indices=[1, 2],
        left_descriptor=["A", "B"],
        right_descriptor=["C", "D"],
    )
    assert op1.get_hash_component() == "cross_add_1_2_A_B_C_D"

    op2 = CrossSeriesInteraction(
        interaction_op=Add(),
        descriptor_indices=[1, 2],
        left_descriptor=["A", "C"],
        right_descriptor=["B", "D"],
    )
    assert op2.get_hash_component() == "cross_add_1_2_A_C_B_D"

    feature1 = Feature(op1, [Column("date"), Column("category1"), Column("category2"), Column("value")])
    feature1.set_display_index(feature1.get_hash())

    feature2 = Feature(op2, [Column("date"), Column("category1"), Column("category2"), Column("value")])
    feature2.set_display_index(feature2.get_hash())

    assert feature1.get_hash() != feature2.get_hash()
    assert feature1.get_display_name() != feature2.get_display_name()


def test_roll_on_different_cross_features():
    # Create two different cross series features
    cross_op1 = CrossSeriesInteraction(
        interaction_op=Add(),
        descriptor_indices=[1],
        left_descriptor=["A"],
        right_descriptor=["B"],
    )

    cross_op2 = CrossSeriesInteraction(
        interaction_op=Multiply(),  # Different operation
        descriptor_indices=[1],
        left_descriptor=["A"],
        right_descriptor=["B"],
    )

    # Create features with these operators
    cross_feature1 = Feature(op=cross_op1, children=[Column("date"), Column("category"), Column("value")])

    cross_feature2 = Feature(op=cross_op2, children=[Column("date"), Column("category"), Column("value")])

    # Create Roll features that use these cross features as input
    # The same Roll operation is applied to both
    from upgini.autofe.timeseries import Roll

    roll_op = Roll(window_size=7, window_unit="D", aggregation="mean")

    # Create Roll features using the cross features as input
    roll_feature1 = Feature(op=roll_op, children=[cross_feature1])
    roll_feature1.set_display_index(roll_feature1.get_hash())

    roll_feature2 = Feature(op=roll_op, children=[cross_feature2])
    roll_feature2.set_display_index(roll_feature2.get_hash())

    # Verify that the display names are different
    assert roll_feature1.get_hash() != roll_feature2.get_hash()
    assert roll_feature1.get_display_name() != roll_feature2.get_display_name()

    # Now test with same operation but different descriptors
    cross_op3 = CrossSeriesInteraction(
        interaction_op=Add(),
        descriptor_indices=[1],
        left_descriptor=["A"],
        right_descriptor=["B"],
    )

    cross_op4 = CrossSeriesInteraction(
        interaction_op=Add(),
        descriptor_indices=[1],
        left_descriptor=["C"],  # Different descriptor
        right_descriptor=["D"],  # Different descriptor
    )

    cross_feature3 = Feature(op=cross_op3, children=[Column("date"), Column("category"), Column("value")])

    cross_feature4 = Feature(op=cross_op4, children=[Column("date"), Column("category"), Column("value")])

    roll_feature3 = Feature(op=roll_op, children=[cross_feature3])
    roll_feature3.set_display_index(roll_feature3.get_hash())

    roll_feature4 = Feature(op=roll_op, children=[cross_feature4])
    roll_feature4.set_display_index(roll_feature4.get_hash())

    assert roll_feature3.get_hash() != roll_feature4.get_hash()
    assert roll_feature3.get_display_name() != roll_feature4.get_display_name()


def test_cross_series_interaction_parse_obj():
    add_op = find_op("+")
    assert add_op is not None

    cross = CrossSeriesInteraction(
        interaction_op=add_op, descriptor_indices=[0], left_descriptor=["temperature"], right_descriptor=["humidity"]
    )

    cross_dict = cross.get_params()
    parsed_cross = CrossSeriesInteraction.parse_obj(cross_dict)

    assert parsed_cross.interaction_op.name == add_op.name
    assert parsed_cross.descriptor_indices == [0]
    assert parsed_cross.left_descriptor == ["temperature"]
    assert parsed_cross.right_descriptor == ["humidity"]

    assert cross.to_formula() == parsed_cross.to_formula()
    assert cross.get_hash_component() == parsed_cross.get_hash_component()
