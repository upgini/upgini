import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Feature, Column
from upgini.autofe.timeseries import Delta, Delta2


def test_delta():
    df = pd.DataFrame(
        {
            "date": ["2024-05-05", "2024-05-06", "2024-05-07", "2024-05-09"],
            "value": [1, 2, 3, 5],
        },
    )

    def check_delta(delta_size: int, delta_unit: str, expected_values: list[float]):
        feature = Feature(
            op=Delta(delta_size=delta_size, delta_unit=delta_unit),
            children=[Column("date"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    check_delta(1, "d", [np.nan, 1.0, 1.0, np.nan])
    check_delta(2, "d", [np.nan, np.nan, 2.0, 2.0])


def test_delta_groups():
    df = pd.DataFrame(
        {
            "date": ["2024-05-06", "2024-05-06", "---", "2024-05-07", "2024-05-07", "2024-05-07"],
            "f1": ["a", "b", "a", "a", "a", "b"],
            "f2": [1, 2, 1, 1, 1, 2],
            "value": [1, 1, 3, 4, 4, 5],
        },
        index=[9, 8, 7, 6, 5, 4],
    )

    def check_delta(delta_size: int, delta_unit: str, expected_values: list[float]):
        feature = Feature(
            op=Delta(delta_size=delta_size, delta_unit=delta_unit),
            children=[Column("date"), Column("f1"), Column("f2"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value", index=df.index)
        assert_series_equal(feature.calculate(df), expected_res)

    check_delta(1, "d", [np.nan, np.nan, np.nan, 3.0, 3.0, 4.0])
    check_delta(2, "d", [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def test_delta2():
    df = pd.DataFrame(
        {
            "date": ["2024-05-05", "2024-05-06", "2024-05-07", "2024-05-08", "2024-05-09"],
            "value": [1, 2, 4, 7, 11],
        },
    )

    def check_delta2(delta_size: int, delta_unit: str, expected_values: list[float]):
        feature = Feature(
            op=Delta2(delta_size=delta_size, delta_unit=delta_unit),
            children=[Column("date"), Column("value")],
        )
        expected_res = pd.Series(expected_values, name="value")
        assert_series_equal(feature.calculate(df), expected_res)

    # First delta: [np.nan, 1.0, 2.0, 3.0, 4.0]
    # Second delta: [np.nan, np.nan, 1.0, 1.0, 1.0]
    check_delta2(1, "d", [np.nan, np.nan, 1.0, 1.0, 1.0])

    # First delta: [np.nan, np.nan, 3.0, 5.0, 7.0]
    # Second delta: [np.nan, np.nan, np.nan, np.nan, 4.0]
    check_delta2(2, "d", [np.nan, np.nan, np.nan, np.nan, 4.0])


def test_delta_from_formula():
    delta = Delta.from_formula("delta_3d")
    assert delta.delta_size == 3
    assert delta.delta_unit == "d"
    assert delta.to_formula() == "delta_3d"

    # Test invalid formulas
    delta = Delta.from_formula("not_a_delta_formula")
    assert delta is None

    delta = Delta.from_formula("delta_abc")
    assert delta is None

    # Test that constructed name matches formula pattern
    delta = Delta(delta_size=5, delta_unit="D")
    assert delta.to_formula() == "delta_5D"


def test_delta2_from_formula():
    delta2 = Delta2.from_formula("delta2_3d")
    assert delta2.delta_size == 3
    assert delta2.delta_unit == "d"
    assert delta2.to_formula() == "delta2_3d"

    # Test invalid formulas
    delta2 = Delta2.from_formula("not_a_delta2_formula")
    assert delta2 is None

    delta2 = Delta2.from_formula("delta2_abc")
    assert delta2 is None

    # Test that constructed name matches formula pattern
    delta2 = Delta2(delta_size=5, delta_unit="D")
    assert delta2.to_formula() == "delta2_5D"


def test_delta_formula():
    # Test to_formula
    op = Delta(delta_size=5, delta_unit="D")
    assert op.to_formula() == "delta_5D"

    # Test from_formula
    op_from_formula = Delta.from_formula("delta_10D")
    assert op_from_formula is not None
    assert op_from_formula.delta_size == 10
    assert op_from_formula.delta_unit == "D"

    # Test lowercase frequency
    op_hour = Delta(delta_size=3, delta_unit="h")
    assert op_hour.to_formula() == "delta_3h"

    op_from_hour = Delta.from_formula("delta_3h")
    assert op_from_hour is not None
    assert op_from_hour.delta_size == 3
    assert op_from_hour.delta_unit == "h"

    # Test with offset
    op_with_offset = Delta(delta_size=5, delta_unit="D", offset_size=1, offset_unit="D")
    assert op_with_offset.to_formula() == "delta_5D_offset_1D"

    op_from_offset_formula = Delta.from_formula("delta_7D_offset_2D")
    assert op_from_offset_formula is not None
    assert op_from_offset_formula.delta_size == 7
    assert op_from_offset_formula.delta_unit == "D"
    assert op_from_offset_formula.offset_size == 2
    assert op_from_offset_formula.offset_unit == "D"

    # Test offset with lowercase frequency
    op_offset_hour = Delta(delta_size=4, delta_unit="h", offset_size=3, offset_unit="h")
    assert op_offset_hour.to_formula() == "delta_4h_offset_3h"

    op_from_offset_hour = Delta.from_formula("delta_4h_offset_3h")
    assert op_from_offset_hour is not None
    assert op_from_offset_hour.delta_size == 4
    assert op_from_offset_hour.delta_unit == "h"
    assert op_from_offset_hour.offset_size == 3
    assert op_from_offset_hour.offset_unit == "h"

    # Test invalid formula
    invalid_op = Delta.from_formula("delta")
    assert invalid_op is None


def test_delta2_formula():
    # Test to_formula
    op = Delta2(delta_size=5, delta_unit="D")
    assert op.to_formula() == "delta2_5D"

    # Test from_formula
    op_from_formula = Delta2.from_formula("delta2_10D")
    assert op_from_formula is not None
    assert op_from_formula.delta_size == 10
    assert op_from_formula.delta_unit == "D"

    # Test lowercase frequency
    op_hour = Delta2(delta_size=3, delta_unit="h")
    assert op_hour.to_formula() == "delta2_3h"

    op_from_hour = Delta2.from_formula("delta2_3h")
    assert op_from_hour is not None
    assert op_from_hour.delta_size == 3
    assert op_from_hour.delta_unit == "h"

    # Test with offset
    op_with_offset = Delta2(delta_size=5, delta_unit="D", offset_size=1, offset_unit="D")
    assert op_with_offset.to_formula() == "delta2_5D_offset_1D"

    op_from_offset_formula = Delta2.from_formula("delta2_7D_offset_2D")
    assert op_from_offset_formula is not None
    assert op_from_offset_formula.delta_size == 7
    assert op_from_offset_formula.delta_unit == "D"
    assert op_from_offset_formula.offset_size == 2
    assert op_from_offset_formula.offset_unit == "D"

    # Test offset with lowercase frequency
    op_offset_hour = Delta2(delta_size=4, delta_unit="h", offset_size=3, offset_unit="h")
    assert op_offset_hour.to_formula() == "delta2_4h_offset_3h"

    op_from_offset_hour = Delta2.from_formula("delta2_4h_offset_3h")
    assert op_from_offset_hour is not None
    assert op_from_offset_hour.delta_size == 4
    assert op_from_offset_hour.delta_unit == "h"
    assert op_from_offset_hour.offset_size == 3
    assert op_from_offset_hour.offset_unit == "h"

    # Test invalid formula
    invalid_op = Delta2.from_formula("delta2")
    assert invalid_op is None


def test_delta_with_offset():
    df = pd.DataFrame(
        {
            "date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
            "value": [1, 2, 3, 4, 5],
        }
    )

    # No offset
    feature_no_offset = Feature(
        op=Delta(delta_size=1, delta_unit="D"),
        children=[Column("date"), Column("value")],
    )
    result_no_offset = feature_no_offset.calculate(df)

    # With offset
    feature_with_offset = Feature(
        op=Delta(delta_size=1, delta_unit="D", offset_size=1, offset_unit="D"),
        children=[Column("date"), Column("value")],
    )
    result_with_offset = feature_with_offset.calculate(df)

    # The offset should shift the result by one day
    # So the result at position i with offset 1 should be similar to the result at position i-1 without offset
    # We skip the first positions as they might be NaN
    for i in range(2, len(df)):
        assert result_with_offset.iloc[i] == result_no_offset.iloc[i - 1] or (
            pd.isna(result_with_offset.iloc[i]) and pd.isna(result_no_offset.iloc[i - 1])
        )
