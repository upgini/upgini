import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Feature, Column
from upgini.autofe.timeseries import Delta
from upgini.autofe.timeseries.delta import Delta2


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
