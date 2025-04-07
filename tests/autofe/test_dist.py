import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.binary import (
    Distance,
)


def test_distance():
    data = pd.DataFrame(
        [
            [np.array([0, 1, 0]), np.array([0, 1, 0])],
            [[0, 1, 0], [0, 1, 0]],
            [np.array([0, 1, 0]), np.array([1, 1, 0])],
            [np.array([0, 1, 0]), np.array([1, 0, 0])],
            [np.array([0, 1, 0]), None],
            [None, np.array([1, 0, 0])],
            [None, None],
        ],
        columns=["v1", "v2"],
    )

    op = Distance()

    expected_values = pd.Series([0.0, 0.0, 0.292893, 1.0, np.nan, np.nan, np.nan])
    actual_values = op.calculate_binary(data.v1, data.v2)
    print(actual_values)

    assert_series_equal(actual_values, expected_values)


def test_empty_distance():
    data = pd.DataFrame({"v1": [None], "v2": [None]})

    op = Distance()

    expected_values = pd.Series([np.nan])
    actual_values = op.calculate_binary(data.v1, data.v2)
    print(actual_values)

    assert_series_equal(actual_values, expected_values)


def test_distance_parse_obj():
    distance = Distance()

    distance_dict = distance.get_params()
    parsed_distance = Distance.parse_obj(distance_dict)

    assert parsed_distance.name == "dist"
    assert parsed_distance.is_binary is True
    assert parsed_distance.to_formula() == "dist"
