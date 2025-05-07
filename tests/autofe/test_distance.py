import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.binary import Distance, Sim
from upgini.autofe.feature import Feature
from upgini.autofe.utils import pydantic_parse_method


def test_distance_calculation():
    data = pd.DataFrame(
        {
            "a": [[1, 0], [0, 1], [1, 1], [0, 0], [None], [3, 4], [3, 4], [None]],
            "b": [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [6, 8], [None], [None]],
        }
    )

    # Expected results:
    # 1. Perpendicular vectors (distance = 1)
    # 2. Perpendicular vectors (distance = 1)
    # 3. Same direction vectors (distance = 0)
    # 4. Zero vectors (undefined, should be None)
    # 5. None in left (should be None)
    # 6. Different vectors but same direction (distance = 0)
    # 7. None in right (should be None)
    # 8. None in both (should be None)
    expected = pd.Series([1.0, 1.0, 0.0, None, None, 0.0, None, None], dtype=np.float64)

    result = Distance().calculate_binary(data["a"], data["b"])

    # Drop NaN values for comparison as they don't equate correctly
    assert_series_equal(result.dropna().reset_index(drop=True), expected.dropna().reset_index(drop=True), atol=1e-6)
    # Check NaN positions match
    assert result.isna().equals(expected.isna())


def test_distance_sim_relationship():
    data = pd.DataFrame(
        {
            "a": [[1, 2], [3, 4], [5, 6]],
            "b": [[2, 3], [3, 4], [10, 12]],
        }
    )

    distance_results = Distance().calculate_binary(data["a"], data["b"])
    sim_results = Sim().calculate_binary(data["a"], data["b"])

    # Sim should be 1 - Distance
    complementary = 1 - distance_results
    assert_series_equal(complementary, sim_results, atol=1e-6)


def test_distance_symmetry():
    data = pd.DataFrame(
        {
            "a": [[1, 2], [3, 4], [5, 6]],
            "b": [[2, 3], [3, 4], [10, 12]],
        }
    )

    forward_results = Distance().calculate_binary(data["a"], data["b"])
    reverse_results = Distance().calculate_binary(data["b"], data["a"])

    assert_series_equal(forward_results, reverse_results, atol=1e-6)


def test_distance_normalization():
    data = pd.DataFrame(
        {
            "a": [[1, 0], [2, 0], [3, 0]],  # Vectors pointing in same direction with different magnitudes
            "b": [[2, 0], [4, 0], [9, 0]],  # Should all have distance = 0
        }
    )

    results = Distance().calculate_binary(data["a"], data["b"])
    expected = pd.Series([0.0, 0.0, 0.0], dtype=np.float64)

    assert_series_equal(results, expected, atol=1e-6)


def test_distance_parse_obj():
    distance = Distance()

    distance_dict = distance.get_params()
    parsed_distance = pydantic_parse_method(Distance)(distance_dict)

    assert parsed_distance.name == "dist"
    assert parsed_distance.is_binary is True
    assert parsed_distance.output_type == "float"
    assert parsed_distance.is_symmetrical is True
    assert parsed_distance.has_symmetry_importance is True
    assert parsed_distance.to_formula() == "dist"


def test_distance_with_feature():
    df = pd.DataFrame(
        {
            "vector1": [[1, 0], [1, 1]],
            "vector2": [[0, 1], [1, 1]],
        }
    )

    feature = Feature.from_formula("dist(vector1,vector2)")
    result = feature.calculate(df)

    expected = pd.Series([1.0, 0.0], dtype=np.float64)
    assert_series_equal(result, expected, atol=1e-6)


def test_distance_edge_cases():
    data = pd.DataFrame(
        {
            "a": [[], [1], [1, 0, 0], [0, 0, 0]],
            "b": [[], [2], [0, 1, 0], [0, 0, 0]],
        }
    )

    results = Distance().calculate_binary(data["a"], data["b"])

    # Empty arrays should give NaN (since norm is 0)
    # 1D arrays should work fine
    # Higher dimension arrays should work
    # Zero vectors should give NaN
    assert np.isnan(results[0])
    assert results[1] == 0.0  # Same direction, different magnitude
    assert results[2] == 1.0  # Perpendicular vectors
    assert np.isnan(results[3])  # Zero vectors
