import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.binary import Distance, Sim
from upgini.autofe.feature import Column, Feature
from upgini.autofe.operand import CalculationContext, OperandKind, OperandValue
from upgini.autofe.utils import pydantic_parse_method


def _list_column_to_matrix(series: pd.Series) -> np.ndarray:
    values = series.tolist()
    valid = [np.asarray(v, dtype=np.float64) for v in values if v is not None and len(v) > 0]
    if not valid:
        return np.empty((len(series), 0))
    dim = valid[0].shape[0]
    matrix = np.full((len(series), dim), np.nan)
    for i, value in enumerate(values):
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            continue
        matrix[i, : arr.size] = arr
    return matrix


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

    result = Distance().calculate(left=data["a"], right=data["b"]).as_series()

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

    distance_results = Distance().calculate(left=data["a"], right=data["b"]).as_series()
    sim_results = Sim().calculate(left=data["a"], right=data["b"]).as_series()

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

    forward_results = Distance().calculate(left=data["a"], right=data["b"]).as_series()
    reverse_results = Distance().calculate(left=data["b"], right=data["a"]).as_series()

    assert_series_equal(forward_results, reverse_results, atol=1e-6)


def test_distance_normalization():
    data = pd.DataFrame(
        {
            "a": [[1, 0], [2, 0], [3, 0]],  # Vectors pointing in same direction with different magnitudes
            "b": [[2, 0], [4, 0], [9, 0]],  # Should all have distance = 0
        }
    )

    results = Distance().calculate(left=data["a"], right=data["b"]).as_series()
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

    results = Distance().calculate(left=data["a"], right=data["b"]).as_series()

    # Empty arrays should give NaN (since norm is 0)
    # 1D arrays should work fine
    # Higher dimension arrays should work
    # Zero vectors should give NaN
    assert np.isnan(results[0])
    assert results[1] == 0.0  # Same direction, different magnitude
    assert results[2] == 1.0  # Perpendicular vectors
    assert np.isnan(results[3])  # Zero vectors


def test_distance_matrix_matches_series():
    data = pd.DataFrame(
        {
            "a": [[1, 0], [0, 1], [1, 1], [0, 0], [None], [3, 4], [3, 4], [None]],
            "b": [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [6, 8], [None], [None]],
        }
    )
    distance = Distance()
    series_result = distance.calculate(left=data["a"], right=data["b"]).as_series()

    matrices = {
        "a": _list_column_to_matrix(data["a"]),
        "b": _list_column_to_matrix(data["b"]),
    }
    ctx = CalculationContext(data, matrices=matrices)
    matrix_result = distance.calculate(
        left=ctx.resolve_operand("a"),
        right=ctx.resolve_operand("b"),
    ).as_series()

    assert matrix_result.index.equals(series_result.index)
    assert_series_equal(matrix_result, series_result, atol=1e-6)


def test_distance_matrix_with_feature():
    matrices = {
        "vector1": np.array([[1.0, 0.0], [1.0, 1.0]]),
        "vector2": np.array([[0.0, 1.0], [1.0, 1.0]]),
    }
    ctx = CalculationContext.from_matrices(matrices)

    feature = Feature.from_formula("dist(vector1,vector2)")
    result = feature.calculate(ctx)

    expected = pd.Series([1.0, 0.0], dtype=np.float64)
    assert_series_equal(result, expected, atol=1e-6)


def test_distance_matrix_only_context_preserves_index():
    index = pd.Index(["x", "y"], name="id")
    matrices = {
        "vector1": np.array([[1.0, 0.0], [1.0, 1.0]]),
        "vector2": np.array([[0.0, 1.0], [1.0, 1.0]]),
    }
    ctx = CalculationContext.from_matrices(matrices, index=index)

    feature = Feature.from_formula("dist(vector1,vector2)")
    result = feature.calculate(ctx)

    expected = pd.Series([1.0, 0.0], index=index, dtype=np.float64)
    assert_series_equal(result, expected, atol=1e-6)


def test_column_resolve_operand_uses_matrix_source():
    matrix = np.array([[1.0, 2.0]])
    ctx = CalculationContext.from_matrices({"emb": matrix})

    operand = Column("emb")._eval(ctx)

    assert operand.kind == OperandKind.MATRIX
    np.testing.assert_array_equal(operand.as_matrix(), matrix)


def test_distance_series_path_when_only_one_matrix_operand():
    data = pd.DataFrame(
        {
            "a": [[1.0, 0.0], [0.0, 1.0]],
            "b": [[0.0, 1.0], [1.0, 0.0]],
        }
    )
    distance = Distance()
    series_result = distance.calculate(left=data["a"], right=data["b"]).as_series()
    ctx = CalculationContext(data, matrices={"a": _list_column_to_matrix(data["a"])})
    mixed_result = distance.calculate(
        left=ctx.resolve_operand("a"),
        right=ctx.resolve_operand("b"),
    ).as_series()

    assert_series_equal(mixed_result, series_result, atol=1e-6)
