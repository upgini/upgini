import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from upgini.autofe.binary import Distance
from upgini.autofe.feature import Column, Feature
from upgini.autofe.operand import CalculationContext, OperandKind, OperandValue, wrap_operand
from upgini.autofe.unary import Embeddings

_EMB_VECTORS = {
    "x": np.array([1.0, 0.0]),
    "y": np.array([0.0, 1.0]),
}


def _fake_emb_unary(self, data: OperandValue) -> OperandValue:
    series = data.as_series()
    matrix = np.stack([_EMB_VECTORS.get(v, np.array([np.nan, np.nan])) for v in series])
    return OperandValue.from_matrix(matrix, index=data.index, source=data.source)


def _fake_emb_unary_ndarray(self, data: OperandValue) -> np.ndarray:
    series = data.as_series()
    return np.stack([_EMB_VECTORS.get(v, np.array([np.nan, np.nan])) for v in series])


def test_wrap_operand_matrix_ndarray():
    index = pd.RangeIndex(2)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    operand = wrap_operand(matrix, index=index, source="a")

    assert operand.kind == OperandKind.MATRIX
    np.testing.assert_array_equal(operand.as_matrix(), matrix)


def test_wrap_operand_ndarray_requires_index():
    with pytest.raises(TypeError, match="index is required"):
        wrap_operand(np.array([[1.0, 0.0]]))


def test_emb_produces_matrix_operand(monkeypatch):
    monkeypatch.setattr(Embeddings, "calculate_unary", _fake_emb_unary)

    ctx = CalculationContext.from_dataframe(pd.DataFrame({"a": ["x"]}))
    emb_feature = Feature.from_formula("emb(a)")
    operand = emb_feature._eval(ctx)

    assert operand.kind == OperandKind.MATRIX
    np.testing.assert_array_equal(operand.as_matrix(), np.array([[1.0, 0.0]]))


def test_dist_emb_pipeline(monkeypatch):
    monkeypatch.setattr(Embeddings, "calculate_unary", _fake_emb_unary)

    data = pd.DataFrame({"a": ["x", "y"], "b": ["y", "x"]})
    feature = Feature.from_formula("dist(emb(a),emb(b))")
    result = feature.calculate(data)

    expected = pd.Series([1.0, 1.0], dtype=np.float64)
    assert_series_equal(result, expected, atol=1e-6)


def test_dist_emb_pipeline_with_ndarray_return(monkeypatch):
    monkeypatch.setattr(Embeddings, "calculate_unary", _fake_emb_unary_ndarray)

    data = pd.DataFrame({"a": ["x", "y"], "b": ["y", "x"]})
    feature = Feature.from_formula("dist(emb(a),emb(b))")
    result = feature.calculate(data)

    expected = pd.Series([1.0, 1.0], dtype=np.float64)
    assert_series_equal(result, expected, atol=1e-6)


def test_column_preserve_kind_matrix_source():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    ctx = CalculationContext.from_matrices({"emb_a": matrix})

    operand = Column("emb_a").calculate(ctx, preserve_kind=True)

    assert isinstance(operand, OperandValue)
    assert operand.kind == OperandKind.MATRIX
    np.testing.assert_array_equal(operand.as_matrix(), matrix)


def test_column_default_returns_series_for_matrix_source():
    matrix = np.array([[1.0, 2.0]])
    ctx = CalculationContext.from_matrices({"emb_a": matrix})

    result = Column("emb_a").calculate(ctx)

    assert isinstance(result, pd.Series)
    np.testing.assert_array_equal(result.iloc[0], np.array([1.0, 2.0]))


def test_feature_preserve_kind_returns_matrix(monkeypatch):
    monkeypatch.setattr(Embeddings, "calculate_unary", _fake_emb_unary)

    data = pd.DataFrame({"a": ["x", "y"]})
    feature = Feature.from_formula("emb(a)")
    result = feature.calculate(data, preserve_kind=True)

    assert isinstance(result, OperandValue)
    assert result.kind == OperandKind.MATRIX
    np.testing.assert_array_equal(result.as_matrix(), np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_column_preserve_kind_returns_matrix_unchanged():
    matrix = np.array([[1.0, np.inf], [np.nan, 2.0]])
    ctx = CalculationContext.from_matrices({"emb_a": matrix})

    result = Column("emb_a").calculate(ctx, preserve_kind=True)

    assert isinstance(result, OperandValue)
    assert result.kind == OperandKind.MATRIX
    np.testing.assert_allclose(result.as_matrix(), matrix, rtol=0, atol=0, equal_nan=True)


def test_feature_preserve_kind_replaces_inf_in_matrix(monkeypatch):
    def fake_emb_inf(self, data: OperandValue) -> OperandValue:
        matrix = np.array([[1.0, np.inf]])
        return OperandValue.from_matrix(matrix, index=data.index, source=data.source)

    monkeypatch.setattr(Embeddings, "calculate_unary", fake_emb_inf)

    data = pd.DataFrame({"a": ["x"]})
    result = Feature.from_formula("emb(a)").calculate(data, preserve_kind=True)

    assert isinstance(result, OperandValue)
    expected = np.array([[1.0, np.nan]])
    np.testing.assert_allclose(result.as_matrix(), expected, rtol=0, atol=0, equal_nan=True)


def test_dist_with_precomputed_matrix_columns(monkeypatch):
    monkeypatch.setattr(Embeddings, "calculate_unary", _fake_emb_unary)

    matrices = {
        "emb_a": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "emb_b": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }
    ctx = CalculationContext.from_matrices(matrices)
    feature = Feature.from_formula("dist(emb_a,emb_b)")
    result = feature.calculate(ctx)

    expected = pd.Series([1.0, 1.0], dtype=np.float64)
    assert_series_equal(result, expected, atol=1e-6)
