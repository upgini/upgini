import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from upgini.autofe.binary import Distance
from upgini.autofe.feature import Feature
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


def test_embeddings_not_implemented():
    data = pd.DataFrame({"a": ["x"]})
    feature = Feature.from_formula("emb(a)")

    with pytest.raises(NotImplementedError, match="Embeddings operator is not implemented"):
        feature.calculate(data)
