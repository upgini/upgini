from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class OperandKind(str, Enum):
    SERIES = "series"
    MATRIX = "matrix"
    ARRAY = "array"


@dataclass
class OperandValue:
    kind: OperandKind
    data: Union[pd.Series, np.ndarray]
    index: pd.Index
    source: Optional[str] = None

    @classmethod
    def from_series(cls, series: pd.Series, source: Optional[str] = None) -> "OperandValue":
        return cls(kind=OperandKind.SERIES, data=series, index=series.index, source=source)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, index: pd.Index, source: Optional[str] = None) -> "OperandValue":
        return cls(kind=OperandKind.MATRIX, data=matrix, index=index, source=source)

    @classmethod
    def from_array(cls, array: np.ndarray, index: pd.Index, source: Optional[str] = None) -> "OperandValue":
        return cls(kind=OperandKind.ARRAY, data=array, index=index, source=source)

    def as_series(self) -> pd.Series:
        if self.kind == OperandKind.SERIES:
            return self.data  # type: ignore[return-value]
        if self.kind == OperandKind.ARRAY:
            return pd.Series(self.data, index=self.index, dtype=np.float64)
        if self.kind == OperandKind.MATRIX:
            return pd.Series(list(self.as_matrix()), index=self.index)
        raise TypeError(f"Cannot convert operand kind {self.kind.value} to series")

    def as_matrix(self) -> np.ndarray:
        if self.kind != OperandKind.MATRIX:
            raise TypeError(f"Cannot convert operand kind {self.kind.value} to matrix")
        return self.data  # type: ignore[return-value]


@dataclass
class CalculationContext:
    data: Optional[pd.DataFrame] = None
    matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    row_index: Optional[pd.Index] = None

    def __post_init__(self) -> None:
        if self.data is None and not self.matrices:
            raise ValueError("CalculationContext requires data and/or matrices")
        if self.matrices:
            lengths = {name: matrix.shape[0] for name, matrix in self.matrices.items()}
            if len(set(lengths.values())) > 1:
                raise ValueError(f"Matrix row counts mismatch: {lengths}")

    @property
    def index(self) -> pd.Index:
        if self.data is not None:
            return self.data.index
        if self.row_index is not None:
            return self.row_index
        if self.matrices:
            nrows = next(iter(self.matrices.values())).shape[0]
            return pd.RangeIndex(nrows)
        raise ValueError("CalculationContext has no index source")

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "CalculationContext":
        return cls(data=data)

    @classmethod
    def from_matrices(
        cls,
        matrices: Dict[str, np.ndarray],
        index: Optional[pd.Index] = None,
    ) -> "CalculationContext":
        return cls(matrices=matrices, row_index=index)

    def resolve_operand(self, name: str) -> OperandValue:
        matrix = self.matrices.get(name)
        if matrix is not None:
            return OperandValue.from_matrix(matrix, index=self.index, source=name)
        if self.data is not None and name in self.data:
            return OperandValue.from_series(self.data[name], source=name)
        raise KeyError(name)


def finalize_operand(operand: OperandValue) -> OperandValue:
    if operand.kind == OperandKind.MATRIX:
        matrix = np.where(np.isinf(operand.as_matrix()), np.nan, operand.as_matrix())
        return OperandValue.from_matrix(matrix, index=operand.index, source=operand.source)
    if operand.kind == OperandKind.ARRAY:
        array = operand.data  # type: ignore[assignment]
        array = np.where(np.isinf(array), np.nan, array)
        return OperandValue.from_array(array, index=operand.index, source=operand.source)
    new_data = operand.as_series()
    if (str(new_data.dtype) == "category") | (str(new_data.dtype) == "object"):
        return OperandValue.from_series(new_data, source=operand.source)
    return OperandValue.from_series(new_data.replace([-np.inf, np.inf], np.nan), source=operand.source)


def operand_to_output(operand: OperandValue, preserve_kind: bool = False) -> Union[pd.Series, OperandValue]:
    finalized = finalize_operand(operand)
    if preserve_kind:
        return finalized
    return finalized.as_series()


def wrap_operand(
    result: Union[pd.Series, np.ndarray, "OperandValue"],
    source: Optional[str] = None,
    index: Optional[pd.Index] = None,
) -> "OperandValue":
    if isinstance(result, OperandValue):
        return result
    if isinstance(result, np.ndarray):
        if index is None:
            raise TypeError("index is required when wrapping ndarray result")
        if result.ndim == 1:
            return OperandValue.from_array(result, index=index, source=source)
        if result.ndim == 2:
            return OperandValue.from_matrix(result, index=index, source=source)
        raise TypeError(f"Cannot wrap ndarray with {result.ndim} dimensions")
    return OperandValue.from_series(result, source=source)
