from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

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
    data: pd.DataFrame

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "CalculationContext":
        return cls(data=data)

    def resolve_operand(self, name: str) -> OperandValue:
        return OperandValue.from_series(self.data[name], source=name)


def wrap_operand(
    result: Union[pd.Series, "OperandValue"],
    source: Optional[str] = None,
) -> "OperandValue":
    if isinstance(result, OperandValue):
        return result
    return OperandValue.from_series(result, source=source)
