import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

from upgini.autofe.operand import PandasOperand, VectorizableMixin


class Min(PandasOperand):
    name = "min"
    is_binary = True
    is_symmetrical = True
    has_symmetry_importance = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return np.minimum(left, right)


class Max(PandasOperand):
    name = "max"
    is_binary = True
    is_symmetrical = True
    has_symmetry_importance = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return np.maximum(left, right)


class Add(PandasOperand, VectorizableMixin):
    name = "+"
    alias = "add"
    is_binary = True
    is_symmetrical = True
    has_symmetry_importance = True
    is_vectorizable = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left + right

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.add(d2, axis=0)


class Subtract(PandasOperand, VectorizableMixin):
    name = "-"
    alias = "sub"
    is_binary = True
    is_symmetrical = True
    has_symmetry_importance = True
    is_vectorizable = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left - right

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.sub(d2, axis=0)


class Multiply(PandasOperand, VectorizableMixin):
    name = "*"
    alias = "mul"
    is_binary = True
    is_symmetrical = True
    has_symmetry_importance = True
    is_vectorizable = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left * right

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.mul(d2, axis=0)


class Divide(PandasOperand, VectorizableMixin):
    name = "/"
    alias = "div"
    is_binary = True
    has_symmetry_importance = True
    is_vectorizable = True
    output_type = "float"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left / right.replace(0, np.nan)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.div(d2.replace(0, np.nan), axis=0)


class Combine(PandasOperand):
    name = "Combine"
    is_binary = True
    has_symmetry_importance = True
    output_type = "object"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        temp = left.astype(str) + "_" + right.astype(str)
        temp[left.isna() | right.isna()] = np.nan
        return pd.Series(temp, index=left.index)


class CombineThenFreq(PandasOperand):
    name = "CombineThenFreq"
    is_binary = True
    is_symmetrical = True
    has_symmetry_importance = True
    output_type = "float"
    is_distribution_dependent = True
    input_type = "discrete"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        temp = left.astype(str) + "_" + right.astype(str)
        temp[left.isna() | right.isna()] = np.nan
        value_counts = temp.value_counts(normalize=True)
        self._loc(temp, value_counts)


class Sim(PandasOperand):
    name = "sim"
    is_binary = True
    output_type = "float"
    is_symmetrical = True
    has_symmetry_importance = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return dot(left, right) / (norm(left) * norm(right))
