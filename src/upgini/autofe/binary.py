import abc
from typing import Optional
import Levenshtein
import numpy as np
import pandas as pd
from jarowinkler import jarowinkler_similarity

from upgini.autofe.operand import PandasOperand, VectorizableMixin


class Min(PandasOperand):
    name: str = "min"
    is_binary: bool = True
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return np.minimum(left, right)


class Max(PandasOperand):
    name: str = "max"
    is_binary: bool = True
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return np.maximum(left, right)


class Add(PandasOperand, VectorizableMixin):
    name: str = "+"
    alias: str = "add"
    is_binary: bool = True
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True
    is_vectorizable: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left + right

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.add(d2, axis=0)


class Subtract(PandasOperand, VectorizableMixin):
    name: str = "-"
    alias: str = "sub"
    is_binary: bool = True
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True
    is_vectorizable: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left - right

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.sub(d2, axis=0)


class Multiply(PandasOperand, VectorizableMixin):
    name: str = "*"
    alias: str = "mul"
    is_binary: bool = True
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True
    is_vectorizable: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left * right

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.mul(d2, axis=0)


class Divide(PandasOperand, VectorizableMixin):
    name: str = "/"
    alias: str = "div"
    is_binary: bool = True
    has_symmetry_importance: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "float"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return left / right.replace(0, np.nan)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return d1.div(d2.replace(0, np.nan), axis=0)


class Combine(PandasOperand):
    name: str = "Combine"
    is_binary: bool = True
    has_symmetry_importance: bool = True
    output_type: Optional[str] = "object"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        temp = left.astype(str) + "_" + right.astype(str)
        temp[left.isna() | right.isna()] = np.nan
        return pd.Series(temp, index=left.index)


class CombineThenFreq(PandasOperand):
    name: str = "CombineThenFreq"
    is_binary: bool = True
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True
    output_type: Optional[str] = "float"
    is_distribution_dependent: bool = True
    input_type: Optional[str] = "discrete"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        temp = left.astype(str) + "_" + right.astype(str)
        temp[left.isna() | right.isna()] = np.nan
        value_counts = temp.value_counts(normalize=True)
        self._loc(temp, value_counts)


class Distance(PandasOperand):
    name: str = "dist"
    is_binary: bool = True
    output_type: Optional[str] = "float"
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return pd.Series(
            1 - self.__dot(left, right) / (self.__norm(left) * self.__norm(right)), index=left.index
        )

    # row-wise dot product
    def __dot(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = left.apply(lambda x: np.array(x))
        right = right.apply(lambda x: np.array(x))
        res = (left.dropna() * right.dropna()).apply(np.sum)
        res = res.reindex(left.index.union(right.index))
        return res

    def __norm(self, vector: pd.Series) -> pd.Series:
        return np.sqrt(self.__dot(vector, vector))


# Left for backward compatibility
class Sim(Distance):
    name: str = "sim"
    is_binary: bool = True
    output_type: Optional[str] = "float"
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return 1 - super().calculate_binary(left, right)


class StringSim(PandasOperand, abc.ABC):
    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        sims = []
        for i in left.index:
            left_i = self._prepare_value(left.get(i))
            right_i = self._prepare_value(right.get(i))
            if left_i is not None and right_i is not None:
                sims.append(self._similarity(left_i, right_i))
            else:
                sims.append(None)

        return pd.Series(sims, index=left.index)

    @abc.abstractmethod
    def _prepare_value(self, value: Optional[str]) -> Optional[str]:
        pass

    @abc.abstractmethod
    def _similarity(self, left: str, right: str) -> float:
        pass


class JaroWinklerSim1(StringSim):
    name: str = "sim_jw1"
    is_binary: bool = True
    input_type: Optional[str] = "string"
    output_type: Optional[str] = "float"
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True

    def _prepare_value(self, value: Optional[str]) -> Optional[str]:
        return value

    def _similarity(self, left: str, right: str) -> float:
        return jarowinkler_similarity(left, right)


class JaroWinklerSim2(StringSim):
    name: str = "sim_jw2"
    is_binary: bool = True
    input_type: Optional[str] = "string"
    output_type: Optional[str] = "float"
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True

    def _prepare_value(self, value: Optional[str]) -> Optional[str]:
        return value[::-1] if value is not None else None

    def _similarity(self, left: str, right: str) -> float:
        return jarowinkler_similarity(left, right)


class LevenshteinSim(StringSim):
    name: str = "sim_lv"
    is_binary: bool = True
    input_type: Optional[str] = "string"
    output_type: Optional[str] = "float"
    is_symmetrical: bool = True
    has_symmetry_importance: bool = True

    def _prepare_value(self, value: Optional[str]) -> Optional[str]:
        return value

    def _similarity(self, left: str, right: str) -> float:
        return 1 - Levenshtein.distance(left, right) / max(len(left), len(right))
