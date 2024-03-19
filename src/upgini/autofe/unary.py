from upgini.autofe.operand import PandasOperand, VectorizableMixin
import numpy as np
import pandas as pd


class Abs(PandasOperand, VectorizableMixin):
    name = "abs"
    is_unary = True
    is_vectorizable = True
    group_index = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return data.abs()

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data.abs()


class Log(PandasOperand, VectorizableMixin):
    name = "log"
    is_unary = True
    is_vectorizable = True
    output_type = "float"
    group_index = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(np.log(np.abs(data.replace(0, np.nan))), 10)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(np.log(data.replace(0, np.nan).abs()), 10)


class Sqrt(PandasOperand, VectorizableMixin):
    name = "sqrt"
    is_unary = True
    is_vectorizable = True
    output_type = "float"
    group_index = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(np.sqrt(np.abs(data)))

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(np.sqrt(data.abs()))


class Square(PandasOperand, VectorizableMixin):
    name = "square"
    is_unary = True
    is_vectorizable = True
    group_index = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return np.square(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.square(data)


class Sigmoid(PandasOperand, VectorizableMixin):
    name = "sigmoid"
    is_unary = True
    is_vectorizable = True
    output_type = "float"
    group_index = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(1 / (1 + np.exp(-data)))

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(1 / (1 + np.exp(-data)))


class Floor(PandasOperand, VectorizableMixin):
    name = "floor"
    is_unary = True
    is_vectorizable = True
    output_type = "int"
    input_type = "continuous"
    group_index = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return np.floor(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.floor(data)


class Residual(PandasOperand, VectorizableMixin):
    name = "residual"
    is_unary = True
    is_vectorizable = True
    input_type = "continuous"
    group_index = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return data - np.floor(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data - np.floor(data)


class Freq(PandasOperand):
    name = "freq"
    is_unary = True
    output_type = "float"
    is_distribution_dependent = True
    input_type = "discrete"

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        value_counts = data.value_counts(normalize=True)
        return self._loc(data, value_counts)
