from typing import Dict, Optional
import numpy as np
import pandas as pd

from upgini.autofe.operand import PandasOperand, VectorizableMixin


class Abs(PandasOperand, VectorizableMixin):
    name: str = "abs"
    is_unary: bool = True
    is_vectorizable: bool = True
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return data.astype(np.float64).abs()
        # return data.abs()

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data.astype(np.float64).abs()
        # return data.abs()


class Log(PandasOperand, VectorizableMixin):
    name: str = "log"
    is_unary: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "float"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(np.log(np.abs(data.replace(0, np.nan))), 10)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(np.log(data.replace(0, np.nan).abs()), 10)


class Sqrt(PandasOperand, VectorizableMixin):
    name: str = "sqrt"
    is_unary: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "float"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(np.sqrt(np.abs(data)))

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(np.sqrt(data.abs()))


class Square(PandasOperand, VectorizableMixin):
    name: str = "square"
    is_unary: bool = True
    is_vectorizable: bool = True
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return np.square(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.square(data)


class Sigmoid(PandasOperand, VectorizableMixin):
    name: str = "sigmoid"
    is_unary: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "float"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(1 / (1 + np.exp(-data)))

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(1 / (1 + np.exp(-data)))


class Floor(PandasOperand, VectorizableMixin):
    name: str = "floor"
    is_unary: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "int"
    input_type: Optional[str] = "continuous"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return np.floor(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.floor(data)


class Residual(PandasOperand, VectorizableMixin):
    name: str = "residual"
    is_unary: bool = True
    is_vectorizable: bool = True
    input_type: Optional[str] = "continuous"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return data - np.floor(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data - np.floor(data)


class Freq(PandasOperand):
    name: str = "freq"
    is_unary: bool = True
    output_type: Optional[str] = "float"
    is_distribution_dependent: bool = True
    input_type: Optional[str] = "discrete"

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        value_counts = data.value_counts(normalize=True)
        return self._loc(data, value_counts)


class Norm(PandasOperand):
    name: str = "norm"
    is_unary: bool = True
    output_type: Optional[str] = "float"
    norm: Optional[float] = None

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        data_dropna = data.dropna()
        if data_dropna.empty:
            return data

        if self.norm is not None:
            normalized_data = data / self.norm
        else:
            self.norm = np.sqrt(np.sum(data * data))
            normalized_data = data / self.norm

        return normalized_data

    def set_params(self, params: Dict[str, str]):
        super().set_params(params)
        if params is not None and "norm" in params:
            self.norm = float(params["norm"])
        return self

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        if self.norm is not None:
            res["norm"] = self.norm
        return res


class Embeddings(PandasOperand):
    name: str = "emb"
    is_unary: bool = True
    input_type: Optional[str] = "string"
    output_type: Optional[str] = "vector"
