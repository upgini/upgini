import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from upgini.autofe.operator import PandasOperator, VectorizableMixin
from upgini.autofe.utils import pydantic_validator


class Abs(PandasOperator, VectorizableMixin):
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


class Log(PandasOperator, VectorizableMixin):
    name: str = "log"
    is_unary: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "float"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(np.log(np.abs(data.replace(0, np.nan))), 10)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(np.log(data.replace(0, np.nan).abs()), 10)


class Sqrt(PandasOperator, VectorizableMixin):
    name: str = "sqrt"
    is_unary: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "float"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(np.sqrt(np.abs(data)))

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(np.sqrt(data.abs()))


class Square(PandasOperator, VectorizableMixin):
    name: str = "square"
    is_unary: bool = True
    is_vectorizable: bool = True
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return np.square(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.square(data)


class Sigmoid(PandasOperator, VectorizableMixin):
    name: str = "sigmoid"
    is_unary: bool = True
    is_vectorizable: bool = True
    output_type: Optional[str] = "float"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return self._round_value(1 / (1 + np.exp(-data)))

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._round_value(1 / (1 + np.exp(-data)))


class Floor(PandasOperator, VectorizableMixin):
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


class Residual(PandasOperator, VectorizableMixin):
    name: str = "residual"
    is_unary: bool = True
    is_vectorizable: bool = True
    input_type: Optional[str] = "continuous"
    group_index: int = 0

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return data - np.floor(data)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data - np.floor(data)


class Freq(PandasOperator):
    name: str = "freq"
    is_unary: bool = True
    output_type: Optional[str] = "float"
    is_distribution_dependent: bool = True
    input_type: Optional[str] = "discrete"

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        value_counts = data.value_counts(normalize=True)
        return self._loc(data, value_counts)


class Norm(PandasOperator):
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


class Embeddings(PandasOperator):
    name: str = "emb"
    is_unary: bool = True
    input_type: Optional[str] = "string"
    output_type: Optional[str] = "vector"


class Bin(PandasOperator):
    name: str = "bin"
    is_unary: bool = True
    output_type: Optional[str] = "category"
    bin_bounds: List[int] = []
    is_categorical: bool = True

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        return data.apply(self._bin, bounds=self.bin_bounds).fillna(-1).astype(int).astype("category")

    def _bin(self, f, bounds):
        if f is None or np.isnan(f):
            return np.nan
        hit = np.where(f >= np.array(bounds))[0]
        if hit.size > 0:
            return np.max(hit) + 1
        else:
            return np.nan

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "bin_bounds": json.dumps(self.bin_bounds),
            }
        )
        return res

    @pydantic_validator("bin_bounds", mode="before")
    def parse_bin_bounds(cls, value):
        if isinstance(value, str):
            return json.loads(value)
        return value
