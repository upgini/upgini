from typing import Dict, List, Optional

import pandas as pd

from upgini.autofe.operator import OperatorRegistry, PandasOperator, VectorizableMixin


class Mean(PandasOperator, VectorizableMixin):
    name: str = "mean"
    output_type: Optional[str] = "float"
    is_vector: bool = True
    group_index: int = 0

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        return pd.DataFrame(data).T.fillna(0).mean(axis=1)


class Sum(PandasOperator, VectorizableMixin):
    name: str = "sum"
    is_vector: bool = True
    group_index: int = 0

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        return pd.DataFrame(data).T.fillna(0).sum(axis=1)


class Vectorize(PandasOperator, VectorizableMixin):
    name: str = "vectorize"
    is_vector: bool = True
    group_index: int = 0

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        return pd.DataFrame(data).T.apply(lambda x: x.to_list(), axis=1)


class OnnxModel(PandasOperator, metaclass=OperatorRegistry):
    name: str = "onnx"
    is_vector: bool = True
    output_type: Optional[str] = "float"
    model_name: str = ""

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "model_name": self.model_name,
            }
        )
        return res
