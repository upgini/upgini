from typing import List, Optional

import pandas as pd

from upgini.autofe.operator import OperatorRegistry, PandasOperator, ParametrizedOperator, VectorizableMixin


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


class OnnxModel(PandasOperator, ParametrizedOperator, metaclass=OperatorRegistry):
    name: str = "onnx"
    score_name: str = "score"
    is_vector: bool = True
    output_type: Optional[str] = "float"

    def to_formula(self) -> str:
        return f"onnx_{self.score_name}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["OnnxModel"]:
        if "(" in formula:
            return None
        if formula.startswith("onnx_"):
            score_name = formula[len("onnx_"):]
            return cls(score_name=score_name)
        elif formula == "onnx":  # for OperatorRegistry
            return cls()
        return None


class CatboostModel(PandasOperator, ParametrizedOperator, metaclass=OperatorRegistry):
    name: str = "catboost"
    score_name: str = "score"
    is_vector: bool = True
    output_type: Optional[str] = "float"

    def to_formula(self) -> str:
        return f"catboost_{self.score_name}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["CatboostModel"]:
        if "(" in formula:
            return None
        if formula.startswith("catboost_"):
            score_name = formula[len("catboost_"):]
            return cls(score_name=score_name)
        elif formula == "catboost":  # for OperatorRegistry
            return cls()
        return None
