from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import abc
import pandas as pd
import numpy as np


class Operand(BaseModel):
    name: str
    alias: Optional[str]
    is_unary: bool = False
    has_symmetry_importance: bool = False
    input_type: Optional[str]
    output_type: Optional[str]
    is_categorical: bool = False
    is_vectorizable: bool = False
    is_grouping: bool = False
    is_binary: bool = False
    is_vector: bool = False
    is_distribution_dependent: bool = False
    params: Optional[Dict[str, str]]

    def set_params(self, params: Dict[str, str]):
        self.params = params
        return self

    def get_params(self) -> Dict[str, str]:
        return self.params


MAIN_COLUMN = "main_column"


class PandasOperand(Operand, abc.ABC):
    def calculate(self, **kwargs) -> pd.Series:
        if self.is_unary:
            return self.calculate_unary(kwargs["data"])
        elif self.is_binary or self.is_grouping:
            return self.calculate_binary(kwargs["left"], kwargs["right"])
        else:
            return self.calculate_vector(kwargs["data"])

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        pass

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        pass

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        pass

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not self.is_vectorizable:
            raise RuntimeError(f"Cannot apply calculate_group: operator {self.name} is not vectorizable")
        else:
            raise RuntimeError(f"Unimplemented calculate_group for operator {self.name}")

    def _loc(self, df_to, df_from):
        df_from.loc[np.nan] = np.nan
        return df_to.fillna(np.nan).apply(lambda x: df_from.loc[x])


class VectorizableMixin(Operand):
    def validate_calculation(self, input_columns: List[str], **kwargs) -> Tuple[str, List[str]]:
        if not kwargs.get(MAIN_COLUMN):
            raise ValueError(f"Expected argument {MAIN_COLUMN} for grouping operator {self.name} not found")
        group_column = kwargs[MAIN_COLUMN]
        value_columns = [col for col in input_columns if col != group_column]

        return group_column, value_columns
