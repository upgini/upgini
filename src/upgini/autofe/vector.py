from typing import List

import pandas as pd

from upgini.autofe.operand import PandasOperand, VectorizableMixin


class Mean(PandasOperand, VectorizableMixin):
    name = "mean"
    output_type = "float"
    is_vector = True
    group_index = 0

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        return pd.DataFrame(data).T.fillna(0).mean(axis=1)


class Sum(PandasOperand, VectorizableMixin):
    name = "sum"
    is_vector = True
    group_index = 0

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        return pd.DataFrame(data).T.fillna(0).sum(axis=1)
