from typing import List, Optional

import pandas as pd

from upgini.autofe.operator import PandasOperator, VectorizableMixin


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
