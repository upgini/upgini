from typing import Optional

import pandas as pd

from upgini.autofe.operand import PandasOperand, VectorizableMixin


class GroupByThenAgg(PandasOperand, VectorizableMixin):
    agg: Optional[str]
    is_vectorizable: bool = True
    is_grouping: bool = True
    is_distribution_dependent: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        temp = left.groupby(right).agg(self.agg)
        return self._loc(right, temp)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]
        temp = d1.groupby(d2).agg(self.agg)
        return temp.merge(d2, how="right", on=[group_column])[value_columns]


class GroupByThenMedian(GroupByThenAgg):
    name: str = "GroupByThenMedian"
    pandas_agg: str = "median"
    is_distribution_dependent: bool = True


class GroupByThenRank(PandasOperand, VectorizableMixin):
    name: str = "GroupByThenRank"
    is_vectorizable: bool = True
    is_grouping: bool = True
    output_type: Optional[str] = "float"
    is_distribution_dependent: bool = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        temp = pd.DataFrame(left[~right.isna()].groupby(right).rank(ascending=True, pct=True)).reset_index()
        return temp.merge(pd.DataFrame(right).reset_index(), how="right", on=["index"])[left.name]

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]
        temp = d1[~d2.isna()].groupby(d2).rank(ascending=True, pct=True)[value_columns].reset_index()
        return temp.merge(d2.reset_index(), how="right", on=["index"])[value_columns]


class GroupByThenNUnique(PandasOperand, VectorizableMixin):
    name: str = "GroupByThenNUnique"
    is_vectorizable: bool = True
    is_grouping: bool = True
    output_type: Optional[str] = "int"
    is_distribution_dependent: bool = True
    input_type: Optional[str] = "discrete"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        nunique = left.groupby(right).nunique()
        return self._loc(right, nunique)

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]
        nunique = d1.groupby(d2).nunique()
        return nunique.merge(d2, how="right", on=[group_column])[value_columns]


class GroupByThenFreq(PandasOperand):
    name: str = "GroupByThenFreq"
    is_grouping: bool = True
    output_type: Optional[str] = "float"
    is_distribution_dependent: bool = True
    input_type: Optional[str] = "discrete"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        def _f(x):
            value_counts = x.value_counts(normalize=True)
            return self._loc(x, value_counts)

        freq = left.groupby(right).apply(_f)
        return pd.Series(freq, index=right.index)
