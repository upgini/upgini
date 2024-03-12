import numpy as np
import pandas as pd

from upgini.autofe.operand import PandasOperand, VectorizableMixin


class DateDiff(PandasOperand, VectorizableMixin):
    name = "date_diff"
    is_binary = True
    has_symmetry_importance = True
    is_vectorizable = True
    unit: str = "D"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        return self.__replace_negative((left - right) / np.timedelta64(1, self.unit))

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        group_column, value_columns = self.validate_calculation(data.columns, **kwargs)
        d1 = data[value_columns]
        d2 = data[group_column]

        return self.__replace_negative(d1.sub(d2, axis=0) / np.timedelta64(1, self.unit))

    def __replace_negative(self, df):
        df[df < 0] = None
        return df


class DateDiffFuture(PandasOperand):
    name = "date_diff_future"
    is_binary = True
    has_symmetry_importance = True
    is_vectorizable = False
    unit: str = "D"

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        future = pd.to_datetime(dict(day=right.dt.day, month=right.dt.month, year=left.dt.year))
        before = future[future < left]
        future[future < left] = pd.to_datetime(dict(day=before.dt.day, month=before.dt.month, year=before.dt.year + 1))
        diff = (future - left) / np.timedelta64(1, self.unit)

        return diff
